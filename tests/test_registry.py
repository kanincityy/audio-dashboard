"""Dependency resolution, cache keying, and the JSON coercion in between.

The ordering guarantees here are what stop voice activity detection running
four times per file, and what stops the signal-to-noise estimator silently
taking its worse path when the user did ask for VAD.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from audio_dashboard import cache, registry  # noqa: E402


# ------------------------------------------------------------------ resolve


def test_requirements_are_pulled_in_and_ordered_first():
    order = registry.resolve(["density"])
    assert order.index("vad") < order.index("density")


def test_transitive_requirements_are_pulled_in():
    # tempo needs a transcript and VAD; neither was asked for.
    order = registry.resolve(["tempo"])
    assert {"transcript", "vad"} <= set(order)
    assert order.index("transcript") < order.index("tempo")
    assert order.index("vad") < order.index("tempo")


def test_implied_lists_only_the_extras():
    assert set(registry.implied(["density"])) == {"vad"}
    assert registry.implied(["vad"]) == []


def test_a_preference_orders_the_run_when_both_are_selected():
    """SNR is better with VAD, so it must run after it — despite sorting first."""
    order = registry.resolve(["snr", "vad"])
    assert order.index("vad") < order.index("snr")


def test_a_preference_is_never_pulled_in_on_its_own():
    """Selecting SNR alone must not silently run the slow VAD step."""
    assert registry.resolve(["snr"]) == ["snr"]
    assert registry.implied(["snr"]) == []


def test_resolution_is_stable_regardless_of_tick_order():
    assert registry.resolve(["density", "levels"]) == registry.resolve(
        ["levels", "density"]
    )


def test_unknown_analysis_raises():
    with pytest.raises(KeyError):
        registry.resolve(["not_a_real_analysis"])


def test_every_requirement_names_a_registered_analysis():
    for analysis in registry.ANALYSES:
        for name in analysis.requires + analysis.prefers:
            assert name in registry.BY_NAME, f"{analysis.name} -> {name}"


def test_every_analysis_belongs_to_a_declared_group():
    groups = {g for g, _ in registry.GROUPS}
    for analysis in registry.ANALYSES:
        assert analysis.group in groups, analysis.name


def test_shared_requirements_are_not_repeated():
    order = registry.resolve(["density", "rt60", "tempo", "truncation"])
    assert order.count("vad") == 1


def test_every_analysis_is_in_registry_once():
    names = [a.name for a in registry.ANALYSES]
    assert len(names) == len(set(names))


def test_every_analysis_is_resolved_once():
    order = registry.resolve([a.name for a in registry.ANALYSES])
    assert len(order) == len(set(order))


# ----------------------------------------------------------------- jsonable


def test_numpy_scalars_and_arrays_become_plain_python():
    result = registry.jsonable(
        {"a": np.float32(1.5), "b": np.array([1, 2]), "c": np.int64(3)}
    )
    assert result == {"a": 1.5, "b": [1, 2], "c": 3}
    assert isinstance(result["a"], float)


def test_non_finite_floats_become_none():
    """NaN is not valid JSON, and a missing metric should read as missing."""
    assert registry.jsonable({"x": float("nan")}) == {"x": None}
    assert registry.jsonable({"x": float("inf")}) == {"x": None}


def test_nested_structures_are_coerced_throughout():
    result = registry.jsonable({"spans": [[np.float64(1.0), np.float64(2.0)]]})
    assert result == {"spans": [[1.0, 2.0]]}


# -------------------------------------------------------------------- cache


def test_cache_key_changes_with_the_analysis_version():
    a = cache.cache_path("abc123", "snr", 1)
    b = cache.cache_path("abc123", "snr", 2)
    assert a != b


def test_cache_key_changes_with_the_file_digest():
    a = cache.cache_path("abc123", "snr", 1)
    b = cache.cache_path("def456", "snr", 1)
    assert a != b


def test_cache_key_changes_with_the_satisfied_optional_inputs():
    """SNR with VAD is a different answer from SNR without it."""
    without = cache.cache_path("abc123", "snr", 1, ())
    with_vad = cache.cache_path("abc123", "snr", 1, ("vad",))
    assert without != with_vad


def test_satisfied_input_order_does_not_change_the_key():
    a = cache.cache_path("abc123", "f0", 1, ("vad", "transcript"))
    b = cache.cache_path("abc123", "f0", 1, ("transcript", "vad"))
    assert a == b


def test_an_analysis_gaining_an_optional_input_is_not_served_the_old_answer(
    tmp_path, monkeypatch
):
    """The bug this key exists to prevent, end to end.

    Run pitch without a transcript, then again with one. The second run must
    recompute rather than hand back the first run's speaker-less result.
    """
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    calls = {"n": 0}

    def fake_f0(bundle, results):
        calls["n"] += 1
        return {"has_transcript": "transcript" in results}

    def fake_transcript(bundle, results):
        return {"words": []}

    monkeypatch.setitem(
        registry.BY_NAME,
        "f0",
        registry.Analysis(
            name="f0",
            group="speaker",
            label="F0",
            fn=fake_f0,
            prefers=("transcript",),
        ),
    )
    monkeypatch.setitem(
        registry.BY_NAME,
        "transcript",
        registry.Analysis(
            name="transcript",
            group="transcript",
            label="T",
            fn=fake_transcript,
            cached=False,
        ),
    )

    first = registry.run(None, "digest", ["f0"])
    assert first["f0"]["has_transcript"] is False

    second = registry.run(None, "digest", ["f0", "transcript"])
    assert second["f0"]["has_transcript"] is True
    assert calls["n"] == 2, "the second run must not be served the first's answer"

    # And the run without the transcript is still a hit, not a recompute.
    registry.run(None, "digest", ["f0"])
    assert calls["n"] == 2


def test_round_trip_through_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    assert cache.read("digest", "snr", 1) is None
    cache.write("digest", "snr", 1, {"snr_db": 12.5})
    assert cache.read("digest", "snr", 1) == {"snr_db": 12.5}
    # A version bump is a miss, not a stale hit.
    assert cache.read("digest", "snr", 2) is None


def test_a_truncated_cache_file_reads_as_a_miss(tmp_path, monkeypatch):
    """An interrupted write should cost a recompute, not crash the app."""
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    cache.cache_path("digest", "snr", 1).write_text('{"snr_db": 12.')
    assert cache.read("digest", "snr", 1) is None


def test_file_digest_follows_content_not_name(tmp_path):
    one = tmp_path / "one.wav"
    two = tmp_path / "two.wav"
    one.write_bytes(b"same bytes")
    two.write_bytes(b"same bytes")
    assert cache.file_digest(one) == cache.file_digest(two)

    two.write_bytes(b"different bytes")
    assert cache.file_digest(one) != cache.file_digest(two)


# ---------------------------------------------------------------------- run


def test_a_failing_extractor_does_not_abort_the_other_analyses(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)

    def boom(bundle, results):
        raise ValueError("boom")

    broken = registry.Analysis(name="broken", group="quality", label="Broken", fn=boom)
    fine = registry.Analysis(
        name="fine", group="quality", label="Fine", fn=lambda b, r: {"ok": True}
    )
    monkeypatch.setitem(registry.BY_NAME, "broken", broken)
    monkeypatch.setitem(registry.BY_NAME, "fine", fine)

    results = registry.run(None, "digest", ["broken", "fine"])
    assert results["fine"] == {"ok": True}
    assert "ValueError: boom" in results["broken"]["error"]


def test_results_are_passed_to_later_analyses(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path)
    seen = {}

    producer = registry.Analysis(
        name="producer", group="quality", label="P", fn=lambda b, r: {"value": 7}
    )

    def consume(bundle, results):
        seen.update(results)
        return {"read": results["producer"]["value"]}

    consumer = registry.Analysis(
        name="consumer",
        group="quality",
        label="C",
        fn=consume,
        requires=("producer",),
    )
    monkeypatch.setitem(registry.BY_NAME, "producer", producer)
    monkeypatch.setitem(registry.BY_NAME, "consumer", consumer)

    results = registry.run(None, "digest", ["consumer"])
    assert results["consumer"]["read"] == 7
    assert "producer" in seen
