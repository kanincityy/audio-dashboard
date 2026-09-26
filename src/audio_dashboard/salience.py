"""Which words are worth flagging as low-confidence.

At the shipped 0.60 threshold the app flags 2.95% of words, and most of them are
not worth a reader's attention: `and` alone is 146 of the 1,591 flagged words in
the corpus sweep, and the top of the list is `and, i, you, the, so, it, a, to`.

**This is a relevance filter, not a calibration fix, and the distinction matters
for how the list is built.** Function words are not low-confidence — measured
over 54,022 words they average 0.9476 against 0.9405 for content words, so they
are slightly *more* confident. They dominate the flagged set only because they
are 55% of all words. Nothing here is correcting a model weakness; it is
declining to draw attention to words whose misrecognition has no consequence.

So the admission test for :data:`FILLER` is not "is this a stopword" but:

    Can mishearing this word change what the record says?

Anything that can, stays flaggable, however common it is. That is the whole
rule, and it is why there is exactly one list here. Words whose mishearing
flips or falsifies the record — `no`, `never`, `£600`, `you`, `and` — are simply
absent from ``FILLER``, and so are flagged like any other word. An earlier
version also kept explicit POLARITY/QUANTITY/PERSON/CONNECTIVE sets and judged
figures and negations against a raised cutoff. Both were dropped: the sets only
restated "not in FILLER" in a form that had to be kept in sync, and a second
cutoff made the highlight rate hard to reason about for a distinction the reader
could not see on screen anyway.

The list is exact tokens rather than stems or parts of speech. A tagger would be
a dependency and a failure mode for what is ultimately a 50-word decision, and an
exact list is auditable by reading it.
"""

from __future__ import annotations

import re
from typing import Any

# Discourse and grammatical filler. Mishearing any of these leaves the meaning of
# the surrounding sentence intact, which is the only reason a word belongs here.
# Everything not in this set is flaggable.
FILLER = frozenset(
    {
        # determiners and articles
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        # copulas and auxiliaries
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        # existential and locative pointers
        "there",
        "here",
        "it",
        "its",
        "itself",
        # hesitation
        "um",
        "uh",
        "er",
        "erm",
        "mm",
        "hmm",
        "eh",
        "ah",
        "oh",
        "mhm",
        "uh-huh",
        "huh",
        "ha",
        "ur",
        "uhm",
        "hmm-hmm",
        "uh-uh",
        "mm-hmm",
        "mm-mm",
    }
)

# Surrounding punctuation goes, apostrophes stay: "don't" must not degrade to
# "do", which sits in FILLER and would silently suppress a negation.
_STRIP = re.compile(r"^[^\w']+|[^\w']+$")


def normalise(text: str) -> str:
    """Lowercase and strip surrounding punctuation, keeping apostrophes."""
    return _STRIP.sub("", str(text or "").lower())


def flaggable(word: dict[str, Any], *, suppress_filler: bool = True) -> bool:
    """Whether this word is eligible to be flagged at all."""
    if not suppress_filler:
        return True
    return normalise(word.get("text", "")) not in FILLER


def is_flagged(
    word: dict[str, Any], threshold: float, *, suppress_filler: bool = True
) -> bool:
    """The single definition of "this word gets highlighted".

    Every consumer — the marks, the timeline, the rug, the readout, the lowest-
    confidence table — goes through here, so a change to the list cannot leave
    one view showing a different set of words from another.

    One cutoff for every word. There is one band on screen and nothing for a
    reader to decode.
    """
    return word["confidence"] < threshold and flaggable(
        word, suppress_filler=suppress_filler
    )


def eligible(
    words: list[dict[str, Any]], *, suppress_filler: bool = True
) -> list[dict[str, Any]]:
    """The words a rate target should be computed over.

    Percentile modes need this: choosing the cutoff from a distribution that
    includes words the app will then refuse to flag makes the rate readout report
    a number the user cannot see on screen.
    """
    if not suppress_filler:
        return list(words)
    return [w for w in words if flaggable(w, suppress_filler=True)]
