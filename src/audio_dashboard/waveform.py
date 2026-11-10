"""A drawable outline of a recording.

Take the minimum and maximum of each bucket to show the waveform
"""

from __future__ import annotations

import numpy as np

DEFAULT_BUCKETS = 4000


def envelope(samples: np.ndarray, buckets: int = DEFAULT_BUCKETS) -> dict:
    """Per-bucket minimum and maximum of a mono signal.

    Returns at most ``buckets`` points; fewer when the signal is shorter.
    ``step_s`` is not included: the caller knows the duration and the count, and
    point ``i`` sits at ``i * duration / points``.
    """
    if buckets < 1:
        raise ValueError("buckets must be at least 1")
    if not len(samples):
        return {"points": 0, "min": [], "max": []}

    step = max(1, len(samples) // buckets)
    usable = (len(samples) // step) * step
    grid = samples[:usable].reshape(-1, step)
    lows = grid.min(axis=1)
    highs = grid.max(axis=1)
    return {
        "points": len(lows),
        "min": np.round(lows, 4).tolist(),
        "max": np.round(highs, 4).tolist(),
    }
