"""Percentile-based normalization: reading a per-photo measurement relative
to the rest of its batch, not just against fixed reference constants.

Absolute reference constants (e.g. "1500 Laplacian variance = tack sharp")
are calibrated against *typical* photos. A trip shot entirely on an overcast
day, indoors, or through a phone's aggressive HDR/denoise pipeline has a
different baseline — every photo might land in the same narrow band of the
absolute scale, making the strongest shots of that trip indistinguishable
from its weakest. Percentile rank within the batch fixes that: it answers
"how does this compare to the *other* photos from this same trip?" — closer
to how a human curator actually judges ("this is the sharpest one we've got
from that overcast morning in the fjords, so it's the keeper").

This is purely a *relative* signal, deliberately blended with (never
replacing) the absolute one in `quality.py` — a trip that's uniformly soft
shouldn't have its best frame inflated to "tack sharp" just for being the
least-bad of the bunch; it should simply out-rank its siblings by more than
the absolute scale alone would allow.
"""
from __future__ import annotations


def percentile_ranks(values: dict[str, float]) -> dict[str, float]:
    """Maps each id to the fractional rank of its value within the batch, in
    [0, 1] (0 = lowest, 1 = highest).

    Ties share the midpoint rank of their span, so a batch of identical
    values resolves to a neutral 0.5 for everyone rather than an arbitrary
    ordering, and a single-photo "batch" is also neutral — there's nothing
    to compare it against.
    """
    if len(values) < 2:
        return {pid: 0.5 for pid in values}

    items = sorted(values.items(), key=lambda pair: pair[1])
    n = len(items)
    ranks: dict[str, float] = {}

    i = 0
    while i < n:
        j = i
        while j + 1 < n and items[j + 1][1] == items[i][1]:
            j += 1
        shared_rank = ((i + j) / 2.0) / (n - 1)
        for k in range(i, j + 1):
            ranks[items[k][0]] = shared_rank
        i = j + 1

    return ranks
