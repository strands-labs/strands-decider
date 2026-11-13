"""The learning-rate schedule: linear warmup, then a cosine from 1.0 down to `floor` of the peak (0.1 by default)."""

from __future__ import annotations

import math


def lr_multiplier(step: int, total_steps: int, warmup_ratio: float, floor: float = 0.1) -> float:
    """The factor the peak learning rate is multiplied by at optimizer step `step` (1-based) of `total_steps`.

    The first `warmup_ratio` of the steps (at least one) rise linearly to 1.0; the rest follow half a cosine from 1.0 at
    the end of the warmup to `floor` at the last step. At the default floor this is the recipe's schedule bit for bit:
    (1 - 0.1) * 0.5 is exactly the 0.45 the recipe wrote.
    """
    warmup = max(1, int(warmup_ratio * total_steps))
    if step <= warmup:
        return step / warmup
    return floor + (1.0 - floor) * 0.5 * (
        1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))
    )
