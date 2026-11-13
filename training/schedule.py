"""The learning-rate schedule: linear warmup, then a cosine from 1.0 down to 0.1 of the peak."""

from __future__ import annotations

import math


def lr_multiplier(step: int, total_steps: int, warmup_ratio: float) -> float:
    """The factor the peak learning rate is multiplied by at optimizer step `step` (1-based) of `total_steps`.

    The first `warmup_ratio` of the steps (at least one) rise linearly to 1.0; the rest follow half a cosine from 1.0 at
    the end of the warmup to 0.1 at the last step.
    """
    warmup = max(1, int(warmup_ratio * total_steps))
    if step <= warmup:
        return step / warmup
    return 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))
