"""The learning-rate schedule: warmup from above zero, cosine down to lr_floor."""

from __future__ import annotations

import pytest
import torch

import training.train as T


def test_the_schedule_starts_above_zero_and_ends_at_the_floor():
    """Optimizer steps count from 1: update 1 warms up, update `total` sits on the floor."""
    warmup, total, floor = 320, 6403, 0.1
    optim = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sched = T._scheduler(optim, warmup, total, floor)
    lrs = []
    for _ in range(total):
        lrs.append(sched.get_last_lr()[0])
        optim.step()
        sched.step()
    assert lrs[0] == pytest.approx(1 / warmup)  # not 0
    assert lrs[warmup - 1] == pytest.approx(1.0)  # peak at update `warmup`
    assert lrs[-1] == pytest.approx(floor)  # exactly the floor at update `total`
    assert all(
        a > b for a, b in zip(lrs[warmup - 1 :], lrs[warmup:], strict=False)
    )  # then monotone down
