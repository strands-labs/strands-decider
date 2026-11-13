import itertools
import struct

import torch
from trainer_support import same_torch

from training import loss, schedule


def f32(h):
    return struct.unpack(">f", bytes.fromhex(h))[0]


def test_schedule_matches_previous(previous):
    for case in previous["schedule"]:
        total, warm = case["total"], case["warmup"]
        got = [repr(schedule.lr_multiplier(s, total, warm)) for s in range(1, total + 1)][
            :: max(1, total // 50)
        ]
        assert got == case["out"]


def test_schedule_shape():
    total = 1000
    f = [schedule.lr_multiplier(s, total, 0.05) for s in range(1, total + 1)]
    assert f[0] == 1 / 50 and f[49] == 1.0
    assert abs(f[-1] - 0.1) < 1e-12
    assert all(a >= b for a, b in itertools.pairwise(f[49:]))


def test_loss_matches_previous_bit_for_bit(previous):
    """Bit for bit under the torch the fixture was made with; within 2 float32 ulps under another, whose CPU kernels
    may round a reduction differently (trainer_support.same_torch)."""
    case = previous["loss"]
    logits = torch.tensor([[f32(v) for v in row] for row in case["logits"]], dtype=torch.float32)
    target = torch.tensor([[f32(v) for v in row] for row in case["target"]], dtype=torch.float32)
    for c in case["cases"]:
        got = [
            struct.pack(">f", float(x)).hex()
            for x in loss.training_loss(logits, target, c["brier_weight"])
        ]
        if same_torch():
            assert got == c["out"]
        else:
            assert all(
                abs(int(a, 16) - int(b, 16)) <= 2 for a, b in zip(got, c["out"], strict=True)
            ), (got, c["out"])


def test_masked_codes_take_no_mass():
    logits = torch.full((1, 255), -1e9)
    logits[0, :2] = torch.tensor([0.0, 0.0])
    target = torch.zeros(1, 255)
    target[0, 0] = 1.0
    out = loss.training_loss(logits, target)
    assert torch.isfinite(out.total) and abs(out.cross_entropy.item() - 0.6931472) < 1e-6
