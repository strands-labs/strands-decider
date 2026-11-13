import itertools
import math
import struct

import torch
from trainer_support import exact

from training import loss, schedule


def f32(h):
    return struct.unpack(">f", bytes.fromhex(h))[0]


def test_schedule_matches_previous(previous):
    for case in previous["schedule"]:
        total, warm = case["total"], case["warmup"]
        got = [schedule.lr_multiplier(s, total, warm) for s in range(1, total + 1)][
            :: max(1, total // 50)
        ]
        if exact():
            assert [repr(x) for x in got] == case["out"]
        else:  # libm's cos may differ in the last bit
            assert all(
                math.isclose(x, float(y), rel_tol=1e-13)
                for x, y in zip(got, case["out"], strict=True)
            )


def test_schedule_shape():
    total = 1000
    f = [schedule.lr_multiplier(s, total, 0.05) for s in range(1, total + 1)]
    assert f[0] == 1 / 50 and f[49] == 1.0
    assert abs(f[-1] - 0.1) < 1e-12
    assert all(a >= b for a, b in itertools.pairwise(f[49:]))


def test_the_default_floor_is_the_recipe_schedule_bit_for_bit():
    def recipe(
        step, total, warmup_ratio
    ):  # the recipe's formula, with its 0.1 and 0.45 written out
        warmup = max(1, int(warmup_ratio * total))
        if step <= warmup:
            return step / warmup
        return 0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))

    for total in (1, 2, 7, 100, 6404):
        for ratio in (0.03, 0.05):
            for step in range(1, total + 1):
                assert schedule.lr_multiplier(step, total, ratio) == recipe(step, total, ratio)
                assert schedule.lr_multiplier(step, total, ratio, 0.1) == recipe(step, total, ratio)


def test_the_floor_is_where_the_cosine_ends():
    for floor in (0.0, 0.25):
        f = [schedule.lr_multiplier(s, 1000, 0.05, floor) for s in range(1, 1001)]
        assert f[49] == 1.0 and abs(f[-1] - floor) < 1e-12
        assert all(a >= b for a, b in itertools.pairwise(f[49:]))


def test_loss_matches_previous_bit_for_bit(previous):
    """Bit for bit under the recipe's pins (trainer_support.exact); within 2 float32 ulps elsewhere, where the CPU
    kernels may round a reduction differently."""
    case = previous["loss"]
    logits = torch.tensor([[f32(v) for v in row] for row in case["logits"]], dtype=torch.float32)
    target = torch.tensor([[f32(v) for v in row] for row in case["target"]], dtype=torch.float32)
    for c in case["cases"]:
        got = [
            struct.pack(">f", float(x)).hex()
            for x in loss.training_loss(logits, target, c["brier_weight"])
        ]
        if exact():
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
