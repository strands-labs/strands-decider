"""Which rows go into which microbatch, and which microbatches into which optimizer step.

The plan depends on nothing but the rows' lengths, the seed, the epochs and the batching settings (GPUs x
accumulation, rows and padded tokens a microbatch, the sort window). Every rank computes it for itself and gets the
same one, and a resumed run picks up the same sequence where it stopped:

  * each epoch the rows are shuffled with random.Random(seed + epoch), cut into windows of `window` rows (by default
    world * accum * batch_size), each window sorted by length and cut into microbatches of at most `batch_size` rows
    and `token_budget` padded tokens (rows x the longest row); the epoch's microbatches are then shuffled with
    random.Random(seed + epoch + 7919);
  * one optimizer step is world * accum consecutive microbatches; the last step of an epoch may hold fewer, and the
    ranks without one run a zero-weight filler so every rank issues the same collectives;
  * microbatch j of a step runs on rank j % world in accumulation slot j // world.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Sequence
from typing import Any, TypeVar

# Tokens added to a row's measured prompt length when it is batched and filtered.
LENGTH_MARGIN = 16

Item = TypeVar("Item")
Step = tuple[int, list[list[int]]]  # (epoch, the step's microbatches as row indices)


def row_length(row: dict) -> int:
    """How many tokens a row counts for, in the microbatches and against --max-length: the prompt length convert.py
    measured under the template (`source.input_tokens`) plus LENGTH_MARGIN. A row without that count falls back to a
    guess from the size of its JSON."""
    measured = row["source"].get("input_tokens")
    if isinstance(measured, int) and not isinstance(measured, bool):
        return measured + LENGTH_MARGIN
    return (
        len(json.dumps([row["state"], row["question"]], ensure_ascii=False)) // 3
        + 192
        + 512 * len(row.get("images", []))
    )


def pack_microbatches(
    items: Sequence[Item],
    batch_size: int,
    token_budget: int,
    length: Callable[[Any], int] = row_length,
) -> list[list[Item]]:
    """`items`, kept in their order, grouped into microbatches. A microbatch is closed when it holds `batch_size` items
    or when one more item would take its padded size (items x the longest) over `token_budget`; an item longer than
    the budget on its own is a microbatch by itself."""
    result: list[list[Item]] = []
    pending: list[Item] = []
    longest = 0
    for item in items:
        size = length(item)
        if pending and (
            len(pending) == batch_size or max(longest, size) * (len(pending) + 1) > token_budget
        ):
            result.append(pending)
            pending, longest = [], 0
        pending.append(item)
        longest = max(longest, size)
    if pending:
        result.append(pending)
    return result


def build_plan(
    lengths: Sequence[int],
    epochs: float,
    seed: int,
    world: int,
    accum: int,
    batch_size: int,
    token_budget: int,
    window: int,
) -> list[Step]:
    """One (epoch, microbatches) entry per optimizer step. A fractional last epoch takes the first
    ceil(fraction x its steps) steps (at least one)."""
    per_step = world * accum
    full, fraction = math.floor(epochs), epochs - math.floor(epochs)
    plan: list[Step] = []
    for epoch in range(full + (1 if fraction > 1e-9 else 0)):
        order = list(range(len(lengths)))
        random.Random(seed + epoch).shuffle(order)
        micro: list[list[int]] = []
        for start in range(0, len(order), window):
            chunk = sorted(order[start : start + window], key=lambda i: lengths[i])
            micro.extend(
                pack_microbatches(chunk, batch_size, token_budget, length=lambda i: lengths[i])
            )
        random.Random(seed + epoch + 7919).shuffle(micro)
        steps = [(epoch, micro[s : s + per_step]) for s in range(0, len(micro), per_step)]
        if epoch == full:
            steps = steps[: max(1, math.ceil(fraction * len(steps)))]
        plan.extend(steps)
    return plan


def plan_sha256(plan: Sequence[Step]) -> str:
    """The plan's identity: a resume must find the same one."""
    payload = [[epoch, micro] for epoch, micro in plan]
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class Placement:
    """Which microbatches of a step each rank runs: microbatch j goes to rank j % world, slot j // world. A
    microbatch keeps its plan index j, and j seeds its augmentation, so the option orders do not depend on the GPU
    count."""

    def __init__(self, world: int, accum: int) -> None:
        self.world, self.accum = world, accum

    def local_work(self, step: Step, rank: int) -> list[tuple[list[int], bool, int]]:
        """[(row indices, real, j)] for `rank`, one entry per accumulation slot; a zero-weight filler (the step's first
        microbatch, real False, j = -1) fills a slot the step has no microbatch for."""
        _epoch, micro = step
        out = []
        for slot in range(self.accum):
            j = slot * self.world + rank
            out.append((micro[j], True, j) if j < len(micro) else (micro[0], False, -1))
        return out


def plan_statistics(plan: Sequence[Step], lengths: Sequence[int], placement: Placement) -> dict:
    """The plan's size and padding, and its sha256."""
    world, accum = placement.world, placement.accum
    rows = [sum(len(mb) for mb in micro) for _e, micro in plan]
    micro = [mb for _e, m in plan for mb in m]
    padded = sum(max(lengths[i] for i in mb) * len(mb) for mb in micro)
    real = sum(lengths[i] for mb in micro for i in mb)
    epochs = sorted({e for e, _m in plan})
    return {
        "optimizer_steps": len(plan),
        "steps_per_epoch": [sum(1 for e, _m in plan if e == epoch) for epoch in epochs],
        "microbatches": len(micro),
        "microbatches_per_step": world * accum,
        "rows_per_step_mean": sum(rows) / len(rows),
        "rows_per_microbatch_mean": sum(len(mb) for mb in micro) / len(micro),
        "estimated_tokens": real,
        "estimated_padded_tokens": padded,
        "padding_fraction_estimate": 1 - real / padded if padded else 0.0,
        "sha256": plan_sha256(plan),
    }
