"""Multi-GPU training under torchrun (DDP) that trains exactly what one GPU trains.

    torchrun --standalone --nproc_per_node=8 -m strands_decider.cli train --config configs/train.yaml

The rule: an N-GPU run takes the optimizer steps a 1-GPU run takes. Step s is micro-batches
s*grad_accum .. s*grad_accum + grad_accum - 1 of the 1-GPU sampler order, always, and the
ranks share those rows out as contiguous slices of the micro-batches (`plan_step`). Three
things keep the gradient the 1-GPU gradient:

  - every rank draws the collator's augmentations for the rows it does not train on
    (`SystemOneCollator.skip`), so each row is rendered as it is on one GPU;
  - each loss term of a slice is weighted by the slice's share of its micro-batch's
    denominator for that term (`Slice`), so the slices of a micro-batch sum to its loss;
  - train.py multiplies the loss by the world size before DDP averages the gradients, so
    the all-reduced gradient is the sum over ranks: the 1-GPU accumulated gradient.

Not exact, and only numerically: dropout masks differ from the 1-GPU masks, and a split
micro-batch runs at a different padded shape (bf16: <= 4e-2 relative gradient difference).

How the rows are shared is free, so it is chosen for speed. Length grouping makes the
micro-batches of one step very unequal (a batch of long documents next to batches of short
rows), and a step takes as long as its slowest rank, so `plan_step` cuts the long
micro-batches finer and gives short ones whole to single ranks.

At world size 1 none of this runs, and train.py is the original single-process loop.
"""
from __future__ import annotations

import contextlib
import functools
import os
from collections.abc import Callable, Iterator
from datetime import timedelta
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from .data.sampling import example_length

# Fixed cost of one forward, in padded prompt characters (the unit of `example_length`).
# Small forwards are launch-bound, so fewer, larger slices win: on 8x A100, 120 steps of
# the v17 mix ran at 0.85 / 1.06 / 1.23 step/s with 512 / 2000 / 6000 here.
FORWARD_OVERHEAD = 6000


def env() -> tuple[int, int, int]:
    """(rank, local_rank, world_size) from torchrun's environment; (0, 0, 1) otherwise."""
    return (int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0")),
            int(os.environ.get("WORLD_SIZE", "1")))


def entry_point(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Run `fn(cfg)` as one rank when torchrun launched this process, else as a plain call.

    As a rank: join the process group (NCCL on GPU, gloo on CPU), silence stdout on ranks
    other than 0 (rank 0 prints the log), wait after `fn` until every rank is done (so none
    returns before rank 0 has saved), and leave the group however `fn` exits.
    """

    @functools.wraps(fn)
    def run(cfg: Any, *args: Any, **kwargs: Any) -> Any:
        rank, local_rank, world = env()
        if world == 1:
            return fn(cfg, *args, **kwargs)
        rows = cfg.micro_batch_size * cfg.grad_accum
        if world > rows:  # fail before the model loads
            raise ValueError(f"world size {world} exceeds the {rows} rows of an optimizer step")
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        # The timeout is a backstop: under torchrun a rank that raises exits, and the
        # launcher then stops the others rather than leaving them in a collective.
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=timedelta(minutes=30))
        try:
            with contextlib.ExitStack() as stack:
                if rank:
                    null = stack.enter_context(open(os.devnull, "w", encoding="utf-8"))
                    stack.enter_context(contextlib.redirect_stdout(null))
                out = fn(cfg, *args, **kwargs)
            dist.barrier(device_ids=[local_rank] if torch.cuda.is_available() else None)
            return out
        finally:
            dist.destroy_process_group()

    return run


def _device() -> torch.device:
    if dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def all_reduce(values: list[float], op: str = "sum") -> list[float]:
    """The sum (or "max") of `values` over ranks; `values` unchanged at world size 1."""
    if env()[2] == 1:
        return list(values)
    t = torch.tensor(values, dtype=torch.float64, device=_device())
    dist.all_reduce(t, op=dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MAX)
    result: list[float] = t.tolist()
    return result


def wrap(model: torch.nn.Module, seed: int) -> torch.nn.Module:
    """The module the training forward goes through: `model` itself, or DDP around it.

    Every rank seeds and builds the model identically, and DDP also broadcasts rank 0's
    parameters as it wraps, so all ranks start from the same weights. Each rank then
    reseeds its CUDA generator with seed + rank: with one seed, equally shaped forwards
    on different ranks would draw the same dropout masks. The CPU generator, which rank 0
    draws the random sampler's order from, is left alone.
    """
    rank, local_rank, world = env()
    if world == 1:
        return model
    ddp = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        # Every trainable parameter (head, LoRA) is reached by every forward, so no
        # unused-parameter search; checkpointing is non-reentrant (train.py), which DDP
        # supports without static_graph. Buffers (rotary tables) never change.
        broadcast_buffers=False,
    )
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed + rank)
    return ddp


# ---- sharing a step's rows out --------------------------------------------------------


def _best_cuts(lens: list[int], k: int) -> tuple[int, list[tuple[int, int]]]:
    """Split a row sequence into k contiguous slices minimising the costliest slice."""
    n = len(lens)

    def cost(a: int, b: int) -> int:
        return (b - a) * max(lens[a:b]) + FORWARD_OVERHEAD

    # best[i, c]: (max cost, cuts) for rows i.. in c slices
    best: dict[tuple[int, int], tuple[int, list[tuple[int, int]]]] = {}
    for i in range(n - 1, -1, -1):
        best[i, 1] = (cost(i, n), [(i, n)])
        for c in range(2, min(k, n - i) + 1):
            cands = []
            for b in range(i + 1, n - c + 2):
                rest, cuts = best[b, c - 1]
                cands.append((max(cost(i, b), rest), [(i, b), *cuts]))
            best[i, c] = min(cands, key=lambda t: t[0])
    return best[0, k]


def plan_step(batches: list[list[int]], lengths: list[int],
              world: int) -> list[list[tuple[int, int, int]]]:
    """Each rank's slices (micro-batch j of the step, row start, row stop) for one step.

    Every rank gets at least one slice and no forward exceeds a micro-batch, so no rank
    needs more memory than the 1-GPU run. Deterministic in its inputs: every rank
    computes the same plan.
    """
    per = [[lengths[i] for i in b] for b in batches]
    assert world <= sum(map(len, per)), "more ranks than rows in the step"

    def assign(n: list[int]) -> tuple[tuple[bool, int], list[list[tuple[int, int, int]]]]:
        slices = []
        for j, k in enumerate(n):
            lens = per[j]
            for a, b in _best_cuts(lens, k)[1]:
                slices.append(((b - a) * max(lens[a:b]) + FORWARD_OVERHEAD, j, a, b))
        slices.sort(key=lambda t: (-t[0], t[1], t[2]))
        load = [0] * world
        ranks: list[list[tuple[int, int, int]]] = [[] for _ in range(world)]
        for c, j, a, b in slices:
            r = min(range(world), key=lambda q: (load[q], q))
            load[r] += c
            ranks[r].append((j, a, b))
        # Fewer slices than ranks leaves a rank idle, so such a plan always scores worse.
        return (len(slices) < world, max(load)), [sorted(x) for x in ranks]

    n = [1] * len(batches)
    best = assign(n)
    # Greedy: add one slice at a time to the micro-batch where it lowers the slowest
    # rank's load most, up to a fixed budget of 2 * world extra slices, and keep the
    # best plan seen on the way.
    for _ in range(min(sum(map(len, per)), len(batches) + 2 * world) - len(batches)):
        cands = [[*n[:j], n[j] + 1, *n[j + 1:]] for j in range(len(n)) if n[j] < len(per[j])]
        if not cands:
            break
        (score, plan), n = min(((assign(c), c) for c in cands), key=lambda t: (t[0][0], t[1]))
        if score < best[0]:
            best = (score, plan)
    # A rank with no slice would skip the step's all-reduce and hang the others.
    assert all(best[1]), "a rank got no slice"
    return best[1]


class Slice(NamedTuple):
    """How one forward's loss terms count toward its micro-batch.

    Each share is the forward's part of its micro-batch's denominator for one term: the
    label-weight sum, the rows the frozen KL covers, the rows with a teacher. A weighted
    mean over the slice times that share is the slice's part of the micro-batch's mean.
    `last` marks the rank's last forward of the optimizer step, whose backward all-reduces.
    """

    weights: float = 1.0
    kl: float = 1.0
    teacher: float = 1.0
    last: bool = True
    micro: int = 0  # the micro-batch's number in the run (1-GPU order), and the slice's
    start: int = 0  # first row in it: where train.py finds a precomputed frozen reference


WHOLE = Slice()  # a whole micro-batch: every term as the 1-GPU loop takes it


def _share(rows: list[Any], a: int, b: int, value: Callable[[Any], float]) -> float:
    whole = sum(value(ex) for ex in rows)
    return sum(value(ex) for ex in rows[a:b]) / whole if whole else 0.0


class StepSlices:
    """This rank's forwards, one epoch per iteration; stands in for the training DataLoader.

    Yields collated batches like the loader, each with its `Slice` under "part". A step is
    a window of grad_accum consecutive micro-batches of the loader's order, across epoch
    boundaries as on one GPU: a window the epoch leaves open is completed by the next one,
    so a step's forwards all come out in the epoch the step ends in. A window still open
    after the last epoch is dropped; one GPU forwards it but never steps on it.

    Construction checks that every rank holds the same data, and fails if not: ranks that
    disagree on the step count wait on each other for ever, and ranks that disagree on the
    rows train some twice and some never, silently.
    """

    def __init__(self, loader: DataLoader, kl_slots: int, grad_accum: int, total_steps: int,
                 kl_skip_kinds: frozenset[str] = frozenset()):
        self.loader = loader
        self.examples = loader.dataset.examples
        self.collate = loader.collate_fn
        self.grad_accum = grad_accum
        self.rank, _, self.world = env()
        self.lengths = [example_length(ex) for ex in self.examples]
        self.kl_slots = kl_slots  # rows with at most this many options get the frozen KL,
        self.kl_skip_kinds = kl_skip_kinds  # unless their kind is one of these
        self.open: list[list[int]] = []
        self.windows = 0  # steps (windows of grad_accum micro-batches) handed out so far

        n_teacher = sum(getattr(ex, "teacher", None) is not None for ex in self.examples)
        facts = [len(self.lengths), sum(self.lengths), n_teacher, total_steps,
                 sum(i * x for i, x in enumerate(self.lengths)) % (2**31 - 1)]
        lo = torch.tensor(facts, dtype=torch.int64, device=_device())
        hi = lo.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if not torch.equal(lo, hi):
            raise RuntimeError(f"ranks hold different training data (rows, total length, "
                               f"teacher rows, steps, length checksum): min {lo.tolist()}, "
                               f"max {hi.tolist()}")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        order: list[Any] = [None]
        if self.rank == 0:
            # The epoch's micro-batches drawn as the 1-GPU loader draws them (a random
            # sampler takes its seed from the global torch RNG), then broadcast. Rank 0's
            # global RNG stays where one GPU's is because rank 0 also runs every
            # validation pass, whose loader draws from it too: keep it that way.
            order[0] = list(DataLoader(range(len(self.examples)),
                                       batch_sampler=self.loader.batch_sampler, collate_fn=list))
        dist.broadcast_object_list(order, src=0)
        for idx in order[0]:
            self.open.append(idx)
            if len(self.open) == self.grad_accum:
                window, self.open = self.open, []
                yield from self._step(window)

    def _step(self, window: list[list[int]]) -> Iterator[dict[str, Any]]:
        mine = plan_step(window, self.lengths, self.world)[self.rank]
        first = self.windows * self.grad_accum
        self.windows += 1
        for j, idx in enumerate(window):
            rows = [self.examples[i] for i in idx]
            done = 0
            for jj, a, b in mine:
                if jj != j:
                    continue
                self.collate.skip(rows[done:a])
                batch = self.collate(rows[a:b])
                batch["part"] = Slice(
                    weights=_share(rows, a, b, lambda ex: ex.weight),
                    kl=_share(rows, a, b, lambda ex: ex.n_options <= self.kl_slots
                              and ex.kind not in self.kl_skip_kinds),
                    teacher=_share(rows, a, b, lambda ex: getattr(ex, "teacher", None) is not None),
                    last=(jj, a, b) == mine[-1],
                    micro=first + j,
                    start=a,
                )
                done = b
                yield batch
            self.collate.skip(rows[done:])
