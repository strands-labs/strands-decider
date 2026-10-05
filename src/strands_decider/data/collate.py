"""Batching: option permutation, tokenisation, and target construction.

This module is where genericity is actually enforced. Every time an example is
drawn, its options are re-permuted and the label is remapped to follow. A head that
tried to memorise "slot 0 means positive" would be wrong half the time, so gradient
descent pushes it toward the only strategy that survives: read the option text at
position k out of the prompt.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import torch

from ..prompting import build_prompt
from ..schema import Question
from .format import Example

# Each row's kind as a number, emitted per batch as "kind_id" so that a loss term can
# select rows by kind (TrainConfig.kl_frozen_skip_kinds).
KIND_IDS = {"noul": 0, "choice": 1, "score": 2}


@dataclass
class CollatorConfig:
    max_length: int = 3072
    num_slots: int = 24
    # "pointer" makes the collator emit option token positions and size the target
    # distribution to the batch's widest option count instead of num_slots.
    head_type: str = "slot"
    shuffle_options: bool = True
    # Score levels are ordered, so they are never permuted -- but reversing the whole
    # rubric (and the label with it) is meaning-preserving and doubles the orderings seen.
    reverse_score_prob: float = 0.5
    # Mass moved to adjacent levels for score targets. Being one level off is a
    # much smaller error than being four off; plain cross-entropy cannot express that.
    ordinal_smoothing: float = 0.1
    # Sample among an example's instruction phrasings each epoch.
    vary_instructions: bool = True
    seed: int = 0


class Draw(NamedTuple):
    """One row as a training step shows it: the option order (None: canonical), the
    question in the drawn phrasing, and the label and soft target in slot order."""

    order: list[int] | None
    question: Question
    label: int
    target: torch.Tensor | None


class SystemOneCollator:
    """Turns a list of Examples into a padded, tokenised training batch."""

    def __init__(self, tokenizer: Any, config: CollatorConfig, *, train: bool = True):
        self.tok = tokenizer
        self.cfg = config
        self.train = train
        self.rng = random.Random(config.seed)

    # ---- option ordering -------------------------------------------------

    def _option_order(self, ex: Example) -> list[int] | None:
        n = ex.n_options
        if not self.cfg.shuffle_options or not self.train:
            return None
        if ex.kind == "score":
            if self.rng.random() < self.cfg.reverse_score_prob:
                return list(reversed(range(n)))
            return None
        order = list(range(n))
        self.rng.shuffle(order)
        return order

    def _instruction(self, ex: Example) -> str | None:
        """Sample a phrasing of the question, the same way option order is sampled.

        Resampling per epoch rather than fixing one at build time is what makes this
        an augmentation: a head that keyed on the exact instruction string would be
        wrong most of the time. Evaluation always uses the canonical phrasing so runs
        stay comparable.
        """
        if not self.train or not self.cfg.vary_instructions:
            return None
        pool = ex.all_instructions()
        return self.rng.choice(pool) if len(pool) > 1 else None

    def draw(self, ex: Example) -> Draw:
        """This row's random rendering choices, drawn from the collator's stream."""
        order = self._option_order(ex)
        question = ex.to_question(self._instruction(ex))
        return Draw(order, question, self._remap_label(ex.label, order),
                    self._target_distribution(ex, ex.label, ex.n_options, order))

    def skip(self, batch: list[Example]) -> None:
        """Advance the random stream exactly as collating `batch` would, rendering nothing.

        A multi-GPU rank calls this for the rows other ranks train on (distributed.py),
        so every row is rendered as it would be on one GPU.
        """
        for ex in batch:  # the draws `__call__` makes, in its order
            self._option_order(ex)
            self._instruction(ex)

    @staticmethod
    def option_token_index(
        offsets: Sequence[Sequence[int]], spans: Sequence[Sequence[int]], base: int
    ) -> list[int]:
        """Last token index of each option's line, for a pointer readout.

        `spans` are character spans within the rendered *question*; `base` shifts them
        into the full prompt, which is the state text followed by that question. The
        last token of the span is the one that has just read the whole option under
        causal attention -- the same position kev reads its `</opt>` marker at.

        Raises when an option has no surviving token: truncation has removed it, and
        scoring it from a neighbour's representation would be silently wrong. Callers
        arrange for this not to happen -- `infer._fit` reserves the question and cuts
        it from the front, so options and `<answer>` are the last things to go.
        """
        out: list[int] = []
        for s, e in spans:
            a, b = base + s, base + e
            last = -1
            for j, (lo, hi) in enumerate(offsets):
                if hi <= lo:  # special/padding tokens carry an empty span
                    continue
                if lo >= a and hi <= b:
                    last = j
            if last < 0:
                raise ValueError(
                    f"option span ({a},{b}) has no tokens left; the prompt was "
                    "truncated through its option list"
                )
            out.append(last)
        return out

    @staticmethod
    def _remap_label(label: int, order: Sequence[int] | None) -> int:
        """`order[k] = original index shown at slot k`, so the new label is its position."""
        if order is None:
            return label
        return list(order).index(label)

    # ---- targets ---------------------------------------------------------

    def _target_distribution(
        self, ex: Example, label: int, n_options: int, order: Sequence[int] | None
    ) -> torch.Tensor | None:
        """Soft target for score questions; None for noul/choice (plain NLL is right there).

        The smoothing is applied in *level* space and then mapped through the slot
        permutation, so a reversed rubric still spreads mass onto the neighbouring
        levels rather than onto arbitrary slots.
        """
        if ex.kind != "score" or self.cfg.ordinal_smoothing <= 0:
            return None

        dist = torch.zeros(self.cfg.num_slots, dtype=torch.float32)
        eps = self.cfg.ordinal_smoothing
        gold_level = ex.label
        neighbours = [lv for lv in (gold_level - 1, gold_level + 1) if 0 <= lv < n_options]

        level_mass = {gold_level: 1.0 - eps}
        if neighbours:
            for lv in neighbours:
                level_mass[lv] = eps / len(neighbours)
        else:
            level_mass[gold_level] = 1.0

        # level index -> slot index under this rendering
        order_list = list(order) if order is not None else list(range(n_options))
        for lv, mass in level_mass.items():
            dist[order_list.index(lv)] += mass
        return dist

    # ---- main entry ------------------------------------------------------

    def __call__(self, batch: list[Example]) -> dict[str, torch.Tensor]:
        texts: list[str] = []
        labels: list[int] = []
        n_slots: list[int] = []
        dists: list[torch.Tensor | None] = []
        weights: list[float] = []
        teachers: list[list[float] | None] = []

        pointer = self.cfg.head_type == "pointer"
        spans: list[Any] = []
        for ex in batch:
            # A pointer head has no fixed slot count, so only the slot head caps options.
            if not pointer and ex.n_options > self.cfg.num_slots:
                raise ValueError(
                    f"example from task {ex.task!r} has {ex.n_options} options but the "
                    f"model has only {self.cfg.num_slots} slots"
                )
            order, question, label, target = self.draw(ex)
            prompt, rq = build_prompt(ex.state, question, option_order=order)
            texts.append(prompt)
            # Option spans are relative to the question; the prompt is state + question.
            spans.append((len(prompt) - len(rq.text), rq.option_spans))
            labels.append(label)
            n_slots.append(ex.n_options)
            dists.append(target)
            weights.append(ex.weight)
            # A teacher's distribution (train.py attaches it) is in canonical option
            # order; slot k shows canonical option order[k], exactly as for the label.
            t = getattr(ex, "teacher", None)
            teachers.append(None if t is None else (t if order is None else [t[i] for i in order]))

        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.max_length,
            return_tensors="pt",
            # Right padding: pool_last_token finds the final real token by mask length,
            # and the shared-prefix cache in infer.py assumes the prompt starts at 0.
            padding_side="right",
            return_offsets_mapping=pointer,
        )

        out: dict[str, torch.Tensor] = {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "n_slots": torch.tensor(n_slots, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "weights": torch.tensor(weights, dtype=torch.float32),
            "kind_id": torch.tensor([KIND_IDS[ex.kind] for ex in batch], dtype=torch.long),
        }
        width = self.cfg.num_slots
        if pointer:
            offs = enc["offset_mapping"].tolist()
            per = [
                self.option_token_index(offs[i], sp, base)
                for i, (base, sp) in enumerate(spans)
            ]
            width = max(len(p) for p in per)
            out["opt_idx"] = torch.tensor(
                [p + [-1] * (width - len(p)) for p in per], dtype=torch.long
            )
        if any(d is not None for d in dists):
            stacked = torch.stack(
                [
                    d
                    if d is not None
                    else torch.nn.functional.one_hot(
                        torch.tensor(lab), num_classes=self.cfg.num_slots
                    ).float()
                    for d, lab in zip(dists, labels, strict=True)
                ]
            )
            # A pointer head emits one logit per option present, so the target must be
            # cut to the batch's widest option count rather than the slot count.
            out["label_dist"] = stacked[:, :width] if pointer else stacked
        if any(t is not None for t in teachers):
            tt = torch.zeros((len(batch), width), dtype=torch.float32)
            for i, t in enumerate(teachers):
                if t is not None:
                    tt[i, : len(t)] = torch.tensor(t, dtype=torch.float32)
            out["teacher"] = tt
            out["has_teacher"] = torch.tensor([t is not None for t in teachers])
        return out
