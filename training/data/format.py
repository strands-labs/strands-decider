"""The on-disk training example, and JSONL round-tripping.

One example is exactly one (state, question, answer) triple -- the same unit the
serving API handles, so nothing about the input distribution shifts between train
and inference. Multi-question requests are simply several examples that happen to
share a state.
"""

from __future__ import annotations

import gzip
import json
import os
import random
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import IO, Any, cast

from strands_decider.schema import ChoiceQuestion, Content, NoulQuestion, Question, ScoreQuestion


@dataclass
class Example:
    """A single supervised item.

    `label` is an index into the *canonical* option order:
      - noul:   0 = false, 1 = true
      - choice: position in `options`
      - score:  the level index (which is also its position in `options`)

    Option shuffling happens at collate time, not here, so one stored example
    yields a different slot binding on every epoch.
    """

    kind: str  # "noul" | "choice" | "score"
    state: Any
    instructions: str
    # Canonical option order: [(name, description), ...]
    options: list[list[str]]
    label: int
    task: str = "unknown"  # provenance, for per-task eval breakdowns
    weight: float = 1.0
    # Alternative phrasings of the same question. Sampled per epoch at collate time,
    # exactly as option order is. Twelve of fourteen tasks originally carried a single
    # instruction string, so the model never saw a task asked two ways -- and held-out
    # accuracy on weak tasks moves by up to 0.15 on rewording alone.
    instruction_variants: list[str] = field(default_factory=list)
    # A soft target for noul/choice: one probability per option, in canonical order. The
    # collator permutes it with the options and trains on it (label_dist); `label` stays
    # the argmax, for metrics. None: one-hot on `label`, as before.
    target: list[float] | None = None
    # Options whose order carries meaning (a score rubric stored as a choice): never shuffled.
    fixed_order: bool = False

    def __post_init__(self) -> None:
        if self.kind not in ("noul", "choice", "score"):
            raise ValueError(f"unknown kind: {self.kind}")
        if not 0 <= self.label < len(self.options):
            raise ValueError(f"label {self.label} out of range for {len(self.options)} options")
        if self.target is not None and (
            self.kind == "score" or len(self.target) != len(self.options)
        ):
            raise ValueError(f"a {self.kind} target of {len(self.target)} for {self.n_options}")

    @property
    def n_options(self) -> int:
        return len(self.options)

    def all_instructions(self) -> list[str]:
        """Canonical phrasing first, then any alternatives."""
        return [self.instructions, *self.instruction_variants]

    def to_question(self, instructions: str | None = None) -> Question:
        """Rebuild the API-level question, so training and serving share one renderer.

        `instructions` overrides the canonical phrasing, which is how the collator
        feeds a sampled variant through without mutating the stored example.
        """
        instr = self.instructions if instructions is None else instructions
        crit: dict[str, Content | None] = {name: desc for name, desc in self.options}
        if self.kind == "noul":
            return NoulQuestion(instructions=instr, criteria=crit)
        if self.kind == "choice":
            return ChoiceQuestion(instructions=instr, criteria=crit)
        return ScoreQuestion(instructions=instr, criteria=[desc for _, desc in self.options])

    def to_request(self, instructions: str | None = None) -> dict[str, Any]:
        """The question as a one-question request JSON, for `strands_decider.core.render`."""
        instr = self.instructions if instructions is None else instructions
        crit: Any = dict(self.options) if self.kind != "score" else [d for _, d in self.options]
        q = {"type": self.kind, "instructions": instr, "criteria": crit}
        return {"state": self.state, "questions": {"q": q}}

    def to_json(self) -> str:
        d = asdict(self)
        # Rows without the new fields write the JSON they always did.
        if d["target"] is None:
            del d["target"]
        if not d["fixed_order"]:
            del d["fixed_order"]
        return json.dumps(d, ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            kind=d["kind"],
            state=d["state"],
            instructions=d["instructions"],
            options=[list(o) for o in d["options"]],
            label=int(d["label"]),
            task=d.get("task", "unknown"),
            weight=float(d.get("weight", 1.0)),
            instruction_variants=list(d.get("instruction_variants") or []),
            target=None if d.get("target") is None else [float(p) for p in d["target"]],
            fixed_order=bool(d.get("fixed_order", False)),
        )


def _open(path: str, mode: str) -> IO[str]:
    """Transparent gzip so large corpora do not need a separate decompress step."""
    if path.endswith(".gz"):
        return cast(IO[str], gzip.open(path, mode + "t", encoding="utf-8"))
    return open(path, mode, encoding="utf-8")


def write_jsonl(path: str, examples: Iterable[Example]) -> int:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    n = 0
    with _open(path, "w") as fh:
        for ex in examples:
            fh.write(ex.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path: str) -> Iterator[Example]:
    with _open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield Example.from_dict(json.loads(line))


def load_examples(paths: list[str]) -> list[Example]:
    out: list[Example] = []
    for p in paths:
        out.extend(read_jsonl(p))
    return out


def split_examples(
    examples: list[Example],
    *,
    val_fraction: float = 0.05,
    seed: int = 0,
    group_by_task: bool = True,
) -> tuple[list[Example], list[Example]]:
    """Split train/val, stratified by task by default.

    Stratifying matters here: the corpus mixes tasks with wildly different sizes,
    and an unstratified 5% slice can miss a small task entirely, which then shows
    up as a misleading per-task eval hole.
    """
    rng = random.Random(seed)
    if not group_by_task:
        shuffled = list(examples)
        rng.shuffle(shuffled)
        cut = int(len(shuffled) * (1 - val_fraction))
        return shuffled[:cut], shuffled[cut:]

    by_task: dict[str, list[Example]] = {}
    for ex in examples:
        by_task.setdefault(ex.task, []).append(ex)

    train: list[Example] = []
    val: list[Example] = []
    for task in sorted(by_task):
        items = by_task[task]
        rng.shuffle(items)
        cut = max(1, int(len(items) * (1 - val_fraction))) if len(items) > 1 else 1
        train.extend(items[:cut])
        val.extend(items[cut:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def held_out_task_split(
    examples: list[Example], held_out: list[str]
) -> tuple[list[Example], list[Example]]:
    """Split by task name, for measuring the thing that actually matters.

    In-distribution accuracy on tasks the model trained on says little about whether
    the generic heads learned to bind to prompt-described options. Holding out whole
    tasks -- unseen label sets, unseen domains -- is the honest test of that.
    """
    hold = set(held_out)
    seen = [e for e in examples if e.task not in hold]
    unseen = [e for e in examples if e.task in hold]
    return seen, unseen
