"""The row format the image builders write, shared by them all.

A row is the in-tree `Example` (kind, state, instructions, options, label, task, weight,
instruction_variants) plus:

  images     image paths, relative to the data root; the image goes inside <state>
  ablation   False here; True only on the image-removed copies that training adds
             (by an image trainer, such as strands_decider.vision_train)
  source     the dataset the row came from (README.md gives each one's licence)
  source_id  what the row was derived from (an image or task id), so that a row, the
             other half of its pair and its image-removed copy stay on one side of the
             validation split
  pair_id    ties the two halves of a minimal pair (same question, other image and answer)
"""

from __future__ import annotations

import json
import random
from typing import Any

from strands_decider.prompting import NOUL_DEFAULT_CRITERIA

# Image yes/no questions are trained exactly as they are served: a noul without criteria.
NOUL_DEFAULT = [[k, v] for k, v in NOUL_DEFAULT_CRITERIA.items()]


def row(kind: str, instructions: str, options: list[list[str]], label: int, *, task: str,
        source: str, images: list[str], source_id: str, state: str = "",
        variants: list[str] | None = None, pair_id: str | None = None) -> dict[str, Any]:
    if not 0 <= label < len(options) or len({o[0] for o in options}) != len(options):
        raise ValueError(f"bad options or label: {options}, {label}")
    return {"kind": kind, "state": state, "instructions": instructions, "options": options,
            "label": label, "task": task, "weight": 1.0,
            "instruction_variants": variants or [], "images": images, "ablation": False,
            "source": source, "source_id": source_id, "pair_id": pair_id}


def yesno(question: str, yes: bool, rng: random.Random, **kw: Any) -> dict[str, Any]:
    """A yes/no question: 60% as a noul with the server's default criteria, 40% as a
    yes/no choice (the form NaturalBench and Image JevBench ask in)."""
    if rng.random() < 0.6:
        return row("noul", question, NOUL_DEFAULT, int(yes), **kw)
    return row("choice", question, [["yes", "Yes"], ["no", "No"]], 0 if yes else 1, **kw)


def write(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[build] {path}: {len(rows)} rows")
