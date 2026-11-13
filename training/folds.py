"""The training file and the held-out folds, as the trainer reads them.

The training file is JSONL, one row per line (`training/convert.py` writes it). Every rank indexes it once (byte
offsets and lengths, with every row checked) and then reads only the rows of its own microbatches.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path

import torch

from strands_decider.core import MAX_CODES

from .batching import row_length
from .metrics import read_jsonl
from .targets import option_labels, target_matrix


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def family_key(row: dict) -> tuple[str, str]:
    """(dataset, family): rows that share it are one item for the held-out folds' purposes."""
    return (str(row["source"].get("dataset", row["suite"])), row["family"])


class TrainIndex:
    """Byte offsets and lengths of the usable rows of a training file.

    Refuses the file on a row without a measured `source.input_tokens` (training/convert.py writes it), a duplicate
    id, an id that is also in a held-out fold, a (dataset, family) shared with a held-out fold (unless allowed), an
    invalid target or more than 255 options. Drops, and counts, rows whose length (prompt + 16) exceeds `max_length`
    and rows with images (the recipe trains on text)."""

    def __init__(
        self,
        path: str | Path,
        limit: int,
        max_length: int,
        protected_ids: set[str],
        protected_families: set[tuple[str, str]],
        allow_family_overlap: bool,
    ) -> None:
        self.path = os.path.abspath(path)
        self.offsets: list[int] = []
        self.lengths: list[int] = []
        self.dropped: dict[str, int] = {"over_max_length": 0, "image_rows": 0}
        self.fixed_order_rows = 0
        prefix = hashlib.sha256()
        ids: set[str] = set()
        family_overlap = 0
        lines = 0
        offset = 0
        with open(self.path, "rb") as stream:
            for raw in stream:
                start, offset = offset, offset + len(raw)
                if limit and lines >= limit:
                    break
                lines += 1
                prefix.update(raw)
                if not raw.strip():
                    continue
                row = json.loads(raw)
                identifier = row["id"]
                if identifier in ids:
                    raise ValueError(f"Duplicate training ID at line {lines}")
                ids.add(identifier)
                if identifier in protected_ids:
                    raise ValueError(
                        f"Training row {lines} has an ID that is also in a held-out fold"
                    )
                if family_key(row) in protected_families:
                    family_overlap += 1
                measured = row["source"].get("input_tokens")
                if not isinstance(measured, int) or isinstance(measured, bool) or measured < 1:
                    raise ValueError(
                        f"Training row {lines} has no measured source.input_tokens: convert the file with "
                        "training.convert"
                    )
                # a bad target fails here, not at step 1,500
                target_matrix([row], torch.device("cpu"))
                if len(option_labels(row["question"])) > MAX_CODES:
                    raise ValueError(f"Training row {lines} has more than {MAX_CODES} options")
                length = row_length(row)
                if length > max_length:
                    self.dropped["over_max_length"] += 1
                    continue
                if row.get("images"):
                    self.dropped["image_rows"] += 1
                    continue
                self.fixed_order_rows += bool(row["source"].get("fixed_order"))
                self.offsets.append(start)
                self.lengths.append(length)
        if family_overlap and not allow_family_overlap:
            raise ValueError(
                f"{family_overlap} training rows share a (dataset, family) with a held-out fold"
            )
        self.lines_read = lines
        self.family_overlap = family_overlap
        self.prefix_sha256 = prefix.hexdigest()
        if not self.offsets:
            raise ValueError("No usable training rows")

    def __len__(self) -> int:
        return len(self.offsets)

    def read(self, indices: Iterable[int]) -> dict[int, dict]:
        out: dict[int, dict] = {}
        with open(self.path, "rb") as stream:
            for index in sorted(set(indices)):
                stream.seek(self.offsets[index])
                out[index] = json.loads(stream.readline())
        return out


def read_fold(path: str | Path | None, limit: int = 0) -> list[dict]:
    """A held-out fold's text rows (rows with images are left out before the limit, so `limit` counts scored rows)."""
    if not path:
        return []
    rows = [row for row in read_jsonl(path) if not row.get("images")]
    return rows[:limit] if limit else rows


def check_conversion(train: str | Path, tokenizer_json: str | Path, prompt: str) -> dict:
    """The training file's conversion manifest (`<train>.manifest.json`, written by training/convert.py), checked: its
    lengths must have been measured with this tokenizer under this template. The trainer also checks the file's
    sha256 against the manifest's."""
    manifest = Path(str(train) + ".manifest.json")
    if not manifest.exists():
        raise ValueError(
            f"{train} has no {manifest.name}: convert it with training.convert, which measures each row's "
            "length under the template"
        )
    record: dict = json.loads(manifest.read_text())
    if record.get("tokenizer_sha256") != sha256_file(tokenizer_json):
        raise ValueError(
            f"{train} was converted with another tokenizer than {tokenizer_json}; convert it for this base"
        )
    if record.get("prompt") != prompt:
        raise ValueError(
            f"{train} was measured under {record.get('prompt')!r}, not {prompt!r}; convert it again"
        )
    return record
