"""The trainer's dev-fold evaluation: the model's logits on the held-out rows, scored by training/metrics.py.

Rows are sorted by length and cut into token-budget microbatches (less padding; the metrics do not depend on the order).
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

import torch

from . import metrics
from . import prompt as prompts
from .batching import pack_microbatches, row_length
from .targets import answer_label, option_index, option_labels


def measure(
    rows: list[dict], tokenizer: Any, prompt: str, max_length: int
) -> tuple[list[dict], int]:
    """(the rows the template renders within `max_length` tokens, no margin, each with `source.input_tokens` set to
    its measured prompt length so dev batching does not depend on how the dev file was converted; the count dropped)."""
    kept = []
    for row in rows:
        try:
            n = prompts.prompt_tokens(row, tokenizer, prompt)
        except (ValueError, TypeError, KeyError):
            continue
        if n <= max_length:
            kept.append({**row, "source": {**row["source"], "input_tokens": n}})
    return kept, len(rows) - len(kept)


@torch.inference_mode()  # type: ignore[untyped-decorator]
def option_logits(model: Any, rows: list[dict], settings: SimpleNamespace) -> list[list[float]]:
    """Each row's valid logits (its first k), in `rows`' order."""
    was_training = model.training
    model.eval()
    result: list[list[float]] = []
    for batch in pack_microbatches(rows, settings.batch_size, settings.token_budget):
        logits = model(model.encode(batch, max_length=settings.max_length)).detach().cpu()
        for row, values in zip(batch, logits, strict=True):
            result.append(values[: len(option_labels(row["question"]))].tolist())
    model.train(was_training)
    return result


def score(
    rows: list[dict], logits: list[list[float]], cal_rows: list[dict], cal_logits: list[list[float]]
) -> dict:
    out: dict[str, Any] = {
        "raw": metrics.summary(
            metrics.score_predictions(metrics.predictions_from_logits(rows, logits))
        )
    }
    if cal_rows:
        answers = [option_index(option_labels(r["question"]), answer_label(r)) for r in cal_rows]
        fitted = metrics.calibrate_temperature(cal_logits, answers)
        out["temperature"] = fitted
        predictions = metrics.predictions_from_logits(rows, logits, fitted)
        out["fitted"] = metrics.summary(metrics.score_predictions(predictions))
    return out


def evaluate(
    model: Any, dev_rows: list[dict], temperature_rows: list[dict], settings: SimpleNamespace
) -> dict:
    """Raw dev metrics, and with a calibration fold the fitted temperature and the dev metrics at it (rank 0 only)."""
    began = time.monotonic()
    rows = sorted(dev_rows, key=row_length)
    logits = option_logits(model, rows, settings)
    cal = sorted(temperature_rows, key=row_length) if temperature_rows else []
    cal_logits = option_logits(model, cal, settings) if cal else []
    out = score(rows, logits, cal, cal_logits)
    out["seconds"] = round(time.monotonic() - began, 1)
    return out
