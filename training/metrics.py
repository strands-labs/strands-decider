"""Scoring the dev fold from the readout's logits: accuracy, negative log-likelihood, Brier score and expected
calibration error over 15 confidence bins; NLL and Brier against a row's soft target too, where it has one; and the
temperature that best calibrates a set of logits."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .targets import Label, answer_label, option_index, option_labels

CALIBRATION_BINS = 15
SUMMARY_KEYS = ("count", "accuracy", "nll", "brier", "ece", "soft_count", "soft_nll", "soft_brier")


def read_jsonl(path: str | Path) -> list[dict]:
    """The rows of a JSONL file, blank lines skipped."""
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def scored_row(
    row: dict,
    probabilities: Sequence[float],
    *,
    prediction: Label | None = None,
    temperature: float = 1.0,
) -> dict:
    """One scored row: its options, probabilities, predicted option (the first most probable), whether it is the label,
    the predicted option's probability (`confidence`), and its soft target (`source.human_distribution`, or a list
    `target`) when it has one."""
    labels = option_labels(row["question"])
    values = [float(value) for value in probabilities]
    if len(values) != len(labels) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError(f"{row['id']}: need one probability in [0, 1] per option")
    if not math.isclose(sum(values), 1, abs_tol=1e-8):
        raise ValueError(f"{row['id']}: the probabilities do not sum to 1")
    label = answer_label(row)
    option_index(labels, label)
    chosen = (
        labels[max(range(len(labels)), key=values.__getitem__)]
        if prediction is None
        else prediction
    )
    chosen_index = option_index(labels, chosen)
    family = row.get("family")
    dataset = row["source"].get("dataset", row["suite"])
    if not isinstance(family, str) or not family or not isinstance(dataset, str):
        raise ValueError(f"{row['id']}: the row names no dataset or family")
    result = {
        "id": row["id"],
        "suite": row["suite"],
        "dataset": dataset,
        "family": family,
        "label": label,
        "options": labels,
        "probabilities": values,
        "prediction": chosen,
        "correct": type(chosen) is type(label) and chosen == label,
        "confidence": values[chosen_index],
        "temperature": temperature,
    }
    human = row["source"].get("human_distribution")
    soft: list[float] | None = None
    if isinstance(human, dict):
        keys = [str(value).lower() if isinstance(value, bool) else str(value) for value in labels]
        if set(human) != set(keys):
            raise ValueError(f"{row['id']}: source.human_distribution's keys are not the options")
        soft = [float(human[key]) for key in keys]
    elif isinstance(row["target"], list):
        soft = [float(value) for value in row["target"]]
    if soft is not None:
        if (
            len(soft) != len(labels)
            or any(not math.isfinite(p) or not 0 <= p <= 1 for p in soft)
            or not math.isclose(sum(soft), 1, abs_tol=1e-6)
        ):
            raise ValueError(f"{row['id']}: the soft target is not a distribution over the options")
        result["soft_target"] = soft
    return result


def check_coverage(rows: Sequence[dict], predictions: Sequence[dict]) -> None:
    """Exactly one prediction per row id, each consistent with its row."""
    expected = {row["id"]: row for row in rows}
    actual = {row["id"]: row for row in predictions}
    if (
        len(expected) != len(rows)
        or len(actual) != len(predictions)
        or expected.keys() != actual.keys()
    ):
        raise ValueError("row ids must be unique, with exactly one prediction for each")
    for identifier, row in expected.items():
        saved = actual[identifier]
        rebuilt = scored_row(
            row,
            saved["probabilities"],
            prediction=saved["prediction"],
            temperature=saved["temperature"],
        )
        for field in ("suite", "dataset", "family", "label", "options", "correct", "confidence"):
            if saved[field] != rebuilt[field]:
                raise ValueError(f"{identifier}: the prediction's {field} does not match its row")


def predictions_from_logits(
    rows: Sequence[dict], logits: Sequence[Sequence[float]], temperature: float = 1.0
) -> list[dict]:
    """Each row's prediction from its valid logits (its first k), softmax at `temperature` in float64."""
    if len(rows) != len(logits) or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("need one logit row per row and a finite temperature above 0")
    result = []
    for row, values in zip(rows, logits, strict=True):
        count = len(option_labels(row["question"]))
        raw = np.asarray(values[:count], dtype=np.float64)
        if len(raw) != count or not np.isfinite(raw).all():
            raise ValueError(f"{row['id']}: fewer logits than options, or one that is not finite")
        shifted = (raw - np.max(raw)) / temperature
        probabilities = np.exp(shifted)
        probabilities /= probabilities.sum()
        prediction = scored_row(row, probabilities.tolist(), temperature=temperature)
        prediction["logits"] = raw.tolist()
        result.append(prediction)
    check_coverage(rows, result)
    return result


def calibrate_temperature(
    logits: Sequence[Sequence[float]], target_indices: Sequence[int]
) -> float:
    """The temperature in [0.05, 20] that minimises the mean NLL of the hard labels: a golden-section search over
    log T (80 iterations), then the best of the search's midpoint, the two bounds and T = 1."""
    if not logits or len(logits) != len(target_indices):
        raise ValueError(
            "calibrating a temperature needs logits and one answer index per row, and at least one row"
        )
    width = max(map(len, logits))
    values = np.full((len(logits), width), -np.inf, dtype=np.float64)
    for index, (row, target) in enumerate(zip(logits, target_indices, strict=True)):
        if not row or not 0 <= target < len(row) or not np.isfinite(row).all():
            raise ValueError(
                "calibrate on each row's finite option logits only, with the answer index among them"
            )
        values[index, : len(row)] = row
    values -= values.max(axis=1, keepdims=True)
    target_logits = values[np.arange(len(values)), np.asarray(target_indices)]

    def mean_nll(log_temperature: float) -> float:
        inverse = math.exp(-log_temperature)
        return float(
            np.mean(np.log(np.exp(values * inverse).sum(axis=1)) - target_logits * inverse)
        )

    low, high = math.log(0.05), math.log(20.0)
    ratio = (math.sqrt(5) - 1) / 2
    left, right = high - ratio * (high - low), low + ratio * (high - low)
    left_loss, right_loss = mean_nll(left), mean_nll(right)
    for _ in range(80):
        if left_loss <= right_loss:
            high, right, right_loss = right, left, left_loss
            left = high - ratio * (high - low)
            left_loss = mean_nll(left)
        else:
            low, left, left_loss = left, right, right_loss
            right = low + ratio * (high - low)
            right_loss = mean_nll(right)
    candidates = [math.log(0.05), (low + high) / 2, math.log(20.0), 0.0]
    return math.exp(min(candidates, key=mean_nll))


def score_predictions(predictions: Sequence[dict]) -> dict:
    """Accuracy, NLL (probabilities floored at 1e-12), Brier, ECE over 15 confidence bins with its reliability table,
    the count of zero probabilities, and soft NLL / Brier over the rows with soft targets."""
    if not predictions:
        raise ValueError("no predictions to score")
    bins: list[list[dict]] = [[] for _ in range(CALIBRATION_BINS)]
    brier, nll, soft_brier, soft_nll = [], [], [], []
    for row in predictions:
        index = option_index(row["options"], row["label"])
        probabilities = row["probabilities"]
        nll.append(-math.log(max(probabilities[index], 1e-12)))
        brier.append(sum((p - int(i == index)) ** 2 for i, p in enumerate(probabilities)))
        bins[min(CALIBRATION_BINS - 1, int(row["confidence"] * CALIBRATION_BINS))].append(row)
        if "soft_target" in row:
            soft = row["soft_target"]
            soft_brier.append(sum((p - q) ** 2 for p, q in zip(probabilities, soft, strict=True)))
            soft_nll.append(
                -sum(q * math.log(max(p, 1e-12)) for p, q in zip(probabilities, soft, strict=True))
            )
    reliability = []
    ece = 0.0
    for index, group in enumerate(bins):
        confidence = sum(row["confidence"] for row in group) / len(group) if group else None
        accuracy = sum(row["correct"] for row in group) / len(group) if group else None
        if confidence is not None and accuracy is not None:
            ece += len(group) / len(predictions) * abs(confidence - accuracy)
        reliability.append(
            {
                "lower": index / CALIBRATION_BINS,
                "upper": (index + 1) / CALIBRATION_BINS,
                "count": len(group),
                "confidence": confidence,
                "accuracy": accuracy,
            }
        )
    result = {
        "count": len(predictions),
        "accuracy": sum(row["correct"] for row in predictions) / len(predictions),
        "ece": ece,
        "brier": sum(brier) / len(predictions),
        "nll": sum(nll) / len(predictions),
        "reliability": reliability,
        "zero_probability_count": sum(p == 0 for row in predictions for p in row["probabilities"]),
    }
    if soft_nll:
        result.update(
            {
                "soft_count": len(soft_nll),
                "soft_nll": sum(soft_nll) / len(soft_nll),
                "soft_brier": sum(soft_brier) / len(soft_brier),
            }
        )
    return result


def summary(result: dict) -> dict:
    """The headline numbers the trainer logs, floats rounded to 5 places."""
    return {
        k: (round(result[k], 5) if isinstance(result[k], float) else result[k])
        for k in SUMMARY_KEYS
        if k in result
    }
