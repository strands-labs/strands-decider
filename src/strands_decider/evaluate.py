"""Evaluation and temperature calibration.

Accuracy alone does not tell you whether this model is usable. The API promises
*calibrated* probabilities, and downstream code routes on confidence thresholds
("< 0.5 -> human"). A model that is 70% accurate but claims 0.99 confidence is
worse than useless there, because it fails silently.

So the metrics here are accuracy plus:

  ECE   expected calibration error -- gap between claimed confidence and observed
        accuracy, bucketed. This is the number that decides whether thresholds mean
        anything.
  NLL   negative log-likelihood, the proper scoring rule.
  MAE   for score questions only: mean absolute error in level units, which respects
        ordinality where accuracy does not.

`fit_temperature` then divides logits by a single scalar chosen to minimise val NLL,
and `fit_temperature_by_kind` fits one temperature per primitive (noul, choice, score)
that minimises ECE where a primitive has enough rows. `calibrate_checkpoint` saves both;
a per-kind value takes precedence where fitted. Neither can change any argmax, so
accuracy is untouched -- they only fix over- or under-confidence, which is exactly the
defect that breaks threshold routing.

`calibrate_checkpoint(..., kinds=[...])` refits only the listed primitives and keeps the
checkpoint's other temperatures, global one included: a yes/no temperature that makes
answers decisive is not refitted, and so not softened, by a set that only needs its
choice or score temperatures corrected.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .data.collate import CollatorConfig, SystemOneCollator
from .data.format import Example
from .modeling import MASK_VALUE, StrandsDeciderModel, config_path, masked_log_softmax
from .schema import derive_confidence, derive_score_confidence
from .train import ExampleDataset

KINDS = ("noul", "choice", "score")


def partition_examples(examples: list[Example], part: str, *, seed: int = 0) -> list[Example]:
    """Deterministically split a corpus into a calibration half and a test half.

    Temperature must not be fitted on the same rows it is then scored on -- doing so
    reports a calibration error the model would not achieve on unseen data. Rather
    than making the caller juggle two files, both `strands-decider calibrate` and `strands-decider eval`
    partition the same file the same way, so `--split calib` and `--split test` are
    guaranteed disjoint across separate invocations.

    The hash is over (seed, position) rather than content, so an example appearing
    twice in a corpus cannot land in both halves.
    """
    if part == "all":
        return examples
    if part not in ("calib", "test"):
        raise ValueError(f"split must be one of calib|test|all, got {part!r}")
    want = 0 if part == "calib" else 1
    return [
        ex for i, ex in enumerate(examples)
        if (hashlib.sha256(f"{seed}:{i}".encode()).digest()[0] & 1) == want
    ]


def sample_examples(examples: list[Example], limit: int, *, seed: int = 0) -> list[Example]:
    """Take `limit` examples representatively, not as a file-order prefix.

    Corpora built by `strands-decider data build` are grouped by task, so truncating the head
    of the list silently evaluates a single task and reports it as an overall number.
    A deterministic shuffle before truncation keeps every task represented in
    proportion, and the fixed seed keeps successive eval runs comparable.
    """
    if limit <= 0 or limit >= len(examples):
        return examples
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


@dataclass
class Prediction:
    task: str
    kind: str
    n_slots: int
    label: int
    probs: list[float]
    ordinal_smoothing: float = 0.0

    @property
    def pred(self) -> int:
        return max(range(len(self.probs)), key=lambda i: self.probs[i])

    @property
    def correct(self) -> bool:
        return self.pred == self.label

    @property
    def confidence(self) -> float:
        """Must match what the server reports, or ECE measures the wrong thing."""
        if self.kind == "score":
            return derive_score_confidence(
                self.probs, ordinal_smoothing=self.ordinal_smoothing
            )
        return derive_confidence(self.probs)


@torch.no_grad()  # type: ignore[untyped-decorator]
def collect_logits(
    model: StrandsDeciderModel,
    examples: list[Example],
    *,
    device: str = "cuda",
    batch_size: int = 16,
    max_length: int = 3072,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[Example]]:
    """Run the model once and keep the logits, so temperature sweeps need no re-forward.

    The logits are untempered. In each row, the columns at or past the row's option
    count hold MASK_VALUE, so a raw argmax over the row lands on one of its options.
    """
    model.eval().to(device)
    coll = SystemOneCollator(
        model.tokenizer,
        CollatorConfig(
            max_length=max_length,
            num_slots=model.config.num_slots,
            head_type=model.config.head_type,
        ),
        train=False,  # no option shuffling: evaluation must be deterministic
    )
    loader = DataLoader(
        ExampleDataset(examples), batch_size=batch_size, shuffle=False, collate_fn=coll
    )

    all_logits, all_labels, all_slots = [], [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            n_slots=batch["n_slots"],
            opt_idx=batch.get("opt_idx"),
            temperature=1.0,  # raw logits; calibration is applied afterwards
        )
        logits = out["logits"].float()
        # A pointer head scores a padded option slot from the hidden state at position
        # 0 (gather_options clamps the -1 index), so in a mixed-width batch these
        # columns hold real values and can win a raw argmax. Mask them here, once, for
        # every reader; masked_log_softmax would write the same value into them.
        cols = torch.arange(logits.size(1), device=logits.device)
        logits = logits.masked_fill(cols >= batch["n_slots"].unsqueeze(1), MASK_VALUE)
        all_logits.append(logits.cpu())
        all_labels.append(batch["labels"].cpu())
        all_slots.append(batch["n_slots"].cpu())

    # A pointer head emits one logit per option, so each batch is only as wide as its
    # own widest question. Pad to the widest across batches before concatenating;
    # MASK_VALUE keeps the extra columns inert under masked_log_softmax.
    width = max(t.size(1) for t in all_logits)
    all_logits = [
        t if t.size(1) == width
        else F.pad(t, (0, width - t.size(1)), value=MASK_VALUE)
        for t in all_logits
    ]
    return (
        torch.cat(all_logits),
        torch.cat(all_labels),
        torch.cat(all_slots),
        examples,
    )


def predictions_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_slots: torch.Tensor,
    examples: list[Example],
    temperature: Any = 1.0,
    ordinal_smoothing: float = 0.0,
) -> list[Prediction]:
    """`temperature` is a scalar, or a {kind: temperature} map for per-primitive fits."""
    if isinstance(temperature, dict):
        t = torch.tensor(
            [float(temperature.get(ex.kind, 1.0)) for ex in examples],
            dtype=torch.float32,
        ).view(-1, 1)
    else:
        t = float(temperature)
    probs = masked_log_softmax(logits / t, n_slots).exp()
    out = []
    for i, ex in enumerate(examples):
        k = int(n_slots[i])
        out.append(
            Prediction(
                task=ex.task,
                kind=ex.kind,
                n_slots=k,
                label=int(labels[i]),
                probs=probs[i, :k].tolist(),
                ordinal_smoothing=ordinal_smoothing,
            )
        )
    return out


def fit_temperature_by_kind(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_slots: torch.Tensor,
    examples: list[Example],
    *,
    min_examples: int = 200,
    objective: str = "ece",
    ordinal_smoothing: float = 0.0,
    kinds: Sequence[str] = KINDS,
) -> dict[str, float]:
    """Fit one temperature per primitive in `kinds`.

    noul, choice and score sit at very different accuracies, and a single scalar
    fitted across all three lands between them -- over-softening the easy primitive
    while under-softening the hard one. Kinds with too few examples to fit reliably
    are left out and fall back to the global temperature.
    """
    out: dict[str, float] = {}
    for kind in kinds:
        idx = [i for i, ex in enumerate(examples) if ex.kind == kind]
        if len(idx) < min_examples:
            continue
        sel = torch.tensor(idx, dtype=torch.long)
        out[kind] = float(
            fit_temperature(
                logits[sel], labels[sel], n_slots[sel],
                objective=objective,
                examples=[examples[i] for i in idx],
                ordinal_smoothing=ordinal_smoothing,
            )
        )
    return out


def expected_calibration_error(preds: list[Prediction], n_bins: int = 10) -> float:
    """Bucket by confidence, compare mean confidence to accuracy within each bucket."""
    if not preds:
        return 0.0
    bins: dict[int, list[Prediction]] = defaultdict(list)
    for p in preds:
        b = min(n_bins - 1, int(p.confidence * n_bins))
        bins[b].append(p)

    total = len(preds)
    ece = 0.0
    for items in bins.values():
        acc = sum(1 for p in items if p.correct) / len(items)
        conf = sum(p.confidence for p in items) / len(items)
        ece += (len(items) / total) * abs(acc - conf)
    return ece


def negative_log_likelihood(preds: list[Prediction]) -> float:
    if not preds:
        return 0.0
    return -sum(math.log(max(1e-12, p.probs[p.label])) for p in preds) / len(preds)


def score_mae(preds: list[Prediction]) -> float | None:
    """Mean absolute error in level units, over score questions only.

    Uses the expected value rather than the argmax, because that is the number the
    API actually returns for a score.
    """
    items = [p for p in preds if p.kind == "score"]
    if not items:
        return None
    total = 0.0
    for p in items:
        expected = sum(i * q for i, q in enumerate(p.probs))
        total += abs(expected - p.label)
    return total / len(items)


def summarise(preds: list[Prediction], *, by_task: bool = True) -> dict[str, object]:
    def block(items: list[Prediction]) -> dict[str, float]:
        if not items:
            return {}
        d: dict[str, float] = {
            "n": len(items),
            "accuracy": sum(1 for p in items if p.correct) / len(items),
            "ece": expected_calibration_error(items),
            "nll": negative_log_likelihood(items),
            "mean_confidence": sum(p.confidence for p in items) / len(items),
        }
        mae = score_mae(items)
        if mae is not None:
            d["score_mae"] = mae
        return d

    result: dict[str, object] = {"overall": block(preds)}
    result["by_kind"] = {
        kind: block([p for p in preds if p.kind == kind]) for kind in ("noul", "choice", "score")
    }
    if by_task:
        tasks = sorted({p.task for p in preds})
        result["by_task"] = {t: block([p for p in preds if p.task == t]) for t in tasks}

    # Does confidence actually separate right from wrong? If accuracy above the 0.9
    # band is not clearly higher than below 0.5, threshold routing buys nothing.
    result["by_confidence_band"] = {
        band: block([p for p in preds if lo <= p.confidence < hi])
        for band, (lo, hi) in {
            "low(<0.5)": (0.0, 0.5),
            "mid(0.5-0.9)": (0.5, 0.9),
            "high(>=0.9)": (0.9, 1.01),
        }.items()
    }
    return result


def ece_at_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_slots: torch.Tensor,
    examples: list[Example],
    temperature: float,
    ordinal_smoothing: float = 0.0,
) -> float:
    preds = predictions_from_logits(
        logits, labels, n_slots, examples, temperature, ordinal_smoothing
    )
    return expected_calibration_error(preds)


def fit_temperature(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_slots: torch.Tensor,
    *,
    lo: float = 0.25,
    hi: float = 6.0,
    iters: int = 60,
    objective: str = "nll",
    examples: list[Example] | None = None,
    ordinal_smoothing: float = 0.0,
) -> float:
    """Fit the temperature that minimises `objective` ("nll" or "ece").

    ECE is the right target when downstream code routes on confidence thresholds,
    and the two objectives genuinely disagree: fitting noul to NLL on this model
    chose T=2.63, which left it under-confident by 0.29 -- worse calibrated than no
    scaling at all, even though the likelihood improved.

    ECE is piecewise-constant in T (it bins), so golden-section search -- which
    assumes a smooth unimodal objective -- can settle on a flat step. A grid is
    slower but reliable; NLL keeps the golden-section path since it is smooth.
    """
    if objective == "ece":
        if examples is None:
            raise ValueError("fitting to ECE needs the examples, to pick per-kind confidence")
        grid = [lo * (hi / lo) ** (i / (iters - 1)) for i in range(iters)]
        return float(min(
            grid,
            key=lambda t: ece_at_temperature(
                logits, labels, n_slots, examples, t, ordinal_smoothing
            ),
        ))
    if objective != "nll":
        raise ValueError(f"objective must be 'nll' or 'ece', got {objective!r}")

    def nll(t: float) -> float:
        lp = masked_log_softmax(logits / t, n_slots)
        return float(-lp.gather(1, labels.view(-1, 1)).mean())

    invphi = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - invphi * (b - a), a + invphi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - invphi * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + invphi * (b - a)
            fd = nll(d)
    return (a + b) / 2


def evaluate_checkpoint(
    checkpoint: str,
    examples: list[Example],
    *,
    device: str = "cuda",
    batch_size: int = 16,
    apply_temperature: bool = True,
) -> dict[str, Any]:
    model = StrandsDeciderModel.load(checkpoint)
    logits, labels, slots, exs = collect_logits(
        model, examples, device=device, batch_size=batch_size,
        max_length=model.config.max_length,
    )
    if apply_temperature:
        by_kind = getattr(model.config, "temperature_by_kind", None) or {}
        t: Any = by_kind or model.config.temperature
    else:
        t = 1.0
    preds = predictions_from_logits(
        logits, labels, slots, exs,
        temperature=t, ordinal_smoothing=model.config.ordinal_smoothing,
    )
    result = summarise(preds)
    result["temperature"] = t
    return result


def calibrate_checkpoint(
    checkpoint: str,
    examples: list[Example],
    *,
    device: str = "cuda",
    batch_size: int = 16,
    kinds: Sequence[str] | None = None,
    objective: str = "ece",
) -> dict[str, Any]:
    """Fit temperature on a held-out split and write it into the checkpoint config.

    With `kinds`, only those primitives' temperatures are refitted (by `objective`); the
    checkpoint's global temperature and its other per-kind ones are kept.
    """
    if kinds is not None and set(kinds) - set(KINDS):
        raise ValueError(f"kinds must be among {KINDS}, got {list(kinds)}")
    if objective not in ("ece", "nll"):
        raise ValueError(f"objective must be 'ece' or 'nll', got {objective!r}")
    import os

    # The fit ends by writing the config json into the checkpoint, so a Hub repo id,
    # which load() accepts, would lose the whole fit at the end. Refuse it first.
    if not os.path.isdir(checkpoint):
        raise FileNotFoundError(
            f"{checkpoint}: calibration writes into the checkpoint, so it needs a local "
            f"directory, not a Hub repo id"
        )
    model = StrandsDeciderModel.load(checkpoint)
    logits, labels, slots, exs = collect_logits(
        model, examples, device=device, batch_size=batch_size,
        max_length=model.config.max_length,
    )

    eps = model.config.ordinal_smoothing
    before = summarise(predictions_from_logits(logits, labels, slots, exs, 1.0, eps))
    if kinds is None:
        t = fit_temperature(logits, labels, slots)
        by_kind = fit_temperature_by_kind(
            logits, labels, slots, exs, objective=objective, ordinal_smoothing=eps
        )
    else:
        t = model.config.temperature
        by_kind = {**model.config.temperature_by_kind, **fit_temperature_by_kind(
            logits, labels, slots, exs, objective=objective, ordinal_smoothing=eps, kinds=kinds
        )}
    global_after = summarise(predictions_from_logits(logits, labels, slots, exs, t, eps))
    after = summarise(predictions_from_logits(logits, labels, slots, exs, by_kind or t, eps))

    model.config.temperature = float(t)
    model.config.temperature_by_kind = by_kind
    # The same file `load` reads: strands_decider_config.json, or the legacy name.
    with open(config_path(checkpoint), "w", encoding="utf-8") as fh:
        fh.write(model.config.to_json())

    return {
        "temperature": float(t),
        "temperature_by_kind": by_kind,
        "before": before["overall"],
        "after_global": global_after["overall"],
        "after": after["overall"],
    }
