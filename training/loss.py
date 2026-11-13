"""The objective: cross-entropy between the readout's softmax and each row's target distribution, plus an optional
Brier term."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F


class LossTerms(NamedTuple):
    """Scalar tensors: what the optimizer minimises, and its two parts (the Brier part is logged even at weight 0)."""

    total: torch.Tensor
    cross_entropy: torch.Tensor
    brier: torch.Tensor


def training_loss(
    logits: torch.Tensor, target: torch.Tensor, brier_weight: float = 0.0
) -> LossTerms:
    """Mean over the rows of -sum(target * log softmax(logits)), plus `brier_weight` times the mean Brier score.

    `logits` and `target` are [rows, 255]. Codes a question does not use carry a -1e9 logit and a zero target, so they
    take no probability and add nothing. The Brier score is always computed and logged; it enters the total only when
    `brier_weight` is nonzero.
    """
    log_probabilities = F.log_softmax(logits.float(), dim=-1)
    cross_entropy = -(target * log_probabilities).sum(-1).mean()
    brier = (log_probabilities.exp() - target).square().sum(-1).mean()
    total = cross_entropy if brier_weight == 0 else cross_entropy + brier_weight * brier
    return LossTerms(total, cross_entropy, brier)
