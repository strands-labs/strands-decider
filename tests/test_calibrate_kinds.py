"""`strands-decider calibrate --kinds`: refit only the listed primitives, keep the rest.

On a tiny Qwen3.5 checkpoint (tests/tiny_qwen35.py), CPU only. The fits read logits
`collect_logits` would return; they are drawn here (over-confident, so every refit moves
off its starting value) rather than forwarded, which on a CPU takes a minute per call.
"""

from __future__ import annotations

import json
import os
import random

import pytest
import torch
from tiny_qwen35 import needs_tiny_qwen35, save_base, save_checkpoint

import strands_decider.evaluate as evaluate
from strands_decider.data.format import Example
from strands_decider.evaluate import calibrate_checkpoint
from strands_decider.modeling import (
    MASK_VALUE,
    StrandsDeciderConfig,
    config_path,
)

pytestmark = needs_tiny_qwen35

KEPT = {"noul": 0.5, "choice": 2.0, "score": 3.0}


def _rows(n: int = 210) -> list[Example]:
    """`n` rows of each primitive: the fit needs 200 of a kind before it fits that kind."""
    rng = random.Random(0)
    out = []
    for i in range(n):
        out.append(Example("noul", f"Ticket {i} is open.", "Is the ticket open?",
                           [["false", "no"], ["true", "yes"]], rng.randrange(2)))
        out.append(Example("choice", f"Route {i}.", "Which queue?",
                           [["a", "billing"], ["b", "support"], ["c", "sales"]], rng.randrange(3)))
        out.append(Example("score", f"Incident {i}.", "How severe?",
                           [["0", "none"], ["1", "minor"], ["2", "major"], ["3", "critical"]],
                           rng.randrange(4)))
    return out


def _logits(model, examples, **_):
    """Raw logits as collect_logits returns them: right 60% of the time, at a margin of 4."""
    g = torch.Generator().manual_seed(0)
    n = torch.tensor([len(ex.options) for ex in examples])
    labels = torch.tensor([ex.label for ex in examples])
    logits = torch.randn(len(examples), 4, generator=g)
    for i, ex in enumerate(examples):
        top = ex.label if torch.rand(1, generator=g) < 0.6 else (ex.label + 1) % len(ex.options)
        logits[i, top] += 4.0
    logits = logits.masked_fill(torch.arange(4) >= n.unsqueeze(1), MASK_VALUE)
    return logits, labels, n, examples


@pytest.fixture
def ckpt(tmp_path_factory, monkeypatch):
    monkeypatch.setattr(evaluate, "collect_logits", _logits)
    base = save_base(str(tmp_path_factory.mktemp("tiny-qwen35")))
    path = save_checkpoint(base, str(tmp_path_factory.mktemp("ckpt")))
    cfg = StrandsDeciderConfig.from_json(config_path(path))
    cfg.temperature, cfg.temperature_by_kind = 1.3, dict(KEPT)
    with open(config_path(path), "w", encoding="utf-8") as fh:
        fh.write(cfg.to_json())
    return path


def _saved(path: str) -> StrandsDeciderConfig:
    return StrandsDeciderConfig.from_json(config_path(path))


def test_listed_kinds_are_refit_and_the_rest_kept(ckpt):
    out = calibrate_checkpoint(ckpt, _rows(), device="cpu", kinds=["choice", "score"], objective="nll")
    cfg = _saved(ckpt)
    assert cfg.temperature == 1.3 and cfg.temperature_by_kind["noul"] == KEPT["noul"]
    for k in ("choice", "score"):
        assert cfg.temperature_by_kind[k] != KEPT[k]
    assert out["temperature_by_kind"] == cfg.temperature_by_kind


def test_without_kinds_every_temperature_is_refit(ckpt):
    calibrate_checkpoint(ckpt, _rows(), device="cpu")
    cfg = _saved(ckpt)
    assert cfg.temperature != 1.3
    assert all(cfg.temperature_by_kind[k] != KEPT[k] for k in KEPT)


def test_a_kind_with_too_few_rows_keeps_its_temperature(ckpt):
    rows = [ex for ex in _rows() if ex.kind != "score"]
    calibrate_checkpoint(ckpt, rows, device="cpu", kinds=["choice", "score"], objective="nll")
    assert _saved(ckpt).temperature_by_kind["score"] == KEPT["score"]


def test_an_unknown_kind_is_refused_before_the_fit(ckpt):
    with pytest.raises(ValueError, match="kinds must be among"):
        calibrate_checkpoint(ckpt, _rows(1), device="cpu", kinds=["yesno"])


def test_an_unknown_objective_is_refused_before_the_fit(ckpt):
    with pytest.raises(ValueError, match="objective must be"):
        calibrate_checkpoint(ckpt, _rows(1), device="cpu", objective="brier")


def test_cli_pools_every_data_file_and_parses_kinds(ckpt, tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from strands_decider.cli import app

    files = []
    for i, part in enumerate((_rows(3)[:4], _rows(3)[4:])):
        p = tmp_path / f"part{i}.jsonl"
        p.write_text("".join(ex.to_json() + "\n" for ex in part))
        files.append(str(p))
    seen = {}

    def fake(checkpoint, examples, **kw):
        seen.update(checkpoint=checkpoint, n=len(examples), **kw)
        return {"temperature": 1.0, "temperature_by_kind": {}, "before": {}, "after": {}}

    monkeypatch.setattr(evaluate, "calibrate_checkpoint", fake)
    res = CliRunner().invoke(app, ["calibrate", ckpt, "--data", files[0], "--data", files[1],
                                   "--split", "all", "--kinds", "choice, score", "--objective", "nll"])
    assert res.exit_code == 0, res.output
    assert (seen["n"], seen["kinds"], seen["objective"]) == (9, ["choice", "score"], "nll")
    res = CliRunner().invoke(app, ["calibrate", ckpt, "--data", files[0], "--split", "all"])
    assert res.exit_code == 0, res.output
    assert (seen["n"], seen["kinds"], seen["objective"]) == (4, None, "ece")
    # --limit caps each file, not the pool
    res = CliRunner().invoke(app, ["calibrate", ckpt, "--data", files[0], "--data", files[1],
                                   "--split", "all", "--limit", "3"])
    assert res.exit_code == 0, res.output
    assert seen["n"] == 6
    assert json.loads(open(os.path.join(ckpt, "strands_decider_config.json")).read())["temperature"] == 1.3
