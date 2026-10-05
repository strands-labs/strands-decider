"""Image temperatures fitted and applied from stored results (evaluation/vision/temps.py):
the logits are recovered exactly from probabilities stored at a known temperature."""
from __future__ import annotations

import json
import math
import random

import pytest
import torch
from vision import temps  # evaluation/ is on the test path


def _rows(t_true: float, t0: float, n: int = 3000, seed: int = 0) -> list[dict]:
    """Rows whose gold is drawn from softmax(z / t_true), stored as run.py stores them at t0."""
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        width = rng.choice([2, 4])
        z = [rng.gauss(0, 3) for _ in range(width)]
        p_true = torch.softmax(torch.tensor(z) / t_true, -1).tolist()
        gold = rng.choices(range(width), p_true)[0]
        rows.append({"id": f"r{k}", "kind": "noul" if width == 2 else "choice", "gold": gold,
                     "probs": torch.softmax(torch.tensor(z, dtype=torch.float64) / t0, -1).tolist()})
    return rows


def test_fit_recovers_the_temperature_per_kind():
    rows = [r for r in _rows(0.6, 1.3) if r["kind"] == "noul"] + \
        [r for r in _rows(2.0, 0.8, seed=1) if r["kind"] == "choice"]
    got = temps.fit(rows, {"noul": 1.3, "choice": 0.8})
    assert got["noul"] == pytest.approx(0.6, rel=0.1)
    assert got["choice"] == pytest.approx(2.0, rel=0.1)


def test_rescale_is_exact():
    rows = _rows(t_true=1.0, t0=0.9, n=50)
    same = temps.rescale(rows, {"noul": 0.9, "choice": 0.9}, {})
    assert all(a["probs"] == pytest.approx(b["probs"], abs=1e-9) for a, b in zip(rows, same, strict=True))
    out = temps.rescale(rows, {"noul": 0.9, "choice": 0.9}, {"noul": 0.45})
    for a, b in zip(rows, out, strict=True):
        z = [0.9 * math.log(p) for p in a["probs"]]
        t = 0.45 if a["kind"] == "noul" else 0.9
        assert b["probs"] == pytest.approx(torch.softmax(torch.tensor(z, dtype=torch.float64) / t, -1).tolist())


def test_apply_rescales_image_rows_and_keeps_blind_rows(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    rows = [dict(r, bench="pope_adversarial") for r in _rows(1.0, 0.9, n=40) if r["kind"] == "noul"]
    for tag in ("strands-v19", "strands-v19-blind"):
        (run / f"{tag}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (run / "summary.json").write_text(json.dumps({"versions": {"torch": "x"},
                                                  "temperatures": {"image": {"noul": 0.9}}}))
    (tmp_path / "t.json").write_text(json.dumps({"noul": 0.45}))
    monkeypatch.setattr("sys.argv", ["temps.py", "apply", "--run", str(run), "--temps",
                                     str(tmp_path / "t.json"), "--out", str(tmp_path / "out")])
    temps.main()
    out = tmp_path / "out"
    assert (out / "strands-v19-blind.jsonl").read_text() == (run / "strands-v19-blind.jsonl").read_text()
    sharp = [json.loads(x) for x in (out / "strands-v19.jsonl").read_text().splitlines()]
    assert all(max(b["probs"]) >= max(a["probs"]) for a, b in zip(rows, sharp, strict=True))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["temperatures"]["image"] == {"noul": 0.45} and summary["versions"] == {"torch": "x"}
