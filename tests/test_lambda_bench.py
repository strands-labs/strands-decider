"""deploy/lambda/bench/: the release check's pass/fail rules (conformance_report.py) and the
report's deduplication and memory table (graviton_bench.py report)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "deploy/lambda/bench"))
import conformance_report
import graviton_bench


def _result(tmp_path: Path, name: str, **fields) -> str:
    row = {"label": "g4", "module": "static", "cpu_part": "0xd4f", "questions": 1032, "top1": 0.9971,
           "mean_abs_dp": 0.00376, "max_abs_dp": 0.0508, **fields}  # fmt: skip
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(row))
    return str(path)


def _check(monkeypatch, capsys, paths: list[str]) -> tuple[int, str]:
    monkeypatch.setattr(sys, "argv", ["conformance_report.py", *paths])
    with pytest.raises(SystemExit) as e:
        conformance_report.main()
    return e.value.code, capsys.readouterr().out


def test_a_module_that_matches_static_passes(tmp_path, monkeypatch, capsys):
    static = _result(tmp_path, "static")
    module = _result(tmp_path, "g4", module="graviton4")
    code, out = _check(monkeypatch, capsys, [static, module])
    assert code == 0
    assert "| Graviton4 | model.gguf | graviton4 | 99.71% | 0.00376 | 0.0508 | pass |" in out


def test_a_module_worse_than_static_fails(tmp_path, monkeypatch, capsys):
    """Three more wrong answers of 1,032 than static, past the two allowed."""
    static = _result(tmp_path, "static")
    module = _result(tmp_path, "g4", module="graviton4", top1=0.9971 - 3 / 1032)
    code, out = _check(monkeypatch, capsys, [static, module])
    assert code == 1
    assert "FAIL: 3 more wrong than static" in out


def test_a_mean_error_over_ten_percent_above_static_fails(tmp_path, monkeypatch, capsys):
    static = _result(tmp_path, "static")
    module = _result(tmp_path, "g4", module="graviton4", mean_abs_dp=0.00376 * 1.2)
    code, out = _check(monkeypatch, capsys, [static, module])
    assert code == 1
    assert "1.20x static's" in out


def test_kleidiai_is_reported_but_does_not_fail(tmp_path, monkeypatch, capsys):
    static = _result(tmp_path, "static")
    kleidiai = _result(tmp_path, "kl", label="g4-kleidiai", module="graviton4", top1=0.9864,
                       mean_abs_dp=0.01079, max_abs_dp=0.1304)  # fmt: skip
    code, out = _check(monkeypatch, capsys, [static, kleidiai])
    assert code == 0
    assert "graviton4 + KleidiAI" in out and "opt-in: 11 more wrong than static" in out


def test_a_module_without_static_on_its_host_fails(tmp_path, monkeypatch, capsys):
    module = _result(tmp_path, "g4", module="graviton4")
    code, out = _check(monkeypatch, capsys, [module])
    assert code == 1
    assert "no static result on this host" in out


def test_the_report_counts_a_result_collected_twice_once_and_tables_memory(tmp_path, capsys):
    host = {"label": "variants", "name": "host", "cpu_part": "0xd4f", "rssanon_mb": 5486,
            "vmhwm_mb": 10363, "where": "ecs-ec2"}  # fmt: skip
    request = {"label": "variants", "name": "text40", "rid": 0, "latency_ms": 2655.0,
               "cpu_part": "0xd4f", "rssanon_mb": 5491, "vmhwm_mb": 13282, "where": "ecs-ec2",
               "answers": {"q": 0.5}}  # fmt: skip
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in (host, request, request)))
    graviton_bench.report(SimpleNamespace(results=[str(path)]))
    out = capsys.readouterr().out
    assert "| ecs-ec2 | Graviton4 | text40 | 2,655 [1] |" in out
    assert "| ecs-ec2 | Graviton4 | variants | 5486 | 5491 | 13282 |" in out
