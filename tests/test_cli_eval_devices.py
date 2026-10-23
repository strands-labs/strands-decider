"""Evaluation CLI device selection, without downloading a checkpoint."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import pytest
import torch
from typer.testing import CliRunner

from strands_decider import cli
from training import evaluate
from training.data.format import Example, read_jsonl, write_jsonl


@pytest.fixture
def rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(
        str(path),
        [
            Example("noul", f"state {i}", "Is this true?", [["false", ""], ["true", ""]], i % 2)
            for i in range(12)
        ],
    )
    return path


@pytest.fixture
def calls(monkeypatch):
    captured = []

    def evaluate_checkpoint(checkpoint, examples, **kwargs):
        captured.append((checkpoint, examples, kwargs))
        return {
            "overall": {"n": len(examples), "accuracy": 0.5, "ece": 0.0, "nll": 0.7},
            "temperature": 1.0,
            "by_kind": {},
            "by_task": {},
            "by_confidence_band": {},
        }

    def calibrate_checkpoint(checkpoint, examples, **kwargs):
        captured.append((checkpoint, examples, kwargs))
        return {"temperature": 1.0, "before": {}, "after": {}}

    monkeypatch.setattr(evaluate, "evaluate_checkpoint", evaluate_checkpoint)
    monkeypatch.setattr(evaluate, "calibrate_checkpoint", calibrate_checkpoint)
    return captured


@pytest.mark.parametrize("command", ["eval", "calibrate"])
@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, "cuda"),
        (True, False, "cuda"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_auto_device_reaches_both_commands(rows, calls, monkeypatch, command, cuda, mps, expected):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
    result = CliRunner().invoke(evaluate.app, [command, "checkpoint", "--data", str(rows)])
    assert result.exit_code == 0, result.output
    [(checkpoint, examples, kwargs)] = calls
    assert checkpoint == "checkpoint"
    assert kwargs["device"] == expected
    part = "test" if command == "eval" else "calib"
    assert examples == evaluate.partition_examples(list(read_jsonl(str(rows))), part)
    assert kwargs["batch_size"] == 16
    if command == "eval":
        assert kwargs["apply_temperature"] is True


@pytest.mark.parametrize("command", ["eval", "calibrate"])
@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:0"])
def test_explicit_device_bypasses_auto_selection(rows, calls, monkeypatch, command, device):
    monkeypatch.setattr(cli, "_auto_device", lambda: pytest.fail("explicit device was ignored"))
    result = CliRunner().invoke(
        evaluate.app,
        [
            command,
            "checkpoint",
            "--data",
            str(rows),
            "--device",
            device,
            "--split",
            "all",
            "--limit",
            "3",
            "--batch-size",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    [(_, examples, kwargs)] = calls
    assert kwargs["device"] == device
    assert kwargs["batch_size"] == 2
    assert examples == evaluate.sample_examples(list(read_jsonl(str(rows))), 3)


def test_eval_keeps_raw_mode_and_report_output(rows, calls, tmp_path):
    out = tmp_path / "reports" / "eval.json"
    result = CliRunner().invoke(
        evaluate.app,
        [
            "eval",
            "checkpoint",
            "--data",
            str(rows),
            "--device",
            "cpu",
            "--split",
            "all",
            "--raw",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert calls[0][2]["apply_temperature"] is False
    assert json.loads(out.read_text())["overall"]["n"] == 12


@pytest.mark.parametrize("command", ["eval", "calibrate"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_eval_and_calibrate_silence_hub_kernels_on_non_cuda(rows, calls, command, device):
    """serve/ask lower hub_kernels to ERROR on cpu/mps; eval/calibrate must too."""
    logger = logging.getLogger("transformers.integrations.hub_kernels")
    original = logger.level
    logger.setLevel(logging.WARNING)
    try:
        result = CliRunner().invoke(
            evaluate.app, [command, "checkpoint", "--data", str(rows), "--device", device]
        )
        assert result.exit_code == 0, result.output
        assert logger.level == logging.ERROR
    finally:
        logger.setLevel(original)


@pytest.mark.parametrize("command", ["eval", "calibrate"])
def test_help_documents_device_selection(command):
    result = CliRunner().invoke(evaluate.app, [command, "--help"])
    assert result.exit_code == 0
    # Rich splits "--device" with ANSI colour codes when FORCE_COLOR is set (GitHub Actions).
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    assert "--device" in plain
    assert "Auto-detected" in plain


@pytest.mark.parametrize("explicit", [False, True])
def test_real_cpu_checkpoint_calibrates_and_evaluates(rows, tmp_path, monkeypatch, explicit):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model

    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel, config_path

    core = Tokenizer(models.WordLevel({"<pad>": 0, "<unk>": 1}, unk_token="<unk>"))
    core.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=core, pad_token="<pad>", unk_token="<unk>")
    base = tmp_path / "base"
    Qwen3Model(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
        )
    ).save_pretrained(base)
    config = StrandsDeciderConfig(
        base_model=str(base),
        use_lora=False,
        pointer_dim=8,
        torch_dtype="float32",
        max_length=128,
    )
    model = StrandsDeciderModel(config, Qwen3Model.from_pretrained(base), tok)
    checkpoint = tmp_path / "checkpoint"
    model.save_pretrained(str(checkpoint))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    device_args = ["--device", "cpu"] if explicit else []
    common = [str(checkpoint), "--data", str(rows), "--batch-size", "2", *device_args]

    result = CliRunner().invoke(evaluate.app, ["calibrate", *common])
    assert result.exit_code == 0, result.output
    saved = json.loads(Path(config_path(str(checkpoint))).read_text(encoding="utf-8"))
    assert 0.25 <= saved["temperature"] <= 6.0
    assert model.config.temperature == 1.0

    out = tmp_path / "eval.json"
    result = CliRunner().invoke(evaluate.app, ["eval", *common, "--out", str(out)])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["temperature"] == (saved["temperature_by_kind"] or saved["temperature"])
    assert report["overall"]["n"] == len(
        evaluate.partition_examples(list(read_jsonl(str(rows))), "test")
    )
