"""The token-readout path end to end on the CPU: decision rows converted, two seeds trained a few steps and merged,
averaged and exported to the spec's checkpoint, which the core loads; and two ranks against one. These are smoke tests
of the whole path: the comparison with the previous trainer's outputs is the fixtures' (test_batching.py and the
others), and the end-to-end comparison with that trainer was run outside the suite (training/README.md). A tiny bf16 Gemma 4 base built in the test
with Gemma 4's real tokenizer (the core renders with it): set STRANDS_DECIDER_GEMMA4_TOKENIZER, or the test skips."""

from __future__ import annotations

import gzip
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def _base(directory: Path, tokenizer: str) -> Path:
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    torch.manual_seed(0)
    text = Gemma4TextConfig(
        vocab_size=262144, vocab_size_per_layer_input=262144, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=32,
        hidden_size_per_layer_input=8, num_kv_shared_layers=1, sliding_window=8, max_position_embeddings=4096,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        final_logit_softcapping=30.0,
    )  # fmt: skip
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text.to_dict(), vision_config=None, audio_config=None)
    )
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() >= 2:
                p.normal_(0, 0.1)
    model.to(torch.bfloat16).save_pretrained(str(directory))
    shutil.copy(tokenizer, directory / "tokenizer.json")
    (directory / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "GemmaTokenizer"})
    )
    return directory


def _rows(directory: Path) -> Path:
    """Decision rows of the three types, in the layout `--mix` reads."""
    rng = random.Random(1)
    words = "parcel refund damaged late invoice customer order shipping billing account".split()

    def sentence(n: int) -> str:
        return " ".join(rng.choice(words) for _ in range(n)).capitalize() + "."

    def row(i: int, split: str) -> dict:
        kind = ("choice", "noul", "score")[i % 3]
        if kind == "noul":
            options = [{"key": "false", "text": None}, {"key": "true", "text": None}]
        else:
            options = [{"key": f"o{j}", "text": sentence(3)} for j in range(rng.choice([2, 3, 4]))]
        p = [rng.random() for _ in options]
        return {"id": f"{split}{i}", "lane": "l", "source": f"ds{i % 4}", "group": f"{split}g{i}", "type": kind,
                "state": sentence(rng.choice([5, 30, 90])), "instructions": sentence(6), "options": options,
                "target": [x / sum(p) for x in p], "meta": {}}  # fmt: skip

    (directory / "train").mkdir(parents=True)
    with gzip.open(directory / "train" / "part-00000.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(row(i, "t")) + "\n" for i in range(96))
    with gzip.open(directory / "dev.jsonl.gz", "wt") as f:
        f.writelines(json.dumps(row(i, "d")) + "\n" for i in range(12))
    return directory


def _env() -> dict:
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "2", "CUDA_VISIBLE_DEVICES": ""}
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


def _run(*args: str) -> None:
    res = subprocess.run(
        [sys.executable, "-m", *args], cwd=ROOT, env=_env(), capture_output=True, text=True
    )
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]


def _train_flags(base: Path, data: Path, output: Path, seed: int) -> list[str]:
    return [
        "training.train", "--base", str(base), "--base-model", "google/gemma-4-E2B-it", "--revision", REVISION,
        "--train", str(data / "train.jsonl"), "--dev", str(data / "dev.jsonl"), "--output", str(output),
        "--rank", "4", "--alpha", "8", "--rows-per-step", "16", "--batch-size", "4", "--token-budget", "2048",
        "--max-length", "1024", "--stop-after", "3", "--eval-every", "3", "--seed", str(seed), "--cpu-threads", "2",
        "--targets", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,per_layer_input_gate",
    ]  # fmt: skip


def test_rows_train_merge_average_and_export(gemma4_tokenizer, tmp_path):
    from strands_decider import core

    base = _base(tmp_path / "base", os.environ["STRANDS_DECIDER_GEMMA4_TOKENIZER"])
    mix = _rows(tmp_path / "mix")
    _run(
        "training.convert",
        "--mix",
        str(mix),
        "--base",
        str(base),
        "--out",
        str(tmp_path / "data"),
        "--workers",
        "1",
    )
    merged = []
    for seed in (1, 2):
        run = tmp_path / f"run-{seed}"
        _run(*_train_flags(base, tmp_path / "data", run, seed))
        steps = [json.loads(line) for line in open(run / "train_log.jsonl")]
        assert [s["step"] for s in steps] == [1, 2, 3] and all(s["loss"] > 0 for s in steps)
        merged.append(run / "merged" / "step-00003")
        record = json.loads((merged[-1] / "training.json").read_text())
        assert (record["prompt"], record["attention_mode"]) == ("strands-chat", "prefix_bidir")
    _run("training.average", "--out", str(tmp_path / "average"), *map(str, merged))
    _run(
        "training.export", "--checkpoint", str(tmp_path / "average"), "--out", str(tmp_path / "export"),
        "--release", "strands-decider-tiny", "--source", "https://github.com/strands-labs/strands-decider",
    )  # fmt: skip
    config = core.DecisionConfig.load(str(tmp_path / "export"))
    assert config.prompt == "strands-chat"
    assert not list((tmp_path / "export").glob("*.py"))


@pytest.mark.distributed
def test_two_ranks_train_what_one_process_trains(gemma4_tokenizer, tmp_path):
    """torchrun with 2 ranks (gloo, rendezvous on 127.0.0.1) against one process, dropout off: the same steps, losses
    and gradient norms to float rounding of the accumulated sums, and the same adapter and readout."""
    import socket

    from safetensors.torch import load_file

    base = _base(tmp_path / "base", os.environ["STRANDS_DECIDER_GEMMA4_TOKENIZER"])
    mix = _rows(tmp_path / "mix")
    _run(
        "training.convert",
        "--mix",
        str(mix),
        "--base",
        str(base),
        "--out",
        str(tmp_path / "data"),
        "--workers",
        "1",
    )
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    one, two = tmp_path / "one", tmp_path / "two"
    flags = ("--dropout", "0", "--no-merge")
    _run(*_train_flags(base, tmp_path / "data", one, 1), *flags)
    _run(
        "torch.distributed.run", "--nnodes=1", "--nproc_per_node=2", "--master-addr=127.0.0.1",
        f"--master-port={port}", "-m", *_train_flags(base, tmp_path / "data", two, 1), *flags,
    )  # fmt: skip
    a = [json.loads(line) for line in open(one / "train_log.jsonl")]
    b = [json.loads(line) for line in open(two / "train_log.jsonl")]
    assert [s["rows"] for s in a] == [s["rows"] for s in b] and len(a) == 3
    for x, y in zip(a, b, strict=True):
        assert y["loss"] == pytest.approx(x["loss"], rel=1e-5)
        assert y["grad_norm"] == pytest.approx(x["grad_norm"], rel=1e-4)
    p = load_file(str(one / "last" / "adapter" / "adapter.safetensors"))
    q = load_file(str(two / "last" / "adapter" / "adapter.safetensors"))
    for k in p:
        torch.testing.assert_close(q[k], p[k], rtol=1e-4, atol=1e-6)
