"""The trainer on several processes trains what one process trains, and v1's sharded labelling writes the
single-process file.

CPU only: a tiny randomly initialised Qwen3 torso with LoRA and the pointer head, and a byte-level BPE tokenizer
trained in the test, so nothing is downloaded. The trainer test launches `python -m training.train` under torchrun at
2 ranks (DDP over gloo, rendezvous on 127.0.0.1) and as one process, dropout off (its masks are drawn per rank), and
compares every step's rows, loss and gradient norm, the dev metrics and the trained weights. It takes about a minute,
so it is marked `distributed` and runs only with `pytest -m distributed` (`-m ""` runs everything).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from tiny_support import _examples, _load_tokenizer, _tokenizer_file, tiny_base

from training.data.format import write_jsonl

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("ddp_corpus")
    train = _examples(150, seed=1)
    write_jsonl(str(d / "train.jsonl"), train)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    _tokenizer_file(str(d / "tokenizer.json"), train)
    return d


# ---------------------------------------------------------------- the trainer on several processes


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _env() -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([os.path.join(ROOT, "src"), ROOT]),
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    return env


@pytest.fixture(scope="module")
def rows(corpus, tmp_path_factory):
    """The corpus converted for the pointer head on the tiny base."""
    d = tmp_path_factory.mktemp("ddp_rows")
    base = tiny_base(d / "base", str(corpus / "tokenizer.json"))
    examples = [*_examples(150, seed=1)]
    for ex in examples:
        ex.weight = 1.0
    write_jsonl(str(d / "train.jsonl"), examples)
    from training import convert

    convert.main(
        [
            "--examples",
            str(d / "train.jsonl"),
            "--base",
            str(base),
            "--out",
            str(d / "rows"),
            "--prompt",
            "numbered",
            "--val-fraction",
            "0.1",
            "--max-length",
            "2048",
        ]
    )
    return base, d / "rows"


def _train(base, rows, out, world: int) -> None:
    flags = [
        "--base",
        str(base),
        "--base-model",
        str(base),
        "--revision",
        "0" * 40,
        "--train",
        str(rows / "train.jsonl"),
        "--dev",
        str(rows / "dev.jsonl"),
        "--output",
        str(out),
        "--head",
        "pointer",
        "--prompt",
        "numbered",
        "--rank",
        "2",
        "--alpha",
        "4",
        "--dropout",
        "0",
        "--pointer-dim",
        "16",
        "--head-dropout",
        "0",
        "--rows-per-step",
        "16",
        "--batch-size",
        "4",
        "--token-budget",
        "8192",
        "--max-length",
        "2048",
        "--epochs",
        "2",
        "--eval-every",
        "4",
        "--save-every",
        "1000",
        "--seed",
        "3",
        "--cpu-threads",
        "1",
    ]
    if world == 1:
        cmd = [sys.executable, "-m", "training.train", *flags]
    else:
        # Static rendezvous on loopback: --standalone resolves the hostname, which a laptop off its network may not.
        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            f"--nproc_per_node={world}",
            "--master-addr=127.0.0.1",
            f"--master-port={_free_port()}",
            "-m",
            "training.train",
            *flags,
        ]
    res = subprocess.run(cmd, env=_env(), cwd=ROOT, capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-5000:]


def _lines(path):
    return [json.loads(line) for line in open(path, encoding="utf-8")]


@pytest.mark.distributed
def test_two_ranks_train_what_one_process_trains(rows, tmp_path):
    base, converted = rows
    _train(base, converted, tmp_path / "one", 1)
    _train(base, converted, tmp_path / "two", 2)
    one, two = (
        _lines(tmp_path / "one" / "train_log.jsonl"),
        _lines(tmp_path / "two" / "train_log.jsonl"),
    )
    assert [s["rows"] for s in one] == [s["rows"] for s in two] and len(one) > 4
    for a, b in zip(one, two, strict=True):
        assert a["epoch"] == b["epoch"] and a["tokens"] == b["tokens"]
        assert b["loss"] == pytest.approx(a["loss"], rel=1e-4)
        assert b["grad_norm"] == pytest.approx(a["grad_norm"], rel=1e-3)
    e1, e2 = (
        _lines(tmp_path / "one" / "eval_log.jsonl"),
        _lines(tmp_path / "two" / "eval_log.jsonl"),
    )
    assert [e["step"] for e in e1] == [e["step"] for e in e2]
    assert e2[-1]["raw"]["nll"] == pytest.approx(e1[-1]["raw"]["nll"], rel=1e-3)
    for name in ("head.safetensors", "lora/adapter_model.safetensors"):
        w1 = load_file(str(tmp_path / "one" / "last" / "checkpoint" / name))
        w2 = load_file(str(tmp_path / "two" / "last" / "checkpoint" / name))
        for k in w1:
            if k == "k.bias":
                # It adds the same to every option's logit, so its gradient is zero up to rounding, and Adam turns
                # that rounding into steps of full size: a random walk on any GPU count, which no logit sees.
                continue
            torch.testing.assert_close(w2[k], w1[k], rtol=1e-4, atol=1e-6)


def test_a_world_across_machines_is_refused(monkeypatch):
    from training import distributed

    monkeypatch.setenv("WORLD_SIZE", "16")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    with pytest.raises(SystemExit, match="one machine"):
        distributed.Process()


def test_skipping_rows_matches_collating_them(corpus):
    """SystemOneCollator.skip leaves the random stream where collating the rows would."""
    from training.data.collate import CollatorConfig, SystemOneCollator

    tok = _load_tokenizer(str(corpus / "tokenizer.json"))
    rows = _examples(64, seed=5)
    cfg = CollatorConfig(seed=7, max_length=2048)
    full, part = SystemOneCollator(tok, cfg), SystemOneCollator(tok, cfg)
    for b in range(0, 64, 8):
        whole = full(rows[b : b + 8])
        part.skip(rows[b : b + 3])
        mid = part(rows[b + 3 : b + 6])
        part.skip(rows[b + 6 : b + 8])
        for j in range(3):
            a, m = whole["input_ids"][3 + j], whole["attention_mask"][3 + j].bool()
            c, mc = mid["input_ids"][j], mid["attention_mask"][j].bool()
            assert a[m].tolist() == c[mc].tolist()
            assert int(whole["labels"][3 + j]) == int(mid["labels"][j])


# ---------------------------------------------------------------- sharded labelling


class _StubLM:
    """A causal-LM stand-in for teacher.label: hidden = mean of the real tokens' embeddings,
    plus (optionally) a term in the padded batch width, the way bf16 kernels make a row's
    numbers depend on its batch. With it, only identical batching reproduces a row."""

    def __init__(self, vocab: int, width_term: float):
        torch.manual_seed(0)
        self.emb, self.width_term = torch.nn.Embedding(vocab, 16), width_term

    def model(self, input_ids, attention_mask, use_cache=False):
        e = self.emb(input_ids) * attention_mask[..., None]
        h = e.cumsum(1) / attention_mask.cumsum(1).clamp_min(1)[..., None]
        return SimpleNamespace(last_hidden_state=h + self.width_term * input_ids.shape[1])

    def get_output_embeddings(self):
        return self.emb


def _stub_teacher(corpus, monkeypatch, width_term=0.0):
    import training.data.teacher as teacher

    tok = _load_tokenizer(str(corpus / "tokenizer.json"))
    tok.chat_template = (
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    model = _StubLM(len(tok), width_term)
    monkeypatch.setattr(teacher, "load", lambda *a, **k: (model, tok))
    return teacher.main


@pytest.mark.parametrize("width_term", [0.0, 1e-3])
def test_teacher_shards_merge_to_the_single_process_file(corpus, tmp_path, monkeypatch, width_term):
    main = _stub_teacher(corpus, monkeypatch, width_term)
    src = str(corpus / "train.jsonl")
    shift = str(corpus / "val.jsonl")
    base = ["--src", src, "--max-batch-tokens", "3000", "--shift-by", shift]

    main([*base, "--out", str(tmp_path / "one.jsonl")])
    for n in (1, 2, 3):
        out = str(tmp_path / f"n{n}.jsonl")
        for i in range(n):
            main([*base, "--out", out, "--num-shards", str(n), "--shard-index", str(i)])
        main([*base, "--out", out, "--num-shards", str(n), "--merge"])
        assert open(out).read() == open(tmp_path / "one.jsonl").read()
        assert open(out[:-6] + "_train.jsonl").read() == open(tmp_path / "one_train.jsonl").read()
    lines = open(tmp_path / "one.jsonl").read().splitlines()
    assert 0 < len(lines) <= 150  # rows with >16 options get no teacher


def test_teacher_shard_resumes_and_merge_refuses_unfinished(corpus, tmp_path, monkeypatch):
    main = _stub_teacher(corpus, monkeypatch)
    base = ["--src", str(corpus / "train.jsonl"), "--max-batch-tokens", "3000"]
    main([*base, "--out", str(tmp_path / "one.jsonl")])

    out = str(tmp_path / "n2.jsonl")
    main([*base, "--out", out, "--num-shards", "2", "--shard-index", "0"])
    main([*base, "--out", out, "--num-shards", "2", "--shard-index", "1"])
    # Interrupt shard 1: half its rows lost, a cut-off line, no .done marker.
    from training.data import shards

    s1 = shards.path(out, 1, 2)
    keep = open(s1).read().splitlines()[:10]
    open(s1, "w").write("\n".join(keep) + "\n" + keep[-1][:7])
    os.remove(s1 + ".done")
    with pytest.raises(SystemExit):
        main([*base, "--out", out, "--num-shards", "2", "--merge"])
    main([*base, "--out", out, "--num-shards", "2", "--shard-index", "1"])  # resume
    main([*base, "--out", out, "--num-shards", "2", "--merge"])
    assert open(out).read() == open(tmp_path / "one.jsonl").read()


def test_replay_shards_merge_to_the_single_process_file(corpus, tmp_path, monkeypatch):
    import training.evaluate as evaluate
    from strands_decider.modeling import MASK_VALUE, StrandsDeciderModel

    def fake_collect(model, examples, *, device="cuda", batch_size=16, max_length=3072):
        # A row's logits depend on its text and on its batch (the batch's widest row).
        out, slots = [], []
        for b in range(0, len(examples), batch_size):
            chunk = examples[b : b + batch_size]
            width = max(len(ex.state) for ex in chunk)
            for ex in chunk:
                g = torch.Generator().manual_seed(len(ex.state) * 31 + ex.n_options)
                row = torch.randn(24, generator=g) + 1e-3 * width
                row[ex.n_options :] = MASK_VALUE
                out.append(row)
                slots.append(ex.n_options)
        return torch.stack(out), None, torch.tensor(slots), examples

    monkeypatch.setattr(evaluate, "collect_logits", fake_collect)
    monkeypatch.setattr(
        StrandsDeciderModel,
        "load",
        staticmethod(lambda path, **k: SimpleNamespace(config=SimpleNamespace(max_length=3072))),
    )
    import training.data.replay as replay

    base = ["ckpt", "--src", str(corpus / "train.jsonl"), "--shift-by", str(corpus / "val.jsonl")]
    replay.main([*base, "--out", str(tmp_path / "one.jsonl")])
    for n in (1, 2, 3, 8):
        out = str(tmp_path / f"n{n}.jsonl")
        for i in range(n):
            replay.main([*base, "--out", out, "--num-shards", str(n), "--shard-index", str(i)])
        replay.main([*base, "--out", out, "--num-shards", str(n), "--merge"])
        assert open(out).read() == open(tmp_path / "one.jsonl").read()
    first = json.loads(open(tmp_path / "one.jsonl").readline())
    assert first["i"] == 20  # offset past the --shift-by file's 20 rows
