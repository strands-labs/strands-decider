"""Multi-GPU training and sharded labelling reproduce the single-process run.

CPU only: DDP over gloo, a tiny randomly initialised Qwen3 torso with LoRA and a pointer
head, and a byte-level BPE tokenizer trained in the test, so nothing is downloaded. The
end-to-end checks launch `train()` under torchrun at world sizes 2 to 8 (8 splits each
micro-batch across ranks) and compare against a plain process:

  - every optimizer step's gradient (captured where train.py clips it, i.e. after the
    all-reduce) equals the 1-process gradient, and every rank starts from its weights;
  - every optimizer step trains on the same rows, rendered identically (same option
    shuffles and instruction phrasings), as the 1-process step;
  - the logged loss curve, epoch numbers and validation loss are the 1-process ones.

Those end-to-end tests launch processes and take minutes, so they are marked
`distributed` and run only with `pytest -m distributed` (`-m ""` runs everything).

This file is also the torchrun worker: `python tests/test_ddp.py worker <cfg.json>`.
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from strands_decider.data.format import Example, write_jsonl
from strands_decider.distributed import plan_step
from strands_decider.train import TrainConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORDS = ("the contract states that payment is due within thirty days of delivery and any "
         "dispute must be raised in writing before the invoice date which party breached "
         "clause obligations notice termination refund shipping order customer seller "
         "question answer evidence claim supported refuted unknown true false option").split()


# ---------------------------------------------------------------- fixtures


def _text(rng: random.Random, n_words: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n_words)).capitalize() + "."


def _examples(n: int, seed: int) -> list:
    rng = random.Random(seed)
    out = []
    for k in range(n):
        kind = ("noul", "choice", "score")[k % 3]
        if kind == "noul":
            options = [["false", _text(rng, 4)], ["true", _text(rng, 4)]]
        elif kind == "choice":
            m = rng.choice([2, 3, 5, 11, 12])  # >9 options: no frozen-KL reference
            options = [[f"opt{j}", _text(rng, 3)] for j in range(m)]
        else:
            m = rng.choice([3, 5])
            options = [[str(j), _text(rng, 3)] for j in range(m)]
        out.append(Example(
            kind=kind,
            # lengths from a few words to a few hundred, so length grouping matters
            state=_text(rng, rng.choice([5, 20, 60, 150])),
            instructions=_text(rng, 6),
            options=options,
            label=rng.randrange(len(options)),
            task=f"task{k % 4}",
            weight=rng.choice([1.0, 1.0, 0.5, 2.0]),
            instruction_variants=[_text(rng, 6)] if k % 2 else [],
        ))
    return out


def _tokenizer_file(path: str, examples: list, vocab_size: int = 400) -> None:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>", "<eos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    from strands_decider.prompting import build_prompt

    texts = [build_prompt(ex.state, ex.to_question())[0] for ex in examples]
    tok.train_from_iterator(texts, trainer)
    tok.save(path)


def _load_tokenizer(path: str):
    from transformers import PreTrainedTokenizerFast

    return PreTrainedTokenizerFast(tokenizer_file=path, pad_token="<pad>", eos_token="<eos>")


def _tiny_model_factory(tokenizer_path: str):
    """Stands in for StrandsDeciderModel.from_pretrained_base: same construction, tiny torso."""
    from transformers import Qwen3Config, Qwen3Model

    from strands_decider.modeling import StrandsDeciderModel

    def build(config, *, device_map=None, attn_implementation=None):
        tok = _load_tokenizer(tokenizer_path)
        qc = Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         head_dim=8, max_position_embeddings=4096, tie_word_embeddings=True)
        torso = Qwen3Model(qc)  # float32, randomly initialised after train()'s seed
        config.lora_dropout = 0.0  # dropout masks cannot match across world sizes
        model = StrandsDeciderModel(config, torso, tok)
        if config.use_lora:
            model.attach_lora()
        return model

    return build


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("ddp_corpus")
    train = _examples(150, seed=1)  # 18 micro-batches of 8: 4 steps + 2 left over
    write_jsonl(str(d / "train.jsonl"), train)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    # rows for the frozen-KL-only term: no label, at most 9 options
    write_jsonl(str(d / "klonly.jsonl"), [ex for ex in _examples(80, seed=9) if ex.n_options <= 9])
    rng = random.Random(3)
    with open(d / "teacher.jsonl", "w", encoding="utf-8") as fh:
        for i, ex in enumerate(train):
            if rng.random() < 0.5:  # only some rows carry a teacher
                p = [rng.random() + 0.01 for _ in range(ex.n_options)]
                fh.write(json.dumps({"i": i, "probs": [x / sum(p) for x in p]}) + "\n")
    _tokenizer_file(str(d / "tokenizer.json"), train)
    _tokenizer_file(str(d / "other_tokenizer.json"), _examples(60, seed=7), vocab_size=300)
    return d


# ---------------------------------------------------------------- worker (torchrun)


def worker(cfg_path: str) -> None:
    """One rank: train with the tiny model, recording gradients and trained rows."""
    torch.set_num_threads(1)
    import strands_decider.train as T
    from strands_decider.data.collate import SystemOneCollator
    from strands_decider.modeling import StrandsDeciderModel

    with open(cfg_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    StrandsDeciderModel.from_pretrained_base = staticmethod(_tiny_model_factory(spec.pop("tokenizer")))
    ref_tokenizer = spec.pop("ref_tokenizer", None)
    if ref_tokenizer:  # kl_frozen_reference: a tiny checkpoint, the same on every rank

        def load(path, **k):
            from strands_decider.modeling import StrandsDeciderConfig

            with torch.random.fork_rng():
                torch.manual_seed(123)
                cfg = StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16)
                return _tiny_model_factory(ref_tokenizer)(cfg)

        StrandsDeciderModel.load = staticmethod(load)
    rank = int(os.environ.get("RANK", "0"))

    grads, init = [], []
    rows = [[]]  # per optimizer step, the rows this rank collated: (unpadded ids, label)
    real_clip = torch.nn.utils.clip_grad_norm_

    def clip(params, max_norm, *a, **k):  # called once per optimizer step
        params = list(params)
        if not grads:  # before the first optimizer step: the initial trainable weights
            init.extend(p.detach().clone() for p in params)
        grads.append([p.grad.detach().clone() for p in params])
        rows.append([])
        return real_clip(params, max_norm, *a, **k)

    torch.nn.utils.clip_grad_norm_ = clip

    real_call = SystemOneCollator.__call__

    def call(self, batch):
        out = real_call(self, batch)
        if self.train:
            rows[-1].extend([
                (tuple(ids[m.bool()].tolist()), int(lab))
                for ids, m, lab in zip(out["input_ids"], out["attention_mask"], out["labels"],
                                       strict=True)
            ])
        return out

    SystemOneCollator.__call__ = call
    real_reference = T._frozen_reference

    def reference(*a, **k):  # renders every row ahead of training: not what a step trains on
        SystemOneCollator.__call__ = real_call
        try:
            return real_reference(*a, **k)
        finally:
            SystemOneCollator.__call__ = call

    T._frozen_reference = reference
    if spec.pop("skew_rank1", False) and rank == 1:  # this rank reads a longer last row
        real_load = T.load_examples

        def skewed(files):
            rows = real_load(files)
            rows[-1].state += " (edited while the run started)"
            return rows

        T.load_examples = skewed
    cfg = TrainConfig(**spec)
    T.train(cfg)
    torch.save({"grads": grads, "rows": rows, "init": init},
               os.path.join(cfg.output_dir, f"rank{rank}.pt"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(corpus, tmp_path, world: int, *, expect_ok: bool = True, **over) -> dict:
    out = tmp_path / f"w{world}"
    spec = dict(
        train_files=[str(corpus / "train.jsonl")], val_files=[str(corpus / "val.jsonl")],
        base_model="tiny", num_slots=24, head_type="pointer", pointer_dim=16,
        kl_frozen_weight=0.3, teacher_file=str(corpus / "teacher.jsonl"), teacher_weight=1.0,
        head_dropout=0.0, max_length=2048, micro_batch_size=8, grad_accum=4, epochs=2,
        group_by_length=True, length_group_mega=2, log_every=1, eval_every=4,
        output_dir=str(out), seed=0, tokenizer=str(corpus / "tokenizer.json"),
    )
    spec.update(over)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg_path = tmp_path / f"cfg{world}.json"
    cfg_path.write_text(json.dumps(spec))
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src"), "OMP_NUM_THREADS": "1",
           "TOKENIZERS_PARALLELISM": "false"}
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    me = os.path.abspath(__file__)
    if world == 1:  # a plain process, not torchrun: the unchanged single-GPU path
        cmd = [sys.executable, me, "worker", str(cfg_path)]
    else:
        # Static rendezvous on loopback: --standalone resolves the hostname, which a
        # laptop off its network may not.
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               f"--nproc_per_node={world}", "--master-addr=127.0.0.1",
               f"--master-port={_free_port()}", me, "worker", str(cfg_path)]
        if sys.platform == "darwin":
            env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    if not expect_ok:
        assert res.returncode != 0
        return {"stdout": res.stdout, "stderr": res.stderr}
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-5000:]
    with open(out / "history.json", encoding="utf-8") as fh:
        history = json.load(fh)
    ranks = [torch.load(out / f"rank{r}.pt") for r in range(world)]
    return {"history": history, "ranks": ranks, "stdout": res.stdout}


def _rows_per_step(ranks: list) -> list:
    """For each optimizer step, the sorted rows all ranks trained on in it."""
    n = len(ranks[0]["grads"])
    return [sorted(row for r in ranks for row in r["rows"][s]) for s in range(n)]


def _epoch_steps(stdout: str) -> list:
    return re.findall(r"epoch (\d+) step (\d+)/", stdout)


def _assert_same_run(ref: dict, got: dict, world: int, *, rtol: float = 1e-4) -> None:
    # 1. composition and rendering of every optimizer step
    n_steps = len(ref["ranks"][0]["grads"])
    assert all(len(r["grads"]) == n_steps for r in got["ranks"])
    assert _rows_per_step(got["ranks"]) == _rows_per_step(ref["ranks"])
    assert all(len(rows) == 32 for rows in _rows_per_step(ref["ranks"]))
    # 2. initial weights identical on every rank and to 1-process; gradients after the
    #    all-reduce identical on every rank and equal to 1-process
    for r in got["ranks"]:
        assert all(torch.equal(x, y) for x, y in zip(r["init"], ref["ranks"][0]["init"], strict=True))
        assert all(torch.equal(x, y) for a, b in zip(r["grads"], got["ranks"][0]["grads"], strict=True)
                   for x, y in zip(a, b, strict=True))
    g_ref = ref["ranks"][0]["grads"]
    for s, (a, b) in enumerate(zip(got["ranks"][0]["grads"], g_ref, strict=True)):
        num = sum(float((x - y).pow(2).sum()) for x, y in zip(a, b, strict=True)) ** 0.5
        den = sum(float(y.pow(2).sum()) for y in b) ** 0.5
        assert num <= rtol * den, f"step {s + 1}: relative gradient error {num / den:.2e}"
    # 3. logged epoch and step numbers, loss curve, KL terms and validation loss
    assert _epoch_steps(got["stdout"]) == _epoch_steps(ref["stdout"])
    assert [h.keys() for h in got["history"]] == [h.keys() for h in ref["history"]]
    for h, g in zip(ref["history"], got["history"], strict=True):
        for k, v in h.items():
            assert g[k] == pytest.approx(v, rel=rtol, abs=1e-7), (h["step"], k)


@pytest.fixture(scope="module")
def plain_run(corpus, tmp_path_factory):
    return _run(corpus, tmp_path_factory.mktemp("w1"), 1)


@pytest.mark.distributed
@pytest.mark.parametrize("world", [2, 3, 4, 8])
def test_ddp_matches_single_process(corpus, plain_run, tmp_path, world):
    got = _run(corpus, tmp_path, world)
    # 18 micro-batches an epoch, 4 whole steps each (the schedule's count), over 2
    # epochs: 8 steps, step 5 straddling the epoch boundary (logged as epoch 1).
    assert len(plain_run["ranks"][0]["grads"]) == 8
    assert [e for e, _ in _epoch_steps(plain_run["stdout"])] == list("00001111")
    _assert_same_run(plain_run, got, world)


@pytest.mark.distributed
@pytest.mark.parametrize("over", [
    dict(group_by_length=False, eval_every=3),                 # epoch 2's order is drawn after evals
    dict(epochs=1, max_steps=10, eval_every=3, save_every=2),  # runs out of data; saves on rank 0
    dict(kl_direction="reverse"),                              # both KL terms reversed
])
def test_ddp_matches_single_process_variants(corpus, tmp_path, over):
    ref = _run(corpus, tmp_path / "a", 1, **over)
    if "max_steps" in over:  # the 2 left-over micro-batches never step
        assert len(ref["ranks"][0]["grads"]) == 4
    _assert_same_run(ref, _run(corpus, tmp_path / "b", 4, **over), 4)


@pytest.mark.distributed
@pytest.mark.parametrize("world", [1, 3, 8])
def test_precomputed_frozen_reference_matches_single_process(corpus, plain_run, tmp_path, world):
    """precompute_frozen_kl: the same rows, rendering, gradients and losses."""
    got = _run(corpus, tmp_path, world, precompute_frozen_kl=True)
    assert "frozen-KL reference for 32 micro-batches" in got["stdout"]
    _assert_same_run(plain_run, got, world)


@pytest.mark.distributed
@pytest.mark.parametrize("tokenizer", ["tokenizer.json", "other_tokenizer.json"])
def test_ddp_frozen_reference_checkpoint_matches_single_process(corpus, tmp_path, tokenizer):
    """kl_frozen_reference: each rank renders its micro-batches with the reference's tokenizer."""
    over = dict(precompute_frozen_kl=True, kl_frozen_reference="tiny-ref",
                ref_tokenizer=str(corpus / tokenizer))
    ref = _run(corpus, tmp_path / "a", 1, **over)
    assert "frozen-KL reference: tiny-ref" in ref["stdout"]
    _assert_same_run(ref, _run(corpus, tmp_path / "b", 3, **over), 3)


@pytest.fixture(scope="module")
def kl_only_plain_run(corpus, tmp_path_factory):
    return _run(corpus, tmp_path_factory.mktemp("klonly"), 1,
                kl_only_files=[str(corpus / "klonly.jsonl")], kl_only_weight=0.7)


@pytest.mark.distributed
@pytest.mark.parametrize("world", [3, 8])
def test_ddp_matches_single_process_kl_only_rows(corpus, kl_only_plain_run, tmp_path, world):
    got = _run(corpus, tmp_path, world, kl_only_files=[str(corpus / "klonly.jsonl")],
               kl_only_weight=0.7)
    _assert_same_run(kl_only_plain_run, got, world)


@pytest.mark.distributed
def test_ddp_frozen_torso(corpus, plain_run, tmp_path):
    """Head-only training: no checkpointing, DDP over a torso with no trainable parameter."""
    ref = _run(corpus, tmp_path / "a", 1, freeze_torso=True)
    got = _run(corpus, tmp_path / "b", 2, freeze_torso=True)
    assert len(ref["ranks"][0]["init"]) < len(plain_run["ranks"][0]["init"])  # no LoRA
    _assert_same_run(ref, got, 2)


@pytest.mark.distributed
def test_ddp_refuses_ranks_that_hold_different_data(corpus, tmp_path):
    res = _run(corpus, tmp_path, 2, expect_ok=False, skew_rank1=True)
    assert "ranks hold different training data" in res["stdout"] + res["stderr"]


def test_train_refuses_more_ranks_than_rows(monkeypatch):
    from strands_decider.train import train

    monkeypatch.setenv("WORLD_SIZE", "33")
    with pytest.raises(ValueError, match="exceeds the 32 rows"):
        train(TrainConfig(micro_batch_size=8, grad_accum=4))


# ---------------------------------------------------------------- unit checks


def _step_batches(seed: int = 0):
    """Micro-batches of a v17-like corpus: mostly short rows, some long documents."""
    from strands_decider.data.sampling import LengthGroupedBatchSampler

    rng = random.Random(seed)
    lengths = [rng.choice([rng.randrange(150, 900)] * 9 + [rng.randrange(2000, 9000)])
               for _ in range(4000)]
    return lengths, LengthGroupedBatchSampler(lengths, 8, mega=50, seed=seed).batches(0)


@pytest.mark.parametrize("world", [2, 3, 4, 5, 8, 16, 32])
def test_plan_shares_out_exactly_the_steps_rows(world):
    """Every row of the step exactly once; each rank >= 1 slice; no slice > a micro-batch."""
    lengths, batches = _step_batches()
    for s in range(0, len(batches) // 4, 7):
        step = batches[4 * s: 4 * s + 4]
        plan = plan_step(step, lengths, world)
        assert len(plan) == world and all(plan)
        seen = sorted((j, i) for p in plan for j, a, b in p for i in range(a, b))
        assert seen == [(j, i) for j in range(4) for i in range(8)]
        assert all(0 < b - a <= 8 for p in plan for _, a, b in p)
        assert plan == plan_step(step, lengths, world)  # deterministic


def test_plan_balances_better_than_even_halves():
    """Slowest rank's padded load, planned vs the fixed layout (W=8: halves of each batch).

    Documents the intent of the planner; the 0.8 threshold is a heuristic and may need
    retuning with FORWARD_OVERHEAD.
    """
    from strands_decider.distributed import FORWARD_OVERHEAD

    lengths, batches = _step_batches()

    def load(rows):
        return len(rows) * max(lengths[i] for i in rows) + FORWARD_OVERHEAD

    planned, halves = [], []
    for s in range(len(batches) // 4):
        step = batches[4 * s: 4 * s + 4]
        plan = plan_step(step, lengths, 8)
        planned.append(max(sum(load(step[j][a:b]) for j, a, b in p) for p in plan))
        halves.append(max(load(b[h:h + 4]) for b in step for h in (0, 4)))
    assert sum(planned) < 0.8 * sum(halves), (sum(planned) / sum(halves))


def test_skipping_rows_matches_collating_them(corpus):
    """SystemOneCollator.skip leaves the random stream where collating the rows would."""
    from strands_decider.data.collate import CollatorConfig, SystemOneCollator

    tok = _load_tokenizer(str(corpus / "tokenizer.json"))
    rows = _examples(64, seed=5)
    cfg = CollatorConfig(num_slots=24, head_type="pointer", seed=7, max_length=2048)
    full, part = SystemOneCollator(tok, cfg), SystemOneCollator(tok, cfg)
    for b in range(0, 64, 8):
        whole = full(rows[b:b + 8])
        part.skip(rows[b:b + 3])
        mid = part(rows[b + 3:b + 6])
        part.skip(rows[b + 6:b + 8])
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
    import strands_decider.data.teacher as teacher

    tok = _load_tokenizer(str(corpus / "tokenizer.json"))
    tok.chat_template = ("{% for m in messages %}<{{ m.role }}>{{ m.content }}\n{% endfor %}"
                         "{% if add_generation_prompt %}<assistant>{% endif %}")
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
        assert (open(out[:-6] + "_train.jsonl").read()
                == open(tmp_path / "one_train.jsonl").read())
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
    from strands_decider.data import shards

    s1 = shards.path(out, 1, 2)
    keep = open(s1).read().splitlines()[: 10]
    open(s1, "w").write("\n".join(keep) + "\n" + keep[-1][:7])
    os.remove(s1 + ".done")
    with pytest.raises(SystemExit):
        main([*base, "--out", out, "--num-shards", "2", "--merge"])
    main([*base, "--out", out, "--num-shards", "2", "--shard-index", "1"])  # resume
    main([*base, "--out", out, "--num-shards", "2", "--merge"])
    assert open(out).read() == open(tmp_path / "one.jsonl").read()


def test_replay_shards_merge_to_the_single_process_file(corpus, tmp_path, monkeypatch):
    import strands_decider.evaluate as evaluate
    from strands_decider.modeling import MASK_VALUE, StrandsDeciderModel

    def fake_collect(model, examples, *, device="cuda", batch_size=16, max_length=3072):
        # A row's logits depend on its text and on its batch (the batch's widest row).
        out, slots = [], []
        for b in range(0, len(examples), batch_size):
            chunk = examples[b:b + batch_size]
            width = max(len(ex.state) for ex in chunk)
            for ex in chunk:
                g = torch.Generator().manual_seed(len(ex.state) * 31 + ex.n_options)
                row = torch.randn(24, generator=g) + 1e-3 * width
                row[ex.n_options:] = MASK_VALUE
                out.append(row)
                slots.append(ex.n_options)
        return torch.stack(out), None, torch.tensor(slots), examples

    monkeypatch.setattr(evaluate, "collect_logits", fake_collect)
    monkeypatch.setattr(StrandsDeciderModel, "load",
                        staticmethod(lambda path, **k: SimpleNamespace(
                            config=SimpleNamespace(max_length=3072))))
    import strands_decider.data.replay as replay

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


if __name__ == "__main__" and sys.argv[1:2] == ["worker"]:
    worker(sys.argv[2])
