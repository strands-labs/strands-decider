"""Evaluation batched by length, and shared out over torchrun ranks, reproduces the plain run.

`collect_logits(sort_by_length=True)` (eval and calibrate) batches rows of similar length
together; the logits must come back in the order of the examples, equal to file-order
batching up to float rounding. Under torchrun each rank forwards its share of the same
batches, and every rank must get exactly the single-process logits and fit.

CPU only, with test_ddp's tiny torso and trained tokenizer. The torchrun checks are marked
`distributed`, as in test_ddp. This file is also their worker:
`python training/tests/test_eval_sharding.py worker <spec.json>`.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest
import torch
from test_ddp import ROOT, _examples, _free_port, _tiny_model_factory, _tokenizer_file

from training.data.format import write_jsonl
from training.data.sampling import example_length
from training.evaluate import eval_batches


def _tiny_model(tokenizer_path: str):
    from strands_decider.modeling import StrandsDeciderConfig

    with torch.random.fork_rng():
        torch.manual_seed(0)  # the same weights in every process
        cfg = StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16)
        return _tiny_model_factory(tokenizer_path)(cfg)


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("eval_corpus")
    rows = _examples(70, seed=4)  # 9 batches of 8, the last short
    write_jsonl(str(d / "eval.jsonl"), rows)
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    return d


def test_eval_batches_default_to_file_order():
    exs = _examples(20, seed=1)
    assert eval_batches(exs, 8) == [list(range(8)), list(range(8, 16)), list(range(16, 20))]


def test_eval_batches_sorted_longest_first_ties_in_file_order():
    exs = _examples(50, seed=1)
    batches = eval_batches(exs, 8, sort_by_length=True)
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(50)) and [len(b) for b in batches] == [8] * 6 + [2]
    keys = [(-example_length(exs[i]), i) for i in flat]
    assert keys == sorted(keys)


def test_sorted_logits_keep_the_order_of_the_examples(corpus):
    from training.data.format import read_jsonl
    from training.evaluate import collect_logits

    torch.set_num_threads(1)
    model = _tiny_model(str(corpus / "tokenizer.json"))
    exs = list(read_jsonl(str(corpus / "eval.jsonl")))
    plain = collect_logits(model, exs, device="cpu", batch_size=8)
    srt = collect_logits(model, exs, device="cpu", batch_size=8, sort_by_length=True)
    assert torch.equal(plain[1], srt[1]) and torch.equal(plain[2], srt[2])
    assert plain[1].tolist() == [ex.label for ex in exs]
    assert plain[2].tolist() == [ex.n_options for ex in exs]
    # The widths can differ (each is the widest batch's); the masked columns agree.
    w = min(plain[0].size(1), srt[0].size(1))
    assert torch.allclose(plain[0][:, :w], srt[0][:, :w], atol=1e-5)
    assert torch.equal(plain[0].argmax(-1), srt[0].argmax(-1))


# ---------------------------------------------------------------- worker (torchrun)


def worker(spec_path: str) -> None:
    """One rank (or a plain process): collect the logits and calibrate, as eval would."""
    torch.set_num_threads(1)
    from strands_decider.modeling import StrandsDeciderModel
    from training import evaluate
    from training.data.format import read_jsonl
    from training.distributed import env, process_group

    with open(spec_path, encoding="utf-8") as fh:
        spec = json.load(fh)
    model = _tiny_model(spec["tokenizer"])
    StrandsDeciderModel.load = staticmethod(lambda path, **k: model)
    exs = list(read_jsonl(spec["data"]))
    with process_group():
        logits, labels, slots, _ = evaluate.collect_logits(
            model, exs, device="cpu", batch_size=8, sort_by_length=True)
        fit = evaluate.calibrate_checkpoint(spec["checkpoint"], exs, device="cpu", batch_size=8)
    rank = env()[0]
    torch.save({"logits": logits, "labels": labels, "slots": slots, "fit": fit},
               os.path.join(spec["out"], f"rank{rank}.pt"))


def _run(corpus, tmp_path, world: int) -> list:
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(parents=True)
    spec = {"tokenizer": str(corpus / "tokenizer.json"), "data": str(corpus / "eval.jsonl"),
            "checkpoint": str(ckpt), "out": str(tmp_path)}
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    paths = [ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "training", "tests")]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(paths),
           "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"}
    for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    me = os.path.abspath(__file__)
    if world == 1:
        cmd = [sys.executable, me, "worker", str(spec_path)]
    else:
        cmd = [sys.executable, "-m", "torch.distributed.run", "--nnodes=1",
               f"--nproc_per_node={world}", "--master-addr=127.0.0.1",
               f"--master-port={_free_port()}", me, "worker", str(spec_path)]
        if sys.platform == "darwin":
            env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-5000:]
    written = os.listdir(ckpt)  # calibrate's config: written by rank 0 alone
    assert len(written) == 1, written
    return [torch.load(tmp_path / f"rank{r}.pt") for r in range(world)]


@pytest.mark.distributed
@pytest.mark.parametrize("world", [2, 3])
def test_sharded_eval_and_calibration_match_one_process(corpus, tmp_path, world):
    ref = _run(corpus, tmp_path / "w1", 1)[0]
    for got in _run(corpus, tmp_path / f"w{world}", world):  # every rank holds every row
        for k in ("logits", "labels", "slots"):
            assert torch.equal(got[k], ref[k]), k
        assert got["fit"] == ref["fit"]


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "worker":
        worker(sys.argv[2])
    else:
        sys.exit(f"usage: {sys.argv[0]} worker <spec.json>")
