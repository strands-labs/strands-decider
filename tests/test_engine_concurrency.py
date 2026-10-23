"""Concurrent requests on one engine get the answers they get alone.

With the server lock lifted on CPU/CUDA, FastAPI's threadpool runs several
`evaluate` calls at once on the same SystemOneEngine. Option offsets kept on the
engine between `_fit` and `_option_idx` were overwritten by a concurrent request,
and the pointer head read another request's option positions. CPU only: a tiny
random torso and a byte-level tokenizer.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.schema import ChoiceQuestion, SystemOneRequest


def _tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        ["<state> <question> <options> <answer> choice"],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=["<pad>", "<eos>"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


@pytest.fixture(scope="module")
def engine():
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    tok = _tokenizer()
    torso = Qwen3Model(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    cfg = StrandsDeciderConfig(
        base_model="tiny", pointer_dim=16, use_lora=False, torch_dtype="float32", max_length=512
    )
    model = StrandsDeciderModel(cfg, torso, tok)
    return SystemOneEngine(model, EngineConfig(device="cpu"))


def _one_question_requests():
    """Single-question requests -- exercise `_slot_probs_batched`."""
    out = []
    for i in range(24):
        opts = [f"option{k}" + "x" * ((i * 7 + k * 3) % 11) for k in range(2 + i % 4)]
        out.append(
            SystemOneRequest(
                state="s" * (5 + (i * 13) % 40),
                questions={
                    "q": ChoiceQuestion(
                        type="choice",
                        instructions="pick " * (1 + i % 5),
                        criteria={o: "" for o in opts},
                    )
                },
            )
        )
    return out


def _multi_question_requests():
    """Multi-question requests -- exercise `_slot_probs_shared_prefix`."""
    out = []
    for i in range(12):
        qs = {}
        for j in range(3):
            opts = [f"opt{k}" + "y" * ((i * 5 + j * 2 + k) % 9) for k in range(2 + (i + j) % 3)]
            qs[f"q{j}"] = ChoiceQuestion(
                type="choice",
                instructions="pick " * (1 + (i + j) % 4),
                criteria={o: "" for o in opts},
            )
        out.append(SystemOneRequest(state="s" * (5 + (i * 11) % 30), questions=qs))
    return out


def _probs(resp, name="q"):
    return resp.answers[name].model_dump()


def _all_probs(resp):
    return {n: a.model_dump() for n, a in resp.answers.items()}


def _run_concurrent(engine, monkeypatch, reqs, extract):
    # Pre-warm so model init doesn't eat the barrier window on cold CI.
    with torch.inference_mode():
        serial = [extract(engine.evaluate(r)) for r in reqs]

    # Hold every thread between tokenising and pointing, so they interleave there each run.
    workers = min(4, len(reqs))
    fit = SystemOneEngine._fit
    # Generous timeout: pool startup + the first concurrent forward on a slow CPU
    # runner can take tens of seconds. If the barrier still breaks, fail loudly
    # rather than silently serialising the test.
    barrier = threading.Barrier(workers, timeout=60)
    broke = threading.Event()
    # The HF fast tokenizer (Rust) raises "Already borrowed" if two threads reconfigure
    # truncation on it at once -- orthogonal to the race under test. Serialise just
    # the tokeniser call; the barrier still interleaves threads at the race-prone spot.
    fit_lock = threading.Lock()

    def slow_fit(self, *a, **kw):
        with fit_lock:
            out = fit(self, *a, **kw)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            broke.set()
        time.sleep(0.01)
        return out

    monkeypatch.setattr(SystemOneEngine, "_fit", slow_fit)

    def run(r):
        with torch.inference_mode():
            return extract(engine.evaluate(r))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        concurrent = list(pool.map(run, reqs))
    assert not broke.is_set(), "barrier broke; threads did not interleave at _fit"
    assert concurrent == serial


def test_concurrent_requests_match_serial_batched(engine, monkeypatch):
    """One question per request -> `_slot_probs_batched` path."""
    _run_concurrent(engine, monkeypatch, _one_question_requests(), _probs)


def test_concurrent_requests_match_serial_shared_prefix(engine, monkeypatch):
    """Multiple questions per request -> `_slot_probs_shared_prefix` path."""
    _run_concurrent(engine, monkeypatch, _multi_question_requests(), _all_probs)


def test_engine_keeps_no_per_request_state(engine):
    with torch.inference_mode():
        engine.evaluate(_one_question_requests()[0])
    assert not hasattr(engine, "_last_offsets")
