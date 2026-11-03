"""The Neuron engine must answer as the torch engine does, from the same checkpoint.

`NeuronEngine` runs the torso as one fixed-shape graph per bucket. Here the graph runs eagerly on
the CPU (`CpuRunner`) with the static kernels the Neuron device uses, so the test covers the
adapter merge, the bucket padding and slicing, and the engine around them, without a device. A
tiny random-weight Qwen3.5 text model is the base, with a LoRA adapter over every projection;
nothing is downloaded. What runs on the device itself is not covered here.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from transformers.models.qwen3_5 import modeling_qwen3_5 as q  # noqa: E402

from strands_decider import neuron_engine, neuron_kernels  # noqa: E402
from strands_decider.infer import load_engine  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.prompting import render_question, render_state  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]
POLICY = (
    "Loans run 21 days and renew twice online unless another patron holds the item. "
    "Accounts with fines above ten dollars cannot renew until the fine is paid. "
)
QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team should handle this?",
        criteria={
            "circulation": "renewals and holds",
            "billing": "fines",
            "reference": "research help",
        },
    ),
    "urgent": NoulQuestion(instructions="Does the patron convey urgency?"),
    "mood": ScoreQuestion(
        instructions="How frustrated is the patron?", criteria=["calm", "annoyed", "angry"]
    ),
}
REQUESTS = [  # one question, and several whose prompts fall in different buckets
    SystemOneRequest(
        state=POLICY + "Can I renew the cookbook?", questions={"team": QUESTIONS["team"]}
    ),
    SystemOneRequest(
        state=POLICY * 6 + "My renewal failed and I leave Friday!", questions=QUESTIONS
    ),
]
BUCKETS = (64, 128, 256, 512)


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    texts = [render_state(r.state) for r in REQUESTS] + [
        render_question(q).text for q in QUESTIONS.values()
    ]
    words = {w for text in texts for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(text)}
    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", *sorted(words)])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
    )


def _checkpoint(tmp_path):
    tok = _tokenizer()
    torch.manual_seed(0)
    base_cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok),
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=32,
        linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )
    base = tmp_path / "base"
    transformers.Qwen3_5ForCausalLM(base_cfg).save_pretrained(base)
    cfg = StrandsDeciderConfig(
        base_model=str(base),
        pointer_dim=16,
        max_length=512,
        torch_dtype="float32",
        lora_r=4,
        lora_alpha=8,
        lora_targets=LORA_TARGETS,
    )
    torso = transformers.Qwen3_5ForCausalLM.from_pretrained(base).model
    model = StrandsDeciderModel(cfg, torso, tok)
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for name, p in model.torso.named_parameters():
            if "lora_B" in name:
                p.normal_(std=0.1)
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return path


def _probabilities(response):
    out = {}
    for name, answer in response.answers.items():
        out[name] = [answer.noul] if answer.type == "noul" else list(answer.probabilities.values())
    return out


@pytest.fixture
def static_kernels(monkeypatch):
    """The kernels the device runs, here for CPU tensors too."""
    monkeypatch.setattr(
        q, "torch_chunk_gated_delta_rule", neuron_kernels.chunk_gated_delta_rule_static
    )
    monkeypatch.setattr(q, "causal_conv1d_fn", neuron_kernels.causal_conv1d_static)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return _checkpoint(tmp_path_factory.mktemp("ckpt"))


def _neuron_cpu(path, buckets=BUCKETS, cores=1):
    return neuron_engine._load(str(path), None, None, buckets, neuron_engine.CpuRunner, cores)


@pytest.mark.parametrize("request_index", range(len(REQUESTS)))
def test_neuron_answers_as_torch_does(checkpoint, static_kernels, request_index):
    request = REQUESTS[request_index]
    expected = load_engine(str(checkpoint), device="cpu").evaluate(request)
    engine = _neuron_cpu(checkpoint)
    got = engine.evaluate(request)
    want, have = _probabilities(expected), _probabilities(got)
    assert have.keys() == want.keys()
    for name in want:
        # The adapter is merged here and unmerged in the torch engine: float rounding only.
        assert have[name] == pytest.approx(want[name], abs=1e-4), name
    assert got.usage.input_tokens >= expected.usage.input_tokens  # no shared prefix here


def test_rows_run_in_their_own_buckets(checkpoint):
    engine = _neuron_cpu(checkpoint)
    shapes = []
    (runner,) = engine.torso.runners

    def record(ids, mask):
        shapes.append(tuple(ids.shape))
        return runner(ids, mask)

    engine.model.torso = neuron_engine.BucketedTorso([record], BUCKETS, 0, engine.torso.config)
    engine.evaluate(REQUESTS[1])
    assert shapes and all(rows == 1 and width in BUCKETS for rows, width in shapes)
    with pytest.raises(ValueError, match="largest bucket"):
        engine.torso.bucket_for(BUCKETS[-1] + 1)


def test_the_engine_reports_neuron_and_has_no_prefix_cache(checkpoint):
    engine = _neuron_cpu(checkpoint)
    assert (engine.cfg.device, engine.cfg.use_prefix_cache) == ("neuron", False)
    assert {p.dtype for p in engine.model.head.parameters()} == {torch.float32}
    assert sorted(engine.warm()) == list(BUCKETS)


def test_buckets_must_cover_max_length(checkpoint):
    with pytest.raises(ValueError, match="max_length"):
        _neuron_cpu(checkpoint, buckets=(64, 128))


def test_a_cache_compiled_by_another_install_is_refused(tmp_path):
    versions = neuron_engine.versions()
    (tmp_path / neuron_engine.MANIFEST).write_text(
        json.dumps({"versions": versions, "buckets": [128, 512]})
    )
    assert neuron_engine.read_manifest(str(tmp_path))["buckets"] == [128, 512]
    (tmp_path / neuron_engine.MANIFEST).write_text(
        json.dumps({"versions": {**versions, "transformers": "0.0.1"}, "buckets": [128]})
    )
    with pytest.raises(RuntimeError, match="compile-neuron"):
        neuron_engine.read_manifest(str(tmp_path))


def test_nki_flash_attention_is_refused(monkeypatch):
    monkeypatch.setenv("TORCH_NEURONX_ENABLE_NKI_SDPA", "1")
    with pytest.raises(RuntimeError, match="TORCH_NEURONX_ENABLE_NKI_SDPA"):
        neuron_engine._set_neuron_env(None)


def test_a_read_only_cache_is_copied(tmp_path):
    cache = tmp_path / "cache"
    (cache / "neff").mkdir(parents=True)
    (cache / "neff" / "graph.neff").write_text("graph")
    assert neuron_engine.writable_cache(str(cache)) == str(cache)
    for path in (cache / "neff", cache):
        path.chmod(0o555)
    try:
        copy = neuron_engine.writable_cache(str(cache))
    finally:
        for path in (cache, cache / "neff"):
            path.chmod(0o755)
    assert copy != str(cache)
    assert open(f"{copy}/neff/graph.neff").read() == "graph"


def test_each_core_gets_its_own_copy_of_the_torso(checkpoint):
    engine = _neuron_cpu(checkpoint, cores=2)
    first, second = engine.torso.runners
    assert (first.core, second.core) == (0, 1)
    a, b = (dict(r.graph.named_parameters()) for r in (first, second))
    assert a.keys() == b.keys()
    assert all(a[k] is not b[k] and torch.equal(a[k], b[k]) for k in a)


def test_concurrent_requests_use_every_core_and_answer_as_alone(checkpoint):
    """Two one-question requests at once each hold a core: both cores run together, and each
    request gets the answer it gets on its own."""
    engine = _neuron_cpu(checkpoint, cores=2)
    requests = [
        SystemOneRequest(state=r.state, questions=dict([next(iter(r.questions.items()))]))
        for r in REQUESTS
    ]
    alone = [_probabilities(engine.evaluate(r)) for r in requests]
    both_busy = threading.Barrier(2, timeout=30)
    used = set()

    def hold(runner):
        def run(ids, mask):
            used.add(runner.core)
            both_busy.wait()  # passes only while the other core is running too
            return runner(ids, mask)

        return run

    torso = engine.torso
    engine.model.torso = neuron_engine.BucketedTorso(
        [hold(r) for r in torso.runners], torso.buckets, torso.pad_id, torso.config
    )
    with ThreadPoolExecutor(2) as pool:
        together = list(pool.map(lambda r: _probabilities(engine.evaluate(r)), requests))
    assert used == {0, 1}
    assert together == alone


def test_a_forward_too_long_for_every_bucket_keeps_the_core_free(checkpoint):
    engine = _neuron_cpu(checkpoint)
    ids = torch.ones((1, BUCKETS[-1] + 1), dtype=torch.long)
    with pytest.raises(ValueError, match="largest bucket"):
        engine.torso(ids, torch.ones_like(ids))
    engine.evaluate(REQUESTS[0])  # would block if the core had not been returned
