"""The MLX engine must answer as the torch engine does, from the same checkpoint.

A tiny random-weight Qwen3.5 text model (Gated DeltaNet and attention layers) is saved as the
base checkpoint, and a decider checkpoint is built on it the way training saves one, with a
LoRA adapter over every projection v19's adapter targets. Both engines load it from disk, so
the test covers the MLX loader, the adapter merge and both of `evaluate`'s paths. Apple silicon
with the mlx extra only; nothing is downloaded.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

if sys.platform != "darwin":
    pytest.skip("MLX runs on Apple silicon only", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

from strands_decider.infer import load_engine  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from strands_decider.prompting import render_question, render_state  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
POLICY = ("Loans run 21 days and renew twice online unless another patron holds the item. "
          "Accounts with fines above ten dollars cannot renew until the fine is paid. ")
QUESTIONS = {
    "team": ChoiceQuestion(instructions="Which team should handle this?",
                           criteria={"circulation": "renewals and holds", "billing": "fines",
                                     "reference": "research help"}),
    "urgent": NoulQuestion(instructions="Does the patron convey urgency?"),
    "mood": ScoreQuestion(instructions="How frustrated is the patron?",
                          criteria=["calm", "annoyed", "angry"]),
}
REQUESTS = [  # one question (the whole-prompt path) and several (the shared-prefix path)
    SystemOneRequest(state=POLICY + "Can I renew the cookbook?", questions={"team": QUESTIONS["team"]}),
    SystemOneRequest(state=POLICY * 6 + "My renewal failed and I leave Friday!", questions=QUESTIONS),
]


def _tokenizer():
    """A word-level tokenizer whose vocabulary covers every rendered prompt, so no position is <unk>."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    texts = [render_state(r.state) for r in REQUESTS] + [render_question(q).text for q in QUESTIONS.values()]
    words = {w for text in texts for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(text)}
    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", *sorted(words)])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>")


def _checkpoint(tmp_path, head_type, torch_dtype="float32"):
    tok = _tokenizer()
    torch.manual_seed(0)
    base_cfg = transformers.Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        # mlx-lm's Gated DeltaNet kernel spreads a key head over 32 lanes: Dk >= 32.
        linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"])
    base = tmp_path / "base"
    transformers.Qwen3_5ForCausalLM(base_cfg).save_pretrained(base)
    cfg = StrandsDeciderConfig(base_model=str(base), head_type=head_type, pointer_dim=16,
                               max_length=512, torch_dtype=torch_dtype, lora_r=4, lora_alpha=8,
                               lora_targets=LORA_TARGETS)
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
        if answer.type == "noul":
            out[name] = [answer.noul]
        else:
            out[name] = list(answer.probabilities.values())
    return out


@pytest.fixture(scope="module", params=["pointer", "slot"])
def engines(request, tmp_path_factory):
    path = _checkpoint(tmp_path_factory.mktemp(request.param), request.param)
    return load_engine(str(path), device="cpu"), load_engine(str(path), device="mlx"), path


@pytest.mark.parametrize("device", ["gpu", "cpu"])
@pytest.mark.parametrize("request_index", range(len(REQUESTS)))
def test_mlx_answers_as_torch_does(engines, request_index, device):
    torch_engine, mlx_engine, _ = engines
    request = REQUESTS[request_index]
    expected = torch_engine.evaluate(request)
    previous = mx.default_device()
    mx.set_default_device(getattr(mx, device))  # mlx-lm picks its Metal kernel by the default device
    try:
        got = mlx_engine.evaluate(request)
    finally:
        mx.set_default_device(previous)
    assert got.usage == expected.usage
    want, have = _probabilities(expected), _probabilities(got)
    assert have.keys() == want.keys()
    for name in want:
        assert have[name] == pytest.approx(want[name], abs=1.5e-4), name  # rounded to 4 places


def test_the_prefix_path_matches_whole_prompts_and_leaves_no_state(engines):
    _, mlx_engine, _ = engines
    request = REQUESTS[1]
    shared = _probabilities(mlx_engine.evaluate(request))
    again = _probabilities(mlx_engine.evaluate(request))
    assert again == shared  # the cached prefix is copied per request, never mutated
    mlx_engine.cfg = dataclasses.replace(mlx_engine.cfg, use_prefix_cache=False)
    try:
        whole = _probabilities(mlx_engine.evaluate(request))
    finally:
        mlx_engine.cfg = dataclasses.replace(mlx_engine.cfg, use_prefix_cache=True)
    for name in shared:
        assert whole[name] == pytest.approx(shared[name], abs=1e-4), name


def test_max_batch_and_strict_window_apply_on_mlx(engines):
    _, mlx_engine, _ = engines
    original = mlx_engine.cfg
    try:
        mlx_engine.cfg = dataclasses.replace(original, max_batch=2)  # three questions: two chunks
        chunked = _probabilities(mlx_engine.evaluate(REQUESTS[1]))
        mlx_engine.cfg = original
        whole = _probabilities(mlx_engine.evaluate(REQUESTS[1]))
        for name in whole:
            assert chunked[name] == pytest.approx(whole[name], abs=1e-4), name
        mlx_engine.cfg = dataclasses.replace(original, strict_window=True)
        too_long = SystemOneRequest(state=POLICY * 100, questions={"team": QUESTIONS["team"]})
        with pytest.raises(ValueError, match="context window"):
            mlx_engine.evaluate(too_long)
    finally:
        mlx_engine.cfg = original


def test_the_adapter_is_merged_into_the_mlx_weights(engines):
    from mlx.utils import tree_flatten
    from safetensors.torch import load_file

    _, mlx_engine, path = engines
    adapter = load_file(str(path / "lora" / "adapter_model.safetensors"))
    base = load_file(str(next((path.parent / "base").glob("*.safetensors"))))
    merged = dict(tree_flatten(mlx_engine._decoder.parameters()))
    checked = 0
    for name, lora_a in adapter.items():
        if not name.endswith(".lora_A.weight"):
            continue
        stem = name.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
        want = base[f"model.{stem}.weight"] + 2.0 * adapter[name.replace("lora_A", "lora_B")] @ lora_a
        have = torch.tensor(merged[f"{stem}.weight"].tolist())
        torch.testing.assert_close(have, want, atol=1e-6, rtol=1e-6)
        checked += 1
    assert checked == sum(1 for k in adapter if k.endswith(".lora_A.weight"))
    assert any(".linear_attn.in_proj_qkv." in k for k in adapter)  # DeltaNet projections are merged too


def _adapter(path, tensors, **config):
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": 4, "lora_alpha": 8, **config}))
    mx.save_safetensors(str(path / "adapter_model.safetensors"),
                        {name: mx.zeros((4, 4)) for name in tensors})
    return str(path)


PAIR = ["base_model.model.layers.0.mlp.up_proj.lora_A.weight",
        "base_model.model.layers.0.mlp.up_proj.lora_B.weight"]


@pytest.mark.parametrize(("tensors", "config", "named"), [
    ([*PAIR, "base_model.model.embed_tokens.lora_embedding_A"], {}, "lora_embedding_A"),
    ([*PAIR, "base_model.model.layers.0.mlp.up_proj.lora_magnitude_vector"], {}, "lora_magnitude_vector"),
    (PAIR[:1], {}, "lora_B"),
    (PAIR, {"rank_pattern": {"up_proj": 8}}, "rank_pattern"),
], ids=["embedding-lora", "dora", "unpaired", "rank-pattern"])
def test_an_adapter_the_merge_cannot_represent_is_refused(tmp_path, tensors, config, named):
    from strands_decider.mlx_engine import merge_lora

    with pytest.raises(ValueError, match=named):
        merge_lora(object(), _adapter(tmp_path / "lora", tensors, **config), "model.")


@pytest.mark.filterwarnings("ignore:Setting `save_embedding_layers`:UserWarning")  # PEFT, saving the fixture
def test_a_checkpoint_with_an_embedding_lora_does_not_load_on_mlx(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "LORA_TARGETS", [*LORA_TARGETS, "embed_tokens"])
    path = _checkpoint(tmp_path, "pointer")
    with pytest.raises(ValueError, match="lora_embedding"):
        load_engine(str(path), device="mlx")


def test_a_bf16_checkpoint_answers_from_another_thread(tmp_path):
    # The server evaluates on a thread pool. The base is fp32 here, so loading casts the torso
    # to bf16, and a cast left lazy belongs to the loading thread's stream.
    engine = load_engine(str(_checkpoint(tmp_path, "pointer", torch_dtype="bfloat16")), device="mlx")
    with ThreadPoolExecutor(1) as pool:
        response = pool.submit(engine.evaluate, REQUESTS[1]).result()
    assert set(response.answers) == set(QUESTIONS)


# ---- the cross-request state cache -----------------------------------------


def _fresh_engine(path, **kwargs):
    from strands_decider.mlx_engine import load_mlx_engine

    return load_mlx_engine(str(path), None, **kwargs)


def test_a_repeated_state_is_encoded_once(tmp_path):
    engine = _fresh_engine(_checkpoint(tmp_path, "pointer"))
    first = _probabilities(engine.evaluate(REQUESTS[1]))
    assert engine.state_encodes == 1  # the multi-question request builds one state prefix
    second = _probabilities(engine.evaluate(REQUESTS[1]))
    assert engine.state_encodes == 1  # the second request reuses the snapshot
    assert second == first


def test_a_single_question_reuses_a_cached_state(tmp_path):
    engine = _fresh_engine(_checkpoint(tmp_path, "pointer"))
    _probabilities(engine.evaluate(REQUESTS[1]))  # populate the cache
    encoded = engine.state_encodes
    single = _probabilities(engine.evaluate(REQUESTS[0]))  # same state, one question
    assert engine.state_encodes == encoded  # the whole-prompt path did not re-encode
    engine._state_cache.clear()  # force the uncached path for the reference answers
    reference = _probabilities(engine.evaluate(REQUESTS[0]))
    for name in reference:
        assert single[name] == pytest.approx(reference[name], abs=1e-4), name


def test_the_state_cache_evicts_least_recently_used(tmp_path):
    engine = _fresh_engine(_checkpoint(tmp_path, "pointer"), state_cache_entries=1)
    _probabilities(engine.evaluate(REQUESTS[1]))  # state A cached
    other = SystemOneRequest(state=POLICY * 2 + "A different question entirely.",
                             questions=QUESTIONS)  # multi-question, so state B populates too
    _probabilities(engine.evaluate(other))  # state B evicts A
    assert engine.state_encodes == 2
    _probabilities(engine.evaluate(REQUESTS[1]))  # A again: must miss after eviction
    assert engine.state_encodes == 3


def test_state_cache_entries_zero_disables_the_cache(tmp_path):
    engine = _fresh_engine(_checkpoint(tmp_path, "pointer"), state_cache_entries=0)
    _probabilities(engine.evaluate(REQUESTS[1]))
    _probabilities(engine.evaluate(REQUESTS[1]))
    assert engine.state_encodes == 2


def test_mlx_loads_the_base_at_the_revision_in_provenance(tmp_path, monkeypatch):
    import strands_decider.mlx_engine as mlx_engine

    path = _checkpoint(tmp_path, "pointer")
    base = str(tmp_path / "base")
    provenance = {"base_model": base, "base_model_revision": "abc123"}
    (path / "provenance.json").write_text(json.dumps(provenance))
    calls, checkpoint_dir = [], mlx_engine.checkpoint_dir

    def record(repo, revision=None):
        calls.append((repo, revision))
        return checkpoint_dir(repo, revision)

    monkeypatch.setattr(mlx_engine, "checkpoint_dir", record)
    load_engine(str(path), device="mlx")
    assert (base, "abc123") in calls
