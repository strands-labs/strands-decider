"""max_batch_tokens on a tiny random-weight Gemma 4 (no download), both serving paths:

- the cap bounds the padded forward (rows x the longest fitted row), not the sum of rows;
- the cap only splits forwards: every question keeps the input it has with the cap off.
"""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4TextConfig"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from tokenizers import Tokenizer, models, pre_tokenizers  # noqa: E402

from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    StrandsDeciderConfig,
    StrandsDeciderModel,
    ensure_bos,
)
from strands_decider.prompting import render_question, render_state  # noqa: E402
from strands_decider.schema import ChoiceQuestion, SystemOneRequest  # noqa: E402


def _model(texts, max_length):
    split = pre_tokenizers.Whitespace()
    words = sorted({w for t in texts for w, _ in split.pre_tokenize_str(t)})
    core = Tokenizer(models.WordLevel(
        {w: i for i, w in enumerate(["<pad>", "<bos>", "<eos>", "<unk>", *words])}, unk_token="<unk>"))
    core.pre_tokenizer = split
    tok = ensure_bos(transformers.PreTrainedTokenizerFast(
        tokenizer_object=core, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>", unk_token="<unk>"))
    torch.manual_seed(0)
    cfg = transformers.Gemma4TextConfig(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=5,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        hidden_size_per_layer_input=8, vocab_size_per_layer_input=len(tok), sliding_window=16,
        layer_types=["sliding_attention"] * 4 + ["full_attention"], num_kv_shared_layers=0,
        pad_token_id=tok.pad_token_id, tie_word_embeddings=True, final_logit_softcapping=30.0)
    cfg._attn_implementation = "eager"
    torso = transformers.Gemma4ForCausalLM(cfg).model.eval()
    return StrandsDeciderModel(StrandsDeciderConfig(
        base_model="tiny-gemma4", use_lora=False, head_type="pointer", pointer_dim=16, num_slots=4,
        max_length=max_length, force_bos=True, torch_dtype="float32"), torso, tok).eval()


def _texts(req):
    return [render_state(req.state), *(render_question(q).text for q in req.questions.values())]


SHORT = ChoiceQuestion(instructions="Choose.", criteria={"yes": None, "no": None})


@pytest.mark.parametrize("prefix", [False, True])
def test_the_cap_bounds_the_padded_forward(prefix):
    # Seven short questions and a long one: their real tokens fit the cap, padded they do not.
    long = ChoiceQuestion(instructions=" ".join(["Consider"] * 400), criteria={"yes": None, "no": None})
    req = SystemOneRequest(state="A fact.", questions={**{f"s{i}": SHORT for i in range(7)}, "long": long})
    model = _model(_texts(req), 1024)
    cap = 726  # the real tokens of all eight rows, BOS included
    engine = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=prefix, max_batch_tokens=cap))
    seen: list[int] = []
    hook = model.torso.register_forward_pre_hook(
        lambda _m, _a, kw: seen.append(kw["attention_mask"].numel()), with_kwargs=True)
    try:
        assert len(engine.evaluate(req).answers) == 8
    finally:
        hook.remove()
    assert len(seen) > 1 and max(seen) <= cap, seen


@pytest.mark.parametrize("prefix", [False, True])
def test_the_cap_keeps_every_input(prefix):
    # At a 256-token window the long question decides how much of the state is kept; a cap
    # that puts the short question in a forward of its own must not refit it.
    long = ChoiceQuestion(instructions=" ".join(["Consider"] * 155), criteria={"yes": None, "no": None})
    state = " ".join(["ordinary"] * 100 + ["important", "evidence"] * 150)
    req = SystemOneRequest(state=state, questions={"short": SHORT, "long": long})
    model = _model(_texts(req), 256)
    a = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=prefix)).evaluate(req)
    capped = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=prefix, max_batch_tokens=256))
    assert [c[0] for c in capped._chunks(render_state(state), [render_question(q) for q in req.questions.values()])] \
        == [slice(0, 1), slice(1, 2)]  # the cap splits the request
    b = capped.evaluate(req)
    for k in req.questions:
        x, y = a.answers[k].probabilities, b.answers[k].probabilities
        assert all(abs(x[o] - y[o]) <= 1e-4 for o in x), (k, x, y)
