"""The opt-in token readout: TokenReadoutHead, the spec's 255-code readout read at the last
prompt token of a strands-chat prompt (strands_decider.core.render), and attention_mode
prefix_bidir."""

from __future__ import annotations

import json
import math
import os
import re
import shutil
from types import SimpleNamespace

import pytest
import torch
from core_support import toy_tokenizer
from test_ddp import _examples, _qwen3_config, _tiny_model_factory, _tokenizer_file
from transformers import PreTrainedTokenizerFast

import training.train as T
from strands_decider import core
from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    TokenReadoutHead,
    lm_head_rows,
    load_head_state,
    prefix_bidir_masks,
)
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import Example, load_examples, write_jsonl
from training.soup import soup

QUESTIONS = {
    "team": {
        "type": "choice",
        "instructions": "Which team?",
        "criteria": {"billing": "money", "tech": None, "sales": "deals"},
    },
    "urgent": {"type": "noul", "instructions": "It is urgent."},
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    d = tmp_path_factory.mktemp("token_readout")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    _tokenizer_file(str(d / "tokenizer.json"), rows)  # the pointer models' (KL reference)
    return d


def _tok(corpus):
    """A tokenizer core.render takes (Gemma 4's control tokens, every code one token) that
    knows the corpus's words."""
    text = (corpus / "a.jsonl").read_text() + (corpus / "val.jsonl").read_text()
    words = re.findall(r"\w+", text + " pay outs fail work now")
    return PreTrainedTokenizerFast(tokenizer_object=toy_tokenizer(words), pad_token="<eos>")


def _chat(tok, request):
    return core.render(request, tok.backend_tokenizer, core.CHAT).questions["q"]


def _readout_factory(corpus):
    """Stands in for StrandsDeciderModel.from_pretrained_base, as _tiny_model_factory."""
    from transformers import Qwen3Model

    def build(config, *, device_map=None, attn_implementation=None):
        tok = _tok(corpus)
        config.lora_dropout = 0.0
        model = StrandsDeciderModel(config, Qwen3Model(_qwen3_config(len(tok))), tok)
        if config.use_lora:
            model.attach_lora()
        return model

    return build


def _readout(corpus, **kw):
    cfg = StrandsDeciderConfig(base_model="tiny", head_type="token_readout")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return _readout_factory(corpus)(cfg)


def test_the_collator_shuffles_only_choices_and_refuses_a_long_prompt(corpus):
    rows = load_examples([str(corpus / "a.jsonl")])
    coll = SystemOneCollator(_tok(corpus), CollatorConfig(render=core.CHAT), train=True)
    orders = [coll._option_order(ex) for ex in rows[:40]]
    assert all(o is None for o, ex in zip(orders, rows, strict=False) if ex.kind != "choice")
    assert any(o is not None for o in orders)
    assert "opt_idx" not in coll(rows[:8])
    with pytest.raises(ValueError, match="longer than max_length"):
        SystemOneCollator(_tok(corpus), CollatorConfig(render=core.CHAT, max_length=20))(rows[:2])


# ---------------------------------------------------------------- the head


def test_the_head_is_255_rows_capped_and_masked_to_the_options(corpus):
    head = TokenReadoutHead(32, softcap=2.0)
    assert head.proj.weight.shape == (255, 32) and head.proj.bias is None
    x = torch.randn(3, 32) * 100
    logits = head(x, 4)
    assert logits.shape == (3, 4) and logits.abs().max() <= 2.0
    torch.testing.assert_close(logits, 2.0 * torch.tanh(head.proj(x)[:, :4] / 2.0))
    model = _readout(corpus).eval()
    b = SystemOneCollator(model.tokenizer, CollatorConfig(render=core.CHAT), train=False)(
        load_examples([str(corpus / "a.jsonl")])[:6]
    )
    with torch.no_grad():
        out = model(**b)
    for row, n in zip(out["log_probs"], b["n_slots"], strict=True):
        assert torch.isinf(row[n:]).all() and row[:n].exp().sum() == pytest.approx(1.0)


def test_under_bf16_autocast_the_cap_runs_in_float32():
    """The product comes back in bf16 under autocast; the cap is then computed in float32, as
    the runtimes compute it (docs/spec.md, section 7.1), not in bf16."""
    torch.manual_seed(0)
    head = TokenReadoutHead(32, softcap=30.0)
    x = torch.randn(64, 32) * 4
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z = head.proj(x)
        logits = head(x, 255)
    assert z.dtype == torch.bfloat16 and logits.dtype == torch.float32
    want = 30.0 * torch.tanh(z.float() / 30.0)
    assert torch.equal(logits, want)
    in_bf16 = (30.0 * torch.tanh(z / 30.0)).float()  # what the head computed before
    assert not torch.equal(in_bf16, want)


@pytest.mark.parametrize("tied", [True, False])
def test_the_readout_is_seeded_from_the_lm_head_rows(tmp_path, tied):
    from transformers import Qwen3ForCausalLM

    cfg = _qwen3_config(50)
    cfg.tie_word_embeddings = tied
    lm = Qwen3ForCausalLM(cfg)
    lm.save_pretrained(str(tmp_path))
    rows = lm_head_rows(StrandsDeciderConfig(base_model=str(tmp_path)), lm.model, [7, 3])
    torch.testing.assert_close(rows, lm.lm_head.weight.detach()[[7, 3]])


# ---------------------------------------------------------------- prefix_bidir


def test_prefix_bidir_masks_per_layer_type():
    cfg = SimpleNamespace(
        _attn_implementation="sdpa",
        layer_types=["linear_attention", "full_attention", "sliding_attention"],
    )
    pad = torch.tensor([[1, 1, 1, 1, 0]])
    m = prefix_bidir_masks(cfg, pad, torch.tensor([2]), torch.float32)
    assert m["linear_attention"] is pad
    want = torch.tensor(
        [[1, 1, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0], [1, 1, 1, 1, 0]]
    ).bool()
    assert (
        torch.equal(m["full_attention"][0, 0], want)
        and m["sliding_attention"] is m["full_attention"]
    )
    eager = prefix_bidir_masks(
        SimpleNamespace(_attn_implementation="eager"), pad, torch.tensor([2]), torch.float32
    )
    assert torch.equal(eager["full_attention"][0, 0] == 0, want)


def test_under_prefix_bidir_the_state_never_sees_the_question(corpus):
    model = _readout(corpus, attention_mode="prefix_bidir", use_lora=False).eval()

    def hidden(state, question):
        q = _chat(model.tokenizer, {"state": state, "questions": {"q": question}})
        ids = torch.tensor([q.ids])
        end = q.state_len
        masks = prefix_bidir_masks(
            model.torso.config, torch.ones_like(ids), torch.tensor([end]), torch.float32
        )
        with torch.no_grad():
            return model.encode(ids, masks), end

    a, end = hidden("pay outs fail now", QUESTIONS["team"])
    b, _ = hidden("pay outs fail now", QUESTIONS["urgent"])
    c, _ = hidden("pay outs work now", QUESTIONS["team"])
    torch.testing.assert_close(a[0, :end], b[0, :end])  # the question does not reach the state
    assert not torch.allclose(a[0, -1], c[0, -1])  # the question reads the state
    assert not torch.allclose(a[0, 1], c[0, 1])  # the state reads itself both ways


# ---------------------------------------------------------------- training


def test_train_soup_and_config_yaml(corpus, tmp_path, monkeypatch):
    import yaml

    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", _readout_factory(corpus))
    raw = dict(
        train_files=[str(corpus / "a.jsonl")],
        val_files=[str(corpus / "val.jsonl")],
        base_model="tiny",
        head_type="token_readout",
        attention_mode="prefix_bidir",
        lr_floor=0.1,
        adam_betas=[0.9, 0.999],
        lora_r=2,
        lora_alpha=4,
        micro_batch_size=4,
        grad_accum=1,
        max_steps=2,
        eval_every=0,
        output_dir=str(tmp_path / "run"),
    )
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(raw))
    cfg = T.TrainConfig.from_yaml(str(tmp_path / "c.yaml"))
    assert (cfg.head_type, cfg.attention_mode, cfg.lr_floor, cfg.adam_betas) == (
        "token_readout",
        "prefix_bidir",
        0.1,
        [0.9, 0.999],
    )
    assert T._lr_lambda(10, 2, 10, 0.1) == pytest.approx(0.1)
    out = T.train(cfg)
    ck = next(r for r, _, f in os.walk(out) if "strands_decider_config.json" in f)
    other = tmp_path / "other"
    shutil.copytree(ck, other)
    heads = [load_head_state(ck)]
    from safetensors.torch import save_file

    save_file({"proj.weight": heads[0]["proj.weight"] + 1.0}, str(other / "head.safetensors"))
    heads.append(load_head_state(str(other)))
    soup([ck, str(other)], str(tmp_path / "soup"))
    souped = StrandsDeciderConfig.from_json(str(tmp_path / "soup" / "strands_decider_config.json"))
    assert (souped.head_type, souped.attention_mode) == ("token_readout", "prefix_bidir")
    torch.testing.assert_close(
        load_head_state(str(tmp_path / "soup"))["proj.weight"],
        (heads[0]["proj.weight"] + heads[1]["proj.weight"]) / 2,
    )


def test_the_engines_refuse_a_token_readout_checkpoint(corpus, tmp_path, monkeypatch):
    from strands_decider import infer
    from strands_decider.hidden_engine import save_assets
    from strands_decider.vision import VisionDeciderModel

    model = _readout(corpus)
    monkeypatch.setattr(StrandsDeciderModel, "load", lambda *a, **k: model)
    with pytest.raises(ValueError, match="separate inference stack"):
        infer.load_engine("x", device="cpu")
    with pytest.raises(ValueError, match="separate inference stack"):
        save_assets(model, str(tmp_path / "assets"), "m")
    (tmp_path / "strands_decider_config.json").write_text(model.config.to_json())
    with pytest.raises(ValueError, match="separate inference stack"):
        VisionDeciderModel.load(str(tmp_path))


# ---------------------------------------------------------------- review guards


def _ck_dir(tmp_path, model):
    d = tmp_path / "ck"
    d.mkdir()
    (d / "strands_decider_config.json").write_text(model.config.to_json())
    return str(d)


def _kl_cfg(corpus, tmp_path, **kw):
    return T.TrainConfig(
        train_files=[str(corpus / "a.jsonl")],
        val_files=[str(corpus / "val.jsonl")],
        base_model="tiny",
        kl_frozen_weight=0.3,
        precompute_frozen_kl=True,
        group_by_length=True,
        micro_batch_size=4,
        grad_accum=1,
        max_steps=1,
        eval_every=0,
        output_dir=str(tmp_path / "run"),
        **kw,
    )


def test_init_from_a_token_readout_with_a_kl_reference_is_refused(corpus, tmp_path, monkeypatch):
    # _init_from sets head_type from the checkpoint; the refusal must come after it.
    model = _readout(corpus)
    ck = _ck_dir(tmp_path, model)
    monkeypatch.setattr(StrandsDeciderModel, "load", lambda *a, **k: model)
    with pytest.raises(ValueError, match="frozen-KL reference yet"):
        T.train(_kl_cfg(corpus, tmp_path, init_from=ck, kl_frozen_reference=ck))


def test_a_kl_reference_of_another_head_type_is_refused(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(
        StrandsDeciderModel,
        "from_pretrained_base",
        staticmethod(_tiny_model_factory(str(corpus / "tokenizer.json"))),
    )
    reference = _readout(corpus)
    monkeypatch.setattr(StrandsDeciderModel, "load", lambda *a, **k: reference)
    with pytest.raises(ValueError, match="has head_type 'token_readout'"):
        T.train(_kl_cfg(corpus, tmp_path, kl_frozen_reference="ref", pointer_dim=16))


def test_the_readout_cap_is_saved_with_the_checkpoint(corpus, monkeypatch):
    from strands_decider import modeling

    monkeypatch.setattr(modeling, "logit_softcap", lambda torso: 30.0)
    model = _readout(corpus)
    assert model.config.readout_softcap == 30.0 and model.head.softcap == 30.0
    saved = StrandsDeciderConfig(**json.loads(model.config.to_json()))
    monkeypatch.setattr(modeling, "logit_softcap", lambda torso: None)  # another base config
    assert modeling.build_head(saved, 32).softcap == 30.0
    plain = StrandsDeciderConfig(base_model="tiny", head_type="token_readout")
    with pytest.raises(ValueError, match="readout_softcap"):
        modeling.build_head(plain, 32)  # never guessed from the base model at load


def test_prefix_bidir_refuses_an_unknown_layer_type():
    cfg = SimpleNamespace(_attn_implementation="sdpa", layer_types=["chunked_attention"])
    with pytest.raises(ValueError, match="chunked_attention"):
        prefix_bidir_masks(cfg, torch.ones(1, 3), torch.tensor([1]), torch.float32)


def _tiny_run(corpus, tmp_path, monkeypatch, **extra):
    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", _readout_factory(corpus))
    raw = dict(
        train_files=[str(corpus / "a.jsonl")],
        val_files=[str(corpus / "val.jsonl")],
        base_model="tiny",
        head_type="token_readout",
        attention_mode="prefix_bidir",
        lora_r=2,
        lora_alpha=4,
        micro_batch_size=4,
        grad_accum=1,
        max_steps=2,
        eval_every=0,
        output_dir=str(tmp_path / "run"),
    )
    return T.TrainConfig(**(raw | extra))


def test_training_turns_off_cudnn_attention(corpus, tmp_path, monkeypatch):
    monkeypatch.setattr(
        torch.backends.cuda, "enable_cudnn_sdp", torch.backends.cuda.enable_cudnn_sdp
    )
    torch.backends.cuda.enable_cudnn_sdp(True)
    try:
        T.train(_tiny_run(corpus, tmp_path, monkeypatch))
        assert not torch.backends.cuda.cudnn_sdp_enabled()
    finally:
        torch.backends.cuda.enable_cudnn_sdp(True)


def test_a_non_finite_gradient_stops_the_run_before_the_update(corpus, tmp_path, monkeypatch):
    real_clip = torch.nn.utils.clip_grad_norm_
    steps = []

    def clip(params, max_norm):
        real_clip(params, max_norm)
        return torch.tensor(float("nan"))

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    monkeypatch.setattr(torch.optim.AdamW, "step", lambda self, *a, **k: steps.append(1))
    cfg = _tiny_run(corpus, tmp_path, monkeypatch)
    with pytest.raises(FloatingPointError, match="step 1: gradient norm is not finite"):
        T.train(cfg)
    assert steps == []  # the optimizer never took the NaN step
    saved = torch.load(os.path.join(cfg.output_dir, "nonfinite-step1-rank0.pt"), weights_only=False)
    assert [f["forward"] for f in saved] == [0] and saved[0]["input_ids"].shape[0] == 4


def test_eval_drops_an_over_long_row_instead_of_stopping(corpus, monkeypatch, capsys):
    """One long row must not cost the whole fold: `calibrate` and `eval` drop it and say so."""
    import training.evaluate as E

    model = _readout_factory(corpus)(
        StrandsDeciderConfig(base_model="tiny", head_type="token_readout")
    )
    monkeypatch.setattr(E, "SystemOneEngine", lambda m, c: SimpleNamespace(model=m))
    rows = load_examples([str(corpus / "a.jsonl")])
    long = Example("noul", "x " * 4000, "Is it long?", [["false", None], ["true", None]], 1)

    def longest(e):  # as training/data/multistep.fits measures a strands-chat row
        return max(len(_chat(model.tokenizer, e.to_request(i)).ids) for i in e.all_instructions())

    cap = max(longest(e) for e in rows)
    logits, labels, _slots, kept = E.collect_logits(
        model, [*rows, long], device="cpu", batch_size=2, max_length=cap
    )
    assert len(kept) == len(rows) and long not in kept
    assert logits.shape[0] == len(kept) == labels.shape[0]
    assert "dropped 1" in capsys.readouterr().out
    with pytest.raises(ValueError, match="longer than max_length"):
        E.collect_logits(model, [long], device="cpu", max_length=cap)


def test_a_token_readout_is_not_switched_to_serving_attention():
    """`load` leaves a token readout's attention alone; a pointer head still gets the override."""
    from strands_decider.modeling import serving_attn_implementation, serving_attn_override

    e2b = SimpleNamespace(model_type="gemma4_text", num_key_value_heads=1)
    assert serving_attn_implementation(e2b) == "eager"  # the serving default for E2B
    readout = StrandsDeciderConfig(base_model="t", head_type="token_readout")
    pointer = StrandsDeciderConfig(base_model="t")
    assert serving_attn_override(readout, e2b, False, None) is None
    assert serving_attn_override(pointer, e2b, False, None) == "eager"
    assert serving_attn_override(pointer, e2b, True, None) is None  # training
    assert serving_attn_override(pointer, e2b, False, "sdpa") is None  # the caller named one


def test_bf16_autocast_trains_with_fp32_master_weights(corpus, tmp_path, monkeypatch):
    """The forward runs in bf16 under autocast; the trained weights and their updates stay fp32."""
    cfg = _tiny_run(corpus, tmp_path, monkeypatch, bf16_autocast=True, log_every=1)
    out = T.train(cfg)
    hist = json.load(open(os.path.join(out, "history.json"), encoding="utf-8"))
    losses = [h["loss"] for h in hist if "loss" in h]
    assert losses and all(math.isfinite(x) for x in losses)
    head = load_head_state(out)["proj.weight"]
    assert head.dtype == torch.float32
