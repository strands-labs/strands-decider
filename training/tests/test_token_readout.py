"""The opt-in token readout: render "lettered", TokenReadoutHead, attention_mode prefix_bidir,
and that the pointer path stays what origin/main computes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from test_ddp import _examples, _load_tokenizer, _qwen3_config, _tiny_model_factory, _tokenizer_file

import training.train as T
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import (
    StrandsDeciderConfig,
    StrandsDeciderModel,
    TokenReadoutHead,
    lm_head_rows,
    load_head_state,
    prefix_bidir_masks,
)
from strands_decider.prompting import (
    CODE_CANDIDATES,
    ChatFrame,
    build_prompt,
    chat_frame,
    lettered_prompt,
    option_codes,
)
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest
from training.data.collate import CollatorConfig, SystemOneCollator
from training.data.format import load_examples, write_jsonl
from training.soup import soup

# A stand-in chat template with Gemma 4's layout; content is a string or a list of text parts.
TEMPLATE = (
    "<bos>{% for m in messages %}<|turn>{{ m['role'] }}\n"
    "{% if m['content'] is string %}{{ m['content'] }}"
    "{% else %}{% for p in m['content'] %}{{ p['text'] }}{% endfor %}{% endif %}<turn|>\n"
    "{% endfor %}{% if add_generation_prompt %}<|turn>model\n<|channel>thought\n<channel|>"
    "{% endif %}"
)
QUESTIONS = {
    "team": ChoiceQuestion(
        instructions="Which team?", criteria={"billing": "money", "tech": None, "sales": "deals"}
    ),
    "urgent": NoulQuestion(instructions="It is urgent."),
    "mood": ScoreQuestion(instructions="How angry?", criteria=["calm", "annoyed", "furious"]),
}
# _pointer_trace on origin/main 87d0fd5, before the token readout existed.
MAIN_RENDERS = "bb75baf142839488b12ab5331f4d7b7c804d48d90c3e574cf71a8a9afc39960f"
MAIN_BATCHES = "7d2ad3f1c9b937565d247346e57ba050e9858f26b43e9aedd32a2d06bf94d44d"
MAIN_LOSSES = [1.2146382, 1.0987360, 1.3282480, 1.2259021, 1.1267848, 1.0307044]
MAIN_PROBS = [0.2482, 0.3714, 0.3804, 0.5091, 0.4687, 0.2211, 0.3102] * 2  # cached, batched


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    from tokenizers import Tokenizer

    d = tmp_path_factory.mktemp("token_readout")
    rows = _examples(150, seed=1)
    write_jsonl(str(d / "a.jsonl"), rows)
    write_jsonl(str(d / "val.jsonl"), _examples(20, seed=2))
    _tokenizer_file(str(d / "tokenizer.json"), rows)
    lettered = Tokenizer.from_file(str(d / "tokenizer.json"))
    lettered.add_tokens(list(CODE_CANDIDATES))  # every code candidate one token
    lettered.save(str(d / "lettered.json"))
    return d


def _tok(corpus):
    tok = _load_tokenizer(str(corpus / "lettered.json"))
    tok.chat_template = TEMPLATE
    return tok


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
    cfg = StrandsDeciderConfig(base_model="tiny", head_type="token_readout", render="lettered")
    for k, v in kw.items():
        setattr(cfg, k, v)
    return _readout_factory(corpus)(cfg)


# ---------------------------------------------------------------- the pointer path


def _pointer_trace(corpus):
    """(sha256 of numbered renders, sha256 of six training batches, their losses, the engine's
    probabilities on both serving paths), on names origin/main has."""
    renders = hashlib.sha256()
    for ex in _examples(150, seed=1):
        for order in (None, list(reversed(range(ex.n_options)))):
            renders.update(build_prompt(ex.state, ex.to_question(), option_order=order)[0].encode())
    rows = load_examples([str(corpus / "a.jsonl")])
    torch.manual_seed(0)
    model = _tiny_model_factory(str(corpus / "tokenizer.json"))(
        StrandsDeciderConfig(base_model="tiny", pointer_dim=16)
    )
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=2048), train=True)
    batches, losses = hashlib.sha256(), []
    for i in range(0, 48, 8):
        b = coll(rows[i : i + 8])
        for k in sorted(b):
            batches.update(f"{k}{b[k].dtype}{tuple(b[k].shape)}".encode())
            batches.update(b[k].contiguous().numpy().tobytes())
        with torch.no_grad():
            out = model(**{k: b[k] for k in b if k not in ("teacher", "has_teacher")})
        losses.append(float(out["loss"]))
    engine = SystemOneEngine(model, EngineConfig(device="cpu"))
    probs = []
    for prefix_cache in (True, False):
        engine.cfg.use_prefix_cache = prefix_cache
        got = engine.evaluate(SystemOneRequest(state="Payouts failing.", questions=QUESTIONS))
        for a in got.answers.values():
            probs += [a.noul] if a.type == "noul" else list(a.probabilities.values())
    return renders.hexdigest(), batches.hexdigest(), losses, probs


def test_the_pointer_path_is_what_origin_main_computes(corpus):
    renders, batches, losses, probs = _pointer_trace(corpus)
    assert (renders, batches) == (MAIN_RENDERS, MAIN_BATCHES)
    assert losses == pytest.approx(MAIN_LOSSES, rel=1e-5)
    assert probs == pytest.approx(MAIN_PROBS, abs=2e-4)  # answers are rounded to 4 places


def test_a_pointer_config_json_has_no_new_keys_and_a_readout_one_round_trips(tmp_path):
    assert "render" not in StrandsDeciderConfig().to_json()
    assert "attention_mode" not in StrandsDeciderConfig().to_json()
    cfg = StrandsDeciderConfig(
        head_type="token_readout", render="lettered", attention_mode="prefix_bidir"
    )
    (tmp_path / "c.json").write_text(cfg.to_json())
    assert StrandsDeciderConfig.from_json(str(tmp_path / "c.json")) == cfg
    with pytest.raises(ValueError, match="goes with"):
        StrandsDeciderConfig(head_type="token_readout")
    with pytest.raises(ValueError, match="attention_mode"):
        StrandsDeciderConfig(attention_mode="prefix_bidir")


# ---------------------------------------------------------------- render "lettered"


def test_lettered_render_golden():
    frame = ChatFrame("<H>", "<T>", CODE_CANDIDATES[:255])
    state, rest = lettered_prompt({"a": "é"}, QUESTIONS["team"], frame, option_order=[2, 0, 1])
    assert state == '<H>State:\n{"a": "é"}'
    assert rest == (
        "\n\nQuestion:\nWhich team?\n\nOptions:\nA: sales: deals\nB: billing: money\nC: tech"
        "\n\nReturn only the letter code of the best option.<T>"
    )
    _, rest = lettered_prompt("s", NoulQuestion(instructions="Up."), frame)
    assert "Options:\nA: No / false\nB: Yes / true\n\n" in rest
    _, rest = lettered_prompt("s", ScoreQuestion(instructions="", criteria=["low", "high"]), frame)
    assert rest.startswith(
        "\n\nQuestion:\nChoose the best matching option.\n\nOptions:\nA: low\nB: high"
    )


def test_codes_are_single_tokens_and_frame_the_chat_template(corpus):
    tok = _tok(corpus)
    assert option_codes(tok) == list(CODE_CANDIDATES[:255])
    frame = chat_frame(tok)
    assert frame.head.startswith("<bos><|turn>system\nClassify") and frame.head.endswith("user\n")
    assert frame.tail == "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"


def test_codes_on_a_real_tokenizer():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it", local_files_only=True)
    except OSError:
        pytest.skip("gemma-4-E2B-it is not in the local Hub cache")
    codes = option_codes(tok)
    assert codes[:3] == ["A", "B", "C"] and len(codes) == 255
    assert chat_frame(tok).tail.endswith("\n")  # the generation prompt, read at its last token


def test_the_collator_shuffles_only_choices_and_marks_the_state(corpus):
    rows = load_examples([str(corpus / "a.jsonl")])
    coll = SystemOneCollator(_tok(corpus), CollatorConfig(render="lettered"), train=True)
    orders = [coll._option_order(ex) for ex in rows[:40]]
    assert all(o is None for o, ex in zip(orders, rows, strict=False) if ex.kind != "choice")
    assert any(o is not None for o in orders)
    b = coll(rows[:8])
    assert "opt_idx" not in b
    for row, end, ex in zip(b["input_ids"], b["prefix_end"], rows[:8], strict=True):
        state = lettered_prompt(ex.state, ex.to_question(), coll.frame)[0]
        seen = coll.tok.decode(row[:end])  # the state's tokens, less one straddling the boundary
        assert state.startswith(seen) and len(state) - len(seen) <= 3
    with pytest.raises(ValueError, match="longer than max_length"):
        SystemOneCollator(_tok(corpus), CollatorConfig(render="lettered", max_length=20))(rows[:2])


# ---------------------------------------------------------------- the head


def test_the_head_is_255_rows_capped_and_masked_to_the_options(corpus):
    head = TokenReadoutHead(32, softcap=2.0)
    assert head.proj.weight.shape == (255, 32) and head.proj.bias is None
    x = torch.randn(3, 32) * 100
    logits = head(x, 4)
    assert logits.shape == (3, 4) and logits.abs().max() <= 2.0
    torch.testing.assert_close(logits, 2.0 * torch.tanh(head.proj(x)[:, :4] / 2.0))
    model = _readout(corpus).eval()
    b = SystemOneCollator(model.tokenizer, CollatorConfig(render="lettered"), train=False)(
        load_examples([str(corpus / "a.jsonl")])[:6]
    )
    with torch.no_grad():
        out = model(**b)
    for row, n in zip(out["log_probs"], b["n_slots"], strict=True):
        assert torch.isinf(row[n:]).all() and row[:n].exp().sum() == pytest.approx(1.0)


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
    frame = chat_frame(model.tokenizer)

    def hidden(state, question):
        s, q = lettered_prompt(state, question, frame)
        ids = model.tokenizer(s + q, add_special_tokens=False, return_tensors="pt")["input_ids"]
        end = len(model.tokenizer(s, add_special_tokens=False)["input_ids"]) - 1
        mask = torch.ones_like(ids)
        masks = prefix_bidir_masks(model.torso.config, mask, torch.tensor([end]), torch.float32)
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
        render="lettered",
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


def test_fits_measures_the_lettered_prompt_with_every_phrasing(corpus):
    from training.data.multistep import fits

    tok = _tok(corpus)
    frame = chat_frame(tok)
    rows = load_examples([str(corpus / "a.jsonl")])[:6]

    def lettered(e, i):
        return len(
            tok(
                "".join(lettered_prompt(e.state, e.to_question(i), frame)), add_special_tokens=False
            )["input_ids"]
        )

    numbered = max(len(tok(build_prompt(e.state, e.to_question())[0])["input_ids"]) for e in rows)
    longest = max(lettered(e, i) for e in rows for i in e.all_instructions())
    assert longest > numbered  # the chat template and system line make it longer
    assert len(fits(rows, tok, numbered)) == len(rows)
    assert fits(rows, tok, numbered, frame) == [
        e for e in rows if max(lettered(e, i) for i in e.all_instructions()) <= numbered
    ]
    assert fits(rows, tok, longest, frame) == rows


def test_the_readout_cap_is_saved_with_the_checkpoint(corpus, monkeypatch):
    from strands_decider import modeling

    monkeypatch.setattr(modeling, "logit_softcap", lambda torso: 30.0)
    model = _readout(corpus)
    assert model.config.readout_softcap == 30.0 and model.head.softcap == 30.0
    saved = StrandsDeciderConfig(**json.loads(model.config.to_json()))
    monkeypatch.setattr(modeling, "logit_softcap", lambda torso: None)  # another base config
    assert modeling.build_head(saved, 32).softcap == 30.0
    plain = StrandsDeciderConfig(base_model="tiny", head_type="token_readout", render="lettered")
    with pytest.raises(ValueError, match="readout_softcap"):
        modeling.build_head(plain, 32)  # never guessed from the base model at load


def test_prefix_bidir_refuses_an_unknown_layer_type():
    cfg = SimpleNamespace(_attn_implementation="sdpa", layer_types=["chunked_attention"])
    with pytest.raises(ValueError, match="chunked_attention"):
        prefix_bidir_masks(cfg, torch.ones(1, 3), torch.tensor([1]), torch.float32)
