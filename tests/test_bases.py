"""A Llama-family torso (MiniCPM5's), on the tiny random-weight base of tests/tiny_bases.py
(nothing is downloaded): the decoder loads alone, LoRA reaches the modules it should, the frozen readout is the LM's own
option-number distribution, training moves the loss, a checkpoint round-trips, and the
shared-prefix path answers as encoding each full prompt does.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import pytest
import torch
import transformers
from tiny_bases import ATTENTION_MLP, save_llama

from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import Example, write_jsonl
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion
from strands_decider.train import TrainConfig, train


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return save_llama(str(tmp_path_factory.mktemp("llama")))


def _model(base_dir: str) -> StrandsDeciderModel:
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="pointer", pointer_dim=16,
                               torch_dtype="float32", max_length=512, lora_targets=ATTENTION_MLP)
    return StrandsDeciderModel.from_pretrained_base(cfg).eval()


def _rows(n: int) -> list[Example]:
    out = []
    for k in range(n):
        if k % 2:
            out.append(Example(kind="noul", state=f"Order {k} shipped late.", instructions="Was it late?",
                               options=[["false", "on time"], ["true", "late"]], label=1, task="late"))
        else:
            out.append(Example(kind="choice", state=f"Ticket {k}: refund please.", instructions="Which team?",
                               options=[["billing", ""], ["shipping", ""], ["sales", ""]], label=0,
                               task="route"))
    return out


def _batch(model: StrandsDeciderModel) -> dict[str, torch.Tensor]:
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=512, head_type="pointer"),
                             train=False)
    return coll(_rows(4))


def _lora_modules(model: StrandsDeciderModel) -> set[tuple[int, str]]:
    found = set()
    for name, _ in model.torso.named_modules():
        m = re.search(r"layers\.(\d+)\..*\.(\w+)\.lora_A$", name)
        if m:
            found.add((int(m.group(1)), m.group(2)))
    return found


def test_the_text_decoder_loads_alone(base):
    model = _model(base)
    assert type(model.torso.base_model.model).__name__ == "LlamaModel"
    assert model.hidden_size(model.torso) == 64
    ids = model.tokenizer("<state>")["input_ids"]
    assert ids[0] == model.tokenizer.bos_token_id  # the tokeniser prepends BOS, as MiniCPM5's does
    assert not StrandsDeciderModel.is_hybrid(model.torso)


def test_lora_reaches_attention_and_mlp_of_every_layer(base):
    assert _lora_modules(_model(base)) == {(i, m) for i in range(2) for m in ATTENTION_MLP}


@torch.no_grad()
def test_forward_shapes(base):
    model = _model(base)
    b = _batch(model)
    out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"], n_slots=b["n_slots"],
                opt_idx=b["opt_idx"], labels=b["labels"])
    assert out["log_probs"].shape == (4, 3)
    assert torch.isfinite(out["loss"])
    p = out["log_probs"].exp()
    torch.testing.assert_close(p.sum(-1), torch.ones(4))
    assert torch.equal(p[1::2, 2], torch.zeros(2))  # the yes/no rows' missing third option


@torch.no_grad()
def test_frozen_readout_is_the_lms_own_option_number_distribution(base):
    """The untied Llama head is read from the checkpoint."""
    model = _model(base)
    b = _batch(model)
    lp, eligible = model.frozen_slot_log_probs(b["input_ids"], b["attention_mask"], b["n_slots"])
    assert bool(eligible.all())
    lm = transformers.LlamaForCausalLM.from_pretrained(base).eval()
    logits = lm(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
    last = logits[torch.arange(4), b["attention_mask"].sum(1) - 1]
    ids = list(model.slot_token_ids().values())
    for r in range(4):
        n = int(b["n_slots"][r])
        torch.testing.assert_close(lp[r, :n], last[r, ids[:n]].log_softmax(-1), atol=1e-5, rtol=1e-5)


def test_option_numbers_past_nine_are_not_read():
    """A vocabulary with single tokens for 10-24 (MiniCPM5's) still reads slots 0-8 only."""
    stub = SimpleNamespace(config=SimpleNamespace(num_slots=24),
                           tokenizer=SimpleNamespace(encode=lambda s, add_special_tokens: [int(s)]))
    assert StrandsDeciderModel.slot_token_ids(stub) == {k: k + 1 for k in range(9)}


def test_three_steps_from_the_raw_base_reduce_the_loss(base, tmp_path):
    write_jsonl(str(tmp_path / "train.jsonl"), _rows(8))
    write_jsonl(str(tmp_path / "val.jsonl"), _rows(2))
    cfg = TrainConfig(base_model=base, train_files=[str(tmp_path / "train.jsonl")],
                      val_files=[str(tmp_path / "val.jsonl")], head_type="pointer", pointer_dim=16,
                      lora_targets=ATTENTION_MLP, max_length=512, kl_frozen_weight=0.3,
                      kl_frozen_skip_kinds=["noul"], micro_batch_size=8, grad_accum=1, epochs=3, max_steps=3,
                      lr=1e-2, head_lr=1e-2, log_every=1, eval_every=0, shuffle_options=False,
                      output_dir=str(tmp_path / "out"))
    out = train(cfg)
    with open(os.path.join(out, "history.json"), encoding="utf-8") as fh:
        losses = [h["loss"] for h in json.load(fh) if "loss" in h]
    assert len(losses) == 3 and losses[-1] < losses[0]
    loaded = StrandsDeciderModel.load(out)
    assert _lora_modules(loaded) == _lora_modules(_model(base))


def test_the_training_config_pins_the_base_revision(monkeypatch):
    """`base_revision` reaches the model's config, which every base download reads."""
    seen = {}

    def stop(cls, config, **kw):
        seen["revision"] = config.base_revision
        raise RuntimeError("stopped before any download")

    monkeypatch.setattr(StrandsDeciderModel, "from_pretrained_base", classmethod(stop))
    with pytest.raises(RuntimeError, match="stopped"):
        train(TrainConfig(base_model="org/base", base_revision="abc123", head_type="pointer"))
    assert seen == {"revision": "abc123"}


@torch.no_grad()
def test_checkpoint_round_trip(base, tmp_path):
    model = _model(base)
    for n, p in model.torso.named_parameters():
        if "lora_B" in n:
            p.normal_(0, 0.05)
    model.save_pretrained(str(tmp_path / "ck"))
    loaded = StrandsDeciderModel.load(str(tmp_path / "ck"))
    loaded.torso.to(torch.float32)
    b = _batch(model)
    kw = {k: b[k] for k in ("input_ids", "attention_mask", "n_slots", "opt_idx")}
    torch.testing.assert_close(loaded(**kw)["log_probs"], model(**kw)["log_probs"])


def test_shared_prefix_answers_as_full_prompts_do(base):
    model = _model(base)
    with torch.no_grad():
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    state = {"ticket": "My card was charged twice for order 123. Please refund one charge.",
             "customer": "since 2019"}
    questions = {
        "route": ChoiceQuestion(instructions="Route it.",
                                criteria={"billing": "money", "shipping": "", "other": "else"}),
        "refund": NoulQuestion(instructions="The customer wants a refund."),
        "urgency": ScoreQuestion(instructions="How urgent?", criteria=["low", "mid", "high"]),
    }
    shared_engine = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=True))
    shared = shared_engine.ask(state, questions)
    assert shared_engine.cfg.use_prefix_cache  # the cache forked; no fallback
    full = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=False)).ask(state, questions)
    assert shared.answers["route"].probabilities == pytest.approx(full.answers["route"].probabilities, abs=1e-3)
    assert shared.answers["refund"].noul == pytest.approx(full.answers["refund"].noul, abs=1e-3)
    assert shared.answers["urgency"].probabilities == pytest.approx(full.answers["urgency"].probabilities,
                                                                    abs=1e-3)

