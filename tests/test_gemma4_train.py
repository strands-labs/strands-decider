"""Training on a Gemma 4 torso, end to end on CPU, from tiny random-weight checkpoints on disk.

The real loader (`from_pretrained_base`: the multimodal class, its text decoder kept,
`host_embeddings`, `force_bos`), LoRA, gradient checkpointing, the frozen-KL anchor and
two optimizer steps of `train()`, then `load()` of the saved checkpoint. One torso per
Gemma 4 layout the configs use: E2B/E4B (per-layer embedding table, KV sharing), the
12B (`gemma4_unified`) and the 26B-A4B (a mixture-of-experts block in every layer).
"""
from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4UnifiedConfig"):
    pytest.skip("this transformers has no Gemma 4 (unified)", allow_module_level=True)

from test_ddp import _examples  # noqa: E402

import strands_decider.train as T  # noqa: E402
from strands_decider.data.format import write_jsonl  # noqa: E402
from strands_decider.modeling import HostEmbedding, StrandsDeciderModel  # noqa: E402

LAYERS = ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention",
          "full_attention"]
GEMMA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _tokenizer(path, rows):
    """A byte-level BPE that, like gemma-4-*-it's, has a BOS token but adds none itself."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    from strands_decider.prompting import build_prompt

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=True)
    trainer = trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<eos>", "<bos>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator([build_prompt(ex.state, ex.to_question())[0] for ex in rows], trainer)
    fast = transformers.PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>",
                                                eos_token="<eos>", bos_token="<bos>")
    fast.save_pretrained(path)
    return fast


def _checkpoint(path, layout, vocab):
    torch.manual_seed(0)
    text = dict(vocab_size=vocab, hidden_size=32, intermediate_size=64,
                num_hidden_layers=len(LAYERS), num_attention_heads=2, num_key_value_heads=1,
                head_dim=16, global_head_dim=16, layer_types=LAYERS, sliding_window=8,
                final_logit_softcapping=30.0, tie_word_embeddings=True, pad_token_id=0,
                max_position_embeddings=4096)
    if layout == "e2b":
        text.update(hidden_size_per_layer_input=8, vocab_size_per_layer_input=vocab,
                    num_kv_shared_layers=1)
    else:
        text.update(hidden_size_per_layer_input=0, num_kv_shared_layers=0)
    if layout == "moe":
        text.update(enable_moe_block=True, num_experts=4, top_k_experts=2,
                    moe_intermediate_size=16)
    if layout == "unified":
        cfg = transformers.Gemma4UnifiedConfig(text_config=text, vision_config=None,
                                               audio_config=None)
        model = transformers.Gemma4UnifiedForConditionalGeneration(cfg)
    else:
        cfg = transformers.Gemma4Config(text_config=text, vision_config=None, audio_config=None)
        model = transformers.Gemma4ForConditionalGeneration(cfg)
    model.save_pretrained(path)


@pytest.mark.parametrize("layout", ["e2b", "unified", "moe"])
def test_gemma4_trains_and_reloads(tmp_path, layout):
    rows = [ex for ex in _examples(80, seed=1) if ex.n_options <= 9]
    write_jsonl(str(tmp_path / "train.jsonl"), rows)
    write_jsonl(str(tmp_path / "val.jsonl"), _examples(10, seed=2))
    base = tmp_path / "base"
    tok = _tokenizer(base, rows)
    assert tok("x")["input_ids"][0] != tok.bos_token_id  # force_bos has work to do
    _checkpoint(base, layout, len(tok))
    out = tmp_path / "out"
    cfg = T.TrainConfig(
        train_files=[str(tmp_path / "train.jsonl")], val_files=[str(tmp_path / "val.jsonl")],
        base_model=str(base), head_type="pointer", pointer_dim=16, lora_targets=GEMMA_TARGETS,
        force_bos=True, host_embeddings=True, kl_frozen_weight=0.3, precompute_frozen_kl=True,
        head_dropout=0.0, max_length=2048, micro_batch_size=4, grad_accum=2, max_steps=2,
        group_by_length=True, length_group_mega=2, log_every=1, eval_every=0,
        gradient_checkpointing=True, output_dir=str(out))
    T.train(cfg)
    history = json.loads((out / "history.json").read_text())
    assert history and all(torch.isfinite(torch.tensor(h["loss"])) for h in history if "loss" in h)

    model = StrandsDeciderModel.load(str(out))
    assert model.tokenizer("x")["input_ids"][0] == model.tokenizer.bos_token_id
    torso = getattr(model.torso, "base_model", model.torso)
    torso = getattr(torso, "model", torso)
    table = getattr(torso, "embed_tokens_per_layer", None)
    assert (table is not None) == (layout == "e2b")
    assert table is None or isinstance(table, HostEmbedding)
    b = T.SystemOneCollator(model.tokenizer, T.CollatorConfig(head_type="pointer"), train=False)(
        rows[:3])
    with torch.no_grad():
        lp = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                   n_slots=b["n_slots"], opt_idx=b["opt_idx"])["log_probs"]
    for r, n in enumerate(b["n_slots"].tolist()):
        assert torch.allclose(lp[r, :n].float().exp().sum(), torch.tensor(1.0), atol=1e-2)


def test_a_research_build_config_with_no_full_weight_targets_loads(tmp_path):
    """The Gemma 4 release folders' config holds `full_weight_targets: []` (LoRA)."""
    from strands_decider.modeling import StrandsDeciderConfig

    path = tmp_path / "strands_decider_config.json"
    path.write_text(json.dumps({"base_model": "google/gemma-4-E4B-it", "force_bos": True,
                                "full_weight_targets": []}))
    assert StrandsDeciderConfig.from_json(str(path)).force_bos
    path.write_text(json.dumps({"full_weight_targets": ["q_proj"]}))
    with pytest.raises(ValueError, match="full_weight_targets"):
        StrandsDeciderConfig.from_json(str(path))


def test_a_token_budget_per_forward_keeps_the_answers(tmp_path):
    """max_batch_tokens splits a request's forwards (long-context serving); every answer is the
    unsplit one up to float rounding, and the chunks respect both caps."""
    from strands_decider.infer import EngineConfig, SystemOneEngine
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, SystemOneRequest

    rows = [ex for ex in _examples(40, seed=1) if ex.n_options <= 9]
    write_jsonl(str(tmp_path / "train.jsonl"), rows)
    write_jsonl(str(tmp_path / "val.jsonl"), _examples(10, seed=2))
    base = tmp_path / "base"
    tok = _tokenizer(base, rows)
    _checkpoint(base, "e2b", len(tok))
    T.train(T.TrainConfig(
        train_files=[str(tmp_path / "train.jsonl")], val_files=[str(tmp_path / "val.jsonl")],
        base_model=str(base), head_type="pointer", pointer_dim=16, lora_targets=GEMMA_TARGETS,
        force_bos=True, head_dropout=0.0, max_length=2048, micro_batch_size=4, grad_accum=1,
        max_steps=1, eval_every=0, output_dir=str(tmp_path / "out")))
    model = StrandsDeciderModel.load(str(tmp_path / "out")).eval()
    state = " ".join(ex.state if isinstance(ex.state, str) else "x" for ex in rows[:3])
    qs = {f"q{i}": (NoulQuestion(instructions=rows[i].instructions + " ok?") if i % 2 else
                    ChoiceQuestion(instructions=rows[i].instructions, criteria={"a": None, "b": "bee", "c": None}))
          for i in range(7)}
    req = SystemOneRequest(state=state, questions=qs)
    for prefix in (True, False):
        plain = SystemOneEngine(model, EngineConfig(device="cpu", max_batch=4, use_prefix_cache=prefix))
        tight = SystemOneEngine(model, EngineConfig(device="cpu", max_batch=4, use_prefix_cache=prefix,
                                                    max_batch_tokens=1))
        a, b = plain.evaluate(req).answers, tight.evaluate(req).answers
        for k in qs:
            x, y = a[k].model_dump(), b[k].model_dump()
            for f in x:
                if isinstance(x[f], dict):
                    assert all(abs(x[f][o] - y[f][o]) <= 2e-3 for o in x[f]), (k, f)
                elif isinstance(x[f], float):
                    assert abs(x[f] - y[f]) <= 2e-3, (k, f)
                else:
                    assert x[f] == y[f], (k, f)
    from strands_decider.infer import render_question, render_state

    rendered = [render_question(q) for q in qs.values()]
    st = render_state(state)
    sizes = lambda e: [len(range(*c.indices(7))) for c in e._chunks(st, rendered)]  # noqa: E731
    assert sizes(tight) == [1] * 7  # the budget binds
    assert sizes(plain) == [4, 3]  # off: max_batch only, as before
