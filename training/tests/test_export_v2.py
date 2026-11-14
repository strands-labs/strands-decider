"""training.export_v2 end to end, on CPU: a tiny bf16 Gemma 4 with a vision tower and Gemma 4's
own tokenizer (STRANDS_DECIDER_GEMMA4_TOKENIZER, else skipped), trained for two seeds on
converted v2 rows, souped and exported; the core's loaders read the result."""

from __future__ import annotations

import json
import os
import random

import pytest
import torch
import typer

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4Config"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from core_support import gemma4_tokenizer_path  # noqa: E402
from safetensors import safe_open  # noqa: E402
from test_gemma4_train import GEMMA_TARGETS, LAYERS  # noqa: E402

import training.train as T  # noqa: E402
from strands_decider import core  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel  # noqa: E402
from training.data import strands_v2  # noqa: E402
from training.data.format import load_examples  # noqa: E402
from training.export_v2 import DECODER, adapter_deltas, export_v2  # noqa: E402
from training.soup import soup  # noqa: E402

HUB_ID = "google/gemma-4-E2B-it"
REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"


def _base(path, tokenizer_json, moe=False):
    tok = transformers.PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_json, bos_token="<bos>", eos_token="<eos>", pad_token="<pad>"
    )
    tok.save_pretrained(path)
    torch.manual_seed(0)
    text = dict(
        vocab_size=len(tok),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=len(LAYERS),
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        global_head_dim=16,
        layer_types=LAYERS,
        sliding_window=8,
        final_logit_softcapping=30.0,
        tie_word_embeddings=True,
        pad_token_id=0,
        max_position_embeddings=4096,
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=len(tok),
        num_kv_shared_layers=1,
    )
    if moe:  # the 26B-A4B's layout: experts in every layer, no per-layer inputs
        text.update(
            hidden_size_per_layer_input=0,
            num_kv_shared_layers=0,
            enable_moe_block=True,
            num_experts=4,
            top_k_experts=2,
            moe_intermediate_size=16,
        )
    vision = transformers.Gemma4VisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
    )
    cfg = transformers.Gemma4Config(text_config=text, vision_config=vision, audio_config=None)
    transformers.Gemma4ForConditionalGeneration(cfg).to(torch.bfloat16).save_pretrained(path)
    (path / "processor_config.json").write_text(json.dumps({"chat_template": "x", "p": 1}))
    return tok


def _v2_rows(n):
    rng = random.Random(0)
    rows = []
    for i in range(n):
        state = f"the log shows {rng.randint(0, 99)} errors in {rng.choice(['api', 'db', 'web'])}"
        q = [
            {
                "type": "noul",
                "criteria": {"false": None, "true": "it failed"},
                "instructions": "Did it fail?",
            },
            {
                "type": "choice",
                "criteria": {"api": None, "db": "the store", "web": None},
                "instructions": "Which part?",
            },
            {"type": "score", "criteria": ["none", "some", "many"], "instructions": "How bad?"},
        ][i % 3]
        target = [rng.random(), [0.2, 0.5, 0.3], [0.1, 0.2, 0.7]][i % 3]
        label = [target > 0.5 if i % 3 == 0 else None, "db", 2][i % 3]
        rows.append(
            {
                "id": str(i),
                "suite": "s",
                "family": "f",
                "state": state,
                "question": q,
                "target": target,
                "label": label,
                "source": {},
            }
        )
    return rows


def _train(tmp_path, base, data, seed):
    out = tmp_path / f"run{seed}"
    T.train(
        T.TrainConfig(
            train_files=[str(data)],
            val_files=[str(data)],
            base_model=str(base),
            base_revision=REVISION,
            head_type="token_readout",
            attention_mode="prefix_bidir",
            lora_targets=GEMMA_TARGETS,
            lora_r=4,
            lora_alpha=8,
            lr=1e-2,
            micro_batch_size=4,
            grad_accum=1,
            max_steps=3,
            max_length=512,
            eval_every=0,
            seed=seed,
            output_dir=str(out),
        )
    )
    return out


def _tensors(d):
    out = {}
    for name in sorted(os.listdir(d)):
        if name.startswith("model") and name.endswith(".safetensors"):
            with safe_open(str(d / name), "pt") as f:
                out.update({k: f.get_tensor(k) for k in f.keys()})
    return out


def test_train_two_seeds_soup_export_and_load_with_the_core(tmp_path):
    base = tmp_path / "base"
    _base(base, gemma4_tokenizer_path())
    src, data = tmp_path / "v2.jsonl", tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _v2_rows(24)), encoding="utf-8")
    assert strands_v2.main([str(src), str(data)])["kept"] == 24
    runs = [_train(tmp_path, base, data, seed) for seed in (1, 2)]
    souped = soup([str(r) for r in runs], str(tmp_path / "soup"))
    out = tmp_path / "out"
    export_v2(souped, str(out), base_model=HUB_ID)

    # (a) the core's loaders take it
    cfg = core.DecisionConfig.load(out)
    core.Readout.load(out, cfg)
    core.load_tokenizer(out, cfg)
    assert cfg.provenance.to_dict() == {"base_model": HUB_ID, "base_revision": REVISION}
    assert cfg.max_length_validated == 512
    assert json.loads((out / "config.json").read_text())["architectures"] == [
        "Gemma4ForConditionalGeneration"
    ]
    assert "chat_template" not in json.loads((out / "processor_config.json").read_text())
    assert not (out / "chat_template.jinja").exists()

    # (b) the base's names and shards; every tensor outside the decoder is the base's
    mine, theirs = _tensors(out), _tensors(base)
    assert sorted(mine) == sorted(theirs)
    assert any(k.startswith("model.vision_tower.") for k in mine)
    for k, t in theirs.items():
        assert mine[k].dtype == torch.bfloat16
        if not k.startswith(DECODER):
            assert torch.equal(mine[k], t), k

    # (c) the merge in float32 computes the unmerged model under prefix_bidir, and the
    # export holds that merge rounded to bf16
    model = StrandsDeciderModel.load(souped).float().eval()
    deltas = {k: d() for k, d in adapter_deltas(souped, model.config).items()}
    assert deltas and max(float(d.abs().max()) for d in deltas.values()) > 0
    for k, d in deltas.items():
        assert torch.equal(mine[k], (theirs[k].float() + d).bfloat16()), k
    batch = T.SystemOneCollator(
        model.tokenizer, T.CollatorConfig(render="strands-chat"), train=False
    )(load_examples([str(data)])[:6])
    keys = ("input_ids", "attention_mask", "n_slots", "prefix_end")
    with torch.no_grad():
        want = model(**{k: batch[k] for k in keys})["logits"]
        model.torso = model.torso.unload()
        params = dict(model.torso.named_parameters())
        for k, d in deltas.items():
            params[k[len(DECODER) :]].add_(d)
        got = model(**{k: batch[k] for k in keys})["logits"]
    finite = torch.isfinite(want)
    torch.testing.assert_close(got[finite], want[finite], rtol=0, atol=1e-4)

    # (d) no local path in any file, and no code
    for name in os.listdir(out):
        data_ = (out / name).read_bytes()
        for p in (tmp_path, os.path.expanduser("~")):
            assert str(p).encode() not in data_, name
        assert not name.endswith(".py")
    record = json.loads((out / "training.json").read_text())
    assert record["soup_of"] == 2 and record["lora"]["rank"] == 4

    # (e) a wrong base is refused, and no output is left behind
    (base / "config.json").write_text(
        (base / "config.json").read_text().replace('"rms_norm_eps"', '"rms_norm_eps" ', 1)
    )
    bad = tmp_path / "bad"
    with pytest.raises(ValueError, match="not the base"):
        export_v2(souped, str(bad), base_model=HUB_ID)
    assert not bad.exists()


def test_no_base_record_for_a_base_that_is_not_on_disk(tmp_path):
    from training.export_v2 import write_base_record

    assert not write_base_record(str(tmp_path), "not/cached-model", None)
    assert not (tmp_path / "base.json").exists()


# ---- a soup of seeds: each merged, every tensor and the readout averaged -------------------


def _seed_run(tmp_path, base, name, seed, **over):
    """A token-readout checkpoint with base.json; seed > 0 gives its adapters and readout
    random values (no training: the export reads only the saved tensors)."""
    from strands_decider.modeling import StrandsDeciderConfig
    from training.export_v2 import write_base_record

    kw = dict(
        base_model=str(base),
        base_revision=REVISION,
        torch_dtype="float32",
        head_type="token_readout",
        attention_mode="prefix_bidir",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_targets=GEMMA_TARGETS,
    )
    cfg = StrandsDeciderConfig(**{**kw, **over})
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    if seed:
        torch.manual_seed(seed)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "lora_B" in n or n.endswith("_b") or n.startswith("head."):
                    p.normal_(0, 0.1)
    path = tmp_path / name
    model.save_pretrained(str(path))
    assert write_base_record(str(path), str(base), REVISION)
    return path


def _readout(d):
    with safe_open(str(d / "readout.safetensors"), "pt") as f:
        return f.get_tensor("weight")


def _loads(out):
    cfg = core.DecisionConfig.load(out)
    core.Readout.load(out, cfg)
    return cfg


def test_a_soup_averages_every_merged_tensor_and_the_readout(tmp_path):
    from strands_decider.modeling import StrandsDeciderConfig, config_path, load_head_state

    base = tmp_path / "base"
    _base(base, gemma4_tokenizer_path())
    runs = [_seed_run(tmp_path, base, f"run{s}", s) for s in (1, 2)]
    out = tmp_path / "soup"
    export_v2([str(r) for r in runs], str(out), base_model=HUB_ID)
    assert _loads(out).temperature == 1.0

    mine, theirs = _tensors(out), _tensors(base)
    assert sorted(mine) == sorted(theirs)
    cfg = StrandsDeciderConfig.from_json(config_path(str(runs[0])))
    deltas = [{k: d() for k, d in adapter_deltas(str(r), cfg).items()} for r in runs]
    assert deltas[0].keys() == deltas[1].keys() and deltas[0]
    for k, t in theirs.items():
        if k not in deltas[0]:
            assert torch.equal(mine[k], t.bfloat16() if k.startswith(DECODER) else t), k
            continue
        merged = [t.float() + d[k] for d in deltas]  # each seed's merged tensor, float32
        assert torch.equal(mine[k], ((merged[0] + merged[1]) / 2).bfloat16()), k
    heads = [load_head_state(str(r))["proj.weight"].float() for r in runs]
    assert torch.equal(_readout(out), ((heads[0] + heads[1]) / 2).bfloat16())
    assert not torch.equal(_readout(out), heads[0].bfloat16())  # the seeds differ
    assert json.loads((out / "training.json").read_text())["soup_of"] == 2


def test_a_soup_of_one_run_twice_is_that_run(tmp_path):
    from typer.testing import CliRunner

    from training.export_v2 import _cli

    base = tmp_path / "base"
    _base(base, gemma4_tokenizer_path())
    run = _seed_run(tmp_path, base, "run", 1)
    one, two = tmp_path / "one", tmp_path / "two"
    export_v2(str(run), str(one), base_model=HUB_ID)
    app = typer.Typer()
    app.command()(_cli)
    res = CliRunner().invoke(app, [str(run), str(run), str(two), "--base-model", HUB_ID])
    assert res.exit_code == 0, res.output
    _loads(one), _loads(two)
    for name in os.listdir(one):
        if name.endswith(".safetensors"):
            assert (one / name).read_bytes() == (two / name).read_bytes(), name


def test_a_soup_refuses_two_recipes(tmp_path):
    base = tmp_path / "base"
    _base(base, gemma4_tokenizer_path())
    a = _seed_run(tmp_path, base, "a", 1)
    b = _seed_run(tmp_path, base, "b", 2, lora_r=2, lora_alpha=4)
    with pytest.raises(ValueError, match="not one recipe"):
        export_v2([str(a), str(b)], str(tmp_path / "soup"), base_model=HUB_ID)
    assert not (tmp_path / "soup").exists()
