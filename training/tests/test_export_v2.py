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

from core_support import gemma4_file  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import load_file, save_file  # noqa: E402
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
# The E2B and E4B configs' targets: the fixture has per-layer inputs, as those sizes do.
E2B_TARGETS = [*GEMMA_TARGETS, "per_layer_input_gate", "per_layer_projection"]
# The image settings that the core counts soft tokens for, and a template the export drops.
PROCESSOR = {
    "chat_template": "x",
    "image_processor": {"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280},
    "image_seq_length": 280,
}
# A decision checkpoint's files, without golden_vectors.jsonl, which the export does not write.
LAYOUT = {
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "readout.safetensors",
    "decision_config.json",
    "LICENSE",
    "NOTICE",
    "README.md",
}


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
    if not moe:
        # Like the E2B and E4B files on the Hub: the base carries k_proj, v_proj and k_norm for
        # its KV-shared layers too, which transformers does not load.
        file = str(path / "model.safetensors")
        tensors, last = load_file(file), len(LAYERS) - 1
        for part in ("k_proj", "v_proj", "k_norm"):
            name = f"{DECODER}layers.{{}}.self_attn.{part}.weight"
            assert name.format(last) not in tensors
            tensors[name.format(last)] = torch.randn_like(tensors[name.format(last - 1)])
        save_file(tensors, file, metadata={"format": "pt"})
    (path / "processor_config.json").write_text(json.dumps(PROCESSOR))
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
            lora_targets=E2B_TARGETS,
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
    _base(base, gemma4_file("tokenizer.json"))
    src, data = tmp_path / "v2.jsonl", tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _v2_rows(24)), encoding="utf-8")
    assert strands_v2.main([str(src), str(data)])["kept"] == 24
    runs = [_train(tmp_path, base, data, seed) for seed in (1, 2)]
    souped = soup([str(r) for r in runs], str(tmp_path / "soup"))
    out = tmp_path / "out"
    export_v2(souped, str(out), base_model=HUB_ID)

    # (a) the core's loaders take it
    cfg = core.DecisionConfig.load(out)
    assert core.load_readout(out).shape == (core.MAX_CODES, 32)
    core.load_tokenizer(out)
    core.check_image_processor(out)
    assert cfg.provenance.model_dump() == {
        "base_model": HUB_ID,
        "base_revision": REVISION,
        "release": None,
    }
    assert cfg.max_length_validated == 512
    assert json.loads((out / "config.json").read_text())["architectures"] == [
        "Gemma4ForConditionalGeneration"
    ]
    assert cfg.inputs == ("text", "image")  # the base's vision encoder, unchanged
    kept = {k: v for k, v in PROCESSOR.items() if k != "chat_template"}
    assert json.loads((out / "processor_config.json").read_text()) == kept  # no template
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):  # the base's bytes
        assert (out / name).read_bytes() == (base / name).read_bytes(), name
    assert set(os.listdir(out)) == LAYOUT  # the layout's files, and no others

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
    for part in ("per_layer_input_gate", "per_layer_projection"):  # merged like the others
        assert any(f".{part}." in k and float(d.abs().max()) > 0 for k, d in deltas.items()), part
    shared = f"{DECODER}layers.{len(LAYERS) - 1}.self_attn."  # the fixture's KV-shared layer
    assert any(k.startswith(shared) for k in deltas)  # its q_proj and o_proj train
    for part in ("k_proj", "v_proj", "k_norm"):  # no adapter there: the base's, as they are
        assert f"{shared}{part}.weight" not in deltas
        assert torch.equal(mine[f"{shared}{part}.weight"], theirs[f"{shared}{part}.weight"])
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
    card = (out / "README.md").read_text()
    assert "rank 4, alpha 8" in card and "of 2 runs" in card
    for seed in (1, 2):  # every run of the soup: seed, data file name, steps, learning rate
        assert f"| {seed} | {seed} | train.jsonl | 3 | 0.01 |" in card
    notice = (out / "NOTICE").read_text()
    assert HUB_ID in notice and REVISION in notice and "average of 2" in notice
    assert "The vision encoder, the vision projection and every other tensor" in notice
    assert (out / "LICENSE").read_text().lstrip().startswith("Apache License")

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
        lora_targets=E2B_TARGETS,
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
    core.load_readout(out)
    return cfg


def test_a_soup_averages_every_merged_tensor_and_the_readout(tmp_path):
    from strands_decider.modeling import StrandsDeciderConfig, config_path, load_head_state

    base = tmp_path / "base"
    _base(base, gemma4_file("tokenizer.json"))
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
    assert "of 2 runs" in (out / "README.md").read_text()
    assert set(os.listdir(out)) == LAYOUT


def test_a_soup_of_one_run_twice_is_that_run(tmp_path):
    from typer.testing import CliRunner

    from training.export_v2 import _cli

    base = tmp_path / "base"
    _base(base, gemma4_file("tokenizer.json"))
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
    _base(base, gemma4_file("tokenizer.json"))
    from training.soup import check_one_recipe

    a = _seed_run(tmp_path, base, "a", 1)
    b = _seed_run(tmp_path, base, "b", 2, lora_r=2, lora_alpha=4)
    c = _seed_run(tmp_path, base, "c", 3, attn_split=1024, head_dropout=0.05)
    check_one_recipe([str(a), str(c)])  # run settings: they do not change the average
    with pytest.raises(ValueError, match="not one recipe"):
        export_v2([str(a), str(b)], str(tmp_path / "soup"), base_model=HUB_ID)
    assert not (tmp_path / "soup").exists()


# ---- temperature, the model card, and what the export refuses ------------------------------


def test_the_temperature_the_model_card_and_the_refusals(tmp_path, capsys):
    from strands_decider.modeling import config_path
    from training.export_v2 import drop_kv_shared_updates

    base = tmp_path / "base"
    _base(base, gemma4_file("tokenizer.json"))
    run = _seed_run(tmp_path, base, "run", 1)

    # A calibrated checkpoint exports its global temperature, and says what it leaves out.
    path = config_path(str(run))
    cfg = json.loads(open(path, encoding="utf-8").read())
    cfg.update(temperature=0.8, temperature_by_kind={"choice": 0.9})
    open(path, "w", encoding="utf-8").write(json.dumps(cfg))
    assert _loads(export_v2(str(run), str(tmp_path / "a"), base_model=HUB_ID)).temperature == 0.8
    assert "are not carried over" in capsys.readouterr().out
    card = tmp_path / "card.md"
    card.write_text("# My model\n", encoding="utf-8")
    out = tmp_path / "b"
    export_v2(str(run), str(out), base_model=HUB_ID, temperature=1.3, model_card_path=str(card))
    assert _loads(out).temperature == 1.3
    assert (out / "README.md").read_bytes() == card.read_bytes()

    # A cap that is not the base config's: config.json is copied, so it must agree.
    other = _seed_run(tmp_path, base, "other", 2, readout_softcap=25.0)
    with pytest.raises(ValueError, match="caps its logits"):
        export_v2(str(other), str(tmp_path / "c"), base_model=HUB_ID)

    # A KV-shared layer's own keys and values: a zero update is dropped, another is refused.
    config = json.loads((base / "config.json").read_text())
    name = f"{DECODER}layers.{len(LAYERS) - 1}.self_attn.k_proj.weight"
    deltas = [{name: lambda: torch.zeros(2, 2), f"{DECODER}layers.0.self_attn.k_proj.weight": None}]
    drop_kv_shared_updates(config, deltas)
    assert list(deltas[0]) == [f"{DECODER}layers.0.self_attn.k_proj.weight"]
    with pytest.raises(ValueError, match="shares an earlier layer's keys and values"):
        drop_kv_shared_updates(config, [{name: lambda: torch.ones(2, 2)}])

    # An image processor that sizes images another way than the core counts them.
    wrong = {**PROCESSOR, "image_processor": {**PROCESSOR["image_processor"], "patch_size": 14}}
    (base / "processor_config.json").write_text(json.dumps(wrong))
    with pytest.raises(ValueError, match="image processor can't take images"):
        export_v2(str(run), str(tmp_path / "e"), base_model=HUB_ID)

    # A base without one of its own files.
    (base / "processor_config.json").unlink()
    with pytest.raises(ValueError, match="has no processor_config"):
        export_v2(str(run), str(tmp_path / "d"), base_model=HUB_ID)
    assert not (tmp_path / "c").exists() and not (tmp_path / "d").exists()


# ---- a decoder that is already merged ------------------------------------------------------


def test_a_merged_decoder_exports_as_the_checkpoint_of_its_adapters(tmp_path, monkeypatch):
    """`--merged-decoder` takes the decoder's tensors under their own names, without the
    KV-shared layers' k_proj, v_proj and k_norm, and the readout. From the weights that the
    adapter export wrote, it writes the same checkpoint."""
    from typer.testing import CliRunner

    import training.export_v2 as X

    base = tmp_path / "base"
    _base(base, gemma4_file("tokenizer.json"))
    run = _seed_run(tmp_path, base, "run", 1)
    one = tmp_path / "one"
    export_v2(str(run), str(one), base_model=HUB_ID, max_length_validated=512)

    shared = X.kv_shared_names(json.loads((base / "config.json").read_text()))
    decoder = {
        k[len(DECODER) :]: v
        for k, v in _tensors(one).items()
        if k.startswith(DECODER) and k not in shared
    }
    assert shared and len(decoder) < sum(k.startswith(DECODER) for k in _tensors(one))
    card = tmp_path / "card.md"
    card.write_text("# A model\n", encoding="utf-8")
    monkeypatch.setattr(X, "base_dir", lambda model, revision: str(base))  # the Hub cache
    kw = dict(
        base_model=HUB_ID,
        base_revision=REVISION,
        max_length_validated=512,
        model_card_path=str(card),
    )

    def merged(name, tensors, readout=None):
        d = tmp_path / name
        d.mkdir()
        save_file(tensors, str(d / "model.safetensors"))
        save_file(
            {"weight": _readout(one) if readout is None else readout},
            str(d / "readout.safetensors"),
        )
        return str(d)

    two = tmp_path / "two"
    X.export_merged(merged("m", decoder), str(two), **kw)
    assert set(os.listdir(two)) == LAYOUT
    a, b = _tensors(one), _tensors(two)
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    assert torch.equal(_readout(one), _readout(two))
    config = lambda d: json.loads((d / "decision_config.json").read_text())  # noqa: E731
    assert config(one) == config(two)
    assert "fine-tuned" in (two / "NOTICE").read_text()
    assert (two / "README.md").read_bytes() == card.read_bytes()

    # The command line: the same export, and a refusal without the flags it needs.
    app = typer.Typer()
    app.command()(X._cli)
    flags = ["--base-model", HUB_ID, "--base-revision", REVISION, "--model-card", str(card)]
    args = [str(tmp_path / "cli"), "--merged-decoder", str(tmp_path / "m"), *flags]
    assert CliRunner().invoke(app, args).exit_code != 0  # no --max-length-validated
    res = CliRunner().invoke(app, [*args, "--max-length-validated", "512", "--temperature", "1.5"])
    assert res.exit_code == 0, res.output
    assert _loads(tmp_path / "cli").temperature == 1.5

    # Every name and shape is checked against the base, and no output is left behind.
    name = next(k for k in decoder if k.endswith("q_proj.weight"))
    cases = {
        "missing": ({k: v for k, v in decoder.items() if k != name}, None, "1 tensors missing"),
        "extra": (
            {**decoder, "layers.99.x": decoder[name].clone()},
            None,
            "1 that the base does not have",
        ),
        "shape": ({**decoder, name: decoder[name][:1]}, None, "the merged decoder has"),
        "readout": (decoder, _readout(one)[:3], "the readout's `weight` must be"),
    }
    for case, (tensors, readout, message) in cases.items():
        with pytest.raises(ValueError, match=message):
            X.export_merged(merged(case, tensors, readout), str(tmp_path / f"out-{case}"), **kw)
        assert not (tmp_path / f"out-{case}").exists()


# ---- an exported model as the start of a fine-tune ------------------------------------------


def test_an_exported_model_evaluates_calibrates_fine_tunes_and_exports_again(tmp_path):
    """The directory that the export writes loads as a model with its own readout and
    temperature. `calibrate` writes its one temperature and no other file. `init_from` trains
    a new adapter on it from its readout, and the result exports with the same base model."""
    import training.evaluate as E
    from strands_decider.modeling import StrandsDeciderConfig, config_path

    base = tmp_path / "base"
    _base(base, gemma4_file("tokenizer.json"))
    src, data = tmp_path / "v2.jsonl", tmp_path / "train.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _v2_rows(24)), encoding="utf-8")
    strands_v2.main([str(src), str(data)])
    released = tmp_path / "released"
    export_v2(str(_train(tmp_path, base, data, 1)), str(released), base_model=HUB_ID)

    model = StrandsDeciderModel.load(str(released))
    assert torch.equal(model.head.proj.weight, _readout(released).float())  # not a new readout
    assert (model.config.attention_mode, model.config.use_lora) == ("prefix_bidir", False)

    rows = load_examples([str(data)])
    fitted = E.calibrate_checkpoint(str(released), rows, device="cpu")["temperature"]
    assert set(os.listdir(released)) == LAYOUT
    assert core.DecisionConfig.load(released).temperature == pytest.approx(fitted)
    assert E.evaluate_checkpoint(str(released), rows, device="cpu")["temperature"] == fitted

    def tune(**over):
        kw = dict(
            train_files=[str(data)],
            val_files=[str(data)],
            init_from=os.path.relpath(released),
            lora_targets=E2B_TARGETS,
            lora_r=4,
            lora_alpha=8,
            lr=1e-2,
            head_lr=0.0,  # the readout stays where the run starts it
            micro_batch_size=4,
            grad_accum=1,
            max_steps=3,
            max_length=512,
            eval_every=0,
            output_dir=str(tmp_path / "tuned"),
        )
        T.train(T.TrainConfig(**{**kw, **over}))
        return tmp_path / "tuned"

    # A config that names another base is refused before the model loads.
    with pytest.raises(ValueError, match="Remove base_model from the config"):
        tune(base_model="google/gemma-4-E4B-it")

    tuned = tune(base_model=HUB_ID)  # the base that the export recorded: accepted
    saved = StrandsDeciderConfig.from_json(config_path(str(tuned)))
    assert saved.base_model == str(released.resolve())  # loads from any directory
    with pytest.raises(ValueError, match="leave --base-model out"):
        export_v2(str(tuned), str(tmp_path / "x"), base_model="some-org/another-model")
    out = tmp_path / "out"
    export_v2(str(tuned), str(out))  # the base's Hub id and revision are the exported model's
    assert set(os.listdir(out)) == LAYOUT
    assert _loads(out).provenance == core.DecisionConfig.load(released).provenance
    mine, theirs = _tensors(out), _tensors(released)
    assert any(not torch.equal(mine[k], theirs[k]) for k in mine if k.startswith(DECODER))
    assert all(torch.equal(mine[k], theirs[k]) for k in mine if not k.startswith(DECODER))
    # The adapter trained, and the readout did not move: the run started from the exported
    # readout, not from a new one.
    assert torch.equal(_readout(out), _readout(released))
