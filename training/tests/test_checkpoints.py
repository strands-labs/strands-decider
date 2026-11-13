"""merge, average and export on a small fake base checkpoint (the arithmetic, the layout, the records)."""

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from training import average, expert_lora, lora, merge

HIDDEN, INTER = 8, 12


def fake_base(directory: Path) -> dict:
    """A base layout with one adapted layer, one expert layer, an encoder tensor and a KV-shared k_proj."""
    torch.manual_seed(0)
    t = {
        "model.language_model.embed_tokens.weight": torch.randn(32, HIDDEN),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.language_model.layers.0.mlp.down_proj.weight": torch.randn(HIDDEN, INTER),
        "model.language_model.layers.0.experts.gate_up_proj": torch.randn(4, 2 * INTER, HIDDEN),
        "model.language_model.layers.0.experts.down_proj": torch.randn(4, HIDDEN, INTER),
        "model.language_model.layers.1.self_attn.k_proj.weight": torch.randn(HIDDEN, HIDDEN),
        "model.vision_tower.encoder.weight": torch.randn(4, 4),
    }
    t = {k: v.to(torch.bfloat16) for k, v in t.items()}
    directory.mkdir(parents=True)
    save_file(t, str(directory / "model.safetensors"), metadata={"format": "pt"})
    (directory / "config.json").write_text(
        json.dumps({"text_config": {"final_logit_softcapping": 30.0}})
    )
    (directory / "tokenizer.json").write_text("{}")
    return t


def fake_adapter(directory: Path, seed: int, prompt="strands-chat") -> dict:
    g = torch.Generator().manual_seed(seed)
    t = {
        "language_model.layers.0.self_attn.q_proj.lora_a": torch.randn(2, HIDDEN, generator=g),
        "language_model.layers.0.self_attn.q_proj.lora_b": torch.randn(HIDDEN, 2, generator=g),
        "language_model.layers.0.mlp.down_proj.lora_a": torch.randn(2, INTER, generator=g),
        "language_model.layers.0.mlp.down_proj.lora_b": torch.randn(HIDDEN, 2, generator=g),
        "language_model.layers.0.experts.gate_up_a": torch.randn(4, 8, HIDDEN, generator=g),
        "language_model.layers.0.experts.gate_up_b": torch.randn(4, 2 * INTER, 8, generator=g),
        "language_model.layers.0.experts.down_a": torch.randn(4, 8, INTER, generator=g),
        "language_model.layers.0.experts.down_b": torch.randn(4, HIDDEN, 8, generator=g),
        "readout.weight": torch.randn(255, HIDDEN, generator=g),
    }
    directory.mkdir(parents=True)
    save_file(t, str(directory / "adapter.safetensors"))
    record = {
        "base_model": "google/gemma-4-E2B-it",
        "revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "prompt": prompt,
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "seed": seed,
        "max_length": 16384,
        "lora": {"rank": 2, "alpha": 4, "dropout": 0.05, "targets": "q_proj,down_proj"},
        "expert_lora": {"rank": 8, "alpha": 16, "dropout": 0.05},
    }
    (directory / "adapter_config.json").write_text(json.dumps(record))
    return t


@pytest.fixture
def fields(monkeypatch):
    monkeypatch.setattr(
        merge,
        "decision_fields",
        lambda base, prompt: {"codes": ["A"], "token_ids": [1], "final_logit_softcapping": 30.0},
    )


def test_merge_arithmetic_and_layout(tmp_path, fields):
    base = fake_base(tmp_path / "base")
    adapter = fake_adapter(tmp_path / "adapter", 1)
    merge.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    out = load_file(str(tmp_path / "merged" / "model.safetensors"))
    assert set(out) == set(base) and all(v.dtype == torch.bfloat16 for v in out.values())
    name = "language_model.layers.0.self_attn.q_proj"
    want = (
        base[f"model.{name}.weight"].float()
        + (adapter[f"{name}.lora_b"] @ adapter[f"{name}.lora_a"]) * 2.0
    ).to(torch.bfloat16)
    assert torch.equal(out[f"model.{name}.weight"], want)
    gu = "model.language_model.layers.0.experts.gate_up_proj"
    want = (
        base[gu].float()
        + 2.0
        * torch.bmm(
            adapter["language_model.layers.0.experts.gate_up_b"],
            adapter["language_model.layers.0.experts.gate_up_a"],
        )
    ).to(torch.bfloat16)
    assert torch.equal(out[gu], want)
    for untouched in (
        "model.vision_tower.encoder.weight",
        "model.language_model.layers.1.self_attn.k_proj.weight",
        "model.language_model.embed_tokens.weight",
    ):
        assert torch.equal(out[untouched], base[untouched])
    readout = load_file(str(tmp_path / "merged" / "readout.safetensors"))
    assert torch.equal(readout["weight"], adapter["readout.weight"].to(torch.bfloat16))
    record = json.loads((tmp_path / "merged" / "training.json").read_text())
    assert (
        record["prompt"] == "strands-chat"
        and record["temperature"] == 1.0
        and record["finetune"]["seed"] == 1
    )
    assert (tmp_path / "merged" / "config.json").exists()


def test_merge_refuses_expert_tensors_without_their_record(tmp_path, fields):
    fake_base(tmp_path / "base")
    fake_adapter(tmp_path / "adapter", 1)
    path = tmp_path / "adapter" / "adapter_config.json"
    record = json.loads(path.read_text())
    record["expert_lora"] = None
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="expert"):
        merge.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_merged_weight_helpers_are_float32():
    w = torch.randn(4, 3).to(torch.bfloat16)
    a, b = torch.randn(2, 3), torch.randn(4, 2)
    assert torch.equal(
        lora.merged_weight(w, a, b, 0.5), (w.float() + (b @ a) * 0.5).to(torch.bfloat16)
    )
    we = torch.randn(2, 4, 3).to(torch.bfloat16)
    ae, be = torch.randn(2, 8, 3), torch.randn(2, 4, 8)
    assert torch.equal(
        expert_lora.merged_weight(we, ae, be, 0.25),
        (we.float() + 0.25 * torch.bmm(be, ae)).to(torch.bfloat16),
    )


def test_average(tmp_path, fields):
    fake_base(tmp_path / "base")
    for s in (1, 2, 3):
        fake_adapter(tmp_path / f"a{s}", s)
        merge.merge(tmp_path / "base", tmp_path / f"a{s}", tmp_path / f"m{s}")
    average.average([tmp_path / f"m{s}" for s in (1, 2, 3)], tmp_path / "avg", "three seeds")
    got = load_file(str(tmp_path / "avg" / "model.safetensors"))
    parts = [load_file(str(tmp_path / f"m{s}" / "model.safetensors")) for s in (1, 2, 3)]
    for k, v in got.items():
        acc = parts[0][k].float()
        acc += parts[1][k].float()
        acc += parts[2][k].float()
        assert torch.equal(v, (acc / 3).to(torch.bfloat16))
    assert torch.equal(
        got["model.vision_tower.encoder.weight"], parts[0]["model.vision_tower.encoder.weight"]
    )
    record = json.loads((tmp_path / "avg" / "training.json").read_text())
    assert record["average"]["seeds"] == [1, 2, 3] and record["average"]["note"] == "three seeds"


def test_average_refuses_mixed_templates(tmp_path, fields):
    fake_base(tmp_path / "base")
    fake_adapter(tmp_path / "a1", 1)
    fake_adapter(tmp_path / "a2", 2, prompt="strands-plain")
    merge.merge(tmp_path / "base", tmp_path / "a1", tmp_path / "m1")
    merge.merge(tmp_path / "base", tmp_path / "a2", tmp_path / "m2")
    with pytest.raises(SystemExit, match="prompt"):
        average.average([tmp_path / "m1", tmp_path / "m2"], tmp_path / "avg")


def test_export_helpers(tmp_path):
    from training import export

    assert export.strip_paths(
        {"_name_or_path": "/x", "a": "/home/u/m", "b": 1, "chat_template": "t"}
    ) == {"b": 1}
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "x.json").write_text('{"a": 1}')
    export.scan(tmp_path / "ok")
    shutil.copytree(tmp_path / "ok", tmp_path / "bad")
    (tmp_path / "bad" / "y.json").write_text('{"path": "/home/someone/model"}')
    with pytest.raises(SystemExit):
        export.scan(tmp_path / "bad")
    text = export.notice(
        "R",
        "google/gemma-4-E2B-it",
        "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
        "strands-chat",
        True,
    )
    assert "equal-weight average" in text and "Gemma is a trademark" in text
    assert export.trained_inputs({"finetune": {"inputs": ["text"]}}) == ["text"]
    assert export.trained_inputs({"finetune": {"seed": 1}}) == [
        "text"
    ]  # a record from before the field
    assert export.trained_inputs({"finetune": {"inputs": ["text", "image"]}}) == ["text", "image"]
    with pytest.raises(SystemExit):
        export.trained_inputs({"finetune": {"inputs": ["image"]}})


def test_merge_keeps_every_other_tensor_and_refuses_a_cast(tmp_path, fields):
    base = fake_base(tmp_path / "base")
    base["model.vision_tower.encoder.weight"][0, :2] = torch.tensor(
        [-0.0, float("nan")]
    )  # bits torch.equal misses
    save_file(base, str(tmp_path / "base" / "model.safetensors"), metadata={"format": "pt"})
    fake_adapter(tmp_path / "adapter", 1)
    merge.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged")
    out = load_file(str(tmp_path / "merged" / "model.safetensors"))
    adapted = {
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.experts.gate_up_proj",
        "model.language_model.layers.0.experts.down_proj",
    }
    for key, value in base.items():
        assert out[key].dtype == value.dtype and out[key].shape == value.shape, key
        if key not in adapted:
            assert out[key].view(torch.int16).equal(value.view(torch.int16)), key  # the same bytes
    tensors = load_file(str(tmp_path / "base" / "model.safetensors"))
    tensors["model.vision_tower.encoder.weight"] = tensors[
        "model.vision_tower.encoder.weight"
    ].float()
    save_file(tensors, str(tmp_path / "base" / "model.safetensors"), metadata={"format": "pt"})
    with pytest.raises(SystemExit, match="bf16"):
        merge.merge(tmp_path / "base", tmp_path / "adapter", tmp_path / "merged-2")
    assert not (tmp_path / "merged-2").exists()


def test_average_refuses_runs_trained_on_different_inputs(tmp_path, fields):
    fake_base(tmp_path / "base")
    for seed in (1, 2):
        fake_adapter(tmp_path / f"adapter-{seed}", seed)
        merge.merge(tmp_path / "base", tmp_path / f"adapter-{seed}", tmp_path / f"merged-{seed}")
    path = tmp_path / "merged-2" / "training.json"
    record = json.loads(path.read_text())
    record["finetune"]["inputs"] = ["text", "image"]
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="inputs"):
        average.average([tmp_path / "merged-1", tmp_path / "merged-2"], tmp_path / "avg")


def fake_merged(directory: Path, finetune: dict) -> None:
    """A merged checkpoint as training.merge writes one, small: the full config, a bf16 shard, the readout, the
    tokenizer files and training.json."""
    from strands_decider import core
    from training import export

    directory.mkdir(parents=True)
    save_file(
        {"model.language_model.embed_tokens.weight": torch.zeros(4, HIDDEN, dtype=torch.bfloat16)},
        str(directory / "model.safetensors"),
        metadata={"format": "pt"},
    )
    save_file(
        {"weight": torch.zeros(len(core.CODES), HIDDEN, dtype=torch.bfloat16)},
        str(directory / export.READOUT_FILE),
    )
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "vision_config": {},
                "_name_or_path": "/home/someone/base",
                "text_config": {
                    "final_logit_softcapping": 30.0,
                    "use_bidirectional_attention": "vision",
                },
            }
        )
    )
    (directory / "tokenizer.json").write_text("{}")
    (directory / "tokenizer_config.json").write_text("{}")
    (directory / merge.RECORD_FILE).write_text(
        json.dumps(
            {
                "base_model": "google/gemma-4-E2B-it",
                "base_revision": "3e22461f65e89153144f8adb70e3b8c2cc9845a7",
                "prompt": "strands-chat",
                "attention_mode": "prefix_bidir",
                "pooling": "last",
                "temperature": 1.0,
                "codes": list(core.CODES),
                "token_ids": list(range(1000, 1000 + len(core.CODES))),
                "final_logit_softcapping": 30.0,
                "finetune": {"max_length": 16384, **finetune},
            }
        )
    )


@pytest.mark.parametrize(
    "finetune, want",
    [
        ({"inputs": ["text"]}, ["text"]),
        ({}, ["text"]),
        ({"inputs": ["text", "image"]}, ["text", "image"]),
    ],
)
def test_export_declares_the_inputs_trained_on(tmp_path, monkeypatch, finetune, want):
    import dataclasses
    import sys

    from strands_decider import core
    from training import export

    fake_merged(tmp_path / "merged", finetune)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export",
            "--checkpoint",
            str(tmp_path / "merged"),
            "--out",
            str(tmp_path / "out"),
            "--release",
            "Test-E2B",
        ],
    )
    export.main()
    decision = json.loads((tmp_path / "out" / "decision_config.json").read_text())
    if "inputs" in {f.name for f in dataclasses.fields(core.DecisionConfig)}:
        assert decision["inputs"] == want
    assert f"| Inputs | {', '.join(want)} |" in (tmp_path / "out" / "README.md").read_text()
    config = json.loads((tmp_path / "out" / "config.json").read_text())
    assert (
        config["text_config"]["use_bidirectional_attention"] == "vision"
        and "_name_or_path" not in config
    )
