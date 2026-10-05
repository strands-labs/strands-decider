"""strands_decider.soup on tiny Qwen3.5 checkpoints (tests/tiny_qwen35.py), CPU only.

* the soup of one checkpoint taken three times is that checkpoint (logits within 1e-5);
* the soup's LoRA update is exactly the mean of the inputs' updates, at rank 3 r;
* the soup loads through StrandsDeciderModel.load and serves through the engine, with
  its calibration reset;
* inputs that do not share a base, a head or an initialisation are refused.
"""

from __future__ import annotations

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import load_file
from tiny_qwen35 import needs_tiny_qwen35, save_base, save_checkpoint

from strands_decider import soup as soup_mod
from strands_decider.data.collate import CollatorConfig, SystemOneCollator
from strands_decider.data.format import Example
from strands_decider.infer import EngineConfig, SystemOneEngine
from strands_decider.modeling import StrandsDeciderModel
from strands_decider.schema import ChoiceQuestion, NoulQuestion

pytestmark = needs_tiny_qwen35


def _train_config(path: str, **over) -> None:
    cfg = {"seed": 0, "init_seed": 0, "continue_from": None, "init_from": None, "head_init": "random"}
    cfg.update(over)
    with open(os.path.join(path, "train_config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Three checkpoints on one base, one init_seed, different adapters and heads."""
    base = save_base(str(tmp_path_factory.mktemp("tiny-qwen35")))
    out = []
    for seed in range(3):
        torch.manual_seed(100 + seed)
        path = save_checkpoint(base, str(tmp_path_factory.mktemp(f"run{seed}")))
        head = torch.load(os.path.join(path, "slot_head.pt"))
        torch.save({k: v + 0.1 * torch.randn_like(v) for k, v in head.items()},
                   os.path.join(path, "slot_head.pt"))
        _train_config(path, seed=seed)
        out.append(path)
    return out


def _logits(path: str) -> torch.Tensor:
    model = StrandsDeciderModel.load(path)
    model.eval()
    rows = [Example(kind="choice", state=f"Ticket {k}: refund please.", instructions="Which team?",
                    options=[["billing", ""], ["shipping", ""], ["sales", ""]], label=0, task="route")
            for k in range(3)]
    coll = SystemOneCollator(model.tokenizer, CollatorConfig(max_length=512, head_type="pointer"),
                             train=False)
    b = coll(rows)
    with torch.no_grad():
        return model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                     n_slots=b["n_slots"], opt_idx=b["opt_idx"])["logits"]


def _delta(path: str) -> dict[str, torch.Tensor]:
    """scaling * B @ A per LoRA module, from the files on disk."""
    with open(os.path.join(path, "lora", "adapter_config.json"), encoding="utf-8") as fh:
        acfg = json.load(fh)
    w = load_file(os.path.join(path, "lora", "adapter_model.safetensors"))
    s = acfg["lora_alpha"] / acfg["r"]
    return {k.replace(".lora_A.weight", ""): s * w[k.replace("lora_A", "lora_B")].double() @ w[k].double()
            for k in w if k.endswith(".lora_A.weight")}


def test_the_soup_of_one_checkpoint_is_that_checkpoint(runs, tmp_path):
    out = str(tmp_path / "soup")
    soup_mod.soup([runs[0]] * 3, out)
    torch.testing.assert_close(_logits(out), _logits(runs[0]), atol=1e-5, rtol=0)


def test_the_soup_update_is_the_mean_update(runs, tmp_path):
    out = str(tmp_path / "soup")
    record = soup_mod.soup(runs, out)
    with open(os.path.join(out, "lora", "adapter_config.json"), encoding="utf-8") as fh:
        acfg = json.load(fh)
    assert (acfg["r"], acfg["lora_alpha"]) == (48, 32)  # 3 x rank 16, scaling 2 -> 2/3
    deltas = [_delta(p) for p in runs]
    mixed = _delta(out)
    assert mixed.keys() == deltas[0].keys() and len(mixed) > 10
    for k, d in mixed.items():
        torch.testing.assert_close(d, sum(x[k] for x in deltas) / 3, atol=1e-10, rtol=1e-9)
    heads = [torch.load(os.path.join(p, "slot_head.pt")) for p in runs]
    for k, v in torch.load(os.path.join(out, "slot_head.pt")).items():
        torch.testing.assert_close(v, sum(h[k] for h in heads) / 3)
    assert "concatenated" in record["method"]
    assert record["versions"]["torch"] == torch.__version__
    with open(os.path.join(out, "soup.json"), encoding="utf-8") as fh:
        assert json.load(fh)["versions"] == record["versions"]


def test_the_soup_loads_serves_and_is_uncalibrated(runs, tmp_path):
    out = str(tmp_path / "soup")
    soup_mod.main(["--out", out, *runs])
    with open(os.path.join(out, "strands_decider_config.json"), encoding="utf-8") as fh:
        cfg = json.load(fh)
    assert cfg["temperature"] == 1.0 and cfg["temperature_by_kind"] == {}  # inputs had some
    assert (cfg["lora_r"], cfg["lora_alpha"]) == (48, 32)
    model = StrandsDeciderModel.load(out)
    engine = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=False))
    resp = engine.ask("Order 7 shipped three days late.", {
        "late": NoulQuestion(instructions="Was the order late?"),
        "team": ChoiceQuestion(instructions="Which team?", criteria={"billing": None, "shipping": None}),
    })
    assert set(resp.answers) == {"late", "team"}
    assert 0 < resp.answers["late"].noul < 1
    assert abs(sum(resp.answers["team"].probabilities.values()) - 1) < 1e-4
    # A soup's inputs can be soups: the soup records the initialisation it shares.
    assert soup_mod.init_identity(out) == soup_mod.init_identity(runs[0])


def _variant(src: str, dst: str, **config) -> str:
    shutil.copytree(src, dst)
    path = os.path.join(dst, "strands_decider_config.json")
    with open(path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    cfg.update(config)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return dst


def test_mismatched_inputs_are_refused(runs, tmp_path):
    def refused(paths, match):
        with pytest.raises((ValueError, FileExistsError), match=match):
            soup_mod.soup(paths, str(tmp_path / f"out{len(os.listdir(tmp_path))}"))

    other_init = _variant(runs[1], str(tmp_path / "i1"))
    _train_config(other_init, init_seed=1)
    refused([runs[0], other_init], "initialised from")
    from_raw_seed = _variant(runs[1], str(tmp_path / "s1"))
    _train_config(from_raw_seed, seed=1, init_seed=None)  # init seeded by seed=1
    refused([runs[0], from_raw_seed], "initialised from")
    continued = _variant(runs[1], str(tmp_path / "c"))
    _train_config(continued, continue_from="checkpoints/v19")
    refused([runs[0], continued], "initialised from")
    refused([runs[0], _variant(runs[1], str(tmp_path / "rev"), base_revision="abc")],
            "base_revision")
    refused([runs[0], _variant(runs[1], str(tmp_path / "h"), head_type="slot")], "head_type")
    refused([runs[0], _variant(runs[1], str(tmp_path / "d"), pointer_dim=16)], "pointer_dim")
    no_tc = _variant(runs[1], str(tmp_path / "notc"))
    os.remove(os.path.join(no_tc, "train_config.json"))
    refused([runs[0], no_tc], "train_config")
    refused([runs[0]], "at least two")
    rslora = [_variant(r, str(tmp_path / f"rs{i}")) for i, r in enumerate(runs[:2])]
    for p in rslora:
        acfg = os.path.join(p, "lora", "adapter_config.json")
        with open(acfg, encoding="utf-8") as fh:
            a = json.load(fh)
        with open(acfg, "w", encoding="utf-8") as fh:
            json.dump({**a, "use_rslora": True}, fh)
    refused(rslora, "use_rslora")
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("x")
    with pytest.raises(FileExistsError):
        soup_mod.soup(runs, str(full))


def test_runs_continued_from_one_checkpoint_soup(runs, tmp_path):
    paths = []
    for i in range(2):
        p = _variant(runs[i], str(tmp_path / f"v{i}"))
        _train_config(p, continue_from="checkpoints/v19", seed=i, init_seed=0 if i else None)
        paths.append(p)
    soup_mod.soup(paths, str(tmp_path / "soup"))
    assert soup_mod.init_identity(str(tmp_path / "soup")) == {"continue_from": "checkpoints/v19"}
