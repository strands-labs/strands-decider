"""training.export_v2 on a tiny Gemma 4 E2B-layout torso, on CPU: the v2 package layout."""

from __future__ import annotations

import json

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4Config"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from core_support import toy_tokenizer  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from test_gemma4_train import GEMMA_TARGETS, _checkpoint  # noqa: E402

from strands_decider.core import CODES  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from training.export_v2 import export_v2  # noqa: E402


def test_export_writes_the_v2_package(tmp_path):
    base, ckpt, out = tmp_path / "base", tmp_path / "ckpt", tmp_path / "out"
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=toy_tokenizer(), pad_token="<eos>")
    tok.save_pretrained(base)
    _checkpoint(base, "e2b", len(tok))
    cfg = StrandsDeciderConfig(
        base_model=str(base),
        base_revision="0" * 40,
        head_type="token_readout",
        render="strands-chat",
        attention_mode="prefix_bidir",
        lora_r=4,
        lora_alpha=8,
        lora_targets=GEMMA_TARGETS,
    )
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    model.save_pretrained(str(ckpt))
    export_v2(str(ckpt), str(out))

    config = json.loads((out / "config.json").read_text())
    assert config["architectures"] == ["Gemma4TextModel"]
    assert config["final_logit_softcapping"] == 30.0
    index = out / "model.safetensors.index.json"
    keys = (
        json.loads(index.read_text())["weight_map"]
        if index.exists()
        else load_file(out / "model.safetensors")
    )
    assert "embed_tokens.weight" in keys and not [k for k in keys if "lora" in k or "model." in k]
    readout = load_file(out / "readout.safetensors")
    assert list(readout) == ["weight"]
    assert readout["weight"].dtype == torch.bfloat16
    assert readout["weight"].shape == (255, config["hidden_size"])
    assert torch.equal(readout["weight"], model.head.proj.weight.detach().bfloat16())
    decision = json.loads((out / "decision_config.json").read_text())
    assert decision["codes"] == list(CODES)
    assert decision["token_ids"] == [tok.encode(c, add_special_tokens=False)[0] for c in CODES]
    assert (decision["prompt"], decision["attention_mode"], decision["revision"]) == (
        "strands-chat",
        "prefix_bidir",
        "0" * 40,
    )
    assert (out / "tokenizer.json").read_bytes() == (ckpt / "tokenizer.json").read_bytes()
