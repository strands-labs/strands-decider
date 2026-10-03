"""StrandsDeciderModel.load: a complete checkpoint loads unchanged. An incomplete or ambiguous one
is refused before the torso loads, and the head pickle is read with weights_only.

CPU only: a tiny random Qwen3 torso stands in for the base model, so nothing is
downloaded."""

import copy
import json
import os
import pickle
import shutil

import pytest
import torch

from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel


class _NotATensor:
    """A global that torch's weights_only unpickler refuses."""


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "a", "b", "c"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>")


@pytest.fixture
def ckpt(tmp_path, monkeypatch):
    """(checkpoint dir, the saved model, the torso loads that StrandsDeciderModel.load made)."""
    from transformers import Qwen3Config, Qwen3Model

    base = Qwen3Model(Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=64,
                                  num_hidden_layers=1, num_attention_heads=4,
                                  num_key_value_heads=2, head_dim=8))
    loads = []

    def load_torso(config, device_map, attn_implementation):
        loads.append(config.base_model)
        return copy.deepcopy(base)

    monkeypatch.setattr(StrandsDeciderModel, "_load_torso", staticmethod(load_torso))
    cfg = StrandsDeciderConfig(base_model="tiny", head_type="pointer", pointer_dim=16, lora_r=2,
                       torch_dtype="float32")
    model = StrandsDeciderModel(cfg, copy.deepcopy(base), _tokenizer())
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_()
    path = tmp_path / "ckpt"
    model.save_pretrained(str(path))
    return path, model, loads


def _same(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return set(sa) == set(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)


def test_a_complete_checkpoint_and_its_safetensors_head_load_unchanged(ckpt):
    path, model, loads = ckpt
    assert _same(StrandsDeciderModel.load(str(path)), model)
    from safetensors.torch import save_file

    save_file(torch.load(path / "slot_head.pt", weights_only=True), str(path / "head.safetensors"))
    os.remove(path / "slot_head.pt")
    assert _same(StrandsDeciderModel.load(str(path)), model)
    assert loads == ["tiny", "tiny"]


def test_a_missing_adapter_is_refused_before_the_torso_loads(ckpt):
    path, _, loads = ckpt
    shutil.rmtree(path / "lora")
    with pytest.raises(FileNotFoundError, match="lora"):
        StrandsDeciderModel.load(str(path))
    assert loads == []


def test_two_head_files_are_refused_before_the_torso_loads(ckpt):
    path, model, loads = ckpt
    from safetensors.torch import save_file

    save_file({k: torch.zeros_like(v) for k, v in model.head.state_dict().items()},
              str(path / "head.safetensors"))
    with pytest.raises(ValueError, match="both head"):
        StrandsDeciderModel.load(str(path))
    assert loads == []


def test_the_head_pickle_is_read_with_weights_only(ckpt, monkeypatch):
    path, model, loads = ckpt
    torch.save(dict(model.head.state_dict(), extra=_NotATensor()), path / "slot_head.pt")
    # Makes torch.load unpickle anything unless the call itself sets weights_only=True.
    monkeypatch.setenv("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        StrandsDeciderModel.load(str(path))
    assert loads == []


def test_a_hub_repo_id_loads_from_the_downloaded_snapshot(ckpt, monkeypatch):
    """A path that is not a local directory is a Hub repo id: snapshot_download gives the
    folder. No network here: the download is stubbed, and a failure names both readings."""
    import strands_decider.modeling as modeling

    path, model, loads = ckpt
    asked = []

    def download(repo_id, revision=None):
        asked.append(repo_id)
        return str(path)

    monkeypatch.setattr(modeling, "snapshot_download", download)
    assert _same(StrandsDeciderModel.load("org/hobson-test"), model)
    assert asked == ["org/hobson-test"] and loads == ["tiny"]

    def refuse(repo_id, revision=None):
        raise OSError("offline")

    monkeypatch.setattr(modeling, "snapshot_download", refuse)
    with pytest.raises(FileNotFoundError, match="not a local directory, and could not be "
                                                "downloaded from the Hugging Face Hub: OSError: offline"):
        StrandsDeciderModel.load(str(path / "typo"))
    assert loads == ["tiny"]


def test_a_local_directory_named_like_a_repo_id_is_not_downloaded(ckpt, monkeypatch, tmp_path):
    import strands_decider.modeling as modeling

    path, model, loads = ckpt
    local = tmp_path / "org" / "hobson-test"
    shutil.copytree(path, local)
    monkeypatch.setattr(modeling, "snapshot_download",
                        lambda repo_id, revision=None: pytest.fail(f"downloaded {repo_id}"))
    monkeypatch.chdir(tmp_path)
    assert _same(StrandsDeciderModel.load("org/hobson-test"), model)
    assert loads == ["tiny"]


def test_hobson_info_accepts_a_repo_id(ckpt, monkeypatch):
    from typer.testing import CliRunner

    import strands_decider.modeling as modeling
    from strands_decider.cli import app

    path, _, loads = ckpt
    monkeypatch.setattr(modeling, "snapshot_download", lambda repo_id, revision=None: str(path))
    res = CliRunner().invoke(app, ["info", "org/hobson-test"])
    assert res.exit_code == 0, res.output
    assert '"base_model": "tiny"' in res.output and loads == []


def test_calibration_refuses_a_repo_id_before_the_fit(ckpt, monkeypatch):
    """calibrate_checkpoint writes hobson_config.json into the checkpoint at the end, so a
    downloaded snapshot would lose the fit. It is refused before anything loads."""
    from strands_decider.evaluate import calibrate_checkpoint

    _, _, loads = ckpt
    with pytest.raises(FileNotFoundError, match="needs a local directory"):
        calibrate_checkpoint("org/hobson-test", [])
    assert loads == []



@pytest.mark.parametrize(("provenance", "pinned"), [
    ({"base_model": "tiny", "base_model_revision": "abc123"}, "abc123"),
    ({"base_model": "other", "base_model_revision": "abc123"}, None),
])
def test_the_base_revision_comes_from_provenance_for_the_same_base(ckpt, monkeypatch, provenance,
                                                                  pinned):
    """A Hugging Face export records the base revision in provenance.json, not in the config.
    load pins the torso to it, unless provenance.json describes another base model."""
    path, _, _ = ckpt
    (path / "provenance.json").write_text(json.dumps(provenance))
    revisions, load_torso = [], StrandsDeciderModel._load_torso

    def record(config, *args):
        revisions.append(config.base_revision)
        return load_torso(config, *args)

    monkeypatch.setattr(StrandsDeciderModel, "_load_torso", staticmethod(record))
    StrandsDeciderModel.load(str(path))
    assert revisions == [pinned]


def test_the_torso_config_and_weights_load_at_the_pinned_revision(monkeypatch):
    import transformers

    revisions = []

    def fake_config(name, **kwargs):
        revisions.append(("config", kwargs.get("revision")))
        return transformers.Qwen3Config()

    def fake_model(name, **kwargs):
        revisions.append(("model", kwargs.get("revision")))
        return transformers.Qwen3Model(transformers.Qwen3Config(
            vocab_size=6, hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8))

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(transformers.AutoModel, "from_pretrained", fake_model)
    config = StrandsDeciderConfig(base_model="tiny", base_revision="abc123", torch_dtype="float32")
    StrandsDeciderModel._load_torso(config, None, None)
    assert revisions == [("config", "abc123"), ("model", "abc123")]
