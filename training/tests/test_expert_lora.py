"""Expert LoRA on a tiny random Gemma 4 mixture-of-experts torso, on CPU."""

from __future__ import annotations

import pytest
import torch

transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Gemma4Config"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from core_support import toy_tokenizer  # noqa: E402
from test_gemma4_train import GEMMA_TARGETS, LAYERS, _checkpoint  # noqa: E402

from strands_decider import expert_lora  # noqa: E402
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel  # noqa: E402
from training.soup import soup  # noqa: E402

E, H, INTER, R = 4, 32, 16, 8  # _checkpoint's "moe" layout: experts, hidden, moe_intermediate


@pytest.fixture
def base(tmp_path):
    path = tmp_path / "base"
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=toy_tokenizer(), pad_token="<eos>")
    tok.save_pretrained(path)
    _checkpoint(path, "moe", len(tok))
    return path


def _model(base, **kw):
    kw = {"base_revision": "0" * 40, **kw}
    cfg = StrandsDeciderConfig(
        base_model=str(base),
        torch_dtype="float32",
        head_type="token_readout",
        attention_mode="prefix_bidir",
        lora_r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        lora_targets=GEMMA_TARGETS,
        **kw,
    )
    return StrandsDeciderModel.from_pretrained_base(cfg).eval()


def _hidden(torso, ids):
    with torch.no_grad():
        return torso(input_ids=ids).last_hidden_state


def _perturb(model):
    torch.manual_seed(1)
    with torch.no_grad():
        for m in expert_lora.modules(model.torso).values():
            m.gate_up_b.normal_(0, 0.5)
            m.down_b.normal_(0, 0.5)


IDS = torch.randint(3, 40, (2, 12), generator=torch.Generator().manual_seed(0))


def test_trainable_count_init_and_frozen_router(base):
    plain, model = _model(base), _model(base, expert_lora=True)
    per_layer = E * (R * H + 2 * INTER * R + R * INTER + H * R)
    extra = model.trainable_parameters()[0] - plain.trainable_parameters()[0]
    assert extra == len(LAYERS) * per_layer
    assert len(expert_lora.modules(model.torso)) == len(LAYERS)
    names = [n for n, p in model.torso.named_parameters() if p.requires_grad]
    assert not [n for n in names if "router" in n or n.endswith(("gate_up_proj", "down_proj"))]
    # B = 0: the same forward as the torso without expert adapters
    torch.testing.assert_close(_hidden(model.torso, IDS), _hidden(plain.torso, IDS))


def test_every_expert_gets_a_gradient(base):
    model = _model(base, expert_lora=True).train()
    model.torso(input_ids=IDS).last_hidden_state.pow(2).sum().backward()
    for m in expert_lora.modules(model.torso).values():
        assert m.gate_up_b.grad is not None and m.gate_up_b.grad.abs().sum() > 0


def test_save_load_merge_and_soup(base, tmp_path):
    model = _model(base, expert_lora=True)
    _perturb(model)
    want = _hidden(model.torso, IDS)
    assert (want - _hidden(_model(base).torso, IDS)).abs().max() > 1e-3  # the adapters act

    ckpt = tmp_path / "ckpt"
    model.save_pretrained(str(ckpt))
    loaded = StrandsDeciderModel.load(str(ckpt))
    assert not any(p.requires_grad for p in loaded.torso.parameters())
    torch.testing.assert_close(_hidden(loaded.torso, IDS), want)

    # a soup of two identical runs computes that run (rank 2r, alpha scaled with it)
    souped = StrandsDeciderModel.load(soup([str(ckpt), str(ckpt)], str(tmp_path / "soup")))
    assert souped.config.expert_lora_r == 2 * R
    torch.testing.assert_close(_hidden(souped.torso, IDS), want, rtol=1e-4, atol=1e-4)

    # the merged base computes what the adapted model does
    merged = loaded.torso.merge_and_unload()
    assert expert_lora.merge_and_unwrap(merged) == len(LAYERS)
    assert not expert_lora.modules(merged)
    torch.testing.assert_close(_hidden(merged, IDS), want, rtol=1e-4, atol=1e-4)


def test_expert_lora_needs_a_token_readout_and_lora():
    with pytest.raises(ValueError, match="expert_lora"):
        StrandsDeciderConfig(expert_lora=True)


def test_train_with_expert_lora_saves_and_reloads(base, tmp_path):
    from test_ddp import _examples

    import training.train as T
    from training.data.format import write_jsonl

    write_jsonl(str(tmp_path / "train.jsonl"), _examples(16, seed=1))
    out = tmp_path / "run"
    T.train(
        T.TrainConfig(
            train_files=[str(tmp_path / "train.jsonl")],
            val_files=[str(tmp_path / "train.jsonl")],
            base_model=str(base),
            head_type="token_readout",
            attention_mode="prefix_bidir",
            lora_targets=GEMMA_TARGETS,
            expert_lora=True,
            expert_lora_alpha=16,
            micro_batch_size=4,
            grad_accum=1,
            max_steps=2,
            eval_every=0,
            output_dir=str(out),
        )
    )
    model = StrandsDeciderModel.load(str(out))
    (m, *_) = expert_lora.modules(model.torso).values()
    assert (m.rank, m.scale) == (R, 16 / R)
    assert m.gate_up_b.abs().sum() > 0  # trained away from B = 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="grouped_mm path is CUDA only")
def test_the_cuda_grouped_path_matches_the_cpu_loop():
    """The fused path (grouped_mm, bfloat16) against the CPU loop, with nonzero adapters."""
    cfg = transformers.Gemma4TextConfig(
        hidden_size=64, moe_intermediate_size=32, num_experts=4, top_k_experts=2
    )
    torch.manual_seed(0)
    experts = transformers.models.gemma4.modeling_gemma4.Gemma4TextExperts(cfg)
    with torch.no_grad():
        experts.gate_up_proj.normal_(0, 0.1)
        experts.down_proj.normal_(0, 0.1)
    layer = expert_lora.ExpertLoRA(experts, R, 16, 0.0)
    with torch.no_grad():
        layer.gate_up_b.normal_(0, 0.1)
        layer.down_b.normal_(0, 0.1)
    x = torch.randn(10, 64)
    idx = torch.randint(0, 4, (10, 2))
    w = torch.rand(10, 2)
    cpu = layer.bfloat16()(x.bfloat16(), idx, w).float()
    gpu = layer.cuda()(x.bfloat16().cuda(), idx.cuda(), w.cuda()).float().cpu()
    torch.testing.assert_close(gpu, cpu, rtol=0.02, atol=0.02)


def test_the_export_folds_the_expert_adapters_as_the_module_merge_does(tmp_path):
    from core_support import gemma4_tokenizer_path
    from test_export_v2 import HUB_ID, REVISION, _base, _tensors

    from training.export_v2 import export_v2, write_base_record

    base = tmp_path / "base"
    _base(base, gemma4_tokenizer_path(), moe=True)
    model = _model(base, expert_lora=True, base_revision=REVISION)
    _perturb(model)
    ckpt, out = tmp_path / "ckpt", tmp_path / "out"
    model.save_pretrained(str(ckpt))
    assert write_base_record(str(ckpt), str(base), REVISION)
    export_v2(str(ckpt), str(out), base_model=HUB_ID)

    merged = model.torso.merge_and_unload()
    assert expert_lora.merge_and_unwrap(merged) == len(LAYERS)
    mine, theirs = merged.state_dict(), _tensors(out)
    for name in [k for k in mine if k.endswith(("experts.gate_up_proj", "experts.down_proj"))]:
        exported = theirs[f"model.language_model.{name}"]
        assert torch.equal(exported, mine[name].bfloat16()), name
        assert not torch.equal(exported, _tensors(base)[f"model.language_model.{name}"])


def test_a_two_seed_soup_export_averages_the_merged_experts(tmp_path):
    from core_support import gemma4_tokenizer_path
    from test_export_v2 import HUB_ID, _base, _loads, _seed_run, _tensors

    from training.export_v2 import export_v2

    base = tmp_path / "base"
    _base(base, gemma4_tokenizer_path(), moe=True)
    runs = [_seed_run(tmp_path, base, f"run{s}", s, expert_lora=True) for s in (1, 2)]
    out = tmp_path / "out"
    export_v2([str(r) for r in runs], str(out), base_model=HUB_ID)
    _loads(out)

    merged = []
    for r in runs:
        torso = StrandsDeciderModel.load(str(r)).float().torso.merge_and_unload()
        assert expert_lora.merge_and_unwrap(torso) == len(LAYERS)
        merged.append(torso.state_dict())
    theirs = _tensors(out)
    names = [k for k in merged[0] if k.endswith(("experts.gate_up_proj", "experts.down_proj"))]
    assert names
    for name in names:
        want = ((merged[0][name].float() + merged[1][name].float()) / 2).bfloat16()
        assert torch.equal(theirs[f"model.language_model.{name}"], want), name
        assert not torch.equal(merged[0][name], merged[1][name])  # the seeds differ
