"""The Neuron decoder's mixture-of-experts layers (Gemma 4 26B-A4B's `enable_moe_block`) against
transformers' own Gemma 4 text model, on a tiny random checkpoint, on the host in float32:
neuron_model.Rank at world 1 (all experts on one rank), joint and split packing, each question's
final-norm row against the reference run of its own prompt under prefix_bidir. No NeuronCore."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("safetensors")


def _tiny_checkpoint(path, moe: bool = True):
    from safetensors.torch import save_file
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    text = dict(
        vocab_size=512,
        hidden_size=64,
        intermediate_size=48,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=16,
        global_head_dim=32,
        attention_k_eq_v=True,
        enable_moe_block=moe,
        num_experts=8,
        top_k_experts=2,
        moe_intermediate_size=24,
        hidden_activation="gelu_pytorch_tanh",
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        sliding_window=8,
        rope_parameters={
            "full_attention": {
                "partial_rotary_factor": 0.25,
                "rope_theta": 1e6,
                "rope_type": "proportional",
            },
            "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
        },
        final_logit_softcapping=30.0,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    cfg = Gemma4TextConfig(**text)
    model = Gemma4TextModel(cfg).float()
    with torch.no_grad():  # random norms, scales and routing, not the init's ones
        for name, p in model.named_parameters():
            if p.dim() == 1 or "scale" in name:
                p.copy_(1.0 + 0.3 * torch.randn_like(p))
            else:
                p.normal_(0.0, 0.08)
        for layer in model.layers:
            layer.layer_scalar.fill_(0.9)
    sd = {"model.language_model." + k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(path / "model.safetensors"))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in sd}})
    )
    cfg.save_pretrained(str(path))
    return model


def _reference_rows(model, state, questions):
    from strands_decider.backends.transformers_attention import ATTENTION, StateSplit, _attention

    transformers.AttentionInterface.register(ATTENTION, _attention)
    model.config._attn_implementation = ATTENTION
    out = []
    for q in questions:
        ids = torch.tensor([state + q])
        mask = {t: StateSplit(len(state)) for t in set(model.config.layer_types)}
        h = model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=torch.arange(ids.shape[1])[None],
            use_cache=False,
        ).last_hidden_state
        out.append(h[0, -1].float().numpy())
    return np.stack(out)


@pytest.mark.parametrize("moe", [True, False])
@pytest.mark.parametrize("joint", [True, False])
def test_decoder_matches_transformers(tmp_path, moe: bool, joint: bool) -> None:
    from strands_decider.backends.neuron import pack, pack_joint
    from strands_decider.backends.neuron_model import Rank

    model = _tiny_checkpoint(tmp_path, moe=moe)
    rng = np.random.default_rng(1)
    state = [int(x) for x in rng.integers(1, 512, 23)]
    questions = [[int(x) for x in rng.integers(1, 512, n)] for n in (5, 1, 9)]
    with torch.no_grad():
        rank = Rank(
            str(tmp_path),
            1,
            0,
            {"device": "cpu", "dtype": "float32", "compile": False, "max_positions": 1024},
        )
        assert bool(rank.top_k) == moe
        b = (
            pack_joint(state, questions, pad_id=0)
            if joint
            else pack(state, questions, pad_id=0, state_buckets=(32,), question_buckets=(16, 32))
        )
        rows, _ = rank.forward(b)
        ref = _reference_rows(model, state, questions)
    np.testing.assert_allclose(rows[: len(questions)], ref, rtol=2e-4, atol=2e-4)


def test_expert_parallel_ranks_sum_to_one_rank(tmp_path) -> None:
    """Two ranks, each with half the experts and half the dense MLP's columns, whose partial
    feed-forward outputs are summed (the reduce), give what one rank with everything gives."""
    from strands_decider.backends.neuron_model import MOE_NAMES, Rank, _ffn_moe

    _tiny_checkpoint(tmp_path, moe=True)
    opts = {"device": "cpu", "dtype": "float32", "compile": False, "max_positions": 64}
    one = Rank(str(tmp_path), 1, 0, opts)
    ranks = [Rank(str(tmp_path), 2, r, opts) for r in (0, 1)]
    assert [r.layers[0]["e_ids"].tolist() for r in ranks] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    h = torch.randn(11, 64, generator=torch.Generator().manual_seed(2))

    def ffn(rank, i, reduce):
        w = rank.layers[i]
        return _ffn_moe(
            h,
            w["w_gu"],
            w["w_down"],
            w["ln_pre_ff"],
            w["ln_post_ff"],
            tuple(w[k] for k in MOE_NAMES),
            one.eps,
            reduce,
            None,
            one.top_k,
        )

    with torch.no_grad():
        for i in range(len(one.layers)):
            want = ffn(one, i, lambda x: x)
            other: list = []
            ffn(ranks[1], i, lambda x, o=other: o.append(x) or x)
            got = ffn(ranks[0], i, lambda x, o=other: x + o[0])
            np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-5, atol=1e-5)
