"""The CPU causal conv must match transformers' reference `causal_conv1d_fn`. No weights needed."""

from __future__ import annotations

import pytest
import torch

q35 = pytest.importorskip("transformers.models.qwen3_5.modeling_qwen3_5")

from strands_decider.cpu_kernels import causal_conv1d_cpu  # noqa: E402

reference = getattr(q35.causal_conv1d_fn, "__wrapped__", q35.causal_conv1d_fn)


@pytest.mark.parametrize("seq", [1, 3, 4, 70, 257])
@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("activation", [None, "silu"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_matches_reference(seq, width, activation, with_bias):
    g = torch.Generator().manual_seed(seq * 10 + width)
    x = torch.randn(2, 48, seq, generator=g)
    w = torch.randn(48, width, generator=g)
    b = torch.randn(48, generator=g) if with_bias else None
    want = reference(x, w, b, activation)
    got = causal_conv1d_cpu(x, w, b, activation)
    assert got.dtype == want.dtype and got.shape == want.shape
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_bf16_input_follows_weight_dtype():
    # The reference computes in the weight dtype and returns the input dtype.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 16, 20, generator=g).to(torch.bfloat16)
    w = torch.randn(16, 4, generator=g)
    want = reference(x, w, None, "silu")
    got = causal_conv1d_cpu(x, w, None, "silu")
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(got, want)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_rounds_no_worse_than_f_conv1d(dtype):
    # F.conv1d accumulates in fp32 and rounds once. Separate half-precision multiplies and
    # adds round at every step, about 3x the reference's error in bf16 and fp16.
    g = torch.Generator().manual_seed(0)
    x = torch.randn(2, 256, 300, generator=g).to(dtype)
    w = torch.randn(256, 4, generator=g).to(dtype)
    b = torch.randn(256, generator=g).to(dtype)

    def conv(x, w, b):
        return torch.nn.functional.conv1d(x, w.unsqueeze(1), b, padding=3, groups=256)[..., :300]

    exact = conv(x.double(), w.double(), b.double())
    want = (conv(x, w, b).double() - exact).abs().max()
    got = causal_conv1d_cpu(x, w, b)
    assert got.dtype == dtype
    assert (got.double() - exact).abs().max() <= 1.5 * want


def test_install_routes_cpu_and_is_idempotent():
    from strands_decider import cpu_kernels

    assert cpu_kernels.install()
    bound = q35.causal_conv1d_fn
    assert cpu_kernels.install()
    assert q35.causal_conv1d_fn is bound  # not wrapped twice
    x, w = torch.randn(1, 8, 9), torch.randn(8, 4)
    torch.testing.assert_close(bound(x, w, None, "silu"), reference(x, w, None, "silu"))


def _tiny_checkpoint(tmp_path):
    """A random-weight Qwen3.5 decider checkpoint: three Gated DeltaNet layers and one attention."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", "a", "b", "c"])}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                  unk_token="<unk>")
    base = tmp_path / "base"
    Qwen3_5ForCausalLM(Qwen3_5TextConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
        layer_types=["linear_attention"] * 3 + ["full_attention"])).save_pretrained(base)
    cfg = StrandsDeciderConfig(base_model=str(base), head_type="pointer", pointer_dim=8,
                               torch_dtype="float32", lora_r=2)
    torso = Qwen3_5ForCausalLM.from_pretrained(base).model
    model = StrandsDeciderModel(cfg, torso, tok)
    model.attach_lora()
    model.save_pretrained(str(tmp_path / "ckpt"))
    return str(tmp_path / "ckpt")


def test_a_cpu_engine_forward_runs_the_kernel(tmp_path, monkeypatch):
    # Fails if install() wraps nothing, if SystemOneEngine stops calling it, or if the
    # forward stops reaching the module function.
    from strands_decider import cpu_kernels
    from strands_decider.infer import load_engine
    from strands_decider.schema import NoulQuestion

    ckpt = _tiny_checkpoint(tmp_path)
    monkeypatch.setattr(q35, "causal_conv1d_fn", reference)
    monkeypatch.setattr(cpu_kernels, "_installed", False)
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[0].device.type)
        return causal_conv1d_cpu(*args, **kwargs)

    monkeypatch.setattr(cpu_kernels, "causal_conv1d_cpu", spy)
    engine = load_engine(ckpt, device="cpu")
    engine.ask("a b c", {"q": NoulQuestion(instructions="a b?")})
    assert calls and set(calls) == {"cpu"}


def test_install_leaves_other_devices_on_the_original(monkeypatch):
    from strands_decider import cpu_kernels

    seen = []

    def original(hidden_states, *args, **kwargs):
        seen.append(hidden_states.device.type)
        return hidden_states

    monkeypatch.setattr(q35, "causal_conv1d_fn", original)
    monkeypatch.setattr(cpu_kernels, "_installed", False)
    monkeypatch.setattr(cpu_kernels, "causal_conv1d_cpu", lambda *a, **k: pytest.fail("ran on meta"))
    assert cpu_kernels.install()
    x = torch.empty(1, 8, 9, device="meta")
    assert q35.causal_conv1d_fn(x, torch.empty(8, 4, device="meta"), None, "silu") is x
    assert seen == ["meta"]
