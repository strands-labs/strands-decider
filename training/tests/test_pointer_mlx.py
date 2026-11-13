"""A pointer-head checkpoint the trainer writes, on a Qwen3.5 torso, answers on v1's MLX engine as on its torch engine.

A tiny random-weight Qwen3.5 text model (Gated DeltaNet and attention layers) is the base; the trainer converts v1
examples, trains two steps of the pointer head with an adapter on every projection v19's adapter targets, and saves a
v1 checkpoint, which both engines load from disk. Apple silicon with the mlx extra only, as tests/test_mlx_engine.py.
"""

from __future__ import annotations

import sys

import pytest

if sys.platform != "darwin":
    pytest.skip("MLX runs on Apple silicon only", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")
transformers = pytest.importorskip("transformers")
if not hasattr(transformers, "Qwen3_5TextConfig"):
    pytest.skip("this transformers has no Qwen3.5", allow_module_level=True)

import torch  # noqa: E402
from tiny_support import _examples, _load_tokenizer, _tokenizer_file, convert, train_pointer  # noqa: E402

from strands_decider.infer import load_engine  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)

TARGETS = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj"
REQUEST = SystemOneRequest(
    state="The contract states that payment is due within thirty days of delivery.",
    questions={
        "who": ChoiceQuestion(
            instructions="Which party breached?",
            criteria={"seller": "the seller", "customer": None},
        ),
        "due": NoulQuestion(instructions="Payment is due within thirty days."),
        "how": ScoreQuestion(
            instructions="How clear is the clause?", criteria=["unclear", "clear", "very clear"]
        ),
    },
)


def _probabilities(response):
    return {
        n: [a.noul] if a.type == "noul" else list(a.probabilities.values())
        for n, a in response.answers.items()
    }


def test_a_trained_qwen3_5_checkpoint_answers_on_mlx_as_on_torch(tmp_path):
    examples = _examples(48, seed=1)
    _tokenizer_file(str(tmp_path / "tokenizer.json"), examples)
    tokenizer = _load_tokenizer(str(tmp_path / "tokenizer.json"))
    torch.manual_seed(0)
    config = transformers.Qwen3_5TextConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_key_head_dim=32, linear_value_head_dim=16,
        layer_types=["linear_attention"] * 3 + ["full_attention"],
    )  # fmt: skip
    base = tmp_path / "base"
    transformers.Qwen3_5ForCausalLM(config).save_pretrained(base)
    tokenizer.save_pretrained(str(base))
    rows = convert(tmp_path, base, examples, "--val-fraction", "0.1")
    flags = ("--targets", TARGETS, "--stop-after", "2", "--eval-every", "100")
    run = train_pointer(rows, base, tmp_path / "run", *flags)
    checkpoint = str(run / "last" / "checkpoint")
    expected = load_engine(checkpoint, device="cpu").evaluate(REQUEST)
    got = load_engine(checkpoint, device="mlx").evaluate(REQUEST)
    want, have = _probabilities(expected), _probabilities(got)
    assert have.keys() == want.keys()
    for name in want:
        # the answers are rounded to 4 places
        assert have[name] == pytest.approx(want[name], abs=1.5e-4), name
