"""A torso split across GPUs (`serve --device-map`) gives the answers it gives on one GPU.

The split is the one accelerate makes for `device_map="auto"`: each layer on its GPU, the
inputs moved to it by hooks. Needs two CUDA GPUs; a tiny random torso.
"""

from __future__ import annotations

import copy

import pytest
import torch
from test_engine_concurrency import _tokenizer

from strands_decider.infer import EngineConfig, SystemOneEngine, spans_gpus
from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
from strands_decider.schema import ChoiceQuestion, SystemOneRequest

TWO_GPUS = torch.cuda.is_available() and torch.cuda.device_count() >= 2


def _model() -> StrandsDeciderModel:
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    tok = _tokenizer()
    torso = Qwen3Model(
        Qwen3Config(
            vocab_size=len(tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    cfg = StrandsDeciderConfig(
        base_model="tiny", pointer_dim=16, use_lora=False, torch_dtype="float32", max_length=512
    )
    return StrandsDeciderModel(cfg, torso, tok)


def _request(n_questions: int) -> SystemOneRequest:
    questions = {
        f"q{j}": ChoiceQuestion(
            type="choice",
            instructions="pick " * (1 + j),
            criteria={f"opt{k}": "" for k in range(2 + j)},
        )
        for j in range(n_questions)
    }
    return SystemOneRequest(state="s" * 20, questions=questions)


def test_one_device_is_not_a_split():
    assert not spans_gpus(_model().torso)


@pytest.mark.skipif(not TWO_GPUS, reason="needs two CUDA GPUs")
@pytest.mark.parametrize("n_questions", [1, 3])  # batched forward, shared-prefix forward
def test_split_torso_answers_as_on_one_gpu(n_questions):
    from accelerate import dispatch_model

    model = _model()
    split = copy.deepcopy(model)
    dispatch_model(
        split.torso,
        device_map={"embed_tokens": 0, "rotary_emb": 0, "layers.0": 0, "layers.1": 1, "norm": 1},
    )
    assert spans_gpus(split.torso)

    one = SystemOneEngine(model, EngineConfig(device="cuda:0")).evaluate(_request(n_questions))
    two = SystemOneEngine(split, EngineConfig(device="cuda:0")).evaluate(_request(n_questions))
    for name, answer in one.answers.items():
        expected = answer.model_dump()["probabilities"]
        got = two.answers[name].model_dump()["probabilities"]
        assert got == pytest.approx(expected, abs=1e-4)
