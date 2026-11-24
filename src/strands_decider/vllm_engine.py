"""vLLM as the backend of `strands_decider.hidden_engine.HiddenStateEngine`.

vLLM runs the torso; `HiddenStateEngine` renders prompts, fits the window, runs the pointer
head (numpy, on the CPU) and shapes the answer. Mirrors `deploy/lambda/function/backend.py`'s
`LlamaBackend`: a `forward()` per request returns the torso's final hidden states at the
positions the head reads (each option's last token, and `<answer>`).

The torso is loaded through vLLM's offline `LLM` with `runner="pooling"` and a pooler config
of `pooling_type="ALL"`, so every prompt token's final hidden state is returned; this backend
slices out the rows the head reads. vLLM's prefix cache is on, so state tokens shared across
the questions of one request are encoded once.

Install with the `vllm` extra:

    pip install 'strands-decider[vllm]'

Then wire it up with `load_vllm_engine`:

    from strands_decider.vllm_engine import load_vllm_engine, VLLMBackendConfig

    engine = load_vllm_engine(
        assets="path/to/save_assets",
        backend=VLLMBackendConfig(model="org/base-torso", dtype="bfloat16"),
    )
    engine.evaluate(request)

`assets` is a `save_assets` directory (tokenizer.json, head.safetensors, head_config.json).
`model` is the torso vLLM loads (a local HF checkpoint or Hub repo id): the base torso's
weights, with any LoRA adapter already merged, or served through vLLM's own LoRA support
(see `extra={"enable_lora": True, ...}`).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hidden_engine import HiddenStateEngine


@dataclass
class VLLMBackendConfig:
    """Arguments for vLLM's offline `LLM`.

    The defaults suit a BF16 torso on one GPU; `extra` passes further kwargs straight to
    `LLM(...)` (for example `enable_lora`, `tokenizer`, `trust_remote_code`).
    """

    model: str
    dtype: str = "bfloat16"
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    enable_prefix_caching: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


class VLLMBackend:
    """`HiddenStateEngine.Backend` backed by vLLM's offline `LLM`.

    One call runs one request's questions: each is submitted as `state + question` and vLLM
    returns the last hidden state at every token position; this backend picks the rows the
    head reads. Several forwards can arrive at once from the engine's thread pool; the lock
    serialises them, as `LlamaBackend` does. vLLM batches within one `encode` call itself.
    """

    def __init__(self, config: VLLMBackendConfig):
        # Imported lazily so the module can be imported (and listed) without vLLM installed.
        from vllm import LLM
        from vllm.config import PoolerConfig

        kwargs: dict[str, Any] = {
            "model": config.model,
            "runner": "pooling",
            "dtype": config.dtype,
            "tensor_parallel_size": config.tensor_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "enforce_eager": config.enforce_eager,
            "enable_prefix_caching": config.enable_prefix_caching,
            "override_pooler_config": PoolerConfig(
                pooling_type="ALL", normalize=False, softmax=False
            ),
        }
        if config.max_model_len is not None:
            kwargs["max_model_len"] = config.max_model_len
        kwargs.update(config.extra)
        self.llm = LLM(**kwargs)
        self._lock = threading.Lock()

    def forward(
        self, state: list[int], questions: list[list[int]], rows: list[list[int]]
    ) -> list[np.ndarray]:
        from vllm import PoolingParams

        base = len(state)
        prompts = [{"prompt_token_ids": state + q} for q in questions]
        with self._lock:
            outputs = self.llm.encode(prompts, pooling_params=PoolingParams(), use_tqdm=False)
        if len(outputs) != len(questions):
            raise RuntimeError(
                f"vLLM returned {len(outputs)} outputs for {len(questions)} questions"
            )
        picked: list[np.ndarray] = []
        for out, q, row in zip(outputs, questions, rows, strict=True):
            data = out.outputs.data  # torch.Tensor [seq_len, hidden]
            seq = int(data.shape[0])
            want = base + len(q)
            if seq != want:
                # Guards against a vLLM build or pooler config that returned the wrong shape:
                # one pooled vector per prompt (pooling_type != ALL), truncation, etc.
                raise RuntimeError(
                    f"vLLM returned {seq} hidden rows for a {want}-token prompt; "
                    "expected one row per token (override_pooler_config pooling_type ALL)"
                )
            idx = [base + r for r in row]
            picked.append(data[idx].float().cpu().numpy().astype(np.float32, copy=False))
        return picked


def load_vllm_engine(
    assets: str,
    backend: VLLMBackendConfig,
    **engine_config: Any,
) -> HiddenStateEngine:
    """`HiddenStateEngine.from_assets` wired to a vLLM backend.

    `assets`: a `save_assets` directory (tokenizer.json, head.safetensors, head_config.json).
    `backend`: vLLM loading options (see `VLLMBackendConfig`).
    `engine_config`: overrides on the saved `HiddenEngineConfig` (`max_request_tokens` and
    friends).
    """
    return HiddenStateEngine.from_assets(assets, VLLMBackend(backend), **engine_config)
