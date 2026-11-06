"""``StrandsDeciderCausalControlForCausalLM``: Gemma 4's own layers (causal, with their sliding windows) under the
checkpoint loading, readout and logits of :class:`strands_decider_vllm.model.StrandsDeciderForCausalLM`."""

from __future__ import annotations

import logging
from typing import Any

from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from strands_decider_vllm import PREFIX_END_KEY
from strands_decider_vllm.model import StrandsDeciderForCausalLM, check_config

logger = logging.getLogger("strands_decider_causal_control")


class CausalControlModelState(DefaultModelState):
    """vLLM's own model state, so every request is causal. A request's prefix_end is ignored, with one warning."""

    _warned = False

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        super().add_request(req_index, new_req_data)
        sp = new_req_data.sampling_params
        if not self._warned and sp is not None and PREFIX_END_KEY in (sp.extra_args or {}):
            self._warned = True
            logger.warning(
                "causal control: vllm_xargs.prefix_end is ignored; this server applies Gemma 4's causal, "
                "sliding-window attention, not the prefix_bidir mask"
            )


class StrandsDeciderCausalControlForCausalLM(StrandsDeciderForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        check_config(vllm_config)
        logger.warning(
            "causal control: Gemma 4's causal, sliding-window attention with the trained readout; a diagnostic, "
            "not a Strands Decider server"
        )
        Gemma4ForCausalLM.__init__(
            self, vllm_config=vllm_config, prefix=prefix
        )  # stock layers, windows kept
        self._init_readout(vllm_config)

    @classmethod
    def get_model_state_cls(cls) -> type[CausalControlModelState]:
        return CausalControlModelState
