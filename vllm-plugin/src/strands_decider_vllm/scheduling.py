"""A decision prompt with images is scheduled whole or not at all.

The mask needs every prompt computed to its end in one step (:mod:`strands_decider_vllm.model_state`). With chunked
prefill off and a step budget of at least ``--max-model-len``, vLLM's scheduler never splits a prompt for lack of
tokens. It still splits one for lack of encoder room: when the encoder's compute budget or its cache cannot take one
of the prompt's images this step, ``Scheduler._try_schedule_encoder_inputs`` cuts the prompt's tokens back to just
before that image and schedules the part in front of it. The encoder cache holds ``--max-num-batched-tokens``
embeddings, and an image's entry is released only once the step that used it has been processed, so under load (with
asynchronous scheduling a step is planned before the previous one's outputs are processed) this happens.

For an engine serving one of the plugin's architectures, :func:`install_whole_prompt_scheduling` wraps that method:
when it would cut a request's tokens, the request is not scheduled this step at all (vLLM's own outcome when an image
cannot be encoded and nothing in front of it can run), the encoder-cache references the attempt took are released, and
the request waits for the next step, when the cache has room again. Engines serving any other model are unchanged.
"""

from __future__ import annotations

from typing import Any

_MARK = "_strands_decider"


def wrap_try_schedule_encoder_inputs(original: Any) -> Any:
    from .engine import serves_decider

    def _try_schedule_encoder_inputs(
        self, request, num_computed_tokens, num_new_tokens, encoder_compute_budget, *args, **kwargs
    ):
        out = original(
            self,
            request,
            num_computed_tokens,
            num_new_tokens,
            encoder_compute_budget,
            *args,
            **kwargs,
        )
        if out[1] == num_new_tokens or not serves_decider(self):
            return out
        # The prompt would be cut before one of its images: schedule none of it this step.
        self.encoder_cache_manager.free(request)
        return [], 0, encoder_compute_budget, [], []

    _try_schedule_encoder_inputs._strands_decider = True
    _try_schedule_encoder_inputs.__wrapped__ = original
    return _try_schedule_encoder_inputs


def install_whole_prompt_scheduling() -> None:
    """Wrap ``vllm.v1.core.sched.scheduler.Scheduler._try_schedule_encoder_inputs`` (idempotent)."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if getattr(Scheduler._try_schedule_encoder_inputs, _MARK, False):
        return
    Scheduler._try_schedule_encoder_inputs = wrap_try_schedule_encoder_inputs(
        Scheduler._try_schedule_encoder_inputs
    )
