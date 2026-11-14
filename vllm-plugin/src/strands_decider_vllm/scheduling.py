"""A decision prompt with images is scheduled whole or not at all, and a prompt held back for encoder room is not
overtaken.

The mask needs every prompt computed to its end in one step (:mod:`strands_decider_vllm.model_state`). With chunked
prefill off and a step budget of at least ``--max-model-len``, vLLM's scheduler never splits a prompt for lack of
tokens. It still splits one for lack of encoder room: when the encoder's compute budget or its cache cannot take one
of the prompt's images this step, ``Scheduler._try_schedule_encoder_inputs`` cuts the prompt's tokens back to just
before that image and schedules the part in front of it. The encoder cache holds ``--max-num-batched-tokens``
embeddings, and an image's entry is released only once the step that used it has been processed, so under load (with
asynchronous scheduling a step is planned before the previous one's outputs are processed) this happens.

For an engine serving one of the plugin's architectures, :func:`install_whole_prompt_scheduling` wraps that method:

- When it would cut a request's tokens, the request is not scheduled this step at all (vLLM's own outcome when an
  image cannot be encoded and nothing in front of it can run). The encoder-cache references the attempt took are
  released, and vLLM's scheduler puts the request back at the head of the waiting queue for the next step.
- For the rest of that step, every later request with an image in the part it would compute waits too, taking
  nothing from the encoder cache; requests without one are scheduled as usual. So a request held back for room is
  not overtaken by smaller image requests that fit in what is left. Once the steps that were in flight when it was
  held back have been processed (with asynchronous scheduling, one step), no request holds an image entry, the whole
  cache is free, and the held-back request is first in the queue. The encoder cache holds at least one whole
  prompt's images (``--max-num-batched-tokens`` embeddings, at least ``--max-model-len``), so it is scheduled then.

Engines serving any other model are unchanged.
"""

from __future__ import annotations

from typing import Any

_MARK = "_strands_decider"
#: Set on a scheduler for the rest of one ``schedule()`` call once a request has been held back for encoder room:
#: vLLM's return value for that request, the shape later held-back requests get.
_HELD = "_strands_decider_held"


def _images_in(request: Any, start: int, end: int) -> bool:
    """Whether the request has an image whose tokens overlap [start, end), the part this step would compute."""
    for f in request.mm_features or ():
        offset, length = f.mm_position.offset, f.mm_position.length
        if offset < end and offset + length > start:
            return True
    return False


def _nothing(shape: tuple, encoder_compute_budget: int) -> tuple:
    """vLLM's "none of this request this step", built from one of its own return values: no encoder inputs, no
    tokens, the encoder budget untouched, and every further element empty."""
    return ([], 0, encoder_compute_budget, *(type(x)() for x in shape[3:]))


def _release(scheduler: Any, request: Any, before: frozenset) -> None:
    """Release the encoder-cache references the attempt took: the request's cached inputs beyond ``before``."""
    ecm = scheduler.encoder_cache_manager
    for input_id in set(ecm.get_cached_input_ids(request)) - before:
        ecm.free_encoder_input(request, input_id)


def wrap_try_schedule_encoder_inputs(original: Any) -> Any:
    from .engine import serves_decider

    def _try_schedule_encoder_inputs(
        self, request, num_computed_tokens, num_new_tokens, encoder_compute_budget, *args, **kwargs
    ):
        call = (request, num_computed_tokens, num_new_tokens, encoder_compute_budget, *args)
        if not serves_decider(self):
            return original(self, *call, **kwargs)
        held = self.__dict__.get(_HELD)
        if held is not None and _images_in(
            request, num_computed_tokens, num_computed_tokens + num_new_tokens
        ):
            # An earlier request of this step is waiting for encoder room: this one waits behind it.
            return _nothing(held, encoder_compute_budget)
        before = frozenset(self.encoder_cache_manager.get_cached_input_ids(request))
        out = original(self, *call, **kwargs)
        if out[1] == num_new_tokens:
            return out
        if request.request_id in getattr(self, "encoder_cache_mismatch_reqs", ()):
            return out  # vLLM fails this request itself (an image id reused for other content)
        # The prompt would be cut before one of its images: schedule none of it this step.
        _release(self, request, before)
        if held is None:
            self.__dict__[_HELD] = tuple(out)
        return _nothing(out, encoder_compute_budget)

    _try_schedule_encoder_inputs._strands_decider = True
    _try_schedule_encoder_inputs.__wrapped__ = original
    return _try_schedule_encoder_inputs


def wrap_schedule(original: Any) -> Any:
    """``Scheduler.schedule``, with the held-back state of :func:`wrap_try_schedule_encoder_inputs` cleared before and
    after each call: it lasts one step."""

    def schedule(self, *args, **kwargs):
        self.__dict__.pop(_HELD, None)
        try:
            return original(self, *args, **kwargs)
        finally:
            self.__dict__.pop(_HELD, None)

    schedule._strands_decider = True
    schedule.__wrapped__ = original
    return schedule


def install_whole_prompt_scheduling() -> None:
    """Wrap ``vllm.v1.core.sched.scheduler.Scheduler``'s ``_try_schedule_encoder_inputs`` and ``schedule``
    (idempotent)."""
    from vllm.v1.core.sched.scheduler import Scheduler

    if not getattr(Scheduler._try_schedule_encoder_inputs, _MARK, False):
        Scheduler._try_schedule_encoder_inputs = wrap_try_schedule_encoder_inputs(
            Scheduler._try_schedule_encoder_inputs
        )
    if not getattr(Scheduler.schedule, _MARK, False):
        Scheduler.schedule = wrap_schedule(Scheduler.schedule)
