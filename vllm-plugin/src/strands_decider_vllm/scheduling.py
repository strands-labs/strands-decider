"""A prompt with images is scheduled whole or not at all, and one held back is not overtaken.

The mask needs each prompt computed whole in one step. With chunked prefill off and a step budget of
at least ``--max-model-len``, vLLM 0.33 still cuts a prompt when the encoder's budget or cache can't
take one of its images this step: ``Scheduler._try_schedule_encoder_inputs`` schedules the tokens in
front of that image. Under load this happens, since an image's cache entry is released only once the
step that used it is processed, and with asynchronous scheduling the next step is planned before.

For engines that serve the plugin's architectures, such a request is scheduled not at all this step
(vLLM's own outcome when nothing in front of the image can run): the encoder-cache references the
attempt took are released, and vLLM puts it back at the head of the queue. For the rest of the step,
later requests with an image in what they would compute wait too, so smaller image requests don't
keep taking the room it needs; text requests run as usual. Once the steps in flight are processed the
whole cache is free, and it holds a prompt's images (``--max-num-batched-tokens`` embeddings, at
least ``--max-model-len``), so the request is scheduled then. vLLM 0.34 defers such a request whole
itself (vLLM #65399).
"""

from __future__ import annotations

from .engine import serves_decider

#: Set on the scheduler for the rest of one ``schedule()`` once a request is held back: vLLM's
#: return value for it, whose shape later held-back requests' "nothing" takes.
_HELD = "_strands_decider_held"


def _nothing(out: tuple, encoder_compute_budget: int) -> tuple:
    """vLLM's "none of this request this step", in the shape of its return value ``out``."""
    return ([], 0, encoder_compute_budget, *(type(x)() for x in out[3:]))


def install() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    original_try, original_schedule = Scheduler._try_schedule_encoder_inputs, Scheduler.schedule

    def _try_schedule_encoder_inputs(
        self, request, num_computed_tokens, num_new_tokens, encoder_compute_budget, *args, **kwargs
    ):
        call = (request, num_computed_tokens, num_new_tokens, encoder_compute_budget, *args)
        if not serves_decider(self.vllm_config):
            return original_try(self, *call, **kwargs)
        held = self.__dict__.get(_HELD)
        end = num_computed_tokens + num_new_tokens
        if held is not None and any(
            f.mm_position.offset < end
            and f.mm_position.offset + f.mm_position.length > num_computed_tokens
            for f in request.mm_features or ()
        ):
            return _nothing(held, encoder_compute_budget)
        ecm = self.encoder_cache_manager
        before = set(ecm.get_cached_input_ids(request))
        out = original_try(self, *call, **kwargs)
        if out[1] == num_new_tokens:
            return out
        for input_id in set(ecm.get_cached_input_ids(request)) - before:
            ecm.free_encoder_input(request, input_id)
        self.__dict__.setdefault(_HELD, tuple(out))
        return _nothing(out, encoder_compute_budget)

    def schedule(self, *args, **kwargs):
        self.__dict__.pop(_HELD, None)  # a request is held back for one step
        return original_schedule(self, *args, **kwargs)

    Scheduler._try_schedule_encoder_inputs = _try_schedule_encoder_inputs
    Scheduler.schedule = schedule
