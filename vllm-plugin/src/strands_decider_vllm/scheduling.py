"""Once vLLM holds an image prompt back for a step, later image prompts wait out that step too.

With chunked prefill off, vLLM 0.34 schedules a prompt with images whole or not at all (vLLM
#65399): when its images don't fit the encoder's budget or cache, it waits for a later step. With
asynchronous scheduling, vLLM's default, the cache still holds the images of the step in flight, so
a steady stream of smaller image prompts would keep taking the room and the held one would never
fit. A step that schedules no new images frees the cache for it. Text prompts run as usual.
"""

from __future__ import annotations

from .engine import serves_decider

#: The scheduler's step in which it last held an image prompt back.
_HELD_AT = "_strands_decider_held_at"


def install() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    schedule_encoder_inputs = Scheduler._try_schedule_encoder_inputs

    def _try_schedule_encoder_inputs(
        self, request, num_computed_tokens, num_new_tokens, encoder_compute_budget, **kwargs
    ):
        args = (request, num_computed_tokens, num_new_tokens, encoder_compute_budget)
        if not serves_decider(self.vllm_config):
            return schedule_encoder_inputs(self, *args, **kwargs)
        end = num_computed_tokens + num_new_tokens
        if getattr(self, _HELD_AT, None) == self.current_step and any(
            f.mm_position.offset < end
            and f.mm_position.offset + f.mm_position.length > num_computed_tokens
            for f in request.mm_features or ()
        ):
            return [], 0, encoder_compute_budget, [], []  # vLLM's "none of it this step"
        out = schedule_encoder_inputs(self, *args, **kwargs)
        if out[1] < num_new_tokens:  # vLLM holds the prompt back rather than cut it
            setattr(self, _HELD_AT, self.current_step)
        return out

    Scheduler._try_schedule_encoder_inputs = _try_schedule_encoder_inputs
