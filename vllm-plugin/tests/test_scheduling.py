"""A decision prompt with images is scheduled whole or not at all: the wrapper around vLLM's encoder-input scheduling
(stubbed here), and the vLLM method it wraps (skipped without vLLM)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from strands_decider_vllm import ARCH, ARCH_MULTIMODAL
from strands_decider_vllm.scheduling import wrap_try_schedule_encoder_inputs


class FakeEncoderCache:
    def __init__(self):
        self.freed = []

    def free(self, request):
        self.freed.append(request.request_id)


class FakeScheduler:
    """vLLM's scheduler as the wrapper sees it: ``_try_schedule_encoder_inputs`` returns (inputs to encode, tokens
    to schedule, remaining encoder budget, external loads, duplicates); here it cuts the tokens to ``cut_to``."""

    def __init__(self, architectures, cut_to=None):
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=architectures)
        )
        self.encoder_cache_manager = FakeEncoderCache()
        self.cut_to = cut_to

    def _try_schedule_encoder_inputs(
        self,
        request,
        num_computed_tokens,
        num_new_tokens,
        encoder_compute_budget,
        shift_computed_tokens=0,
    ):
        n = num_new_tokens if self.cut_to is None else self.cut_to
        return [0, 1], n, encoder_compute_budget - 516, [], []


wrapped = wrap_try_schedule_encoder_inputs(FakeScheduler._try_schedule_encoder_inputs)
REQUEST = SimpleNamespace(request_id="r1")


def test_a_whole_prompt_is_scheduled_as_vllm_plans_it():
    s = FakeScheduler([ARCH_MULTIMODAL])
    assert wrapped(s, REQUEST, 0, 900, 8192, shift_computed_tokens=0) == ([0, 1], 900, 7676, [], [])
    assert s.encoder_cache_manager.freed == []


@pytest.mark.parametrize("cut_to", [3, 0, 899])
def test_a_prompt_cut_before_an_image_is_not_scheduled_at_all(cut_to):
    # vLLM would schedule the tokens in front of an image it cannot encode this step; the mask needs the whole
    # prompt in one step, so none of it runs now, and the encoder-cache references the attempt took are released.
    s = FakeScheduler([ARCH_MULTIMODAL], cut_to=cut_to)
    assert wrapped(s, REQUEST, 0, 900, 8192) == ([], 0, 8192, [], [])
    assert s.encoder_cache_manager.freed == ["r1"]


def test_other_models_keep_vllms_partial_schedule():
    s = FakeScheduler(["Gemma4ForConditionalGeneration"], cut_to=3)
    assert wrapped(s, REQUEST, 0, 900, 8192) == ([0, 1], 3, 7676, [], [])
    assert s.encoder_cache_manager.freed == []


def test_the_decoder_layout_is_covered_too():
    # It never has encoder inputs, so vLLM never cuts its prompts here; were it to, the same rule applies.
    s = FakeScheduler([ARCH], cut_to=3)
    assert wrapped(s, REQUEST, 0, 900, 8192)[1] == 0


# --- against vLLM (skipped without it) ------------------------------------------------------------------------------


def test_the_wrapped_method_is_vllms_and_installs_once():
    sched = pytest.importorskip("vllm.v1.core.sched.scheduler")
    import inspect

    from strands_decider_vllm.scheduling import install_whole_prompt_scheduling

    params = list(inspect.signature(sched.Scheduler._try_schedule_encoder_inputs).parameters)
    assert params[:5] == [
        "self",
        "request",
        "num_computed_tokens",
        "num_new_tokens",
        "encoder_compute_budget",
    ]
    install_whole_prompt_scheduling()
    fn = sched.Scheduler._try_schedule_encoder_inputs
    install_whole_prompt_scheduling()
    assert sched.Scheduler._try_schedule_encoder_inputs is fn and fn._strands_decider
    original = inspect.getsource(fn.__wrapped__)
    # The cuts the wrapper turns into "not this step": the encoder budget or cache is exhausted, or the item cannot
    # be split (disable_chunked_mm_input), each of which shortens num_new_tokens to before the image.
    assert "num_new_tokens = min(" in original and "can_allocate" in original
    # The scheduler requeues a request for which no token is scheduled, rather than running a zero-token step.
    schedule = inspect.getsource(sched.Scheduler.schedule)
    assert "if num_new_tokens == 0:" in schedule and "skip_request(request_queue)" in schedule
    # Releasing the references of the attempt is what the encoder cache manager's free does.
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager

    assert "free_encoder_input" in inspect.getsource(EncoderCacheManager.free)
    assert "request_cached_ids" in inspect.getsource(EncoderCacheManager.free_encoder_input)
