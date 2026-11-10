"""A decision prompt with images is scheduled whole or not at all, and one held back for encoder room is not overtaken:
the wrapper around vLLM's encoder-input scheduling against a stand-in, then vLLM's own Scheduler driven step by step
with a full encoder cache (skipped without vLLM; no GPU needed)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from strands_decider_vllm import ARCH, ARCH_MULTIMODAL
from strands_decider_vllm import scheduling as scheduling_mod
from strands_decider_vllm.scheduling import wrap_schedule, wrap_try_schedule_encoder_inputs


class FakeEncoderCache:
    """The two calls the wrapper makes: a request's cached input ids, and releasing one of them."""

    def __init__(self):
        self.refs: dict[str, set[int]] = {}
        self.freed: list[tuple[str, int]] = []

    def get_cached_input_ids(self, request):
        return set(self.refs.get(request.request_id, ()))

    def free_encoder_input(self, request, input_id):
        self.refs[request.request_id].discard(input_id)
        self.freed.append((request.request_id, input_id))


class FakeScheduler:
    """vLLM's scheduler as the wrapper sees it: ``_try_schedule_encoder_inputs`` returns (inputs to encode, tokens to
    schedule, remaining encoder budget, external loads, duplicates); here it takes a reference on input 0 (a cache
    hit) and cuts the tokens to ``cut_to``."""

    def __init__(self, architectures, cut_to=None):
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(architectures=architectures)
        )
        self.encoder_cache_manager = FakeEncoderCache()
        self.cut_to = cut_to
        self.calls = 0

    def _try_schedule_encoder_inputs(
        self,
        request,
        num_computed_tokens,
        num_new_tokens,
        encoder_compute_budget,
        shift_computed_tokens=0,
    ):
        self.calls += 1
        if not request.mm_features:  # vLLM returns a request without images unchanged
            return [], num_new_tokens, encoder_compute_budget, [], []
        self.encoder_cache_manager.refs.setdefault(request.request_id, set()).add(0)
        n = num_new_tokens if self.cut_to is None else self.cut_to
        return [1], n, encoder_compute_budget - 516, [], []

    def schedule(self, requests):
        return [wrapped(self, r, 0, 900, 8192, shift_computed_tokens=0) for r in requests]


wrapped = wrap_try_schedule_encoder_inputs(FakeScheduler._try_schedule_encoder_inputs)
FakeScheduler.schedule = wrap_schedule(FakeScheduler.schedule)


def request(rid="r1", images=((10, 280),)):
    return SimpleNamespace(
        request_id=rid,
        mm_features=[
            SimpleNamespace(mm_position=SimpleNamespace(offset=o, length=n)) for o, n in images
        ],
    )


def test_a_whole_prompt_is_scheduled_as_vllm_plans_it():
    s = FakeScheduler([ARCH_MULTIMODAL])
    assert wrapped(s, request(), 0, 900, 8192, shift_computed_tokens=0) == (
        [1],
        900,
        7676,
        [],
        [],
    )
    assert s.encoder_cache_manager.freed == []


@pytest.mark.parametrize("cut_to", [3, 0, 899])
def test_a_prompt_cut_before_an_image_is_not_scheduled_at_all(cut_to):
    # vLLM would schedule the tokens in front of an image it cannot encode this step; the mask needs the whole
    # prompt in one step, so none of it runs now, and the encoder-cache reference the attempt took is released.
    s = FakeScheduler([ARCH_MULTIMODAL], cut_to=cut_to)
    assert wrapped(s, request(), 0, 900, 8192) == ([], 0, 8192, [], [])
    assert s.encoder_cache_manager.freed == [("r1", 0)]


def test_references_held_before_the_attempt_are_kept():
    s = FakeScheduler([ARCH_MULTIMODAL], cut_to=3)
    s.encoder_cache_manager.refs["r1"] = {0}  # taken in an earlier step
    wrapped(s, request(), 0, 900, 8192)
    assert s.encoder_cache_manager.freed == [] and s.encoder_cache_manager.refs["r1"] == {0}


def test_the_result_has_vllms_shape():
    class Longer(FakeScheduler):
        def _try_schedule_encoder_inputs(self, request, *args, **kwargs):
            out = FakeScheduler._try_schedule_encoder_inputs(self, request, *args, **kwargs)
            return (*out, [])

    w = wrap_try_schedule_encoder_inputs(Longer._try_schedule_encoder_inputs)
    assert w(Longer([ARCH_MULTIMODAL], cut_to=3), request(), 0, 900, 8192) == (
        [],
        0,
        8192,
        [],
        [],
        [],
    )


def test_later_image_requests_wait_behind_a_held_back_one():
    s = FakeScheduler([ARCH_MULTIMODAL], cut_to=3)
    out = s.schedule([request("big"), request("small"), request("text", images=())])
    assert out[0] == out[1] == ([], 0, 8192, [], [])
    assert s.calls == 2, "the small image request waits without asking vLLM, so it takes nothing"
    assert out[2][1] == 900, "a request without images is scheduled as vLLM plans it"
    s.cut_to = None
    assert s.schedule([request("small")])[0][1] == 900, "the next step starts clear"
    # An image wholly before the part a request computes (its state cached) does not hold it back.
    assert not scheduling_mod._images_in(request(), 290, 900)
    assert scheduling_mod._images_in(request(), 289, 900)


def test_other_models_keep_vllms_partial_schedule():
    s = FakeScheduler(["Gemma4ForConditionalGeneration"], cut_to=3)
    assert wrapped(s, request(), 0, 900, 8192) == ([1], 3, 7676, [], [])
    assert s.encoder_cache_manager.freed == []


def test_the_decoder_layout_is_covered_too():
    # It never has encoder inputs, so vLLM never cuts its prompts here; were it to, the same rule applies.
    s = FakeScheduler([ARCH], cut_to=3)
    assert wrapped(s, request(), 0, 900, 8192)[1] == 0


# --- vLLM's own Scheduler (skipped without vLLM) ----------------------------------------------------------------------


def test_the_wrapped_methods_are_vllms_and_install_once():
    sched = pytest.importorskip("vllm.v1.core.sched.scheduler")
    import inspect

    from strands_decider_vllm.scheduling import install_whole_prompt_scheduling

    install_whole_prompt_scheduling()
    fn, schedule = sched.Scheduler._try_schedule_encoder_inputs, sched.Scheduler.schedule
    install_whole_prompt_scheduling()
    assert sched.Scheduler._try_schedule_encoder_inputs is fn and fn._strands_decider
    assert sched.Scheduler.schedule is schedule and schedule._strands_decider
    params = list(inspect.signature(fn.__wrapped__).parameters)
    assert params[:5] == [
        "self",
        "request",
        "num_computed_tokens",
        "num_new_tokens",
        "encoder_compute_budget",
    ]


TINY_LLAMA = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "num_hidden_layers": 1,
    "vocab_size": 1024,
    "max_position_embeddings": 4096,
    "torch_dtype": "float32",
}
IMAGE = 280  # one image's embeddings, the most a Gemma 4 image has


@pytest.fixture
def vllm_scheduler(tmp_path):
    """vLLM's own Scheduler (AsyncScheduler with ``asynchronous``) for a decider engine, with an encoder cache of
    ``cache`` embeddings, built from a tiny local config. The encoder cache and budget are set after construction:
    vLLM would size them from a multimodal model's processor, which these tests do not need."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm.v1.core.sched.scheduler")
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    from strands_decider_vllm import engine as engine_mod
    from strands_decider_vllm.scheduling import install_whole_prompt_scheduling

    install_whole_prompt_scheduling()
    (tmp_path / "config.json").write_text(json.dumps(TINY_LLAMA))

    def make(cache, *, max_model_len=1200, asynchronous=False, decider=True, block=16):
        model_config = ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=max_model_len,
            dtype="float32",
        )
        scheduler_config = SchedulerConfig(
            max_num_seqs=64,
            max_num_batched_tokens=max_model_len,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            enable_chunked_prefill=False,
            async_scheduling=asynchronous,
        )
        cache_config = CacheConfig(block_size=block)
        cache_config.num_gpu_blocks = 2048  # what the engine sets after profiling
        vllm_config = VllmConfig(
            model_config=model_config, scheduler_config=scheduler_config, cache_config=cache_config
        )
        kv = KVCacheConfig(
            num_blocks=2048,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["layer"],
                    FullAttentionSpec(
                        block_size=block, num_kv_heads=1, head_size=8, dtype=torch.float32
                    ),
                )
            ],
        )
        s = scheduler_config.get_scheduler_cls()(
            vllm_config=vllm_config,
            kv_cache_config=kv,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=block,
        )
        s.encoder_cache_manager = EncoderCacheManager(cache_size=cache)
        s.max_num_encoder_input_tokens = max_model_len
        s.__dict__[engine_mod._SERVES_DECIDER] = decider
        return s

    return make


def mm_request(rid, images, length):
    """A request of ``length`` tokens with an image of IMAGE embeddings at each (identifier, offset)."""
    from vllm import SamplingParams
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.request import Request

    features = [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier=ident,
            mm_position=PlaceholderRange(offset=offset, length=IMAGE),
            mm_hash=ident,
        )
        for ident, offset in images
    ]
    return Request(
        request_id=rid,
        prompt_token_ids=[5] * length,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=features,
    )


def finish(s, out):
    """The model runner's output for a step: one sampled token per scheduled request."""
    from vllm.v1.outputs import ModelRunnerOutput

    ids = list(out.num_scheduled_tokens)
    s.update_from_output(
        out,
        ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={r: i for i, r in enumerate(ids)},
            sampled_token_ids=[[7] for _ in ids],
        ),
    )


@pytest.mark.parametrize("decider", [False, True])
def test_vllms_scheduler_cuts_before_an_image_and_the_wrapper_waits_for_the_whole_prompt(
    vllm_scheduler, decider
):
    # A has two images (560 of 600 embeddings) and is in flight; B shares A's first image and needs 280 more.
    s = vllm_scheduler(600, decider=decider)
    a = mm_request("A", [("x", 10), ("a2", 300)], 600)
    b = mm_request("B", [("x", 10), ("b2", 300)], 600)
    s.add_request(a)
    first = s.schedule()
    assert first.num_scheduled_tokens == {"A": 600}
    s.add_request(b)
    second = s.schedule()  # before A's step is processed, as with asynchronous scheduling
    if not decider:
        # vLLM alone schedules B up to its second image: half a prompt, which the mask cannot serve.
        assert second.num_scheduled_tokens == {"B": 300}
        return
    ecm = s.encoder_cache_manager
    assert "B" not in second.num_scheduled_tokens
    assert ecm.get_cached_input_ids(b) == set() and ecm.cached["x"] == {"A"}, (
        "the reference the attempt took on the shared image is released"
    )
    finish(s, first)
    third = s.schedule()
    assert third.num_scheduled_tokens == {"B": 600}, "whole, once A's images are released"
    assert third.scheduled_encoder_inputs["B"] == [1], "the shared image is reused"


@pytest.mark.parametrize("asynchronous, hold_others", [(False, True), (True, True), (True, False)])
def test_a_held_back_four_image_request_is_not_starved(
    vllm_scheduler, monkeypatch, asynchronous, hold_others
):
    # A 4-image request (1,120 of 1,200 embeddings) arrives while three 1-image requests are in flight, and two
    # more 1-image requests arrive every step. With asynchronous scheduling a step is planned before the previous
    # one is processed. The control (hold_others False: later image requests are not held back) shows the
    # starvation: each step fits smaller requests into the room left, and the big one never fits.
    if not hold_others:
        monkeypatch.setattr(scheduling_mod, "_images_in", lambda *a: False)
    s = vllm_scheduler(1200, asynchronous=asynchronous)
    n = 0

    def small():
        nonlocal n
        n += 1
        s.add_request(mm_request(f"s{n}", [(f"img{n}", 10)], 300))

    for _ in range(3):
        small()
    in_flight = [s.schedule()]
    assert len(in_flight[0].num_scheduled_tokens) == 3
    s.add_request(mm_request("big", [(f"big{i}", 10 + 290 * i) for i in range(4)], 1150))
    scheduled_at = None
    for step in range(1, 9):
        small()
        small()
        if not asynchronous:
            finish(s, in_flight.pop())
        out = s.schedule()
        in_flight.append(out)
        if asynchronous:
            finish(s, in_flight.pop(0))
        if "big" in out.num_scheduled_tokens:
            assert out.num_scheduled_tokens["big"] == 1150, "scheduled whole"
            scheduled_at = step
            break
    if hold_others:
        assert scheduled_at is not None and scheduled_at <= 2, scheduled_at
    else:
        assert scheduled_at is None, f"the control was scheduled at step {scheduled_at}"
