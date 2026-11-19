"""A prompt with images is scheduled whole or not at all, and one held back is not overtaken: vLLM's
own Scheduler, driven step by step with a full encoder cache. No GPU needed."""

from __future__ import annotations

import dataclasses
import json

import pytest

from strands_decider_vllm import scheduling

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
def scheduler(tmp_path, monkeypatch):
    """vLLM's Scheduler (or AsyncScheduler) with an encoder cache of ``cache`` embeddings, set after
    construction: vLLM would size it from a multimodal model's processor, which isn't needed here."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("vllm.v1.core.sched.scheduler")
    from vllm.config import CacheConfig, ModelConfig, SchedulerConfig, VllmConfig
    from vllm.v1.core.encoder_cache_manager import EncoderCacheManager
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.structured_output import StructuredOutputManager

    import strands_decider_vllm

    strands_decider_vllm.register()
    (tmp_path / "config.json").write_text(json.dumps(TINY_LLAMA))

    def make(cache, *, asynchronous=False, decider=True, max_model_len=1200):
        monkeypatch.setattr(scheduling, "serves_decider", lambda vllm_config: decider)
        model = ModelConfig(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            max_model_len=max_model_len,
            dtype="float32",
        )
        sched = SchedulerConfig(
            max_num_seqs=64,
            max_num_batched_tokens=max_model_len,
            max_model_len=max_model_len,
            is_encoder_decoder=False,
            enable_chunked_prefill=False,
            async_scheduling=asynchronous,
        )
        cache_config = CacheConfig(block_size=16)
        cache_config.num_gpu_blocks = 2048  # what the engine sets after profiling
        config = VllmConfig(model_config=model, scheduler_config=sched, cache_config=cache_config)
        spec = FullAttentionSpec(block_size=16, num_kv_heads=1, head_size=8, dtype=torch.float32)
        kv = KVCacheConfig(
            num_blocks=2048, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["l"], spec)]
        )
        s = sched.get_scheduler_cls()(
            vllm_config=config,
            kv_cache_config=kv,
            structured_output_manager=StructuredOutputManager(config),
            block_size=16,
        )
        s.encoder_cache_manager = EncoderCacheManager(cache_size=cache)
        s.max_num_encoder_input_tokens = max_model_len
        return s

    return make


def request(rid, images, length):
    """A request of ``length`` tokens with an image of IMAGE embeddings at each (id, offset)."""
    from vllm import SamplingParams
    from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
    from vllm.v1.request import Request

    features = [
        MultiModalFeatureSpec(
            data=None,
            modality="image",
            identifier=i,
            mm_position=PlaceholderRange(offset=offset, length=IMAGE),
            mm_hash=i,
        )
        for i, offset in images
    ]
    return Request(
        request_id=rid,
        prompt_token_ids=[5] * length,
        sampling_params=SamplingParams(max_tokens=1),
        pooling_params=None,
        mm_features=features,
    )


def finish(s, out):
    """The model runner's output for a step: one token per scheduled request."""
    from vllm.v1.outputs import ModelRunnerOutput

    ids = list(out.num_scheduled_tokens)
    kwargs: dict = {"req_ids": ids, "sampled_token_ids": [[7] for _ in ids]}
    if "req_id_to_index" in {f.name for f in dataclasses.fields(ModelRunnerOutput)}:
        kwargs["req_id_to_index"] = {r: i for i, r in enumerate(ids)}  # gone in vLLM 0.34
    s.update_from_output(out, ModelRunnerOutput(**kwargs))


@pytest.mark.parametrize("decider", [False, True])
def test_a_prompt_vllm_would_cut_before_an_image_waits_whole(scheduler, decider):
    # A (two images, 560 of 600 embeddings) is in flight; B shares A's first image and needs 280
    # more; C, text only, is behind B.
    s = scheduler(600, decider=decider)
    s.add_request(request("A", [("x", 10), ("a2", 300)], 600))
    first = s.schedule()
    b = request("B", [("x", 10), ("b2", 300)], 600)
    s.add_request(b)
    s.add_request(request("C", [], 300))
    second = s.schedule()  # before A's step is processed, as with asynchronous scheduling
    if not decider:
        import vllm

        # vLLM 0.33 alone schedules B up to its second image, half a prompt; 0.34 defers B whole.
        half = vllm.__version__.startswith("0.33.")
        assert second.num_scheduled_tokens == ({"B": 300, "C": 300} if half else {"C": 300})
        return
    assert second.num_scheduled_tokens == {"C": 300}
    ecm = s.encoder_cache_manager
    assert ecm.get_cached_input_ids(b) == set() and ecm.cached["x"] == {"A"}, "B's reference went"
    finish(s, first)
    third = s.schedule()
    assert third.num_scheduled_tokens == {"B": 600}
    assert third.scheduled_encoder_inputs["B"] == [1], "the shared image is reused"


@pytest.mark.parametrize("asynchronous", [False, True])
def test_a_held_back_request_is_not_starved_by_smaller_ones(scheduler, asynchronous):
    # A 4-image request (1,120 of 1,200 embeddings) arrives while three 1-image requests are in
    # flight, and two more arrive every step. Without holding the later ones back too, each step
    # fits smaller requests into the room left and the big one is never scheduled.
    s = scheduler(1200, asynchronous=asynchronous)
    n = 0

    def small():
        nonlocal n
        n += 1
        s.add_request(request(f"s{n}", [(f"img{n}", 10)], 300))

    for _ in range(3):
        small()
    in_flight = [s.schedule()]
    s.add_request(request("big", [(f"big{i}", 10 + 290 * i) for i in range(4)], 1150))
    for _ in range(2):
        small()
        small()
        if not asynchronous:
            finish(s, in_flight.pop())
        out = s.schedule()
        in_flight.append(out)
        if asynchronous:
            finish(s, in_flight.pop(0))
        if "big" in out.num_scheduled_tokens:
            assert out.num_scheduled_tokens["big"] == 1150, "whole"
            return
    pytest.fail("the 4-image request wasn't scheduled within two steps")
