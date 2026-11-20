"""What the server starts with: vLLM's engine arguments for a checkpoint, the refusals of checkpoints
it can't serve, and the loopback defaults."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import huggingface_hub
import pytest
from strands_decider.core.images import MAX_IMAGE_BYTES

from strands_decider_vllm import SERVER_CONFIG_KEY, cli
from strands_decider_vllm.cli import parser
from strands_decider_vllm.settings import settings

VISION_OFF = {"use_bidirectional_attention": None}
MULTIMODAL = ("StrandsDeciderForConditionalGeneration", {"text_config": VISION_OFF})


@pytest.mark.parametrize(
    "inputs, layout, hf_overrides, limit_mm_per_prompt",
    [
        (("text", "image"), "multimodal", MULTIMODAL, {"image": 8193, "video": 0, "audio": 0}),
        (("text",), "multimodal", MULTIMODAL, {"image": 0, "video": 0, "audio": 0}),
        (("text",), "decoder", ("StrandsDeciderForCausalLM", VISION_OFF), {}),
    ],
)
def test_the_engine_gets_the_plugins_architecture_and_flags(
    make_settings, inputs, layout, hf_overrides, limit_mm_per_prompt
):
    e = make_settings(inputs=inputs, layout=layout).engine_args
    architecture, overrides = hf_overrides
    assert e.hf_overrides == {"architectures": [architecture], **overrides}
    assert e.limit_mm_per_prompt == limit_mm_per_prompt
    assert (e.dtype, e.enable_chunked_prefill, e.enable_prefix_caching) == ("bfloat16", False, True)
    assert (e.logprobs_mode, e.max_logprobs) == ("processed_logprobs", 255)
    assert e.prefix_caching_hash_algo == "sha256" and e.additional_config == {
        SERVER_CONFIG_KEY: True
    }


@pytest.mark.parametrize(
    "flags, lengths",
    [
        ((), (8192, 8193, 8193)),
        (("--max-length", "4096"), (4096, 4097, 4097)),
        (("--max-num-batched-tokens", "16384"), (8192, 8193, 16384)),
    ],
)
def test_the_context_leaves_a_position_for_the_answer(make_settings, flags, lengths):
    # vLLM refuses a prompt that fills the context (tests/test_gpu.py).
    s = make_settings(*flags)
    e = s.engine_args
    assert (s.max_length, e.max_model_len, e.max_num_batched_tokens) == lengths


def test_the_image_token_budget_and_its_limit(make_settings):
    """The checkpoint's budget is the default and the limit; vLLM plans its encoder cache for the
    limit. A higher limit warns, a lower one lowers the default, and a text-only checkpoint plans
    for no images."""
    s = make_settings()
    assert (s.token_budget, s.max_token_budget) == (280, 280)
    assert s.engine_args.mm_processor_kwargs == {"max_soft_tokens": 280}
    with pytest.warns(UserWarning, match="past the checkpoint's 280"):
        s = make_settings("--max-image-token-budget", "1120")
    assert (s.token_budget, s.max_token_budget) == (280, 1120)
    assert s.engine_args.mm_processor_kwargs == {"max_soft_tokens": 1120}
    s = make_settings("--max-image-token-budget", "140")
    assert (s.token_budget, s.max_token_budget) == (140, 140)
    assert make_settings(inputs=("text",)).engine_args.mm_processor_kwargs is None
    with pytest.raises(SystemExit):
        make_settings("--max-image-token-budget", "300")


def test_the_body_limit_follows_the_checkpoints_inputs(make_settings):
    image = 4 * math.ceil(MAX_IMAGE_BYTES / 3)  # one of the core's largest images in base64
    assert 4 * image + 8 * 2**20 <= make_settings().max_body < 5 * image
    assert make_settings(inputs=("text",)).max_body == 8 * 2**20
    s = make_settings("--max-body-mb", "300", "--max-buffered-mb", "64")
    assert s.max_buffered == s.max_body == 300 * 2**20, "the budget holds any one body"


@pytest.mark.parametrize(
    "file, change, problem",
    [
        ("decision_config.json", {"attention_mode": "causal"}, "attention_mode"),
        ("config.json", {"architectures": ["LlamaForCausalLM"]}, "no Strands Decider layout"),
        ("processor_config.json", {"image_processor": {"patch_size": 14}}, "image processor"),
    ],
)
def test_a_checkpoint_it_cant_serve_stops_the_server(checkpoint, file, change, problem):
    path = Path(checkpoint())
    content = json.loads((path / file).read_text())
    (path / file).write_text(json.dumps({**content, **change}))
    with pytest.raises(SystemExit, match=problem):
        settings(parser().parse_args(["serve", str(path)]))


def test_a_hub_checkpoint_is_served_at_the_commit_its_files_were_read_at(
    monkeypatch, checkpoint, tmp_path_factory
):
    commit = "0123456789" * 4
    snapshot = tmp_path_factory.mktemp("hub") / "snapshots" / commit
    snapshot.parent.mkdir()
    Path(checkpoint()).rename(snapshot)
    calls = []
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda *a, **k: calls.append(k) or str(snapshot)
    )
    # Offline, vLLM would look the repo id up in the local cache.
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", False)
    e = settings(parser().parse_args(["serve", "org/model", "--revision", "main"])).engine_args
    assert (e.model, e.revision) == ("org/model", commit)
    assert calls[0]["revision"] == "main" and "decision_config.json" in calls[0]["allow_patterns"]


def test_the_server_binds_loopback_and_vllms_sockets_stay_there(monkeypatch):
    assert parser().parse_args(["serve", "/ckpt"]).host == "127.0.0.1"
    monkeypatch.setenv("VLLM_HOST_IP", "10.0.0.1")  # the operator's own value is kept
    for key in ("GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr("strands_decider_vllm.server.serve", lambda args: None)
    cli.main(["serve", "/ckpt"])
    assert os.environ["VLLM_HOST_IP"] == "10.0.0.1"
    assert os.environ["GLOO_SOCKET_IFNAME"] == os.environ["NCCL_SOCKET_IFNAME"] == "lo"
