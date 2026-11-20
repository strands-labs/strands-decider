"""The plugin end to end on a GPU, which a maintainer runs before a release (README, Tests):

    pytest -m gpu tests/test_gpu.py

Tiny random-weight checkpoints of every layout, with Gemma 4's real configs, tokenizer and image
processor, are served by ``strands-decider-vllm serve`` and ``vllm serve``, and every answer is
checked against transformers' own model under the mask. Each test runs twice: with both head sizes
64, which FlashAttention 2 and 3 run, and with Gemma 4's two head sizes, which select FlashAttention
4 (Hopper or later). Needs Hub access to google/gemma-4-E2B-it and google/gemma-4-12B-it.

The tolerances are measured on H100, with headroom: a tiny model's bf16 answers differ from the bf16
reference on at most 1.7% of top-1 answers and by a mean |dp| of at most 0.0081, where a wrong mask
moves them by a mean of 0.15 to 0.36; a cached answer differs from its cold one by at most 0.0138.
"""

from __future__ import annotations

import base64
import contextlib
import functools
import io
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU"),
]

BASES = {  # each layout's Gemma 4 base, at the revision the tolerances were measured with
    "multimodal": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
    "unified": ("google/gemma-4-12B-it", "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"),
    "decoder": ("google/gemma-4-E2B-it", "3e22461f65e89153144f8adb70e3b8c2cc9845a7"),
}
SMALL_TEXT = dict(
    hidden_size=128,
    intermediate_size=256,
    num_hidden_layers=6,
    layer_types=["sliding_attention", "sliding_attention", "full_attention"] * 2,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=64,
    global_head_dim=64,
    sliding_window=64,  # shorter than most prompts, so a window left on would change the answers
)
MIN_TOP1, MAX_MEAN_DP, CACHED_DP = 0.95, 0.015, 0.02
DECODER = {"architectures": ["StrandsDeciderForCausalLM"], "use_bidirectional_attention": None}
LOOPBACK = {"VLLM_HOST_IP": "127.0.0.1", "GLOO_SOCKET_IFNAME": "lo", "NCCL_SOCKET_IFNAME": "lo"}
WORDS = "order refund delayed carrier label package damaged customer invoice photo seller".split()


@pytest.fixture(scope="module", params=["equal-heads", "gemma4-heads"])
def checkpoints(request, tmp_path_factory):
    """Tiny checkpoints of the three layouts, with equal head sizes or Gemma 4's two."""
    gemma4_heads = request.param == "gemma4-heads"
    if gemma4_heads and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Gemma 4's two head sizes need FlashAttention 4, on Hopper or later")
    root = tmp_path_factory.mktemp(request.param)
    return {layout: tiny_checkpoint(root / layout, layout, gemma4_heads) for layout in BASES}


def tiny_checkpoint(out: Path, layout: str, gemma4_heads: bool) -> Path:
    """The layout's base config with a few small layers, transformers' seeded random weights, a
    random readout and a decision_config.json. The multimodal base gets the "vision" mode that the
    12B, 26B-A4B and 31B bases set, and the plugin turns off."""
    import transformers
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file
    from strands_decider.core import CODE_TOKEN_IDS, CODES

    repo, revision = BASES[layout]
    files = ("config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json")
    base = {f: Path(hf_hub_download(repo, f, revision=revision)) for f in files}
    config = json.loads(base["config.json"].read_text())
    text = {**config["text_config"], **SMALL_TEXT}
    text.pop("per_layer_config", None)
    if gemma4_heads:
        text["global_head_dim"] = 2 * text["head_dim"]
    if layout == "unified":
        text.update(num_kv_shared_layers=0, num_global_key_value_heads=1)
    else:  # E2B's KV-shared layers and per-layer embeddings
        text.update(num_kv_shared_layers=2, hidden_size_per_layer_input=16)
    torch.manual_seed(0)
    if layout == "decoder":
        cfg = transformers.Gemma4TextConfig(**{k: v for k, v in text.items() if k != "model_type"})
        cfg.architectures = ["Gemma4TextModel"]
        model = transformers.Gemma4TextModel(cfg)
    else:
        vision = dict(config["vision_config"])
        if layout == "multimodal":
            text["use_bidirectional_attention"] = "vision"
            vision.update(hidden_size=64, intermediate_size=128, num_hidden_layers=2, head_dim=32)
            vision.update(num_attention_heads=2, num_key_value_heads=2, global_head_dim=32)
            config_cls = transformers.Gemma4Config
            model_cls = transformers.Gemma4ForConditionalGeneration
        else:
            hidden = SMALL_TEXT["hidden_size"]
            vision.update(mm_embed_dim=hidden, output_proj_dims=hidden)
            config_cls = transformers.Gemma4UnifiedConfig
            model_cls = transformers.Gemma4UnifiedForConditionalGeneration
        full = {**config, "text_config": text, "vision_config": vision, "audio_config": None}
        cfg = config_cls(**{k: v for k, v in full.items() if k != "model_type"})
        cfg.architectures = config["architectures"]
        model = model_cls(cfg)
    smooth_attention(model)
    model.to(torch.bfloat16).save_pretrained(out)
    (out / "generation_config.json").unlink(missing_ok=True)
    shutil.copy(base["tokenizer.json"], out)
    if layout != "decoder":
        shutil.copy(base["processor_config.json"], out)
    tokenizer_config = json.loads(base["tokenizer_config.json"].read_text())
    tokenizer_config.pop("chat_template", None)
    (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    readout = (torch.randn(255, SMALL_TEXT["hidden_size"]) * 0.25).to(torch.bfloat16)
    save_file({"weight": readout}, str(out / "readout.safetensors"))
    decision = {"codes": list(CODES), "token_ids": list(CODE_TOKEN_IDS), "temperature": 1.0}
    decision.update(attention_mode="prefix_bidir", pooling="last", format_version=1)
    if layout != "decoder":  # an intermediate decoder's file predates the core's other fields
        decision.update(
            interface="strands-decider",
            prompt="strands-chat",
            inputs=["text", "image"],
            final_logit_softcapping=text["final_logit_softcapping"],
            max_length_validated=4096,
            provenance={"base_model": repo, "base_revision": revision},
        )
    (out / "decision_config.json").write_text(json.dumps(decision))
    return out


def smooth_attention(model) -> None:
    """Sets every q_norm and k_norm weight so that attention logits have a standard deviation of
    1.5. Gemma 4 scales q.k by 1.0, so with its norms at 1 a random model's logits have one of
    sqrt(head size), a near-hard maximum that bf16 rounding flips; these keep attention smooth, and
    far enough from uniform that the mask changes the answers."""
    with torch.no_grad():
        for name, m in model.named_modules():
            if name.endswith(("q_norm", "k_norm")) and getattr(m, "weight", None) is not None:
                m.weight.fill_((1.5 / math.sqrt(m.weight.numel())) ** 0.5)


class Reference:
    """transformers' own model for the layout, in bf16 with SDPA, one prompt per forward pass, with
    the prefix_bidir mask on every layer, and the core's readout at the last token."""

    def __init__(self, checkpoint: Path) -> None:
        import transformers
        from safetensors.torch import load_file

        config = json.loads((checkpoint / "config.json").read_text())
        architecture = config["architectures"][0]
        model = getattr(transformers, architecture).from_pretrained(
            checkpoint, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        self.decoder = (model if architecture == "Gemma4TextModel" else model.model).cuda().eval()
        if architecture == "Gemma4UnifiedForConditionalGeneration":
            self.processor = transformers.Gemma4UnifiedImageProcessor.from_pretrained(checkpoint)
        elif architecture == "Gemma4ForConditionalGeneration":
            self.processor = transformers.Gemma4ImageProcessorPil.from_pretrained(checkpoint)
        self.readout = load_file(str(checkpoint / "readout.safetensors"))["weight"].cuda()
        self.softcap = config.get("text_config", config)["final_logit_softcapping"]
        self.temperature = json.loads((checkpoint / "decision_config.json").read_text())[
            "temperature"
        ]

    @torch.inference_mode()
    def probabilities(self, q, images=()) -> list[float]:
        """A rendered question's code probabilities; ``images`` are its decoded images."""
        from strands_decider.core import resize_image, resized_size

        j = torch.arange(len(q.ids), device="cuda")
        mask = ((j[None, :] < q.state_len) | (j[:, None] >= q.state_len))[None, None]
        kwargs = dict(
            input_ids=torch.tensor([q.ids], device="cuda"),
            attention_mask={"full_attention": mask, "sliding_attention": mask},
            use_cache=False,
        )
        if images:
            resized = [resize_image(im, *resized_size(*im.size)) for im in images]
            pixels = self.processor(images=resized, do_resize=False, return_tensors="pt")
            counts = torch.as_tensor(pixels["num_soft_tokens_per_image"]).view(-1).tolist()
            assert counts == [im.tokens for im in q.images], "the processor counts as the core"
            kwargs.update(
                pixel_values=pixels["pixel_values"].to("cuda", torch.bfloat16),
                image_position_ids=pixels["image_position_ids"].cuda(),
            )
        h = self.decoder(**kwargs).last_hidden_state[0, -1]
        z = torch.nn.functional.linear(h, self.readout).float()
        z = torch.tanh(z / self.softcap) * self.softcap / self.temperature
        return torch.softmax(z[: len(q.code_token_ids)], dim=-1).tolist()


def photo(rng: random.Random, width: int, height: int, fmt: str) -> str:
    """A synthetic photo as base64, different for every draw of ``rng``."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.new("RGB", (width, height), tuple(rng.randrange(256) for _ in range(3)))
    draw = ImageDraw.Draw(img)
    for _ in range(30):
        x, y = rng.randrange(width), rng.randrange(height)
        corner = (x + rng.randrange(1, width // 2 + 2), y + rng.randrange(1, height // 2 + 2))
        draw.rectangle((x, y, *corner), fill=tuple(rng.randrange(256) for _ in range(3)))
    img = img.filter(ImageFilter.GaussianBlur(2))
    buf = io.BytesIO()
    (img.convert("P", palette=Image.ADAPTIVE) if fmt == "GIF" else img).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def words(rng: random.Random, n: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(n))


def questions(rng: random.Random) -> dict:
    options = dict.fromkeys(rng.sample(WORDS, rng.randint(2, 6)))
    return {
        "kind": {"type": "choice", "instructions": "What is it about?", "criteria": options},
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "tone": {"type": "score", "instructions": "How upset?", "criteria": ["calm", "angry"]},
    }


@pytest.fixture(scope="module")
def image_requests():
    """Requests with short, empty and long states and 0 to 4 images of mixed sizes, aspect ratios
    and formats."""
    rng = random.Random(0)
    sizes = [(640, 480), (480, 640), (300, 40), (64, 48), (1000, 1000), (96, 1200), (1600, 1200)]
    formats = ["PNG", "JPEG", "WEBP", "GIF"]
    return [
        {
            "state": words(rng, rng.choice([0, 12, 300])),
            "images": [photo(rng, *rng.choice(sizes), rng.choice(formats)) for _ in range(i % 5)],
            "questions": questions(rng),
        }
        for i in range(20)
    ]


@pytest.fixture(scope="module")
def cache_requests():
    """A photo, then another of its size, then a further question and a repeat on the first."""
    rng = random.Random(1)
    first, other, state = photo(rng, 640, 480, "PNG"), photo(rng, 640, 480, "PNG"), words(rng, 600)
    q1, q2 = {"urgent": questions(rng)["urgent"]}, {"kind": questions(rng)["kind"]}
    bodies = [(first, q1), (other, q1), (first, q2), (first, q1)]
    return [{"state": state, "images": [image], "questions": q} for image, q in bodies]


@pytest.fixture(scope="module")
def longest_request():
    """The longest prompt of this module, which the multimodal server's --max-length is set to."""
    rng = random.Random(2)
    images = [photo(rng, 1000, 1000, "JPEG") for _ in range(4)]
    return {"state": words(rng, 900), "images": images, "questions": questions(rng)}


def rendered_questions(body: dict, checkpoint: Path) -> dict:
    from strands_decider.core import Request, render

    return render(Request.model_validate(body), tokenizer(checkpoint)).questions


@functools.cache
def tokenizer(checkpoint: Path):
    from strands_decider.core import load_tokenizer

    return load_tokenizer(checkpoint)


class Server:
    """A server in its own process group, bound to loopback with vLLM's sockets there too."""

    def __init__(self, command: list[str], log: Path) -> None:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self.url, self.log = f"http://127.0.0.1:{port}", log
        with open(log, "w") as out:
            self.proc = subprocess.Popen(
                [*command, "--host", "127.0.0.1", "--port", str(port)],
                stdout=out,
                stderr=subprocess.STDOUT,
                env={**os.environ, **LOOPBACK},
                start_new_session=True,
            )

    def wait_ready(self) -> None:
        deadline = time.monotonic() + 1200
        while not healthy(self.url):
            assert self.proc.poll() is None, f"exited:\n{self.log.read_text()[-3000:]}"
            assert time.monotonic() < deadline, f"not ready:\n{self.log.read_text()[-3000:]}"
            time.sleep(2)

    def stop(self) -> None:
        """Stops the server's process group: vLLM's engine processes may outlive the server."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            self.proc.wait(60)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, signal.SIGKILL)  # whatever didn't stop
        self.proc.wait()


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            return r.status == 200
    except OSError:
        return False


def post(url: str, body: dict) -> tuple[int, dict | str]:
    data = json.dumps(body).encode()
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode(errors="replace")


def vllm_serve(checkpoint: Path, *flags: str, hf_overrides: dict = DECODER) -> list[str]:
    """The README's ``vllm serve`` command for a bare decoder, at a 4,096-token context."""
    return [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", str(checkpoint),
        "--served-model-name", "m", "--dtype", "bfloat16", "--seed", "0",
        "--hf-overrides", json.dumps(hf_overrides),
        "--max-model-len", "4097", "--max-num-batched-tokens", "4097",
        "--no-enable-chunked-prefill", "--enable-prefix-caching", "--enable-prompt-tokens-details",
        "--generation-config", "vllm", "--logprobs-mode", "processed_logprobs",
        "--max-logprobs", "255", *flags,
    ]  # fmt: skip


def test_vllm_serve_refuses_unsafe_settings_at_startup(checkpoints, tmp_path):
    # First in the module, before the servers below take their share of the GPU.
    decoder, multimodal = checkpoints["decoder"], checkpoints["multimodal"]
    cases = {
        "--no-enable-chunked-prefill": ["--enable-chunked-prefill"],
        "--max-num-scheduled-tokens 2048 <": ["--max-num-scheduled-tokens", "2048"],
        "cudagraph_mode FULL": ["--compilation-config", '{"cudagraph_mode": "FULL"}'],
        "--logprobs-mode raw_logprobs": ["--logprobs-mode", "raw_logprobs"],
        "--dtype torch.float16": ["--dtype", "float16"],
        "--kv-cache-dtype fp8": ["--kv-cache-dtype", "fp8"],
        "--quantization fp8": ["--quantization", "fp8"],
        "--enable-prompt-embeds": ["--enable-prompt-embeds"],
    }
    commands = {message: vllm_serve(decoder, *flags) for message, flags in cases.items()}
    commands["PrefixLM mode is on"] = vllm_serve(
        multimodal, hf_overrides={"architectures": ["StrandsDeciderForConditionalGeneration"]}
    )
    commands["are not on FLASH_ATTN"] = vllm_serve(decoder, "--attention-backend", "TRITON_ATTN")
    memory = ["--gpu-memory-utilization", "0.05"]
    # All at once: each stops at the model's construction, before it loads weights.
    servers = {
        m: Server([*c, *memory], tmp_path / f"{i}.log") for i, (m, c) in enumerate(commands.items())
    }
    try:
        for message, server in servers.items():
            server.proc.wait(900)
            assert server.proc.returncode != 0 and message in server.log.read_text(), message
    finally:  # a server that started after all must not outlive the test
        for server in servers.values():
            server.stop()


@pytest.fixture(scope="module")
def servers(checkpoints, image_requests, cache_requests, longest_request, tmp_path_factory):
    """The plugin's server on the multimodal and unified checkpoints, and ``vllm serve`` on the
    decoder, side by side on one GPU."""
    lengths = {
        id(body): max(
            len(q.ids) for q in rendered_questions(body, checkpoints["multimodal"]).values()
        )
        for body in [*image_requests, *cache_requests, longest_request]
    }
    max_length = lengths.pop(id(longest_request))
    assert max(lengths.values()) < max_length
    plugin = [sys.executable, "-m", "strands_decider_vllm", "serve"]
    memory = ["--gpu-memory-utilization", "0.15"]
    commands = {
        "multimodal": [*plugin, str(checkpoints["multimodal"]), "--max-length", str(max_length)],
        "unified": [*plugin, str(checkpoints["unified"])],
        "decoder": vllm_serve(checkpoints["decoder"]),
    }
    logs, started = tmp_path_factory.mktemp("logs"), {}
    try:
        # One at a time: vLLM's memory profiling at startup takes the GPU's other use as its own.
        for name, command in commands.items():
            started[name] = Server([*command, *memory], logs / f"{name}.log")
            started[name].wait_ready()
        yield started
    finally:
        for server in started.values():
            server.stop()


def agreement(pairs: list[tuple[list[float], list[float]]]) -> tuple[float, float]:
    """Top-1 agreement over (answer, reference) pairs, and the mean of each pair's largest |dp|."""
    top1 = [p.index(max(p)) == r.index(max(r)) for p, r in pairs]
    dps = [max(abs(a - b) for a, b in zip(p, r, strict=True)) for p, r in pairs]
    return sum(top1) / len(pairs), sum(dps) / len(dps)


def vector(answer: dict) -> list[float]:
    """An answer's probabilities in code order: a noul's (false, true), the others' as listed."""
    if answer["type"] == "noul":
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


@pytest.mark.parametrize("layout", ["multimodal", "unified"])
def test_the_plugins_server_answers_as_the_reference(checkpoints, servers, image_requests, layout):
    from PIL import Image

    url = servers[layout].url + "/v1/systemone"
    with ThreadPoolExecutor(8) as pool:  # concurrent, so that the engine batches them
        responses = list(pool.map(lambda body: post(url, body), image_requests))
    reference = Reference(checkpoints[layout])
    pairs = []
    for body, (status, response) in zip(image_requests, responses, strict=True):
        assert status == 200, response
        for name, q in rendered_questions(body, checkpoints[layout]).items():
            images = [Image.open(io.BytesIO(im.data)).convert("RGB") for im in q.images]
            pairs.append((vector(response["answers"][name]), reference.probabilities(q, images)))
    top1, mean_dp = agreement(pairs)
    assert top1 >= MIN_TOP1 and mean_dp <= MAX_MEAN_DP, (top1, mean_dp, len(pairs))


def test_a_prompt_of_max_length_is_answered_and_a_longer_one_refused(servers, longest_request):
    # vLLM refuses a prompt that fills its context: hence --max-model-len = --max-length + 1.
    url = servers["multimodal"].url + "/v1/systemone"
    assert post(url, longest_request)[0] == 200
    status, detail = post(url, {**longest_request, "state": longest_request["state"] + " order"})
    assert status == 422 and "exceeds the limit" in detail


def test_the_prefix_cache_reuses_a_states_blocks_and_nothing_an_image_changes(
    checkpoints, servers, cache_requests
):
    url = servers["multimodal"].url + "/v1/systemone"
    cold, other_image, further, repeat = [post(url, body)[1] for body in cache_requests]
    q = rendered_questions(cache_requests[0], checkpoints["multimodal"])["urgent"]
    cached = [r["usage"]["cached_tokens"] for r in (cold, other_image, further, repeat)]
    assert cached[:2] == [0, 0], "a different photo of the same size reuses nothing"
    # vLLM reuses whole blocks of A tokens: an exact repeat all but its last token's.
    block = next((a for a in (16, 32, 64, 128, 256) if cached[3] == (len(q.ids) - 1) // a * a), 0)
    assert block, f"an exact repeat of {len(q.ids)} tokens reused {cached[3]}"
    assert cached[2] == q.state_len // block * block, "a further question reuses the state's blocks"
    dp = abs(repeat["answers"]["urgent"]["noul"] - cold["answers"]["urgent"]["noul"])
    assert dp <= CACHED_DP, "a cached answer is the cold one"


def test_vllm_serve_answers_token_ids_as_the_reference(checkpoints, servers, image_requests):
    url = servers["decoder"].url + "/v1/completions"
    reference = Reference(checkpoints["decoder"])
    texts = [{**body, "images": []} for body in image_requests[:12]]
    questions = [
        q for body in texts for q in rendered_questions(body, checkpoints["decoder"]).values()
    ]
    pairs = []
    for q in questions:
        status, response = post(url, completion(q))
        assert status == 200, response
        top = response["choices"][0]["logprobs"]["top_logprobs"][0]
        e = [math.exp(top[f"token_id:{t}"]) for t in q.code_token_ids]
        pairs.append(([x / sum(e) for x in e], reference.probabilities(q)))
    top1, mean_dp = agreement(pairs)
    assert top1 >= MIN_TOP1 and mean_dp <= MAX_MEAN_DP, (top1, mean_dp, len(pairs))
    status, response = post(url, {**completion(questions[0]), "temperature": 0.7})
    assert status == 400 and "strands-decider: temperature" in response
    assert healthy(servers["decoder"].url), "a refusal leaves the engine serving"


def completion(q) -> dict:
    """The README's /v1/completions request for one rendered question."""
    return {
        "model": "m",
        "prompt": list(q.ids),
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "logprobs": len(q.code_token_ids),
        "allowed_token_ids": list(q.code_token_ids),
        "return_tokens_as_token_ids": True,
        "vllm_xargs": {"prefix_end": q.state_len},
    }


def test_every_server_listens_on_loopback_only(servers):
    import psutil

    for name, server in servers.items():
        root = psutil.Process(server.proc.pid)
        listening = {
            c.laddr.ip
            for p in [root, *root.children(recursive=True)]
            for c in p.net_connections(kind="inet")
            if c.status == psutil.CONN_LISTEN
        }
        assert listening and listening <= {"127.0.0.1", "::1", "::ffff:127.0.0.1"}, (
            name,
            listening,
        )
