"""``strands-decider-vllm serve``: the System One request, images included, on vLLM's engine in process.

vLLM 0.33's ``/v1/completions`` takes token ids but no images, its chat endpoints render a chat template (which the
specification forbids for these prompts), and its token-in-token-out endpoint lets a client name an image by an id of
its choosing, which vLLM then uses as the image's cache key. This server takes v1's request instead and does the rest:

1. It renders the request with the core library (``strands_decider.core.render``) and the checkpoint's prompt
   template. Each image becomes ``<|image>``, N ``<|image|>`` and ``<image|>`` first in the state, and S counts them.
2. It decodes each image as v1 does (first frame, EXIF orientation, transparency onto white, RGB) and resizes it to
   the specification's W x H with Pillow's bicubic resampling, the reference resize for every size (section 11.1).
   vLLM's processor then runs with its own resize off (``do_resize=False``), so the pixels the encoder gets are the
   reference's; W and H are multiples of 48, and the processor's count of soft tokens is N.
3. For each question it sends vLLM the prompt with each image's line collapsed to the single ``<|image|>`` that vLLM's
   Gemma 4 processor expands, the images as multimodal data, the image file's sha256 as each image's id (so vLLM's
   multimodal hash, and the plugin's cache digest, is the specification's key: the file's content), and S in
   ``vllm_xargs.prefix_end``. The first question runs alone and the others then together, so that they reuse the
   state's cached blocks.
4. It checks that vLLM's expanded prompt is the rendered one, token for token, so that N and S are the core's.
5. It reads each question's code probabilities from the processed logprobs and answers in v1's response format
   (``strands_decider.core.read``).

Endpoints: ``POST /v1/systemone`` and ``POST /invocations`` (the same handler, for SageMaker), ``GET /ping`` and
``GET /health``. A request the server or the engine refuses gets HTTP 400 with ``{"detail": "..."}``; the engine keeps
serving. The server has no authentication and binds 127.0.0.1 unless told otherwise: expose it beyond the machine only
behind the deployment's own authentication, such as SageMaker's IAM-signed endpoint or an authenticating proxy.
"""

from __future__ import annotations

import asyncio
import io
import json
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import ARCH_FOR_BASE, PREFIX_END_KEY, SERVER_CONFIG_KEY, __version__, logger

#: Gemma 4's image control tokens.
IMAGE_OPEN, IMAGE_SOFT, IMAGE_CLOSE = "<|image>", "<|image|>", "<image|>"
#: The image processor settings the core's N is computed with (specification, section 3.2).
PROCESSOR_SETTINGS = {"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280}
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
MAX_IMAGES = 4
LOOPBACK = ("127.0.0.1", "::1", "localhost")


class Refused(ValueError):
    """A request the server will not answer (HTTP 400)."""


class Mismatch(RuntimeError):
    """vLLM's expanded prompt differs from the rendered one (HTTP 500: a bug, never the client's)."""


@dataclass(frozen=True)
class Checkpoint:
    """The files the server reads beside the engine: decision_config.json, config.json, processor_config.json and
    the tokenizer, from a local directory or a Hub repo."""

    model: str
    revision: str | None
    files: dict[str, Path]

    @classmethod
    def open(cls, model: str, revision: str | None = None) -> Checkpoint:
        names = (
            "decision_config.json",
            "config.json",
            "processor_config.json",
            "tokenizer.json",
        )
        files: dict[str, Path] = {}
        if Path(model).is_dir():
            for n in names:
                if (Path(model) / n).is_file():
                    files[n] = Path(model) / n
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            for n in names:
                try:
                    files[n] = Path(hf_hub_download(model, n, revision=revision))
                except EntryNotFoundError:
                    pass
        for n in ("decision_config.json", "config.json", "tokenizer.json"):
            if n not in files:
                raise SystemExit(f"strands-decider-vllm serve: {model!r} has no {n}")
        return cls(model, revision, files)

    def json(self, name: str) -> dict:
        return json.loads(self.files[name].read_text())


def plugin_architecture(config: dict) -> str:
    """The plugin's architecture for a checkpoint, from the base architecture its config.json names."""
    archs = config.get("architectures") or []
    for a in archs:
        if a in ARCH_FOR_BASE:
            return ARCH_FOR_BASE[a]
    raise SystemExit(
        f"strands-decider-vllm serve: config.json names {archs}; a Strands Decider checkpoint is a Gemma 4 model "
        f"({', '.join(ARCH_FOR_BASE)})"
    )


def processor_problem(processor_config: dict | None) -> str | None:
    """Why image requests must be refused for this image processor configuration, or None."""
    if processor_config is None:
        return "the checkpoint has no processor_config.json"
    ip = processor_config.get("image_processor") or {}
    bad = {k: ip.get(k) for k, v in PROCESSOR_SETTINGS.items() if ip.get(k) != v}
    if bad:
        return (
            f"the checkpoint's image processor has {bad}; N is defined for {PROCESSOR_SETTINGS} "
            "(specification, section 3.2)"
        )
    return None


def decode_image(data: bytes) -> Any:
    """One image file as v1's ``vision.decode_image`` decodes it: its first frame, upright by its EXIF orientation,
    transparent areas composited onto white, as RGB. The core has already checked its format and size."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
    except Exception as e:
        raise Refused(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    try:
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise Refused(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(white, rgba)
    return img.convert("RGB")


def resized_size(width: int, height: int) -> tuple[int, int]:
    """Section 3.2's W and H: the largest size with sides that are multiples of 48 pixels and area at most
    645,120 pixels at the image's aspect ratio, in IEEE double precision as the specification writes it. The image
    then has (W / 48) x (H / 48) soft tokens, the core's N."""
    side = PROCESSOR_SETTINGS["pooling_kernel_size"] * PROCESSOR_SETTINGS["patch_size"]
    target = PROCESSOR_SETTINGS["max_soft_tokens"] * side * side
    f = math.sqrt(target / (height * width))
    h = math.floor((f * height) / side) * side
    w = math.floor((f * width) / side) * side
    if h == 0:
        h, w = (
            side,
            min(math.floor(width / height) * side, PROCESSOR_SETTINGS["max_soft_tokens"] * side),
        )
    elif w == 0:
        w, h = (
            side,
            min(math.floor(height / width) * side, PROCESSOR_SETTINGS["max_soft_tokens"] * side),
        )
    return w, h


def prepare_image(data: bytes, tokens: int) -> Any:
    """The decoded image, resized to W x H with Pillow's bicubic resampling (specification, section 11.1)."""
    from PIL import Image

    img = decode_image(data)
    w, h = resized_size(*img.size)
    side = PROCESSOR_SETTINGS["pooling_kernel_size"] * PROCESSOR_SETTINGS["patch_size"]
    if (w // side) * (h // side) != tokens:
        raise Mismatch(
            f"a {img.size[0]} x {img.size[1]} image resizes to {w} x {h}, not to the core's {tokens} tokens"
        )
    return img.resize((w, h), Image.Resampling.BICUBIC)


def collapse(state_ids: tuple[int, ...], images: Any, image_id: int) -> list[int]:
    """The state ids with each image's line, ``<|image>`` N ``<|image|>`` ``<image|>``, replaced by the one
    ``<|image|>`` that vLLM's Gemma 4 processor expands back to it."""
    out: list[int] = []
    pos = 0
    for im in images:
        out.extend(state_ids[pos : im.start - 1])
        out.append(image_id)
        pos = im.end + 1
    out.extend(state_ids[pos:])
    return out


def probabilities(logprobs: dict, code_ids: tuple[int, ...]) -> list[float]:
    """The question's code probabilities from the sampled position's processed logprobs (renormalized over its k
    codes, which ``allowed_token_ids`` already restricts them to)."""
    lp = [logprobs[t].logprob if t in logprobs else -math.inf for t in code_ids]
    m = max(lp)
    if not math.isfinite(m):
        raise Mismatch("no code of the question has a finite logprob")
    e = [math.exp(x - m) for x in lp]
    s = sum(e)
    return [x / s for x in e]


class Decider:
    """The engine and what the server needs around it."""

    def __init__(
        self,
        engine: Any,
        checkpoint: Checkpoint,
        *,
        served_name: str,
        max_length: int,
        prefix_caching: bool,
        image_problem: str | None,
    ) -> None:
        from strands_decider.core import DecisionConfig
        from tokenizers import Tokenizer

        self.engine = engine
        self.checkpoint = checkpoint
        self.config = DecisionConfig.from_dict(checkpoint.json("decision_config.json"))
        self.tokenizer = Tokenizer.from_file(str(checkpoint.files["tokenizer.json"]))
        self.image_id = self.tokenizer.token_to_id(IMAGE_SOFT)
        self.served_name = served_name
        self.max_length = max_length
        self.prefix_caching = prefix_caching
        self.image_problem = image_problem

    @property
    def takes_images(self) -> bool:
        return "image" in self.config.inputs

    async def _question(
        self, q: Any, images: list[Any], rendered_images: Any
    ) -> tuple[list[float], int]:
        """One question as one engine request: its code probabilities, and the prompt tokens vLLM served from its
        prefix cache."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        ids = collapse(q.state_ids, rendered_images, self.image_id) + list(q.question_ids)
        prompt: dict[str, Any] = {"prompt_token_ids": ids}
        if images:
            prompt["multi_modal_data"] = {"image": images}
            prompt["multi_modal_uuids"] = {"image": [im.sha256 for im in rendered_images]}
            # The images are already at W x H (prepare_image): the processor's own resize stays off.
            prompt["mm_processor_kwargs"] = {"do_resize": False}
        k = len(q.code_token_ids)
        sp = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            logprobs=k,
            allowed_token_ids=list(q.code_token_ids),
            extra_args={PREFIX_END_KEY: q.state_len},
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        final = None
        async for out in self.engine.generate(prompt, sp, f"sd-{uuid.uuid4().hex}"):
            final = out
        if final is None or not final.outputs or not final.outputs[0].logprobs:
            raise Mismatch("the engine returned no logprobs")
        if list(final.prompt_token_ids) != list(q.ids):
            raise Mismatch(
                f"vLLM's expanded prompt ({len(final.prompt_token_ids)} tokens) differs from the rendered one "
                f"({len(q.ids)} tokens): its image token counts are not the core's N"
            )
        return probabilities(final.outputs[0].logprobs[0], tuple(q.code_token_ids)), int(
            getattr(final, "num_cached_tokens", None) or 0
        )

    async def decide(self, body: Any) -> dict[str, Any]:
        from strands_decider.core import read, render
        from strands_decider.core.request import has_images

        if has_images(body):
            if not self.takes_images:
                raise Refused(
                    "this checkpoint does not take images: its decision_config.json inputs are "
                    + json.dumps(list(self.config.inputs))
                )
            if self.image_problem:
                raise Refused(f"image requests are refused: {self.image_problem}")
        try:
            rendered = render(body, self.tokenizer, self.config.prompt, max_length=self.max_length)
        except ValueError as e:
            raise Refused(str(e)) from e
        images = [prepare_image(im.data, im.tokens) for im in rendered.images]
        questions = list(rendered.questions.values())
        first, rest = questions[:1], questions[1:]
        if not self.prefix_caching:
            first, rest = [], questions
        results = [await self._question(q, images, rendered.images) for q in first]
        results += await asyncio.gather(*(self._question(q, images, rendered.images) for q in rest))
        answers = read([p for p, _ in results], None, self.config, rendered, inputs="probabilities")
        return {
            "model": self.served_name,
            "answers": answers,
            # v1's usage, plus the prompt tokens served from vLLM's prefix cache over all questions.
            "usage": {
                "input_tokens": rendered.input_tokens,
                "output_tokens": len(answers),
                "cached_tokens": sum(c for _, c in results),
            },
        }


def engine_args(checkpoint: Checkpoint, a: Any) -> Any:
    """The engine's settings: the plugin's required flags (README, Serving) for this checkpoint's layout."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    config = checkpoint.json("config.json")
    arch = plugin_architecture(config)
    dc = checkpoint.json("decision_config.json")
    takes_images = "image" in (dc.get("inputs") or [])
    if "text_config" in config:  # the multimodal layout: the "vision" mode is in the text config
        overrides: dict[str, Any] = {
            "architectures": [arch],
            "text_config": {"use_bidirectional_attention": None},
        }
        limits = {"image": MAX_IMAGES if takes_images else 0, "video": 0, "audio": 0}
    else:
        overrides = {"architectures": [arch], "use_bidirectional_attention": None}
        limits = None
    max_model_len = a.max_model_len or int(dc.get("max_length_validated") or 8192)
    kwargs: dict[str, Any] = dict(
        model=checkpoint.model,
        revision=checkpoint.revision,
        served_model_name=[a.served_model_name],
        dtype=a.dtype,
        seed=0,
        hf_overrides=overrides,
        max_model_len=max_model_len,
        max_num_batched_tokens=max(max_model_len, a.max_num_batched_tokens or 0),
        enable_chunked_prefill=False,
        enable_prefix_caching=a.enable_prefix_caching,
        prefix_caching_hash_algo="sha256",
        generation_config="vllm",
        logprobs_mode="processed_logprobs",
        max_logprobs=255,
        gpu_memory_utilization=a.gpu_memory_utilization,
        tensor_parallel_size=a.tensor_parallel_size,
        disable_custom_all_reduce=a.tensor_parallel_size > 1,
        enforce_eager=a.enforce_eager,
        additional_config={SERVER_CONFIG_KEY: __version__},
    )
    if a.max_num_seqs:
        kwargs["max_num_seqs"] = a.max_num_seqs
    if a.block_size:
        kwargs["block_size"] = a.block_size
    if limits is not None:
        kwargs["limit_mm_per_prompt"] = limits
        kwargs["mm_processor_cache_gb"] = a.mm_processor_cache_gb
    if a.enable_prefix_caching:
        kwargs["prefix_cache_retention_interval"] = None
    return AsyncEngineArgs(**kwargs)


def build_app(decider_factory: Any) -> Any:
    """The FastAPI app. ``decider_factory`` is an async callable that starts the engine and returns a
    :class:`Decider` (``lifespan`` start), and the app shuts the engine down on exit."""
    from contextlib import asynccontextmanager

    from vllm.exceptions import VLLMValidationError

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.decider = await decider_factory()
        try:
            yield
        finally:
            app.state.decider.engine.shutdown()

    app = FastAPI(title="strands-decider-vllm", version=__version__, lifespan=lifespan)

    async def health(request: Request) -> JSONResponse:
        d: Decider = request.app.state.decider
        try:
            await d.engine.check_health()
        except Exception as e:  # the engine is dead
            raise HTTPException(status_code=503, detail=f"engine unhealthy: {e}") from e
        return JSONResponse(
            {
                "status": "ok",
                "model": d.served_name,
                "inputs": list(d.config.inputs),
                "prompt": d.config.prompt,
                "max_length": d.max_length,
                "prefix_cache": d.prefix_caching,
                "version": __version__,
            }
        )

    async def systemone(request: Request) -> JSONResponse:
        d: Decider = request.app.state.decider
        started = time.perf_counter()
        try:
            body = await request.json()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"the body is not JSON: {e}") from e
        try:
            payload = await d.decide(body)
        except (Refused, VLLMValidationError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except ValueError as e:  # the engine's input processing refused the request
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Mismatch as e:
            logger.error("strands-decider: %s", e)
            raise HTTPException(status_code=500, detail=str(e)) from e
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(payload)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/ping", health, methods=["GET"])
    app.add_api_route("/v1/systemone", systemone, methods=["POST"])
    app.add_api_route("/invocations", systemone, methods=["POST"])
    return app


def serve(a: Any) -> None:
    """Start the engine and the HTTP server (``strands-decider-vllm serve``)."""
    if a.host not in LOOPBACK:
        logger.warning(
            "strands-decider-vllm serve: binding %s. The server has no authentication: anyone who can reach this "
            "address can send it requests. Put the deployment's own authentication in front of it (SageMaker's "
            "IAM-signed endpoint, or an authenticating proxy) or bind 127.0.0.1 and use an SSH port forward.",
            a.host,
        )
    checkpoint = Checkpoint.open(a.model, a.revision)
    dc = checkpoint.json("decision_config.json")
    from strands_decider.core import DecisionConfig

    try:
        config = DecisionConfig.from_dict(dc)
    except ValueError as e:
        raise SystemExit(
            f"strands-decider-vllm serve: decision_config.json: {e}. The server renders requests with the core "
            "library and needs the specification's schema (section 8). Serve an intermediate checkpoint with "
            "`vllm serve` and pre-rendered token ids instead (README, Serving)."
        ) from e
    image_problem = None
    if "image" in config.inputs:
        pc = (
            checkpoint.json("processor_config.json")
            if "processor_config.json" in checkpoint.files
            else None
        )
        image_problem = processor_problem(pc)
        if image_problem:
            logger.error("strands-decider: image requests will be refused: %s", image_problem)
    args = engine_args(checkpoint, a)
    max_length = a.max_length or config.max_length_validated
    if max_length > args.max_model_len:
        raise SystemExit(
            f"strands-decider-vllm serve: --max-length {max_length} exceeds --max-model-len {args.max_model_len}"
        )

    async def start() -> Decider:
        from vllm.v1.engine.async_llm import AsyncLLM

        engine = AsyncLLM.from_engine_args(args)
        return Decider(
            engine,
            checkpoint,
            served_name=a.served_model_name,
            max_length=max_length,
            prefix_caching=a.enable_prefix_caching,
            image_problem=image_problem,
        )

    import uvicorn

    uvicorn.run(build_app(start), host=a.host, port=a.port, log_level="info")


def prepare_environment(host: str) -> None:
    """The settings vLLM reads from the environment at import: Model Runner V2, which the mask needs, and, for a
    loopback server, vLLM's internal sockets on loopback too. A value the caller set is kept."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    if host in LOOPBACK:
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
