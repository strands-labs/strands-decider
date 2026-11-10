"""``strands-decider-vllm serve``: the System One request, images included, on vLLM's engine in process.

vLLM 0.33's ``/v1/completions`` takes token ids but no images, its chat endpoints render a chat template (which the
specification forbids for these prompts), and its token-in-token-out endpoint lets a client name an image by an id of
its choosing, which vLLM then uses as the image's cache key. This server takes v1's request instead and does the rest:

1. It renders the request with the core library (``strands_decider.core.render``) and the checkpoint's prompt
   template. Each image becomes ``<|image>``, N ``<|image|>`` and ``<image|>`` first in the state, and S counts them.
2. It decodes each image as v1 does (first frame, EXIF orientation, transparency onto white, RGB) and resizes it to
   the specification's W x H, computed from the decoded size, with the core's ``resize_image``: Pillow's bicubic
   resampling, the width's pass and then the height's, the reference resize for every size (section 11.1).
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

Parsing, rendering and image decoding run on a small thread pool (``--preprocess-workers``), never on the event loop,
so ``/ping`` and ``/health`` answer while images are decoded; vLLM runs its own image preprocessing on its renderer's
threads. A body over ``--max-body-mb`` gets HTTP 413, and uvicorn answers 503 past ``--limit-concurrency``
connections.

Endpoints: ``POST /v1/systemone`` and ``POST /invocations`` (the same handler, for SageMaker), ``GET /ping`` and
``GET /health``. A request the server, the core library or the engine refuses gets HTTP 400 with
``{"detail": "..."}``; any other failure is logged and gets HTTP 500 with only an error id. The engine keeps serving
either way. The server has no authentication and binds 127.0.0.1 unless told otherwise: expose it beyond the machine
only behind the deployment's own authentication, such as SageMaker's IAM-signed endpoint or an authenticating proxy.
"""

from __future__ import annotations

import asyncio
import inspect
import io
import json
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
#: The core library this server is written against, as pip writes it (pyproject.toml's ``serve`` extra).
CORE_REQUIREMENT = "strands-decider>=0.3.1.dev0,<0.5"

#: The default body limit is four image files of IMAGE_FILE_BYTES each, base64-encoded, plus TEXT_BYTES of text and
#: JSON. 32 MiB holds a photo of the core's largest size (4,096 x 4,096 pixels) as JPEG, WebP or a compressed PNG.
IMAGE_FILE_BYTES = 32 * 2**20
TEXT_BYTES = 16 * 2**20


def base64_size(n: int) -> int:
    """The length of ``n`` bytes in base64, with room for a ``data:`` URI prefix and the JSON string's quotes."""
    return 4 * math.ceil(n / 3) + 64


def default_max_body(takes_images: bool) -> int:
    """The body limit for a checkpoint: the text budget, plus four image files when it takes images (187 MiB)."""
    return TEXT_BYTES + (MAX_IMAGES * base64_size(IMAGE_FILE_BYTES) if takes_images else 0)


class Refused(ValueError):
    """A request the server will not answer (HTTP 400)."""


class Mismatch(RuntimeError):
    """vLLM's expanded prompt differs from the rendered one (HTTP 500: a bug, never the client's)."""


# --- the core library -----------------------------------------------------------------------------------------------


def stage_argument(read: Any) -> str:
    """The name of ``read``'s argument for the rows' stage: ``stage``, or ``inputs`` in a core from before the
    rename (which still accepts it, with a DeprecationWarning, after it)."""
    return "stage" if "stage" in inspect.signature(read).parameters else "inputs"


def require_core() -> SimpleNamespace:
    """What the server uses from the core library (``strands_decider.core``, ``CORE_REQUIREMENT``), or SystemExit
    naming what is missing; ``serve`` calls it before the engine starts. ``read`` names the rows' stage ``stage=``;
    a core that still calls it ``inputs=`` is given that name."""
    try:
        from strands_decider import core
        from strands_decider.core.request import has_images
    except ImportError as e:
        raise SystemExit(
            f"strands-decider-vllm serve needs the core library, strands_decider.core ({CORE_REQUIREMENT}): "
            f"install it with pip install 'strands-decider-vllm[serve]' ({e})"
        ) from e
    missing = [n for n in ("render", "read", "DecisionConfig") if not hasattr(core, n)]
    if missing:
        raise SystemExit(
            f"strands-decider-vllm serve: the installed core library has no {', '.join(missing)}; the server "
            f"needs {CORE_REQUIREMENT}"
        )
    return SimpleNamespace(
        render=core.render,
        read=core.read,
        DecisionConfig=core.DecisionConfig,
        has_images=has_images,
        stage={stage_argument(core.read): "probabilities"},
        # What a newer core provides; the server's own functions stand in for them without it.
        check_tokenizer=getattr(core, "check_tokenizer", None),
        check_image_processor=getattr(core, "check_image_processor", None),
        check_decoded_size=getattr(core, "check_decoded_size", None),
        resize_image=getattr(core, "resize_image", None) or bicubic,
        resized_size=getattr(core, "resized_size", None) or resized_size,
    )


# --- the checkpoint -------------------------------------------------------------------------------------------------

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Checkpoint:
    """The files the server reads beside the engine: decision_config.json, config.json, processor_config.json and
    the tokenizer, from a local directory or a Hub repo. For a Hub repo ``revision`` is the commit the files were
    read at, and the engine is given that commit, so that both read the same one."""

    model: str
    revision: str | None
    files: dict[str, Path]

    NAMES = ("decision_config.json", "config.json", "processor_config.json", "tokenizer.json")
    REQUIRED = ("decision_config.json", "config.json", "tokenizer.json")

    @classmethod
    def open(cls, model: str, revision: str | None = None) -> Checkpoint:
        files: dict[str, Path] = {}
        if Path(model).is_dir():
            for n in cls.NAMES:
                if (Path(model) / n).is_file():
                    files[n] = Path(model) / n
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            # config.json first: its place in the cache, snapshots/<commit>/config.json, names the commit the
            # revision resolves to now; every other file, and the engine's weights, are then read at that commit.
            files["config.json"] = Path(hf_hub_download(model, "config.json", revision=revision))
            commit = files["config.json"].parent.name
            if not _COMMIT.match(commit):
                raise SystemExit(
                    f"strands-decider-vllm serve: could not resolve {model!r} at revision {revision!r} to a commit"
                )
            revision = commit
            for n in cls.NAMES:
                if n in files:
                    continue
                try:
                    files[n] = Path(hf_hub_download(model, n, revision=revision))
                except EntryNotFoundError:
                    pass
        for n in cls.REQUIRED:
            if n not in files:
                raise SystemExit(f"strands-decider-vllm serve: {model!r} has no {n}")
        return cls(model, revision, files)

    def json(self, name: str) -> dict:
        return json.loads(self.files[name].read_text())

    def tokenizer(self) -> Any:
        from tokenizers import Tokenizer

        return Tokenizer.from_file(str(self.files["tokenizer.json"]))


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


def processor_problem(processor_config: dict | None, check: Any = None) -> str | None:
    """Why image requests must be refused for this image processor configuration, or None. ``check`` is the core
    library's ``check_image_processor`` when it has one."""
    if processor_config is None:
        return "the checkpoint has no processor_config.json"
    if check is not None:
        try:
            check(processor_config)
        except ValueError as e:
            return str(e)
        return None
    ip = processor_config.get("image_processor") or {}
    bad = {k: ip.get(k) for k, v in PROCESSOR_SETTINGS.items() if ip.get(k) != v}
    if bad:
        return (
            f"the checkpoint's image processor has {bad}; N is defined for {PROCESSOR_SETTINGS} "
            "(specification, section 3.2)"
        )
    return None


# --- images ---------------------------------------------------------------------------------------------------------


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


def bicubic(img: Any, width: int, height: int) -> Any:
    """Section 11.1's resize, for a core without ``resize_image`` (which is the same): Pillow's bicubic resampling,
    the width's pass first and then the height's, each left out when its size does not change. Pillow 10 to 12 do
    this in one call; Pillow 13 may do the height first, so it is two calls, which give the same bytes on every
    version."""
    from PIL import Image

    if img.size[0] != width:
        img = img.resize((width, img.size[1]), Image.Resampling.BICUBIC)
    if img.size[1] != height:
        img = img.resize((width, height), Image.Resampling.BICUBIC)
    return img


def prepare_image(data: bytes, tokens: int, target: Any = None, resize: Any = None) -> Any:
    """The decoded image, resized to W x H (specification, section 11.1). W and H come from the decoded size, after
    the orientation: ``target(width, height)`` gives them, the core library's ``check_decoded_size`` for this image
    when it has one (which refuses a decoded size that is neither the header's nor its transpose), else section
    3.2's arithmetic. ``resize`` is the core's ``resize_image`` when it has one, else :func:`bicubic`, the same
    rule."""
    img = decode_image(data)
    try:
        w, h = target(*img.size) if target is not None else resized_size(*img.size)
    except ValueError as e:  # the core refuses the decoded size
        raise Refused(str(e)) from e
    side = PROCESSOR_SETTINGS["pooling_kernel_size"] * PROCESSOR_SETTINGS["patch_size"]
    if (w // side) * (h // side) != tokens:
        raise Mismatch(
            f"a {img.size[0]} x {img.size[1]} image resizes to {w} x {h}, not to the core's {tokens} tokens"
        )
    return (resize or bicubic)(img, w, h)


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


# --- requests -------------------------------------------------------------------------------------------------------


def parse_body(raw: bytes) -> dict:
    """The request's JSON object, or Refused."""
    try:
        body = json.loads(raw)
    except (ValueError, RecursionError) as e:  # RecursionError: nested past the parser's limit
        raise Refused(f"the body is not JSON: {e}") from e
    if not isinstance(body, dict):
        raise Refused(f"the body must be a JSON object, not {type(body).__name__}")
    return body


def refusal(e: BaseException) -> bool:
    """Whether ``e`` refuses the request (HTTP 400): the server's or the core library's refusal (Refused), or the
    engine's (vLLM's VLLMValidationError, which AsyncLLM.generate raises for a request its input processing or its
    engine core refuses). Anything else is a server fault."""
    from vllm.exceptions import VLLMValidationError

    return isinstance(e, (Refused, VLLMValidationError))


async def all_or_none(coros: list[Any]) -> list[Any]:
    """The coroutines' results, run concurrently. On the first failure the others are cancelled (vLLM aborts each
    cancelled engine request) and that failure is raised."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


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
        max_body: int | None = None,
        preprocess_workers: int = 4,
        tokenizer: Any = None,
        core: SimpleNamespace | None = None,
    ) -> None:
        self.core = core or require_core()
        self.engine = engine
        self.checkpoint = checkpoint
        self.config = self.core.DecisionConfig.from_dict(checkpoint.json("decision_config.json"))
        self.tokenizer = tokenizer if tokenizer is not None else checkpoint.tokenizer()
        self.image_id = self.tokenizer.token_to_id(IMAGE_SOFT)
        self.served_name = served_name
        self.max_length = max_length
        self.prefix_caching = prefix_caching
        self.image_problem = image_problem
        self.max_body = (
            max_body
            if max_body is not None
            else default_max_body(self.takes_images and not image_problem)
        )
        # Parsing, rendering and decoding run on this pool, at most preprocess_workers requests at once; the others
        # wait on a semaphore, where a cancelled request leaves nothing queued.
        self.preprocess_workers = max(1, preprocess_workers)
        self._pool = ThreadPoolExecutor(
            self.preprocess_workers, thread_name_prefix="strands-decider"
        )
        self._slots: asyncio.Semaphore | None = None
        self._slots_loop: Any = None

    @property
    def takes_images(self) -> bool:
        return "image" in self.config.inputs

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def prepare(self, body: Any) -> tuple[Any, list[Any]]:
        """The rendered request and its images resized to W x H. Blocking: it runs on the pool. ``body`` is the raw
        body or its parsed JSON object."""
        if isinstance(body, (bytes, bytearray)):
            body = parse_body(bytes(body))
        elif not isinstance(body, dict):
            raise Refused(f"the body must be a JSON object, not {type(body).__name__}")
        if self.core.has_images(body):
            if not self.takes_images:
                raise Refused(
                    "this checkpoint does not take images: its decision_config.json inputs are "
                    + json.dumps(list(self.config.inputs))
                )
            if self.image_problem:
                raise Refused(f"image requests are refused: {self.image_problem}")
        try:
            rendered = self.core.render(
                body, self.tokenizer, self.config.prompt, max_length=self.max_length
            )
        except ValueError as e:  # the core library refuses the request
            raise Refused(str(e)) from e
        check = self.core.check_decoded_size
        return rendered, [
            prepare_image(
                im.data,
                im.tokens,
                self.core.resized_size if check is None else (lambda w, h, im=im: check(im, w, h)),
                self.core.resize_image,
            )
            for im in rendered.images
        ]

    async def _offload(self, fn: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        if self._slots is None or self._slots_loop is not loop:
            self._slots, self._slots_loop = asyncio.Semaphore(self.preprocess_workers), loop
        async with self._slots:
            return await loop.run_in_executor(self._pool, fn, *args)

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
        """The answers to one request. ``body`` is its raw JSON body or the parsed object."""
        rendered, images = await self._offload(self.prepare, body)
        questions = list(rendered.questions.values())
        first, rest = questions[:1], questions[1:]
        if not self.prefix_caching:
            first, rest = [], questions
        results = [await self._question(q, images, rendered.images) for q in first]
        results += await all_or_none([self._question(q, images, rendered.images) for q in rest])
        answers = self.core.read(
            [p for p, _ in results], None, self.config, rendered, **self.core.stage
        )
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


async def read_body(request: Request, limit: int) -> bytes:
    """The request's body, refused with HTTP 413 past ``limit`` bytes: by its Content-Length before reading, and by
    what arrives while reading."""
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            n = int(declared)
        except ValueError:
            raise HTTPException(status_code=400, detail="Content-Length is not a number") from None
        if n > limit:
            raise HTTPException(
                status_code=413,
                detail=f"the body is {n} bytes; this server takes at most {limit} (--max-body-mb)",
            )
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(
                status_code=413,
                detail=f"the body is over {limit} bytes, the most this server takes (--max-body-mb)",
            )
    return bytes(body)


def build_app(decider_factory: Any) -> Any:
    """The FastAPI app. ``decider_factory`` is an async callable that starts the engine and returns a
    :class:`Decider` (``lifespan`` start), and the app shuts the engine down on exit."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.decider = await decider_factory()
        try:
            yield
        finally:
            app.state.decider.close()
            app.state.decider.engine.shutdown()

    app = FastAPI(title="strands-decider-vllm", version=__version__, lifespan=lifespan)

    async def health(request: Request) -> JSONResponse:
        d: Decider = request.app.state.decider
        try:
            await d.engine.check_health()
        except Exception as e:  # the engine is dead
            logger.error("strands-decider: engine unhealthy: %s", e)
            raise HTTPException(status_code=503, detail="engine unhealthy") from e
        return JSONResponse(
            {
                "status": "ok",
                "model": d.served_name,
                "inputs": list(d.config.inputs),
                "prompt": d.config.prompt,
                "max_length": d.max_length,
                "max_body_bytes": d.max_body,
                "prefix_cache": d.prefix_caching,
                "version": __version__,
            }
        )

    async def systemone(request: Request) -> JSONResponse:
        d: Decider = request.app.state.decider
        started = time.perf_counter()
        raw = await read_body(request, d.max_body)
        try:
            payload = await d.decide(raw)
        except Exception as e:
            if refusal(e):
                raise HTTPException(status_code=400, detail=str(e)) from e
            error_id = uuid.uuid4().hex[:16]
            logger.exception("strands-decider: request failed (error id %s)", error_id)
            return JSONResponse(
                {"detail": "internal server error", "error_id": error_id}, status_code=500
            )
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(payload)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/ping", health, methods=["GET"])
    app.add_api_route("/v1/systemone", systemone, methods=["POST"])
    app.add_api_route("/invocations", systemone, methods=["POST"])
    return app


# --- startup --------------------------------------------------------------------------------------------------------


def engine_args(checkpoint: Checkpoint, a: Any, max_model_len: int) -> Any:
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


def lengths(a: Any, max_length_validated: int) -> tuple[int, int]:
    """(max_length, max_model_len): the longest prompt served (``--max-length``, default the checkpoint's validated
    length) and the engine's context (``--max-model-len``, default one token more). vLLM 0.33 refuses a prompt of
    exactly max_model_len tokens, which leaves no position for the answer token, so max_length must be below it."""
    max_length = a.max_length or max_length_validated
    max_model_len = a.max_model_len or max_length + 1
    if max_length >= max_model_len:
        raise SystemExit(
            f"strands-decider-vllm serve: --max-length {max_length} must be below --max-model-len {max_model_len}: "
            "vLLM refuses a prompt that fills the context, because the answer token needs one more position"
        )
    return max_length, max_model_len


@dataclass(frozen=True)
class Settings:
    """What ``serve`` starts with, checked before the engine starts."""

    checkpoint: Checkpoint
    core: SimpleNamespace
    tokenizer: Any
    engine_args: Any
    max_length: int
    image_problem: str | None
    max_body: int


def settings(a: Any) -> Settings:
    """The checkpoint's files, the core library and the engine's settings for ``a`` (the CLI's arguments), or
    SystemExit."""
    core = require_core()
    checkpoint = Checkpoint.open(a.model, a.revision)
    try:
        config = core.DecisionConfig.from_dict(checkpoint.json("decision_config.json"))
    except ValueError as e:
        raise SystemExit(
            f"strands-decider-vllm serve: decision_config.json: {e}. The server renders requests with the core "
            "library and needs the specification's schema (section 8). Serve an intermediate checkpoint with "
            "`vllm serve` and pre-rendered token ids instead (README, Serving)."
        ) from e
    tokenizer = checkpoint.tokenizer()
    if core.check_tokenizer is not None:
        try:
            core.check_tokenizer(tokenizer, config)
        except ValueError as e:
            raise SystemExit(f"strands-decider-vllm serve: tokenizer.json: {e}") from e
    image_problem = None
    if "image" in config.inputs:
        pc = (
            checkpoint.json("processor_config.json")
            if "processor_config.json" in checkpoint.files
            else None
        )
        image_problem = processor_problem(pc, core.check_image_processor)
        if image_problem:
            logger.error("strands-decider: image requests will be refused: %s", image_problem)
    max_length, max_model_len = lengths(a, config.max_length_validated)
    max_body = (
        int(a.max_body_mb * 2**20)
        if a.max_body_mb
        else default_max_body("image" in config.inputs and not image_problem)
    )
    return Settings(
        checkpoint,
        core,
        tokenizer,
        engine_args(checkpoint, a, max_model_len),
        max_length,
        image_problem,
        max_body,
    )


def decider(engine: Any, s: Settings, a: Any) -> Decider:
    """The server's :class:`Decider` on ``engine``, from :func:`settings`."""
    return Decider(
        engine,
        s.checkpoint,
        served_name=a.served_model_name,
        max_length=s.max_length,
        prefix_caching=a.enable_prefix_caching,
        image_problem=s.image_problem,
        max_body=s.max_body,
        preprocess_workers=a.preprocess_workers,
        tokenizer=s.tokenizer,
        core=s.core,
    )


def serve(a: Any) -> None:
    """Start the engine and the HTTP server (``strands-decider-vllm serve``)."""
    if a.host not in LOOPBACK:
        logger.warning(
            "strands-decider-vllm serve: binding %s. The server has no authentication: anyone who can reach this "
            "address can send it requests. Put the deployment's own authentication in front of it (SageMaker's "
            "IAM-signed endpoint, or an authenticating proxy) or bind 127.0.0.1 and use an SSH port forward.",
            a.host,
        )
    s = settings(a)

    async def start() -> Decider:
        from vllm.v1.engine.async_llm import AsyncLLM

        return decider(AsyncLLM.from_engine_args(s.engine_args), s, a)

    import uvicorn

    uvicorn.run(
        build_app(start),
        host=a.host,
        port=a.port,
        log_level="info",
        limit_concurrency=a.limit_concurrency,
    )


def prepare_environment() -> None:
    """The settings vLLM reads from the environment at import: Model Runner V2, which the mask needs, and vLLM's
    internal sockets on loopback, whatever address the HTTP server binds (the engine runs on this machine). A value
    the caller set is kept."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
