"""``strands-decider-vllm serve``: the System One request, images included, on vLLM's engine in process.

vLLM 0.33's own endpoints don't fit: ``/v1/completions`` takes token ids but no images, its chat endpoints apply a
chat template on top of the core's prompt, and its token-in endpoint lets a client pick an image's id, which vLLM
uses as the image's cache key. This server renders each request with the core (``strands_decider.core``), decodes
and resizes each image with the core's ``resize_image`` (vLLM's processor then runs with its own resize off), and
sends vLLM one request per question: the prompt with each image collapsed to the one ``<|image|>`` vLLM's processor
expands, each image's file sha256 as its id, and the state boundary S in ``vllm_xargs.prefix_end``. It checks that
vLLM's expanded prompt is the rendered one, token for token, and reads the answers from the processed logprobs.

Parsing, rendering and image decoding run on worker threads, never on the event loop. Large bodies share a byte
budget (``--max-buffered-mb``), so buffered bodies are bounded in aggregate. A refused request gets HTTP 422 and a
server fault 500 with only an error id; the engine keeps serving either way. The server has no authentication and
binds 127.0.0.1 by default: expose it only behind the deployment's own authentication, such as SageMaker's IAM-signed
endpoint or an authenticating proxy.
"""

from __future__ import annotations

import asyncio
import collections
import io
import json
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from . import ARCH_FOR_BASE, PREFIX_END_KEY, SERVER_CONFIG_KEY, __version__, logger

#: Gemma 4's image placeholder, which vLLM's processor expands.
IMAGE_SOFT = "<|image|>"
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
LOOPBACK = ("127.0.0.1", "::1", "localhost")

#: The default body limit: room for four of the core's largest image files (16 MiB each) in base64, plus 8 MiB of
#: text (93 MiB). It limits bytes, not images.
IMAGE_FILE_BYTES = 16 * 2**20
BODY_IMAGE_FILES = 4
TEXT_BYTES = 8 * 2**20
#: Bodies up to this size are read at once; a larger one first reserves its size from the budget of buffered bytes.
SMALL_BODY = 2**20
BUFFERED_BYTES = 256 * 2**20
#: Seconds a body may go without a chunk arriving, and a large body may wait for the budget.
BODY_READ_TIMEOUT = 30.0


def default_max_body(takes_images: bool) -> int:
    base64_file = (
        4 * math.ceil(IMAGE_FILE_BYTES / 3) + 64
    )  # with room for a data URI prefix and quotes
    return TEXT_BYTES + (BODY_IMAGE_FILES * base64_file if takes_images else 0)


class Refused(ValueError):
    """A request the server will not answer (HTTP 422)."""


class Mismatch(RuntimeError):
    """vLLM's expanded prompt differs from the rendered one (HTTP 500: a bug, never the client's)."""


def errors(e: ValidationError) -> str:
    """pydantic's errors as ``where: what``, for a 422."""
    out = []
    for err in e.errors(include_url=False, include_input=False):
        where, what = ".".join(map(str, err["loc"])), err["msg"].removeprefix("Value error, ")
        out.append(f"{where}: {what}" if where else what)
    return "; ".join(out)


def _core() -> Any:
    try:
        from strands_decider import core
    except ImportError as e:
        raise SystemExit(
            "strands-decider-vllm serve needs the core library: pip install 'strands-decider-vllm[serve]' "
            f"({e})"
        ) from e
    return core


# --- the checkpoint -------------------------------------------------------------------------------------------------

_COMMIT = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class Checkpoint:
    """The files the server reads beside the engine, from a local directory or a Hub repo. For a Hub repo,
    ``revision`` is the commit the files were read at, and the engine is given the same commit."""

    model: str
    revision: str | None
    files: dict[str, Path]

    NAMES = ("decision_config.json", "config.json", "processor_config.json", "tokenizer.json")
    REQUIRED = ("decision_config.json", "config.json", "tokenizer.json")

    @classmethod
    def open(cls, model: str, revision: str | None = None) -> Checkpoint:
        files: dict[str, Path] = {}
        if Path(model).is_dir():
            files = {n: Path(model) / n for n in cls.NAMES if (Path(model) / n).is_file()}
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            # config.json first: its cache path, snapshots/<commit>/config.json, names the commit the revision
            # resolves to now, and every other file is read at that commit.
            files["config.json"] = Path(hf_hub_download(model, "config.json", revision=revision))
            revision = files["config.json"].parent.name
            if not _COMMIT.match(revision):
                raise SystemExit(
                    f"strands-decider-vllm serve: could not resolve {model!r} to a commit"
                )
            for n in cls.NAMES:
                if n in files:
                    continue
                try:
                    files[n] = Path(hf_hub_download(model, n, revision=revision))
                except EntryNotFoundError:
                    pass
        missing = [n for n in cls.REQUIRED if n not in files]
        if missing:
            raise SystemExit(f"strands-decider-vllm serve: {model!r} has no {', '.join(missing)}")
        return cls(model, revision, files)

    def text(self, name: str) -> str:
        return self.files[name].read_text(encoding="utf-8")


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


# --- images ---------------------------------------------------------------------------------------------------------


def decode_image(data: bytes) -> Any:
    """One image file's pixels: its first frame, upright by its EXIF orientation, transparent areas on white (as
    v1 decodes images), as RGB. The core has already checked its format and size from the header."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise Refused(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def prepare_image(image: Any) -> Any:
    """A rendered request's image, decoded and resized to the core's W x H, from its decoded size."""
    core = _core()
    img = decode_image(image.data)
    try:
        w, h = core.check_decoded_size(image, *img.size)
    except ValueError as e:
        raise Refused(str(e)) from e
    return core.resize_image(img, w, h)


def without_image_files(rendered: Any) -> Any:
    """``rendered`` without its images' files, once their pixels are decoded, so a request doesn't hold them while
    the engine runs."""
    if not rendered.images:
        return rendered
    images = tuple(replace(im, data=b"") for im in rendered.images)
    return replace(
        rendered, questions={n: replace(q, images=images) for n, q in rendered.questions.items()}
    )


def collapse(state_ids: tuple[int, ...], images: Any, image_id: int) -> list[int]:
    """The state ids with each image's ``<|image>`` N ``<|image|>`` ``<image|>`` replaced by the one ``<|image|>``
    that vLLM's Gemma 4 processor expands back to it."""
    out: list[int] = []
    pos = 0
    for im in images:
        out.extend(state_ids[pos : im.start - 1])
        out.append(image_id)
        pos = im.end + 1
    out.extend(state_ids[pos:])
    return out


def probabilities(logprobs: dict, code_ids: tuple[int, ...]) -> list[float]:
    """A question's code probabilities from the sampled position's processed logprobs, renormalized over its codes."""
    lp = [logprobs[t].logprob if t in logprobs else -math.inf for t in code_ids]
    m = max(lp)
    if not math.isfinite(m):
        raise Mismatch("no code of the question has a finite logprob")
    e = [math.exp(x - m) for x in lp]
    s = sum(e)
    return [x / s for x in e]


# --- requests -------------------------------------------------------------------------------------------------------


def refusal(e: BaseException) -> bool:
    """Whether ``e`` refuses the request (HTTP 422): the server's refusal, or the engine's VLLMValidationError."""
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


class ByteBudget:
    """A budget of buffered body bytes, granted in arrival order, so a large body isn't starved by smaller ones."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(1, capacity)
        self.used = 0
        self._waiters: collections.deque[tuple[int, asyncio.Future]] = collections.deque()

    async def take(self, n: int, timeout: float | None = None) -> int:
        """Reserve ``n`` bytes (at most the whole budget) and return the reservation; asyncio.TimeoutError after
        ``timeout`` seconds, having taken nothing."""
        n = min(n, self.capacity)
        if not self._waiters and self.used + n <= self.capacity:
            self.used += n
            return n
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append((n, fut))
        try:
            await asyncio.wait_for(fut, timeout)
        except BaseException:
            if fut.done() and not fut.cancelled():
                self.give(n)  # granted just as it was cancelled or timed out
            else:
                try:
                    self._waiters.remove((n, fut))
                except ValueError:
                    pass
                self._grant()
            raise
        return n

    def give(self, n: int) -> None:
        self.used -= n
        self._grant()

    def _grant(self) -> None:
        while self._waiters and self.used + self._waiters[0][0] <= self.capacity:
            n, fut = self._waiters.popleft()
            if not fut.done():
                self.used += n
                fut.set_result(None)


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
        max_buffered: int = BUFFERED_BYTES,
        body_read_timeout: float = BODY_READ_TIMEOUT,
        preprocess_workers: int = 4,
        tokenizer: Any = None,
    ) -> None:
        core = _core()
        self.engine = engine
        self.checkpoint = checkpoint
        self.config = core.DecisionConfig.model_validate_json(
            checkpoint.text("decision_config.json")
        )
        if tokenizer is None:
            tokenizer = core.load_tokenizer(checkpoint.files["tokenizer.json"].parent)
        self.tokenizer = tokenizer
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
        self.max_buffered = max(max_buffered, self.max_body)
        self.body_read_timeout = body_read_timeout
        # Two lanes of threads, rendering and image decoding, so a request without images never waits behind a
        # decode. Each lane's semaphore admits at most preprocess_workers at once, so nothing queues in the pool.
        self.preprocess_workers = max(1, preprocess_workers)
        self._pool = ThreadPoolExecutor(
            2 * self.preprocess_workers, thread_name_prefix="strands-decider"
        )
        self._loop: Any = None
        self._lanes: dict[str, asyncio.Semaphore] = {}
        self._budget = ByteBudget(self.max_buffered)

    @property
    def takes_images(self) -> bool:
        return "image" in self.config.inputs

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    @property
    def budget(self) -> ByteBudget:
        self._bind()
        return self._budget

    def _bind(self) -> asyncio.AbstractEventLoop:
        """The running loop, with the lanes and the budget bound to it (asyncio primitives belong to one loop)."""
        loop = asyncio.get_running_loop()
        if self._loop is not loop:
            self._loop = loop
            self._lanes = {
                lane: asyncio.Semaphore(self.preprocess_workers) for lane in ("render", "decode")
            }
            self._budget = ByteBudget(self.max_buffered)
        return loop

    async def _run(self, lane: str, fn: Any, *args: Any) -> Any:
        loop = self._bind()
        async with self._lanes[lane]:
            return await loop.run_in_executor(self._pool, fn, *args)

    def render_request(self, body: bytes | bytearray | Any) -> Any:
        """The rendered request, from the raw body or a core ``Request`` (blocking: it runs on the render lane)."""
        core = _core()
        if isinstance(body, (bytes, bytearray)):
            try:
                body = core.Request.model_validate_json(body)
            except ValidationError as e:
                raise Refused(errors(e)) from None
        if body.images:
            if not self.takes_images:
                raise Refused("this checkpoint does not take images")
            if self.image_problem:
                raise Refused(f"image requests are refused: {self.image_problem}")
        try:
            return core.render(body, self.tokenizer, max_length=self.max_length)
        except ValueError as e:  # an image or the prompt's length
            raise Refused(str(e)) from e

    async def prepare(self, body: Any) -> tuple[Any, list[Any]]:
        """The rendered request, its image files dropped, and its images resized to W x H."""
        rendered = await self._run("render", self.render_request, body)
        del body
        if not rendered.images:
            return rendered, []
        images = await self._run("decode", lambda: [prepare_image(im) for im in rendered.images])
        return without_image_files(rendered), images

    async def _question(
        self, q: Any, images: list[Any], rendered_images: Any
    ) -> tuple[list[float], int]:
        """One question as one engine request: its code probabilities, and the prompt tokens served from vLLM's
        prefix cache."""
        from vllm import SamplingParams
        from vllm.sampling_params import RequestOutputKind

        prompt: dict[str, Any] = {
            "prompt_token_ids": collapse(q.state_ids, rendered_images, self.image_id)
            + list(q.question_ids)
        }
        if images:
            prompt["multi_modal_data"] = {"image": images}
            prompt["multi_modal_uuids"] = {"image": [im.sha256 for im in rendered_images]}
            prompt["mm_processor_kwargs"] = {"do_resize": False}  # already at W x H
        sp = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            min_p=0.0,
            logprobs=len(q.code_token_ids),
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
                f"({len(q.ids)} tokens): its image token counts are not the core's"
            )
        cached = int(getattr(final, "num_cached_tokens", None) or 0)
        return probabilities(final.outputs[0].logprobs[0], tuple(q.code_token_ids)), cached

    async def answer(self, rendered: Any, images: list[Any]) -> dict[str, Any]:
        """The answers to a prepared request. The first question runs alone, so the others reuse its cached state."""
        questions = list(rendered.questions.values())
        first, rest = (questions[:1], questions[1:]) if self.prefix_caching else ([], questions)
        results = [await self._question(q, images, rendered.images) for q in first]
        results += await all_or_none([self._question(q, images, rendered.images) for q in rest])
        answers = _core().read([p for p, _ in results], self.config, rendered)
        return {
            "model": self.served_name,
            "answers": answers,
            "usage": {
                "input_tokens": rendered.input_tokens,
                "output_tokens": len(answers),
                "cached_tokens": sum(c for _, c in results),
            },
        }

    async def decide(self, body: Any) -> dict[str, Any]:
        """The answers to one request: its raw JSON body or a core ``Request``."""
        rendered, images = await self.prepare(body)
        del body
        return await self.answer(rendered, images)

    async def decide_http(self, request: Request) -> dict[str, Any]:
        """The answers to one HTTP request. A large body's reservation from the byte budget is given back once the
        body is parsed and its images decoded, before the engine runs."""
        reserved = 0
        try:
            raw, reserved = await read_body(
                request, self.max_body, self.budget, timeout=self.body_read_timeout
            )
            prepared = await self.prepare(raw)
            del raw
        finally:
            if reserved:
                self.budget.give(reserved)
        return await self.answer(*prepared)


def too_large(limit: int, n: int | None = None) -> HTTPException:
    what = f"is {n} bytes" if n is not None else f"is over {limit} bytes"
    return HTTPException(
        status_code=413,
        detail=f"the body {what}; this server takes at most {limit} (--max-body-mb)",
    )


async def read_body(
    request: Request,
    limit: int,
    budget: ByteBudget | None = None,
    *,
    timeout: float | None = BODY_READ_TIMEOUT,
) -> tuple[bytearray, int]:
    """The request's body and the bytes it reserved from ``budget``; the caller gives the reservation back.

    - HTTP 413 past ``limit`` bytes, by Content-Length before reading and by what arrives while reading.
    - A body declared over SMALL_BODY reserves its size before reading; one without a length reads SMALL_BODY
      freely, then reserves the limit, and returns what it doesn't use.
    - HTTP 503 when a reservation has waited ``timeout`` seconds, and 408 when no chunk arrives for ``timeout``
      seconds, so a stalled or vanished client gives its reservation back.
    """
    declared = request.headers.get("content-length")
    n = None
    if declared is not None:
        try:
            n = int(declared)
        except ValueError:
            raise HTTPException(status_code=400, detail="Content-Length is not a number") from None
        if n > limit:
            raise too_large(limit, n)

    async def reserve(size: int) -> int:
        assert budget is not None
        try:
            return await budget.take(size, timeout)
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="the server is busy with other large requests; retry",
                headers={"Retry-After": "5"},
            ) from None

    reserved = 0
    try:
        if budget is not None and n is not None and n > SMALL_BODY:
            reserved = await reserve(n)
        body = bytearray()
        chunks = request.stream().__aiter__()
        while True:
            try:
                chunk = await asyncio.wait_for(chunks.__anext__(), timeout)
            except StopAsyncIteration:
                break
            except TimeoutError:
                raise HTTPException(
                    status_code=408,
                    detail=f"no part of the body arrived for {timeout:g} s (--body-read-timeout)",
                ) from None
            body += chunk
            if len(body) > limit:
                raise too_large(limit)
            if budget is not None and not reserved and len(body) > SMALL_BODY:
                reserved = await reserve(limit)
        if budget is not None and reserved > len(body) and n is None:
            budget.give(reserved - len(body))
            reserved = len(body)
    except BaseException:  # a 408, 413 or 503, a client that disconnected, a cancellation
        if reserved and budget is not None:
            budget.give(reserved)
        raise
    return body, reserved


def build_app(decider_factory: Any) -> Any:
    """The FastAPI app. ``decider_factory`` is an async callable that starts the engine and returns a
    :class:`Decider`; the app shuts the engine down on exit."""
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
        except Exception as e:
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
                "max_buffered_bytes": d.max_buffered,
                "prefix_cache": d.prefix_caching,
                "version": __version__,
            }
        )

    async def systemone(request: Request) -> JSONResponse:
        d: Decider = request.app.state.decider
        started = time.perf_counter()
        try:
            payload = await d.decide_http(request)
        except HTTPException:
            raise
        except ClientDisconnect:
            logger.info("strands-decider: a client disconnected before its body arrived")
            return JSONResponse({"detail": "the client disconnected"}, status_code=400)
        except Exception as e:
            if refusal(e):
                raise HTTPException(status_code=422, detail=str(e)) from e
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


def engine_args(checkpoint: Checkpoint, a: Any, max_model_len: int, takes_images: bool) -> Any:
    """The engine's settings: the plugin's required flags (README, Serving) for this checkpoint's layout."""
    from vllm.engine.arg_utils import AsyncEngineArgs

    config = json.loads(checkpoint.text("config.json"))
    arch = plugin_architecture(config)
    if (
        "text_config" in config
    ):  # the multimodal layout: the base's "vision" mode is in the text config
        overrides: dict[str, Any] = {
            "architectures": [arch],
            "text_config": {"use_bidirectional_attention": None},
        }
        # No count of images: the prompt's length bounds them, and vLLM's profiling caps them per prompt at
        # max_model_len over an image's most tokens.
        limits: dict[str, int] | None = {
            "image": max_model_len if takes_images else 0,
            "video": 0,
            "audio": 0,
        }
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
    length) and the engine's context (``--max-model-len``, default one token more: vLLM 0.33 refuses a prompt that
    fills the context, which leaves no position for the answer token)."""
    max_length = a.max_length or max_length_validated
    max_model_len = a.max_model_len or max_length + 1
    if max_length >= max_model_len:
        raise SystemExit(
            f"strands-decider-vllm serve: --max-length {max_length} must be below --max-model-len {max_model_len}"
        )
    return max_length, max_model_len


@dataclass(frozen=True)
class Settings:
    """What ``serve`` starts with, checked before the engine starts."""

    checkpoint: Checkpoint
    tokenizer: Any
    engine_args: Any
    max_length: int
    image_problem: str | None
    max_body: int
    max_buffered: int


def settings(a: Any) -> Settings:
    """The checkpoint's files and the engine's settings for ``a`` (the CLI's arguments), or SystemExit."""
    core = _core()
    checkpoint = Checkpoint.open(a.model, a.revision)
    try:
        config = core.DecisionConfig.model_validate_json(checkpoint.text("decision_config.json"))
    except ValidationError as e:
        raise SystemExit(
            f"strands-decider-vllm serve: decision_config.json: {errors(e)}. Serve an intermediate checkpoint with "
            "`vllm serve` and pre-rendered token ids instead (README, Serving)."
        ) from None
    try:
        tokenizer = core.load_tokenizer(checkpoint.files["tokenizer.json"].parent)
    except ValueError as e:
        raise SystemExit(f"strands-decider-vllm serve: tokenizer.json: {e}") from e
    takes_images = "image" in config.inputs
    image_problem = None
    if takes_images:
        try:
            core.check_image_processor(checkpoint.files["processor_config.json"])
        except (KeyError, ValueError) as e:
            image_problem = (
                "the checkpoint has no processor_config.json" if isinstance(e, KeyError) else str(e)
            )
            logger.error("strands-decider: image requests will be refused: %s", image_problem)
    max_length, max_model_len = lengths(a, config.max_length_validated)
    max_body = (
        int(a.max_body_mb * 2**20)
        if a.max_body_mb
        else default_max_body(takes_images and not image_problem)
    )
    return Settings(
        checkpoint,
        tokenizer,
        engine_args(checkpoint, a, max_model_len, takes_images),
        max_length,
        image_problem,
        max_body,
        int(a.max_buffered_mb * 2**20) if a.max_buffered_mb else max(BUFFERED_BYTES, max_body),
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
        max_buffered=s.max_buffered,
        body_read_timeout=a.body_read_timeout,
        preprocess_workers=a.preprocess_workers,
        tokenizer=s.tokenizer,
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
    internal sockets on loopback, whatever address the HTTP server binds. A value the caller set is kept."""
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
    os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("NCCL_SOCKET_IFNAME", "lo")
