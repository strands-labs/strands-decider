"""``strands-decider-vllm serve``: the System One request, images included, on vLLM's engine in
process (README, Serving).

vLLM 0.33's endpoints don't fit: ``/v1/completions`` takes token ids but no images, the chat
endpoints apply a chat template over the core's prompt, and the token-in endpoint lets a client
choose an image's cache key. So the server renders each request with the core, decodes and resizes
its images, and sends vLLM one request per question: the prompt with each image collapsed to the one
``<|image|>`` vLLM's processor expands, each image's file sha256 as its id, and S in
``vllm_xargs.prefix_end``. It checks that vLLM's expanded prompt is the rendered one.

The server has no authentication and binds 127.0.0.1 by default. Expose it only behind the
deployment's own authentication: SageMaker's IAM-signed endpoint or an authenticating proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import io
import json
import math
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect
from strands_decider import core
from strands_decider.core.images import IMAGE_FORMATS, MAX_IMAGE_BYTES
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind

from . import ARCHITECTURES, PREFIX_END_KEY, SERVER_CONFIG_KEY, __version__, logger

IMAGE_ID = core.CONTROL_TOKEN_IDS["<|image|>"]
CONFIG_FILES = ("config.json", "decision_config.json", "processor_config.json", "tokenizer.json")
#: Bodies up to this size are read at once; a larger one first reserves its size from the budget.
SMALL_BODY = 2**20
TEXT_BYTES = 8 * 2**20


def default_max_body(takes_images: bool) -> int:
    """8 MiB of text, and four of the core's largest image files in base64 (93 MiB)."""
    base64_file = 4 * math.ceil(MAX_IMAGE_BYTES / 3) + 64  # with room for a data URI and quotes
    return TEXT_BYTES + (4 * base64_file if takes_images else 0)


class Refused(ValueError):
    """A request the server won't answer (HTTP 422)."""


class Mismatch(RuntimeError):
    """vLLM's expanded prompt differs from the rendered one: a bug, never the client's (HTTP 500)."""


def errors(e: ValidationError) -> str:
    """pydantic's errors as ``where: what``, for a 422."""
    out = []
    for err in e.errors(include_url=False, include_input=False):
        where, what = ".".join(map(str, err["loc"])), err["msg"].removeprefix("Value error, ")
        out.append(f"{where}: {what}" if where else what)
    return "; ".join(out)


# --- images ---------------------------------------------------------------------------------------


def decode_image(data: bytes) -> Any:
    """A request image's pixels, as v1 decodes them: its first frame, upright by its EXIF
    orientation, transparency on white, RGB. The core has checked its header."""
    from PIL import Image, ImageOps

    try:
        img = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:  # Pillow raises more than OSError for a bad file
        raise Refused(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def prepare_image(image: core.RenderedImage) -> Any:
    """The image decoded and resized to the core's W x H, which vLLM's processor then keeps."""
    img = decode_image(image.data)
    try:
        width, height = core.check_decoded_size(image, *img.size)
    except ValueError as e:
        raise Refused(str(e)) from e
    return core.resize_image(img, width, height)


def collapse(state_ids: tuple[int, ...], images: tuple[core.RenderedImage, ...]) -> list[int]:
    """The state with each image's ``<|image>``, N ``<|image|>``, ``<image|>`` replaced by the one
    ``<|image|>`` that vLLM's Gemma 4 processor expands back to them."""
    out: list[int] = []
    pos = 0
    for im in images:
        out += [*state_ids[pos : im.start - 1], IMAGE_ID]
        pos = im.end + 1
    return out + list(state_ids[pos:])


# --- the checkpoint and the engine's settings -----------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """What the server starts with: the CLI's arguments, checked against the checkpoint."""

    config: core.DecisionConfig
    tokenizer: Any
    engine_args: AsyncEngineArgs
    served_name: str
    max_length: int
    max_body: int
    max_buffered: int
    body_read_timeout: float
    preprocess_workers: int

    @property
    def takes_images(self) -> bool:
        return "image" in self.config.inputs


def checkpoint_dir(model: str, revision: str | None) -> tuple[Path, str | None]:
    """The directory of the checkpoint's config files, and the revision the engine loads: for a Hub
    repo, the commit those files were read at."""
    if Path(model).is_dir():
        return Path(model), revision
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model, revision=revision, allow_patterns=list(CONFIG_FILES)))
    return path, path.name  # .../snapshots/<commit>


def settings(a: argparse.Namespace) -> Settings:
    """The server's settings, or SystemExit for a checkpoint or flags it can't serve."""
    try:
        path, revision = checkpoint_dir(a.model, a.revision)
        config = core.DecisionConfig.load(path)
        tokenizer = core.load_tokenizer(path)
        if "image" in config.inputs:
            core.check_image_processor(path)
        base = json.loads((path / "config.json").read_text("utf-8"))["architectures"][0]
        if base not in ARCHITECTURES:
            raise ValueError(f"config.json names {base}, which is no Strands Decider layout")
    except (OSError, ValueError) as e:  # pydantic's ValidationError is a ValueError
        raise SystemExit(f"strands-decider-vllm serve: {a.model}: {e}") from None
    max_length = a.max_length or config.max_length_validated
    # vLLM refuses a prompt that fills the context: it leaves no position for the answer.
    max_model_len = a.max_model_len or max_length + 1
    if max_length >= max_model_len:
        raise SystemExit(
            f"strands-decider-vllm serve: --max-length {max_length} must be below "
            f"--max-model-len {max_model_len}"
        )
    takes_images = "image" in config.inputs
    # The plugin's architecture, and the base's "vision" mode off (model.check_config).
    vision_off = {"use_bidirectional_attention": None}
    multimodal = base != "Gemma4TextModel"
    engine_args = AsyncEngineArgs(
        model=a.model,
        revision=revision,
        served_model_name=[a.served_model_name],
        dtype=a.dtype,
        seed=0,
        hf_overrides={
            "architectures": [ARCHITECTURES[base][0]],
            **({"text_config": vision_off} if multimodal else vision_off),
        },
        max_model_len=max_model_len,
        max_num_batched_tokens=max(max_model_len, a.max_num_batched_tokens or 0),
        max_num_seqs=a.max_num_seqs,
        enable_chunked_prefill=False,
        enable_prefix_caching=a.enable_prefix_caching,
        prefix_caching_hash_algo="sha256",
        # Cache checkpoints at every block of the sliding-window KV groups.
        prefix_cache_retention_interval=None if a.enable_prefix_caching else 0,
        block_size=a.block_size,
        generation_config="vllm",
        logprobs_mode="processed_logprobs",
        max_logprobs=255,
        gpu_memory_utilization=a.gpu_memory_utilization,
        tensor_parallel_size=a.tensor_parallel_size,
        disable_custom_all_reduce=a.tensor_parallel_size > 1,
        enforce_eager=a.enforce_eager,
        additional_config={SERVER_CONFIG_KEY: __version__},
        # No count of images: the prompt's length bounds them. ({} is vLLM's default.)
        limit_mm_per_prompt=(
            {"image": max_model_len if takes_images else 0, "video": 0, "audio": 0}
            if multimodal
            else {}
        ),
    )
    max_body = int(a.max_body_mb * 2**20) if a.max_body_mb else default_max_body(takes_images)
    return Settings(
        config=config,
        tokenizer=tokenizer,
        engine_args=engine_args,
        served_name=a.served_model_name,
        max_length=max_length,
        max_body=max_body,
        max_buffered=max(int(a.max_buffered_mb * 2**20), max_body),
        body_read_timeout=a.body_read_timeout,
        preprocess_workers=max(1, a.preprocess_workers),
    )


# --- requests -------------------------------------------------------------------------------------


async def all_or_none(coros: list[Any]) -> list[Any]:
    """The coroutines' results; on the first failure the others are cancelled (vLLM aborts their
    engine requests) and the failure is raised."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class Decider:
    """The engine, and what the server does around it."""

    def __init__(self, engine: Any, s: Settings) -> None:
        self.engine, self.s = engine, s
        # Two pools, so that a request without images never waits behind an image's decode.
        self._render = ThreadPoolExecutor(s.preprocess_workers, "strands-decider-render")
        self._decode = ThreadPoolExecutor(s.preprocess_workers, "strands-decider-decode")
        self.budget = ByteBudget(s.max_buffered)

    def close(self) -> None:
        for pool in (self._render, self._decode):
            pool.shutdown(wait=False, cancel_futures=True)

    def render(self, body: bytes | bytearray) -> core.RenderedRequest:
        try:
            request = core.Request.model_validate_json(body)
        except ValidationError as e:
            raise Refused(errors(e)) from None
        if request.images and not self.s.takes_images:
            raise Refused("this checkpoint does not take images")
        try:
            return core.render(request, self.s.tokenizer, max_length=self.s.max_length)
        except ValueError as e:  # an image, or the prompt's length
            raise Refused(str(e)) from e

    async def prepare(self, body: bytes | bytearray) -> tuple[core.RenderedRequest, list[Any]]:
        """The rendered request without its image files, and its images resized to W x H."""
        loop = asyncio.get_running_loop()
        rendered = await loop.run_in_executor(self._render, self.render, body)
        if not rendered.images:
            return rendered, []
        images = await loop.run_in_executor(
            self._decode, lambda: [prepare_image(im) for im in rendered.images]
        )
        # The files go, so that a request holds only its pixels while the engine runs.
        files_dropped = tuple(replace(im, data=b"") for im in rendered.images)
        questions = {n: replace(q, images=files_dropped) for n, q in rendered.questions.items()}
        return replace(rendered, questions=questions), images

    async def question(
        self, q: core.RenderedQuestion, images: list[Any]
    ) -> tuple[list[float], int]:
        """One question as one engine request: its codes' probabilities, and how many prompt tokens
        came from vLLM's prefix cache."""
        prompt: dict[str, Any] = {
            "prompt_token_ids": collapse(q.state_ids, q.images) + list(q.question_ids)
        }
        if images:
            prompt["multi_modal_data"] = {"image": images}
            prompt["multi_modal_uuids"] = {"image": [im.sha256 for im in q.images]}
            prompt["mm_processor_kwargs"] = {"do_resize": False}  # already at W x H
        sp = SamplingParams(
            max_tokens=1,
            logprobs=len(q.code_token_ids),
            allowed_token_ids=list(q.code_token_ids),
            extra_args={PREFIX_END_KEY: q.state_len},
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        outputs = [o async for o in self.engine.generate(prompt, sp, f"sd-{uuid.uuid4().hex}")]
        final = outputs[-1]  # FINAL_ONLY: the one output
        if list(final.prompt_token_ids) != list(q.ids):
            raise Mismatch(
                f"vLLM expanded the prompt to {len(final.prompt_token_ids)} tokens, not the "
                f"rendered {len(q.ids)}: its image token counts are not the core's"
            )
        logprobs = final.outputs[0].logprobs[0]
        probs = [math.exp(logprobs[t].logprob) for t in q.code_token_ids]
        return probs, final.num_cached_tokens or 0

    async def answer(self, rendered: core.RenderedRequest, images: list[Any]) -> dict[str, Any]:
        questions = list(rendered.questions.values())
        # The first question runs alone, so that the others reuse its cached state.
        alone = 1 if self.s.engine_args.enable_prefix_caching else 0
        results = [await self.question(q, images) for q in questions[:alone]]
        results += await all_or_none([self.question(q, images) for q in questions[alone:]])
        answers = core.read([p for p, _ in results], self.s.config, rendered)
        return {
            "model": self.s.served_name,
            "answers": answers,
            "usage": {
                "input_tokens": rendered.input_tokens,
                "output_tokens": len(answers),
                "cached_tokens": sum(c for _, c in results),
            },
        }

    async def decide(self, body: bytes | bytearray) -> dict[str, Any]:
        return await self.answer(*await self.prepare(body))

    async def decide_http(self, request: Request) -> dict[str, Any]:
        """A large body's reservation is given back once its images are decoded."""
        body, reserved = await read_body(
            request, self.s.max_body, self.budget, self.s.body_read_timeout
        )
        try:
            prepared = await self.prepare(body)
        finally:
            del body
            if reserved:
                self.budget.give(reserved)
        return await self.answer(*prepared)


class ByteBudget:
    """The body bytes that large requests buffer at once, granted in arrival order so that a large
    body isn't starved by smaller ones."""

    def __init__(self, capacity: int) -> None:
        self.capacity, self.used = capacity, 0
        self._waiters: collections.deque[tuple[int, asyncio.Future]] = collections.deque()

    async def take(self, n: int, timeout: float | None = None) -> int:
        """Reserves ``n`` bytes, at most the whole budget, and returns the reservation; TimeoutError
        after ``timeout`` seconds, having taken nothing."""
        n = min(n, self.capacity)
        if not self._waiters and self.used + n <= self.capacity:
            self.used += n
            return n
        waiter = (n, asyncio.get_running_loop().create_future())
        self._waiters.append(waiter)
        try:
            await asyncio.wait_for(waiter[1], timeout)
        except BaseException:
            if waiter[1].done() and not waiter[1].cancelled():
                self.give(n)  # granted just as the wait ended
            elif waiter in self._waiters:
                self._waiters.remove(waiter)
                self._grant()
            raise
        return n

    def give(self, n: int) -> None:
        self.used -= n
        self._grant()

    def _grant(self) -> None:
        while self._waiters and self.used + self._waiters[0][0] <= self.capacity:
            n, future = self._waiters.popleft()
            if not future.done():  # a cancelled wait takes nothing
                self.used += n
                future.set_result(None)


def too_large(limit: int) -> HTTPException:
    return HTTPException(413, f"the body is over {limit} bytes (--max-body-mb)")


async def read_body(
    request: Request, limit: int, budget: ByteBudget, timeout: float
) -> tuple[bytearray, int]:
    """The body, and the bytes it reserved from ``budget``, which the caller gives back. A body
    declared over SMALL_BODY reserves its size before it's read; one without a length reserves
    ``limit`` once past SMALL_BODY and gives back what it doesn't use. HTTP 413 past ``limit``, 408
    when no chunk arrives for ``timeout`` seconds and 503 when a reservation waits that long, so a
    stalled or vanished client gives its reservation back."""
    declared = request.headers.get("content-length")
    n = int(declared) if declared is not None else None  # the HTTP parser has checked it
    if n is not None and n > limit:
        raise too_large(limit)

    async def reserve(size: int) -> int:
        try:
            return await budget.take(size, timeout)
        except TimeoutError:
            raise HTTPException(
                503, "busy with other large requests; retry", headers={"Retry-After": "5"}
            ) from None

    reserved = 0
    try:
        if n is not None and n > SMALL_BODY:
            reserved = await reserve(n)
        body = bytearray()
        chunks = request.stream()
        while True:
            try:
                body += await asyncio.wait_for(anext(chunks), timeout)
            except StopAsyncIteration:
                break
            except TimeoutError:
                raise HTTPException(
                    408, f"no part of the body arrived for {timeout:g} s (--body-read-timeout)"
                ) from None
            if len(body) > limit:
                raise too_large(limit)
            if not reserved and len(body) > SMALL_BODY:
                reserved = await reserve(limit)
        if n is None and reserved > len(body):
            budget.give(reserved - len(body))
            reserved = len(body)
    except BaseException:  # a 408, 413 or 503, a client that left, a cancellation
        if reserved:
            budget.give(reserved)
        raise
    return body, reserved


# --- the app --------------------------------------------------------------------------------------


def build_app(start: Any) -> FastAPI:
    """The HTTP app. ``start`` is an async callable that starts the engine and returns a Decider."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.decider = await start()
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
            raise HTTPException(503, "engine unhealthy") from e
        return JSONResponse(
            {
                "status": "ok",
                "model": d.s.served_name,
                "inputs": list(d.s.config.inputs),
                "max_length": d.s.max_length,
                "max_body_bytes": d.s.max_body,
                "version": __version__,
            }
        )

    async def systemone(request: Request) -> JSONResponse:
        started = time.perf_counter()
        try:
            payload = await request.app.state.decider.decide_http(request)
        except HTTPException:
            raise
        except ClientDisconnect:
            return JSONResponse({"detail": "the client disconnected"}, status_code=400)
        except (Refused, VLLMValidationError) as e:  # the engine core's refusals, too
            raise HTTPException(422, str(e)) from e
        except Exception:
            error_id = uuid.uuid4().hex[:16]
            logger.exception("strands-decider: request failed (error id %s)", error_id)
            return JSONResponse(
                {"detail": "internal server error", "error_id": error_id}, status_code=500
            )
        payload["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return JSONResponse(payload)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/ping", health, methods=["GET"])  # SageMaker's
    app.add_api_route("/v1/systemone", systemone, methods=["POST"])
    app.add_api_route("/invocations", systemone, methods=["POST"])  # SageMaker's
    return app


def serve(a: argparse.Namespace) -> None:
    import uvicorn
    from vllm.v1.engine.async_llm import AsyncLLM

    if a.host not in ("127.0.0.1", "::1", "localhost"):
        logger.warning(
            "strands-decider-vllm serve: binding %s. The server has no authentication: anyone who "
            "can reach this address can send it requests. Put the deployment's own authentication "
            "in front of it (SageMaker's IAM-signed endpoint, an authenticating proxy), or bind "
            "127.0.0.1 and use an SSH port forward.",
            a.host,
        )
    s = settings(a)

    async def start() -> Decider:
        return Decider(AsyncLLM.from_engine_args(s.engine_args), s)

    uvicorn.run(
        build_app(start),
        host=a.host,
        port=a.port,
        log_level="info",
        limit_concurrency=a.limit_concurrency,
    )
