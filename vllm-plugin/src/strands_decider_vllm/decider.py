"""A System One request answered on vLLM's engine: the core renders it, the server decodes and
resizes its images, and each question is one engine request.

vLLM 0.34's endpoints don't fit: ``/v1/completions`` takes token ids but no images, the chat
endpoints apply a chat template over the core's prompt, and the token-in endpoint lets a client
choose an image's cache key. So each engine request is the question's prompt with each image as the
one ``<|image|>`` vLLM's processor expands, each image's file sha256 as its id, and S in
``extra_args``; its expanded prompt must be the rendered one.
"""

from __future__ import annotations

import asyncio
import io
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any

from PIL import Image, ImageOps
from pydantic import ValidationError
from strands_decider import core
from strands_decider.core.images import IMAGE_FORMATS
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from . import PREFIX_END_KEY
from .settings import Settings

_IMAGE_ID = core.CONTROL_TOKEN_IDS["<|image|>"]


class Refused(ValueError):
    """A request the server won't answer (HTTP 422)."""


@dataclass(frozen=True)
class Prepared:
    """A rendered request without its image files, and its images' pixels at the core's W x H."""

    rendered: core.RenderedRequest
    pixels: list[Image.Image]


class Decider:
    def __init__(self, engine: Any, settings: Settings) -> None:
        self.engine, self.settings = engine, settings
        # Two pools, so that a request without images never waits behind an image's decode.
        self._render_pool = ThreadPoolExecutor(settings.preprocess_workers, "decider-render")
        self._decode_pool = ThreadPoolExecutor(settings.preprocess_workers, "decider-decode")

    def close(self) -> None:
        for pool in (self._render_pool, self._decode_pool):
            pool.shutdown(wait=False, cancel_futures=True)

    async def prepare(self, body: bytes | bytearray) -> Prepared:
        """The request rendered and its images decoded, off the event loop; Refused if invalid."""
        loop = asyncio.get_running_loop()
        rendered = await loop.run_in_executor(self._render_pool, self._render, body)
        if not rendered.images:
            return Prepared(rendered, [])
        pixels = await loop.run_in_executor(
            self._decode_pool, lambda: [_pixels(im) for im in rendered.images]
        )
        # The files go, so that a request holds only its pixels while the engine runs.
        no_files = tuple(replace(im, data=b"") for im in rendered.images)
        questions = {n: replace(q, images=no_files) for n, q in rendered.questions.items()}
        return Prepared(replace(rendered, questions=questions), pixels)

    async def answer(self, prepared: Prepared) -> dict[str, Any]:
        """The response: every question's answer, and the prompt tokens vLLM's cache served."""
        questions = list(prepared.rendered.questions.values())
        # The first question runs alone, so that the others reuse its cached state.
        first = await self._ask(questions[0], prepared.pixels)
        try:
            async with asyncio.TaskGroup() as group:
                rest = [group.create_task(self._ask(q, prepared.pixels)) for q in questions[1:]]
        except ExceptionGroup as failed:  # the first failure cancelled the others
            raise failed.exceptions[0] from None
        results = [first, *(task.result() for task in rest)]
        answers = core.read(
            [probs for probs, _ in results], self.settings.config, prepared.rendered
        )
        return {
            "model": self.settings.served_name,
            "answers": answers,
            "usage": {
                "input_tokens": prepared.rendered.input_tokens,
                "output_tokens": len(answers),
                "cached_tokens": sum(cached for _, cached in results),
            },
        }

    def _render(self, body: bytes | bytearray) -> core.RenderedRequest:
        try:
            request = core.Request.model_validate_json(body)
        except ValidationError as e:
            raise Refused(_describe(e)) from None
        if request.images and not self.settings.takes_images:
            raise Refused("this checkpoint does not take images")
        try:
            return core.render(
                request,
                self.settings.tokenizer,
                max_length=self.settings.max_length,
                token_budget=self.settings.token_budget,
                max_token_budget=self.settings.max_token_budget,
            )
        except ValueError as e:  # an image, or the prompt's length
            raise Refused(str(e)) from e

    async def _ask(
        self, q: core.RenderedQuestion, pixels: list[Image.Image]
    ) -> tuple[list[float], int]:
        """One question as one engine request: its codes' probabilities, and how many of its prompt
        tokens came from vLLM's prefix cache."""
        prompt: dict[str, Any] = {"prompt_token_ids": _one_id_per_image(q) + list(q.question_ids)}
        if pixels:
            prompt["multi_modal_data"] = {"image": pixels}
            # The budget is part of an image's name: vLLM's caches key on it, and the same file
            # at another budget has other pixels and another number of placeholders.
            uuids = [f"{im.sha256}-{im.token_budget}" for im in q.images]
            prompt["multi_modal_uuids"] = {"image": uuids}
            prompt["mm_processor_kwargs"] = {  # already at each image's W x H
                "do_resize": False,
                "max_soft_tokens": max(im.token_budget for im in q.images),
            }
        params = SamplingParams(
            max_tokens=1,
            logprobs=len(q.code_token_ids),
            allowed_token_ids=list(q.code_token_ids),
            extra_args={PREFIX_END_KEY: q.state_len},
            detokenize=False,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        outputs = [o async for o in self.engine.generate(prompt, params, f"sd-{uuid.uuid4().hex}")]
        output = outputs[-1]  # FINAL_ONLY: the one output
        if list(output.prompt_token_ids) != list(q.ids):
            raise RuntimeError(
                f"vLLM expanded the prompt to {len(output.prompt_token_ids)} tokens, not the "
                f"rendered {len(q.ids)}: its image token counts are not the core's"
            )
        logprobs = output.outputs[0].logprobs[0]
        probs = [math.exp(logprobs[t].logprob) for t in q.code_token_ids]
        return probs, output.num_cached_tokens or 0


def _describe(e: ValidationError) -> str:
    """pydantic's errors as ``where: what``."""
    out = []
    for err in e.errors(include_url=False, include_input=False):
        where, what = ".".join(map(str, err["loc"])), err["msg"].removeprefix("Value error, ")
        out.append(f"{where}: {what}" if where else what)
    return "; ".join(out)


def _pixels(image: core.RenderedImage) -> Image.Image:
    """A request image decoded as v1 decodes it (its first frame, upright by its EXIF orientation,
    transparency on white, RGB), then resized to the core's W x H."""
    try:
        img = Image.open(io.BytesIO(image.data), formats=IMAGE_FORMATS)
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:  # Pillow raises more than OSError for a bad file
        raise Refused(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    img = img.convert("RGB")
    try:
        width, height = core.check_decoded_size(image, *img.size)
    except ValueError as e:
        raise Refused(str(e)) from e
    return core.resize_image(img, width, height)


def _one_id_per_image(q: core.RenderedQuestion) -> list[int]:
    """The state with each image's ``<|image>``, N ``<|image|>``, ``<image|>`` replaced by the one
    ``<|image|>`` that vLLM's Gemma 4 processor expands back to them."""
    out: list[int] = []
    pos = 0
    for im in q.images:
        out += [*q.state_ids[pos : im.start - 1], _IMAGE_ID]
        pos = im.end + 1
    return out + list(q.state_ids[pos:])
