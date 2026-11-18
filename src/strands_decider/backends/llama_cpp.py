"""llama.cpp as a decision backend.

The decoder is a GGUF and, for images, the vision tower is an mmproj GGUF, both run through
`deploy/lambda/decision/shim.c` (llama.cpp at the pinned commit, with patches/0001). The shim
decodes the state once, its text and image rows as one non-causal batch, then each question on
top of it under prefix_bidir with no sliding window, and returns each question's final-norm hidden
state at its last token. The core's readout does the rest:

    config = DecisionConfig.load(artifact)
    backend = LlamaCppBackend("libdecision.so", "model.gguf", mmproj="mmproj.gguf",
                              processor_config="processor_config.json")
    evaluate(request, load_tokenizer(artifact), load_readout(artifact), config, backend)

Torch-free: numpy, tokenizers, pydantic and, for images, Pillow.
"""

from __future__ import annotations

import ctypes
import io
import json
import os
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.codes import CONTROL_TOKEN_IDS
from ..core.images import (
    IMAGE_FORMATS,
    IMAGE_SOFT,
    MAX_SOFT_TOKENS,
    PATCH_SIZE,
    check_decoded_size,
    check_image_processor,
    resize_image,
)
from ..core.runtime import NO_VISION

if TYPE_CHECKING:
    from PIL import Image

    from ..core.prompt import RenderedImage

_I = ctypes.POINTER(ctypes.c_int)
_F = ctypes.POINTER(ctypes.c_float)
_D = ctypes.POINTER(ctypes.c_double)

# The shim's return codes.
_FORWARD_ERRORS = {
    1: "bad arguments",
    2: "could not create a llama.cpp context",
    3: "the state's decode failed",
    4: "a question's decode failed",
    5: "a question's decode gave no hidden state",
    6: "could not remove a question's cells",
    7: "could not allocate the batch",
    10: "more runs of image placeholders than images",
    11: "a run of image placeholders shorter than its image",
    12: "could not build the mixed batch",
    13: "fewer runs of image placeholders than images, or a run longer than its image",
    14: "the images' row counts are not positive or do not add up",
}
_ENCODE_ERRORS = {
    1: "bad arguments",
    2: "could not allocate the image",
    3: "mtmd could not tokenize the image",
    4: "mtmd made no image chunk",
    5: "the vision tower failed",
    6: "the image has more soft tokens than allowed",
}


def decode_pixels(data: bytes) -> Image.Image:
    """An image file as RGB pixels: the first frame, its EXIF orientation applied, and transparent
    areas on white, as v1 decodes images. A file Pillow can't read or decode is a ValueError."""
    from PIL import Image, ImageOps

    try:
        img: Image.Image = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
    except Exception as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    try:
        img.load()
        img = ImageOps.exif_transpose(img)  # a malformed EXIF block raises here
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def prepare_pixels(image: RenderedImage) -> Image.Image:
    """A request image decoded and resized with the core's `resize_image`: what the vision tower reads."""
    img = decode_pixels(image.data)
    w, h = check_decoded_size(image, *img.size)
    return resize_image(img, w, h)


def _ints(xs: list[int]) -> Any:
    return (ctypes.c_int * len(xs))(*xs)


class LlamaCppBackend:
    """A decision checkpoint's decoder on llama.cpp. `forward` returns each question's final-norm
    hidden state at its last token, float32. Without `mmproj` it serves text only; with it,
    `processor_config` is the checkpoint's processor_config.json. Calls run one at a time, since
    each uses every thread; `last_timings` holds the last call's milliseconds."""

    def __init__(
        self,
        library: str | os.PathLike[str],
        model: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        processor_config: str | os.PathLike[str] | None = None,
        threads: int | None = None,
    ):
        lib = ctypes.CDLL(os.fspath(library))
        lib.sd_load.argtypes = [ctypes.c_char_p, ctypes.c_int, _I]
        lib.sd_load.restype = ctypes.c_void_p
        lib.sd_free.argtypes = [ctypes.c_void_p]
        lib.sd_free.restype = None
        lib.sd_load_vision.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        lib.sd_load_vision.restype = ctypes.c_int
        lib.sd_n_embd_inp.argtypes = [ctypes.c_void_p]
        lib.sd_n_embd_inp.restype = ctypes.c_int
        lib.sd_vision_info.argtypes = [ctypes.c_void_p, _F, _F, _I]
        lib.sd_vision_info.restype = ctypes.c_int
        lib.sd_encode_image.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, _F, ctypes.c_int, _I, _I, _I, _D,
        ]  # fmt: skip
        lib.sd_encode_image.restype = ctypes.c_int
        lib.sd_forward.argtypes = [
            ctypes.c_void_p, _I, ctypes.c_int, ctypes.c_int, _F, _I, ctypes.c_int, ctypes.c_int, _I, _I,
            ctypes.c_int, _F, _D,
        ]  # fmt: skip
        lib.sd_forward.restype = ctypes.c_int
        self._lib = lib
        self._lock = threading.Lock()
        self.last_timings: dict[str, Any] = {}
        n_embd = ctypes.c_int(0)
        self._handle = lib.sd_load(
            os.fsencode(model), threads or os.cpu_count() or 1, ctypes.byref(n_embd)
        )
        if not self._handle:
            raise RuntimeError(f"llama.cpp could not load {os.fspath(model)}")
        self.n_embd = n_embd.value
        self.has_vision = mmproj is not None
        if mmproj is None:
            return
        if processor_config is None:
            raise ValueError("a vision tower needs the checkpoint's processor_config.json")
        try:
            with open(processor_config, encoding="utf-8") as f:
                processor = json.load(f)
            check_image_processor(processor)
            rc = lib.sd_load_vision(self._handle, os.fsencode(mmproj), MAX_SOFT_TOKENS)
            if rc != 0:
                raise RuntimeError(f"llama.cpp could not load the vision tower {mmproj} ({rc})")
            self.n_embd_inp = lib.sd_n_embd_inp(self._handle)
            self._check_normalisation(processor.get("image_processor", processor))
        except Exception:
            self.close()
            raise

    def _check_normalisation(self, processor: dict[str, Any]) -> None:
        """That the mmproj normalises pixels and cuts patches as the image processor does. The
        converter writes these into the mmproj, so a llama.cpp pin that changes them shows here."""
        mean, std, patch = (ctypes.c_float * 3)(), (ctypes.c_float * 3)(), ctypes.c_int(0)
        rc = self._lib.sd_vision_info(self._handle, mean, std, ctypes.byref(patch))
        if rc != 0:
            raise RuntimeError(f"could not read the vision tower's preprocessing ({rc})")
        normalise = processor.get("do_normalize")
        want = (
            processor.get("image_mean") if normalise else [0, 0, 0],
            processor.get("image_std") if normalise else [1, 1, 1],
            PATCH_SIZE,
        )
        got = (list(mean), list(std), patch.value)
        if not (
            np.allclose(got[0], want[0]) and np.allclose(got[1], want[1]) and got[2] == want[2]
        ):
            raise RuntimeError(
                f"the vision tower preprocesses with {got}, the processor with {want}"
            )

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._lib.sd_free(self._handle)
                self._handle = None

    def _encode_image(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
        """An RGB image's soft-token rows, [N, n_embd_inp] float32, and the milliseconds taken."""
        out = np.empty((MAX_SOFT_TOKENS, self.n_embd_inp), dtype=np.float32)
        # mtmd also reports its grid and wrap tokens; the prompt places the image's own tokens.
        n, grid, wrap = ctypes.c_int(0), (ctypes.c_int * 2)(), (ctypes.c_int * 2)()
        ms = (ctypes.c_double * 2)()
        rc = self._lib.sd_encode_image(
            self._handle, image.tobytes(), image.width, image.height, out.ctypes.data_as(_F),
            MAX_SOFT_TOKENS, ctypes.byref(n), grid, wrap, ms,
        )  # fmt: skip
        if rc != 0:
            raise RuntimeError(f"sd_encode_image failed: {_ENCODE_ERRORS.get(rc, rc)}")
        return out[: n.value].copy(), {"preprocess_ms": ms[0], "encode_ms": ms[1]}

    def _image_rows(self, images: tuple[RenderedImage, ...]) -> np.ndarray:
        """Each image's soft-token rows, in order; an image repeated in the request is encoded once."""
        rows, timings, cache = [], [], {}
        for im in images:
            if im.sha256 not in cache:
                started = time.perf_counter()
                pixels = prepare_pixels(im)
                prepared_ms = (time.perf_counter() - started) * 1000
                embd, ms = self._encode_image(pixels)
                if len(embd) != im.tokens:
                    raise RuntimeError(
                        f"the vision tower gives a {pixels.width} x {pixels.height} image "
                        f"{len(embd)} soft tokens; the prompt holds {im.tokens}"
                    )
                cache[im.sha256] = embd
                timings.append({"size": list(pixels.size), "prepare_ms": prepared_ms, **ms})
            rows.append(cache[im.sha256])
        self.last_timings.update(images=timings, n_images=len(images))
        return np.ascontiguousarray(np.concatenate(rows), dtype=np.float32)

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: tuple[RenderedImage, ...] = (),
    ) -> np.ndarray:
        """Each question run after the shared state under prefix_bidir, read at its last token:
        [len(questions), n_embd] float32."""
        if images and not self.has_vision:
            raise ValueError(NO_VISION)
        with self._lock:
            if not self._handle:
                raise RuntimeError("the backend is closed")
            self.last_timings = {}
            embd = self._image_rows(images) if images else None
            spans = [im.tokens for im in images]
            lens = [len(q) for q in questions]
            out = np.empty((len(questions), self.n_embd), dtype=np.float32)
            ms = (ctypes.c_double * (len(questions) + 2))()
            rc = self._lib.sd_forward(
                self._handle,
                _ints(state),
                len(state),
                CONTROL_TOKEN_IDS[IMAGE_SOFT],
                embd.ctypes.data_as(_F) if embd is not None else None,
                _ints(spans) if spans else None,
                len(spans),
                sum(spans),
                _ints([t for q in questions for t in q]),
                _ints(lens),
                len(lens),
                out.ctypes.data_as(_F),
                ms,
            )
            if rc != 0:
                raise RuntimeError(f"sd_forward failed: {_FORWARD_ERRORS.get(rc, rc)}")
            self.last_timings.update(context_ms=ms[0], state_ms=ms[1], question_ms=list(ms[2:]))
        return out
