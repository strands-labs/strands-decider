"""llama.cpp as a runtime of the decision interface (docs/spec.md, The runtime interface).

`LlamaCppBackend` is a `strands_decider.core.Backend` over a converted decision checkpoint: the
decoder as a GGUF and, for images, the vision tower and its projection as an mmproj GGUF. It runs
llama.cpp through a small C library, `deploy/lambda/decision/shim.c`, built against llama.cpp
at the commit and with the patches that `deploy/lambda/decision/` pins. The shim decodes the state
once, its text tokens and image rows as one non-causal batch, then each question on top of it,
under the prefix_bidir mask with no sliding window, and returns the final-norm hidden state at
each question's last token. The core's readout does the rest:

    config = DecisionConfig.load(artifact)
    readout = Readout.load(artifact, config)
    backend = LlamaCppBackend("libdecision.so", "model.gguf", mmproj="mmproj.gguf",
                              processor_config="processor_config.json")
    evaluate(request, tokenizer, readout, config, backend)

Images are prepared as section 11.1 says: decoded as v1 decodes them (first frame, EXIF
orientation, transparency onto white, RGB), resized with Pillow's bicubic resampling, width then
height, to the size the image processor gives the decoded image, and handed to the vision tower at
that size, which mtmd then copies unchanged and scales to [0, 1]. The vision tower's grid of soft
tokens is checked against the image's W x H, and its normalisation against the processor's.

A GGUF Q8_0 decoder conforms by docs/spec.md section 10's bound for that format, measured at E2B
with this runtime: 99.08% top-1 agreement and a mean |dp| of 0.0104 on 1,200 prompts.

Torch-free: numpy, the standard library and, for images only, Pillow.
"""

from __future__ import annotations

import ctypes
import io
import json
import os
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..core.images import (
    IMAGE_CLOSE,
    IMAGE_FORMATS,
    IMAGE_OPEN,
    IMAGE_SOFT,
    MAX_SOFT_TOKENS,
    PATCH_SIZE,
    POOLING_KERNEL_SIZE,
    check_decoded_size,
    check_image_processor,
    resize_image,
)
from ..core.readout import HIDDEN
from ..core.request import NO_VISION

if TYPE_CHECKING:
    from PIL import Image

    from ..core.prompt import RenderedImage

# The three image control tokens by id (docs/spec.md, From text to token ids).
CONTROL_IDS = {IMAGE_OPEN: 255999, IMAGE_SOFT: 258880, IMAGE_CLOSE: 258882}


class ImageError(ValueError):
    """An image of the request that cannot be read or decoded: the request's error, not the
    runtime's. Every other failure in the backend is a RuntimeError."""


# v1's messages for a file that passes the header checks but not Pillow (vision.decode_image).
NOT_READABLE = f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {{}}"
NOT_DECODABLE = "image could not be decoded: {}"

_I = ctypes.POINTER(ctypes.c_int)
_F = ctypes.POINTER(ctypes.c_float)
_D = ctypes.POINTER(ctypes.c_double)

# sd_forward's and sd_encode_image's failures, by return code.
_FORWARD_ERRORS = {
    1: "bad arguments",
    2: "could not create a llama.cpp context",
    3: "the state's decode failed",
    4: "a question's decode failed",
    5: "a question's decode gave no hidden state",
    6: "could not remove a question's cells",
    10: "more runs of image placeholders than images",
    11: "a run of image placeholders shorter than its image",
    12: "could not build the mixed batch",
    7: "could not allocate the batch",
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
    """An image file as v1 decodes it (`strands_decider.vision.decode_image`): the first frame of
    an animated file, its EXIF orientation applied, transparent areas composited onto white, as RGB.
    A file Pillow cannot read or decode gets v1's message, as a ValueError."""
    from PIL import Image, ImageOps

    try:
        img: Image.Image = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
    except Exception as e:
        raise ImageError(NOT_READABLE.format(e)) from e
    try:
        img.load()
        # A malformed EXIF block raises SyntaxError or struct.error here.
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ImageError(NOT_DECODABLE.format(e)) from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def prepare_pixels(image: RenderedImage) -> Image.Image:
    """A request image's file, decoded (`decode_pixels`), its decoded size checked against its
    header (`strands_decider.core.check_decoded_size`), and resized by the spec's bicubic resize
    (`strands_decider.core.resize_image`, width then height) to the W x H the image processor
    gives the decoded size: what the vision tower reads (docs/spec.md, Images, steps 1 and 2)."""
    img = decode_pixels(image.data)
    try:
        w, h = check_decoded_size(image, *img.size)
    except ValueError as e:
        raise ImageError(str(e)) from e
    resized: Image.Image = resize_image(img, w, h)
    return resized


def _ints(xs: Sequence[int]) -> Any:
    return (ctypes.c_int * len(xs))(*[int(x) for x in xs])


class LlamaCppBackend:
    """The decoder of a decision checkpoint on llama.cpp, a `strands_decider.core.Backend` that
    returns `hidden` rows: the final-norm hidden state at each question's last token, float32.

    library: the shim (`deploy/lambda/decision/shim.c`) built against the pinned, patched llama.cpp.
    model: the decoder's GGUF, a stock conversion of the checkpoint (quantized or not).
    mmproj: the vision tower's mmproj GGUF, for images; without it `has_vision` is False and
        `forward` refuses images with v1's text-only message.
    processor_config: the checkpoint's processor_config.json, checked when `mmproj` is given.
    threads: CPU threads per decode; the machine's CPUs by default.

    Calls run one at a time; each already uses every thread. `last_timings` holds the last call's
    milliseconds: the images' preprocessing and encoding, the context, the state and each question.
    """

    outputs = HIDDEN

    def __init__(
        self,
        library: str | os.PathLike[str],
        model: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        processor_config: str | os.PathLike[str] | None = None,
        threads: int | None = None,
    ):
        processor: dict[str, Any] = {}
        if mmproj is not None:
            if processor_config is None:
                raise ValueError("a vision tower needs the checkpoint's processor_config.json")
            check_image_processor(processor_config)
            with open(processor_config, encoding="utf-8") as f:
                cfg = json.load(f)
            processor = cfg.get("image_processor", cfg)
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
        n_embd = ctypes.c_int(0)
        self._handle = lib.sd_load(
            os.fsencode(model), int(threads or os.cpu_count() or 1), ctypes.byref(n_embd)
        )
        if not self._handle:
            raise RuntimeError(f"llama.cpp could not load {os.fspath(model)}")
        self.n_embd = n_embd.value
        self.has_vision = False
        if mmproj is not None:
            rc = lib.sd_load_vision(self._handle, os.fsencode(mmproj), MAX_SOFT_TOKENS)
            if rc != 0:
                self.close()
                raise RuntimeError(
                    f"llama.cpp could not load the vision tower {os.fspath(mmproj)} ({rc})"
                )
            self.has_vision = True
            self.n_embd_inp = lib.sd_n_embd_inp(self._handle)
            try:
                self._check_vision(processor)
            except Exception:
                self.close()
                raise
        self.last_timings: dict[str, Any] = {}

    def _check_vision(self, processor: dict[str, Any]) -> None:
        """The vision tower normalises and cuts pixels as the image processor does: the mean and
        std it subtracts and divides by, after scaling to [0, 1], are the processor's when it
        normalises, and 0 and 1 when it does not; its patches are the processor's."""
        mean, std, patch = (ctypes.c_float * 3)(), (ctypes.c_float * 3)(), ctypes.c_int(0)
        rc = self._lib.sd_vision_info(self._handle, mean, std, ctypes.byref(patch))
        if rc != 0:
            raise RuntimeError(
                f"could not read the vision tower's preprocessing from its mmproj ({rc})"
            )
        normalise = bool(processor.get("do_normalize"))
        want_mean = (
            [float(x) for x in processor.get("image_mean", [0, 0, 0])] if normalise else [0.0] * 3
        )
        want_std = (
            [float(x) for x in processor.get("image_std", [1, 1, 1])] if normalise else [1.0] * 3
        )
        got = ([round(x, 6) for x in mean], [round(x, 6) for x in std], patch.value)
        want = ([round(x, 6) for x in want_mean], [round(x, 6) for x in want_std], PATCH_SIZE)
        if got != want:
            raise RuntimeError(
                f"the vision tower normalises with mean {got[0]}, std {got[1]} and {got[2]}-pixel "
                f"patches; the image processor with mean {want[0]}, std {want[1]} and {want[2]}"
            )

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._lib.sd_free(self._handle)
                self._handle = None

    def encode_image(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
        """One prepared RGB image's soft-token rows, [N, n_embd_inp] float32, and the timings."""
        with self._lock:
            if not self._handle:
                raise RuntimeError("the backend is closed")
            return self._encode_image(image)

    def _encode_image(self, image: Image.Image) -> tuple[np.ndarray, dict[str, float]]:
        """encode_image, under the lock: mtmd's output buffer is shared."""
        if not self.has_vision:
            raise RuntimeError("no vision tower is loaded")
        rgb = image.tobytes()
        if image.mode != "RGB" or len(rgb) != image.width * image.height * 3:
            raise RuntimeError(f"expected an RGB image, not mode {image.mode}")
        out = np.empty((MAX_SOFT_TOKENS, self.n_embd_inp), dtype=np.float32)
        n, grid = ctypes.c_int(0), (ctypes.c_int * 2)()
        wrap, ms = (ctypes.c_int * 2)(), (ctypes.c_double * 2)()
        rc = self._lib.sd_encode_image(
            self._handle, rgb, image.width, image.height, out.ctypes.data_as(_F), MAX_SOFT_TOKENS,
            ctypes.byref(n), grid, wrap, ms,
        )  # fmt: skip
        if rc != 0:
            raise RuntimeError(f"sd_encode_image failed: {_ENCODE_ERRORS.get(rc, rc)}")
        if list(wrap) != [CONTROL_IDS[IMAGE_OPEN], CONTROL_IDS[IMAGE_CLOSE]]:
            raise RuntimeError(
                f"the vision tower wraps an image in {list(wrap)}, not <|image> and <image|>"
            )
        # mtmd gives Gemma 4's soft tokens as one run, n x 1, which cannot show the size it encoded
        # at; tests/test_llama_cpp_shim.py and the image build's shim check hold the rows to
        # transformers' instead. A tower that reports a two-dimensional grid must report W x H's.
        side = POOLING_KERNEL_SIZE * PATCH_SIZE
        want = [image.width // side, image.height // side]
        if list(grid) != want and list(grid) != [n.value, 1]:
            raise RuntimeError(
                f"the vision tower encodes a {image.width} x {image.height} image as a "
                f"{grid[0]} x {grid[1]} grid of soft tokens, not {want[0]} x {want[1]}"
            )
        return out[: n.value].copy(), {"preprocess_ms": ms[0], "encode_ms": ms[1]}

    def _image_rows(
        self, state: Sequence[int], images: Sequence[RenderedImage]
    ) -> tuple[np.ndarray, list[int]]:
        """Each image prepared and encoded, in order, its rows checked against its span."""
        soft = CONTROL_IDS[IMAGE_SOFT]
        spans = []
        for im in images:
            ok = (
                im.start >= 1
                and im.end < len(state)
                and state[im.start - 1] == CONTROL_IDS[IMAGE_OPEN]
                and state[im.end] == CONTROL_IDS[IMAGE_CLOSE]
                and all(t == soft for t in state[im.start : im.end])
            )
            if not ok:
                raise RuntimeError(
                    f"image span [{im.start}, {im.end}) is not a line of placeholders in the state"
                )
            spans.append(im.tokens)
        if sum(1 for t in state if t == soft) != sum(spans):
            raise RuntimeError("the state holds image placeholders outside its images' spans")
        rows, timings, cache = [], [], {}
        for im in images:
            started = time.perf_counter()
            if im.sha256 not in cache:
                pixels = prepare_pixels(im)
                prepared = (time.perf_counter() - started) * 1000
                w, h = pixels.size
                side = POOLING_KERNEL_SIZE * PATCH_SIZE
                if (w // side) * (h // side) != im.tokens:
                    raise RuntimeError(
                        f"a {w} x {h} image gives {(w // side) * (h // side)} soft tokens, not {im.tokens}"
                    )
                embd, ms = self._encode_image(pixels)
                if len(embd) != im.tokens:
                    raise RuntimeError(
                        f"the vision tower gives a {w} x {h} image {len(embd)} soft tokens; the "
                        f"image processor gives it {im.tokens}"
                    )
                cache[im.sha256] = embd
                timings.append({"size": [w, h], "prepare_ms": prepared, **ms})
            rows.append(cache[im.sha256])
        self.last_timings["images"] = timings
        self.last_timings["n_images"] = len(images)
        return np.ascontiguousarray(np.concatenate(rows), dtype=np.float32), spans

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: Sequence[RenderedImage] = (),
    ) -> np.ndarray:
        """Each question run after the shared state under prefix_bidir, read at its last token:
        [len(questions), n_embd] float32, the final-norm hidden states. `images` are the state's,
        each with its file and the span of its placeholders in `state`."""
        if not state or not questions or any(not q for q in questions):
            raise RuntimeError("forward needs a state and questions, none of them empty")
        if images and not self.has_vision:
            raise ValueError(NO_VISION)
        with self._lock:
            if not self._handle:
                raise RuntimeError("the backend is closed")
            self.last_timings = {}
            embd, spans = self._image_rows(state, images) if images else (None, [])
            lens = [len(q) for q in questions]
            out = np.empty((len(questions), self.n_embd), dtype=np.float32)
            ms = (ctypes.c_double * (len(questions) + 2))()
            rc = self._lib.sd_forward(
                self._handle,
                _ints(state),
                len(state),
                CONTROL_IDS[IMAGE_SOFT],
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
            self.last_timings.update(
                {"context_ms": ms[0], "state_ms": ms[1], "question_ms": list(ms[2:])}
            )
        return out
