"""llama.cpp as a decision backend.

The decoder is a GGUF and, for images, the vision tower an mmproj GGUF, both run through
`deploy/lambda/decision/shim.c` (llama.cpp at the pinned commit, with patches/0001). The shim
decodes the state once, its text and image rows as one non-causal batch, then each question on
top of it under prefix_bidir with no sliding window, and returns each question's final-norm hidden
state at its last token. The core's readout does the rest:

    backend = LlamaCppBackend("libdecision.so", "model.gguf", mmproj="mmproj.gguf")
    evaluate(request, load_tokenizer(ckpt), load_readout(ckpt), DecisionConfig.load(ckpt), backend)

Torch-free: numpy and, for images, Pillow.
"""

from __future__ import annotations

import ctypes
import io
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
    check_decoded_size,
    resize_image,
)
from ..core.runtime import NO_VISION

if TYPE_CHECKING:
    from PIL import Image

    from ..core.prompt import RenderedImage

_P, _S, _N = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int
_I, _F, _D = ctypes.POINTER(_N), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_double)
# The shim's functions: their parameters and result. A call that fails returns what failed.
_SHIM = {
    "sd_load": ([_S, _N, _I, _I], _P),
    "sd_free": ([_P], None),
    "sd_load_vision": ([_P, _S, _N], _S),
    "sd_encode_image": ([_P, _S, _N, _N, _N, _F], _S),
    "sd_forward": ([_P, _I, _N, _N, _F, _N, _I, _I, _N, _F, _D], _S),
}


def decode_pixels(data: bytes) -> Image.Image:
    """An image file whose header the core has read, as RGB pixels: the first frame, its EXIF
    orientation applied, and transparent areas on white, as v1 decodes images. Pixels that don't
    decode (a truncated file, a malformed EXIF block) are a ValueError: the request's fault."""
    from PIL import Image, ImageOps

    try:
        img: Image.Image = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        img.load()
        img = ImageOps.exif_transpose(img)
    except Exception as e:  # Pillow raises more than OSError for a bad file
        raise ValueError(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def prepare_pixels(image: RenderedImage) -> Image.Image:
    """A request image decoded and resized with the core's `resize_image`: what the vision tower reads."""
    img = decode_pixels(image.data)
    return resize_image(img, *check_decoded_size(image, *img.size))


def _ints(xs: list[int]) -> Any:
    return (ctypes.c_int * len(xs))(*xs)


def _check(err: bytes | None, what: str) -> None:
    if err:
        raise RuntimeError(f"{what}: {err.decode()}")


class LlamaCppBackend:
    """A decision checkpoint's decoder on llama.cpp. `forward` returns each question's final-norm
    hidden state at its last token, float32. Without `mmproj` it serves text only. Calls run one
    at a time, since each uses every thread; `last_timings` holds the last call's milliseconds."""

    def __init__(
        self,
        library: str | os.PathLike[str],
        model: str | os.PathLike[str],
        *,
        mmproj: str | os.PathLike[str] | None = None,
        threads: int | None = None,
    ):
        lib = ctypes.CDLL(os.fspath(library))
        for name, (argtypes, restype) in _SHIM.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = argtypes, restype
        self._lib = lib
        self._lock = threading.Lock()
        self.last_timings: dict[str, Any] = {}
        n_embd, n_embd_inp = ctypes.c_int(), ctypes.c_int()
        self._handle = lib.sd_load(
            os.fsencode(model),
            threads or os.cpu_count() or 1,
            ctypes.byref(n_embd),
            ctypes.byref(n_embd_inp),
        )
        if not self._handle:
            raise RuntimeError(f"llama.cpp could not load {os.fspath(model)}")
        self._n_embd, self._n_embd_inp = n_embd.value, n_embd_inp.value
        self._vision = mmproj is not None
        if mmproj is not None:
            err = lib.sd_load_vision(self._handle, os.fsencode(mmproj), MAX_SOFT_TOKENS)
            if err:
                self.close()
                _check(err, f"llama.cpp could not load the vision tower {os.fspath(mmproj)}")

    def close(self) -> None:
        with self._lock:
            if self._handle:
                self._lib.sd_free(self._handle)
                self._handle = None

    def _image_rows(self, images: tuple[RenderedImage, ...]) -> np.ndarray:
        """The images' soft-token rows end to end, [placeholders, n_embd_inp] float32; an image
        repeated in the request is encoded once."""
        rows: dict[str, np.ndarray] = {}
        timings = []
        for im in images:
            if im.sha256 in rows:
                continue
            started = time.perf_counter()
            pixels = prepare_pixels(im)
            prepared = time.perf_counter()
            out = np.empty((im.tokens, self._n_embd_inp), dtype=np.float32)
            err = self._lib.sd_encode_image(
                self._handle, pixels.tobytes(), pixels.width, pixels.height, im.tokens,
                out.ctypes.data_as(_F),
            )  # fmt: skip
            _check(err, f"the vision tower, on a {pixels.width} x {pixels.height} image")
            rows[im.sha256] = out
            timings.append({
                "size": list(pixels.size),
                "prepare_ms": (prepared - started) * 1000,
                "encode_ms": (time.perf_counter() - prepared) * 1000,
            })  # fmt: skip
        self.last_timings["images"] = timings
        ordered = [rows[im.sha256] for im in images]
        return np.concatenate(ordered) if ordered else np.empty((0, self._n_embd_inp), np.float32)

    def forward(
        self,
        state: list[int],
        questions: list[list[int]],
        images: tuple[RenderedImage, ...] = (),
    ) -> np.ndarray:
        """Each question run after the shared state under prefix_bidir, read at its last token:
        [len(questions), n_embd] float32. An image whose pixels don't decode is a ValueError,
        raised before any token runs."""
        if images and not self._vision:
            raise ValueError(NO_VISION)
        with self._lock:
            if not self._handle:
                raise RuntimeError("the backend is closed")
            self.last_timings = {}
            embd = self._image_rows(images)
            out = np.empty((len(questions), self._n_embd), dtype=np.float32)
            ms = (ctypes.c_double * (len(questions) + 2))()
            err = self._lib.sd_forward(
                self._handle, _ints(state), len(state), CONTROL_TOKEN_IDS[IMAGE_SOFT],
                embd.ctypes.data_as(_F), len(embd),
                _ints([t for q in questions for t in q]), _ints([len(q) for q in questions]),
                len(questions), out.ctypes.data_as(_F), ms,
            )  # fmt: skip
            _check(err, "sd_forward")
            self.last_timings.update(context_ms=ms[0], state_ms=ms[1], question_ms=ms[2:])
        return out
