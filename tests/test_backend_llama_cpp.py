"""The llama.cpp backend without llama.cpp: its pixels against the core's pixel goldens, and what
reaches a stand-in for the shim's C API. tests/test_llama_cpp_shim.py runs the real library.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import json
import types
from dataclasses import replace

import numpy as np
import pytest
from core_support import FIXTURES, PIXEL_FILES, image_bytes

from strands_decider.backends import llama_cpp
from strands_decider.backends.llama_cpp import LlamaCppBackend, image_pixels
from strands_decider.core import (
    CONTROL_TOKEN_IDS,
    NO_VISION,
    RenderedImage,
    decode_image,
    resized_size,
)

pytest.importorskip("PIL.Image")
SOFT = CONTROL_TOKEN_IDS["<|image|>"]
N_EMBD, N_EMBD_INP = 6, 4


def _image(name: str) -> RenderedImage:
    data = image_bytes(name)
    e = decode_image(base64.b64encode(data).decode())
    return RenderedImage(data, e.format, e.width, e.height, e.sha256, 1, e.tokens)


@pytest.mark.parametrize("name", sorted(PIXEL_FILES))
def test_the_pixels_are_the_goldens(name):
    """v1's decode, then the core's resize to the processor's size: the core's pixel goldens
    (tests/fixtures/core/image_pixels.json)."""
    golden = json.loads((FIXTURES / "image_pixels.json").read_text("utf-8"))["images"][name]
    pixels = image_pixels(_image(name))
    assert [list(pixels.size), hashlib.sha256(pixels.tobytes()).hexdigest()] == [
        golden["resized"], golden["resized_sha256"]
    ]  # fmt: skip


class FakeShim:
    """The shim's C API in Python, recording what reaches it."""

    def __init__(self):
        self.fail: dict[str, bytes] = {}  # function name -> the error it returns
        self.encoded: list[tuple[int, int, int]] = []
        self.forwarded: list[dict] = []

    def sd_load(self, path, threads, n_embd, n_embd_inp):
        n_embd._obj.value, n_embd_inp._obj.value = N_EMBD, N_EMBD_INP
        return None if "sd_load" in self.fail else 1234

    def sd_free(self, handle):
        pass

    def sd_cpu_backend(self):
        return b"libggml-cpu-armv8.6_1.so"

    def sd_load_vision(self, handle, path, max_tokens):
        return self.fail.get("sd_load_vision")

    def sd_encode_image(self, handle, rgb, w, h, n_rows, out):
        assert len(rgb) == w * h * 3
        self.encoded.append((w, h, n_rows))
        for i in range(n_rows * N_EMBD_INP):
            out[i] = len(self.encoded)  # each row says which encode made it
        return self.fail.get("sd_encode_image")

    def sd_forward(
        self, handle, state, n_state, image_token, embd, n_rows, q, q_lens, n_q, out, ms
    ):
        lens = [q_lens[i] for i in range(n_q)]
        self.forwarded.append({
            "state": [state[i] for i in range(n_state)], "image_token": image_token,
            "rows": [embd[r * N_EMBD_INP] for r in range(n_rows)],
            "questions": [q[i] for i in range(sum(lens))], "lens": lens,
        })  # fmt: skip
        for i in range(n_q * N_EMBD):
            out[i] = i // N_EMBD  # each row says which question it answers
        return self.fail.get("sd_forward")


@pytest.fixture
def shim(monkeypatch):
    """A FakeShim behind ctypes.CDLL. Its functions are partials, which take argtypes and restype
    as a library's functions do."""
    fake = FakeShim()

    def cdll(path):
        names = [n for n in dir(fake) if n.startswith("sd_")]
        return types.SimpleNamespace(**{n: functools.partial(getattr(fake, n)) for n in names})

    monkeypatch.setattr(llama_cpp.ctypes, "CDLL", cdll)
    return fake


def test_forward_passes_the_state_questions_and_image_rows(shim):
    backend = LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf", threads=2)
    out = backend.forward([2, 10], [[20, 21], [22]])
    assert out.shape == (2, N_EMBD) and out.dtype == np.float32 and list(out[:, 0]) == [0, 1]
    assert shim.forwarded[-1] == {
        "state": [2, 10], "image_token": SOFT, "rows": [], "questions": [20, 21, 22], "lens": [2, 1],
    }  # fmt: skip
    # The spans are the core's to place; the shim fills the placeholders in order.
    a, b = _image("checker-64x48.png"), _image("rotated-80x40.jpg")
    state = [2, *[SOFT] * a.tokens, *[SOFT] * b.tokens, *[SOFT] * a.tokens]
    backend.forward(state, [[30]], (a, b, a))
    # Each distinct file is encoded once, at the processor's size of its upright pixels (the JPEG
    # is shown 40 x 80), and its rows go where the request puts it.
    assert shim.encoded == [(*resized_size(64, 48), a.tokens), (*resized_size(40, 80), b.tokens)]
    assert shim.forwarded[-1]["rows"] == [1] * a.tokens + [2] * b.tokens + [1] * a.tokens
    assert shim.forwarded[-1]["state"] == state


def test_a_text_backend_refuses_images(shim):
    backend = LlamaCppBackend("lib.so", "m.gguf")
    with pytest.raises(ValueError, match=NO_VISION):
        backend.forward([2], [[1]], (_image("checker-64x48.png"),))


@pytest.mark.parametrize(
    "broken",
    [
        lambda im: replace(im, data=im.data[:-30]),  # the header reads, the pixels don't
        lambda im: replace(im, width=im.width // 2),  # pixels of another size than the header's
    ],
)
def test_an_image_that_does_not_decode_is_refused_before_the_shim(shim, broken):
    backend = LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf")
    with pytest.raises(ValueError, match=r"^image could not be decoded"):
        backend.forward([2], [[1]], (broken(_image("checker-64x48.png")),))
    assert shim.encoded == [] and shim.forwarded == []


def test_the_cpu_backend_is_the_shims(shim):
    """The CPU backend llama.cpp runs on is what the shim says it loaded."""
    backend = llama_cpp.LlamaCppBackend("libdecision.so", "model.gguf")
    assert backend.cpu_backend == "libggml-cpu-armv8.6_1.so"


@pytest.mark.parametrize("function", ["sd_load", "sd_load_vision", "sd_encode_image", "sd_forward"])
def test_a_failing_shim_is_a_runtime_error(shim, function):
    shim.fail[function] = b"it broke"
    with pytest.raises(
        RuntimeError, match="could not load" if function == "sd_load" else "it broke"
    ):
        backend = LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf")
        backend.forward([2], [[1]], (_image("checker-64x48.png"),))


def test_a_closed_backend_refuses_calls(shim):
    backend = LlamaCppBackend("lib.so", "m.gguf")
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.forward([2], [[1]])
