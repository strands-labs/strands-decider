"""The llama.cpp backend (strands_decider.backends.llama_cpp) without llama.cpp.

The image preparation is the backend's own and is tested against Pillow and the core: the pixel
goldens, the decoded-size check and the two-pass bicubic resize. The shim is replaced by a stand-in
that implements its C API in Python, so the marshalling around it, the image spans, and the checks
of mtmd's soft-token count, grid, wrapping tokens and normalisation are tested too. The real
library, built against the pinned llama.cpp, is tests/test_llama_cpp_shim.py's to test: the mask,
state reuse, image rows and answers against transformers.
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import io
import json

import numpy as np
import pytest
from core_support import FIXTURES, IMAGE_FILES, image_bytes

from strands_decider.backends import llama_cpp
from strands_decider.backends.llama_cpp import (
    CONTROL_IDS,
    ImageError,
    LlamaCppBackend,
    decode_pixels,
    prepare_pixels,
)
from strands_decider.core import HIDDEN, RenderedImage, decode_image, resized_size, soft_tokens

Image = pytest.importorskip("PIL.Image")
BOI, SOFT, EOI = CONTROL_IDS["<|image>"], CONTROL_IDS["<|image|>"], CONTROL_IDS["<image|>"]
N_EMBD, N_EMBD_INP = 6, 4


# ---- the image's pixels -----------------------------------------------------------------------


def _encode(img, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kw)
    return buf.getvalue()


def _request_image(data: bytes) -> RenderedImage:
    e = decode_image(base64.b64encode(data).decode())
    return RenderedImage(data, e.format, e.width, e.height, e.sha256, 1, e.tokens)


@pytest.mark.parametrize("name", sorted(IMAGE_FILES))
def test_prepared_pixels_are_the_goldens(name):
    """Decoded and resized to the bytes of tests/fixtures/core/image_pixels.json (docs/spec.md,
    Images): the decode, the orientation, and Pillow's bicubic resize to resized_size's size."""
    golden = json.loads((FIXTURES / "image_pixels.json").read_text(encoding="utf-8"))["images"][
        name
    ]
    data = image_bytes(name)
    decoded = decode_pixels(data)
    assert list(decoded.size) == golden["decoded"]
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == golden["decoded_sha256"]
    prepared = prepare_pixels(_request_image(data))
    assert prepared.mode == "RGB" and list(prepared.size) == golden["resized"]
    assert prepared.size == resized_size(*decoded.size)
    assert hashlib.sha256(prepared.tobytes()).hexdigest() == golden["resized_sha256"]
    w, h = prepared.size
    assert (w // 48) * (h // 48) == _request_image(data).tokens


@pytest.mark.parametrize("size", [(1080, 2400), (2400, 1080), (3000, 2000), (640, 40)])
def test_the_resize_is_the_width_then_the_height(size):
    """Each pass its own call, as the spec's rule says: Pillow 13 does the height's pass first,
    in one call, when the height shrinks by more than twice as much as the width."""
    img = Image.new("RGB", size)
    img.putdata(
        [
            ((x * 7 + y) % 256, (x * y) % 256, (x + 3 * y) % 256)
            for y in range(size[1])
            for x in range(size[0])
        ]
    )
    prepared = prepare_pixels(_request_image(_encode(img, "PNG")))
    w, h = resized_size(*size)
    want = img.resize((w, size[1]), Image.Resampling.BICUBIC).resize(
        (w, h), Image.Resampling.BICUBIC
    )
    assert prepared.size == (w, h) and prepared.tobytes() == want.tobytes()


def test_decode_pixels_applies_the_exif_orientation():
    raw = Image.open(io.BytesIO(image_bytes("rotated-80x40.jpg")))
    assert raw.size == (80, 40)
    assert decode_pixels(image_bytes("rotated-80x40.jpg")).size == (40, 80)
    # Resized to the decoded shape, not the header's.
    assert prepare_pixels(_request_image(image_bytes("rotated-80x40.jpg"))).size == (528, 1104)


def test_decode_pixels_puts_transparency_on_white_and_takes_the_first_frame():
    rgba = Image.new("RGBA", (10, 10), (255, 0, 0, 0))
    assert decode_pixels(_encode(rgba, "PNG")).getpixel((5, 5)) == (255, 255, 255)
    frames = [Image.new("RGB", (8, 8), c) for c in ((0, 0, 255), (0, 255, 0))]
    gif = io.BytesIO()
    frames[0].save(gif, format="GIF", save_all=True, append_images=frames[1:])
    assert decode_pixels(gif.getvalue()).getpixel((4, 4)) == (0, 0, 255)


def test_decode_pixels_refuses_with_v1s_messages():
    png = _encode(Image.new("RGB", (16, 16), (1, 2, 3)), "PNG")
    with pytest.raises(ValueError, match=r"^image could not be read as PNG/JPEG/WEBP/GIF: "):
        decode_pixels(b"not an image at all")
    with pytest.raises(ValueError, match=r"^image could not be read as PNG/JPEG/WEBP/GIF: "):
        decode_pixels(_encode(Image.new("RGB", (4, 4)), "BMP"))  # not one of the four formats
    cut = png[: len(png) - 30]  # IDAT cut short: the header is sound, the pixels are not
    with pytest.raises(ValueError, match=r"^image could not (be decoded|be read)"):
        decode_pixels(cut)


def test_a_decoded_size_that_is_not_the_headers_is_refused():
    small, large = (
        _encode(Image.new("RGB", (16, 8)), "PNG"),
        _encode(Image.new("RGB", (32, 8)), "PNG"),
    )
    forged = RenderedImage(large, "PNG", 16, 8, "0" * 64, 1, soft_tokens(16, 8))
    assert prepare_pixels(_request_image(small)).size == resized_size(16, 8)
    with pytest.raises(ValueError, match=r"^image could not be decoded: its pixels are 32x8"):
        prepare_pixels(forged)


# ---- the backend over a stand-in shim --------------------------------------------------------


class FakeShim:
    """sd_* in Python. A question's hidden state is a function of the state, the question and
    the image rows, so the tests can tell what reached the shim."""

    def __init__(self, *, tokens_for=None, grid_for=None, wrap=(BOI, EOI), fail=0, info=None):
        self.tokens_for = tokens_for or (lambda w, h: (w // 48) * (h // 48))
        self.grid_for = grid_for or (lambda w, h: (w // 48, h // 48))
        self.wrap, self.fail = wrap, fail
        self.info = info or ([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 16)
        self.calls: list[dict] = []
        self.encoded: list[tuple[int, int]] = []
        self.freed = False

    def sd_load(self, path, threads, n_embd):
        self.loaded = (path, threads)
        n_embd._obj.value = N_EMBD
        return 1234

    def sd_free(self, handle):
        self.freed = True

    def sd_load_vision(self, handle, path, max_tokens):
        self.vision = (path, max_tokens)
        return 0

    def sd_n_embd_inp(self, handle):
        return N_EMBD_INP

    def sd_vision_info(self, handle, mean, std, patch):
        m, sd, ps = self.info
        for i in range(3):
            mean[i], std[i] = m[i], sd[i]
        patch._obj.value = ps
        return 0

    def sd_encode_image(self, handle, rgb, w, h, out, cap, n_tok, grid, wrap, ms):
        assert len(rgb) == w * h * 3
        self.encoded.append((w, h))
        n = self.tokens_for(w, h)
        n_tok._obj.value = n
        grid[0], grid[1] = self.grid_for(w, h)
        wrap[0], wrap[1] = self.wrap
        for i in range(min(n, cap) * N_EMBD_INP):
            out[i] = float(len(self.encoded) * 1000 + i)
        ms[0], ms[1] = 1.0, 2.0
        return 0

    def sd_forward(
        self,
        handle,
        state,
        n_state,
        image_token,
        embd,
        rows,
        n_images,
        n_rows,
        q,
        q_lens,
        n_q,
        out,
        ms,
    ):
        lens = [q_lens[i] for i in range(n_q)]
        flat = [q[i] for i in range(sum(lens))]
        rows_ = [rows[i] for i in range(n_images)] if n_images else []
        assert n_rows == sum(rows_)
        e = [embd[i] for i in range(sum(rows_) * N_EMBD_INP)] if n_images else []
        call = {"state": [state[i] for i in range(n_state)], "image_token": image_token,
                "lens": lens, "q": flat, "rows": rows_, "embd": e}  # fmt: skip
        self.calls.append(call)
        if self.fail:
            return self.fail
        off = 0
        for i, n in enumerate(lens):
            for j in range(N_EMBD):
                out[i * N_EMBD + j] = float(
                    sum(call["state"]) % 97 + sum(flat[off : off + n]) % 89 + j + sum(e) % 7
                )
            off += n
        for i in range(n_q + 2):
            ms[i] = float(i)
        return 0


class FakeFunction:
    """What ctypes.CDLL gives: a callable whose argtypes and restype the backend sets."""

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *args):
        return self.fn(*args)


class FakeLib:
    def __init__(self, shim):
        for name in dir(shim):
            if name.startswith("sd_"):
                setattr(self, name, FakeFunction(getattr(shim, name)))


@pytest.fixture
def shim(monkeypatch):
    s = FakeShim()
    monkeypatch.setattr(llama_cpp.ctypes, "CDLL", lambda path: FakeLib(s))
    return s


@pytest.fixture
def processor_config(tmp_path):
    p = tmp_path / "processor_config.json"
    p.write_text(
        json.dumps(
            {
                "image_processor": {
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                }
            }
        )
    )
    return p


def _rendered(data: bytes, start: int) -> RenderedImage:
    e = decode_image(base64.b64encode(data).decode())
    return RenderedImage(data, e.format, e.width, e.height, e.sha256, start, e.tokens)


def _state_with(images: list[bytes], text=(7, 8, 9)) -> tuple[list[int], list[RenderedImage]]:
    state, rendered = [2, 10], []
    for data in images:
        state.append(BOI)
        im = _rendered(data, len(state))
        state += [SOFT] * im.tokens + [EOI, 11]
        rendered.append(im)
    return state + list(text), rendered


def test_a_text_backend(shim):
    backend = LlamaCppBackend("lib.so", "model.gguf", threads=3)
    assert shim.loaded == (b"model.gguf", 3)
    assert backend.outputs == HIDDEN and backend.inputs == ("text",)
    out = backend.forward([2, 5, 6], [[20, 21], [22]])
    assert out.shape == (2, N_EMBD) and out.dtype == np.float32
    (call,) = shim.calls
    assert call["state"] == [2, 5, 6] and call["lens"] == [2, 1] and call["q"] == [20, 21, 22]
    assert call["rows"] == [] and call["embd"] == []
    assert backend.last_timings == {"context_ms": 0.0, "state_ms": 1.0, "question_ms": [2.0, 3.0]}
    png = image_bytes("checker-64x48.png")
    state, images = _state_with([png])
    with pytest.raises(RuntimeError, match="no vision tower"):
        backend.forward(state, [[1]], images)
    with pytest.raises(RuntimeError, match="none of them empty"):
        backend.forward([1], [[]])
    backend.close()
    assert shim.freed
    with pytest.raises(RuntimeError, match="closed"):
        backend.forward([1], [[2]])


def test_a_vision_backend_needs_a_good_processor_config(shim, tmp_path, processor_config):
    with pytest.raises(ValueError, match="processor_config"):
        LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf")
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "image_processor": {
                    "patch_size": 14,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                }
            }
        )
    )
    with pytest.raises(ValueError, match="patch_size"):
        LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf", processor_config=bad)
    backend = LlamaCppBackend(
        "lib.so", "m.gguf", mmproj="mm.gguf", processor_config=processor_config
    )
    assert backend.inputs == ("text", "image") and shim.vision == (b"mm.gguf", 280)


def test_images_reach_the_shim_prepared_and_in_order(shim, processor_config):
    backend = LlamaCppBackend(
        "lib.so", "m.gguf", mmproj="mm.gguf", processor_config=processor_config
    )
    a, b = image_bytes("checker-64x48.png"), image_bytes("rotated-80x40.jpg")
    state, images = _state_with([a, b, a])
    out = backend.forward(state, [[30, 31]], images)
    assert out.shape == (1, N_EMBD)
    (call,) = shim.calls
    assert call["state"] == state and call["image_token"] == SOFT
    assert call["rows"] == [im.tokens for im in images] == [266, soft_tokens(80, 40), 266]
    # Each distinct file is prepared and encoded once, at the processor's size, the EXIF
    # orientation applied (the JPEG is shown 40 x 80); its rows go in the images' order.
    assert shim.encoded == [resized_size(64, 48), resized_size(40, 80)]
    rows = np.array(call["embd"], dtype=np.float32).reshape(-1, N_EMBD_INP)
    second = 266 + soft_tokens(80, 40)
    assert rows[0, 0] == 1000 and rows[266, 0] == 2000 and rows[second, 0] == 1000
    assert [t["size"] for t in backend.last_timings["images"]] == [
        list(resized_size(64, 48)),
        list(resized_size(40, 80)),
    ]


def test_the_vision_towers_soft_tokens_must_be_the_cores(monkeypatch, processor_config):
    png = image_bytes("checker-64x48.png")
    state, images = _state_with([png])
    for shim, match in [
        (FakeShim(tokens_for=lambda w, h: 70), "gives a 912 x 672 image 70 soft tokens"),
        (FakeShim(wrap=(BOI, 5)), "wraps an image"),
        (
            FakeShim(grid_for=lambda w, h: (h // 48, w // 48)),
            "a 14 x 19 grid of soft tokens, not 19 x 14",
        ),
        (FakeShim(grid_for=lambda w, h: (133, 2)), "a 133 x 2 grid"),
        (FakeShim(fail=13), "fewer runs of image placeholders"),
        (FakeShim(fail=14), "row counts are not positive"),
    ]:
        monkeypatch.setattr(llama_cpp.ctypes, "CDLL", lambda path, s=shim: FakeLib(s))
        backend = LlamaCppBackend(
            "lib.so", "m.gguf", mmproj="mm.gguf", processor_config=processor_config
        )
        with pytest.raises(RuntimeError, match=match):
            backend.forward(state, [[1]], images)


def test_image_spans_must_be_lines_of_placeholders(shim, processor_config):
    backend = LlamaCppBackend(
        "lib.so", "m.gguf", mmproj="mm.gguf", processor_config=processor_config
    )
    png = image_bytes("checker-64x48.png")
    state, images = _state_with([png])
    moved = [
        RenderedImage(im.data, im.format, im.width, im.height, im.sha256, im.start + 1, im.tokens)
        for im in images
    ]
    with pytest.raises(RuntimeError, match="not a line of placeholders"):
        backend.forward(state, [[1]], moved)
    with pytest.raises(RuntimeError, match="outside its images' spans"):
        backend.forward([*state, SOFT], [[1]], images)
    # An image that passes the header checks and fails to decode is the request's error.
    cut = png[: len(png) - 30]
    bad_state, bad = _state_with([png])
    bad = [
        RenderedImage(cut, b.format, b.width, b.height, b.sha256, b.start, b.tokens) for b in bad
    ]
    with pytest.raises(ImageError, match=r"^image could not"):
        backend.forward(bad_state, [[1]], bad)
    assert shim.calls == []


@pytest.mark.parametrize(
    ("info", "processor", "match"),
    [
        (([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], 16), {}, "mean \\[0.5, 0.5, 0.5\\]"),
        (([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 14), {}, "14-pixel"),
        (([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], 16), {"do_normalize": True, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}, "std \\[0.5"),
    ],
)  # fmt: skip
def test_the_vision_towers_normalisation_must_be_the_processors(
    monkeypatch, tmp_path, info, processor, match
):
    s = FakeShim(info=info)
    monkeypatch.setattr(llama_cpp.ctypes, "CDLL", lambda path: FakeLib(s))
    cfg = tmp_path / "processor_config.json"
    cfg.write_text(
        json.dumps(
            {
                "image_processor": {
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                    **processor,
                }
            }
        )
    )
    with pytest.raises(RuntimeError, match=match):
        LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf", processor_config=cfg)
    assert s.freed  # the handle is not leaked
    # A normalising processor with the tower's own mean and std is fine.
    ok = FakeShim(info=([0.5] * 3, [0.5] * 3, 16))
    monkeypatch.setattr(llama_cpp.ctypes, "CDLL", lambda path: FakeLib(ok))
    cfg.write_text(
        json.dumps(
            {
                "image_processor": {
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "max_soft_tokens": 280,
                    "do_normalize": True,
                    "image_mean": [0.5] * 3,
                    "image_std": [0.5] * 3,
                }
            }
        )
    )
    LlamaCppBackend("lib.so", "m.gguf", mmproj="mm.gguf", processor_config=cfg)


def test_a_one_dimensional_run_of_soft_tokens_is_accepted(monkeypatch, processor_config):
    s = FakeShim(grid_for=lambda w, h: ((w // 48) * (h // 48), 1))  # as mtmd reports Gemma 4's
    monkeypatch.setattr(llama_cpp.ctypes, "CDLL", lambda path: FakeLib(s))
    backend = LlamaCppBackend(
        "lib.so", "m.gguf", mmproj="mm.gguf", processor_config=processor_config
    )
    state, images = _state_with([image_bytes("checker-64x48.png")])
    assert backend.forward(state, [[1]], images).shape == (1, N_EMBD)


def test_encode_image_and_close_take_the_lock(shim, processor_config):
    backend = LlamaCppBackend(
        "lib.so", "m.gguf", mmproj="mm.gguf", processor_config=processor_config
    )
    pixels = prepare_pixels(_rendered(image_bytes("checker-64x48.png"), 1))
    with backend._lock:  # held, as by a forward on another thread
        t = __import__("threading").Thread(target=backend.encode_image, args=(pixels,))
        t.start()
        t.join(0.2)
        assert t.is_alive() and shim.encoded == []
    t.join(5)
    assert shim.encoded == [pixels.size]
    backend.close()
    with pytest.raises(RuntimeError, match="closed"):
        backend.encode_image(pixels)


def test_the_shim_failing_is_an_error(monkeypatch):
    for rc, match in [(2, "could not create a llama.cpp context"), (99, "99")]:
        s = FakeShim(fail=rc)
        monkeypatch.setattr(llama_cpp.ctypes, "CDLL", lambda path, s=s: FakeLib(s))
        with pytest.raises(RuntimeError, match=match):
            LlamaCppBackend("lib.so", "m.gguf").forward([1], [[2]])


def test_the_shim_declares_what_the_backend_calls():
    """Every sd_* the backend binds is defined in shim.c, with as many parameters."""
    import re
    from pathlib import Path

    src = (Path(__file__).parents[1] / "deploy" / "lambda" / "decision" / "shim.c").read_text()
    defined = {
        m.group(1): len([p for p in m.group(2).split(",") if p.strip()])
        for m in re.finditer(r"^(?:void \*|int|void) (sd_\w+)\(([^)]*)\)", src, re.M | re.S)
    }
    bound = {}

    class Recorder:
        def __getattr__(self, name):
            fn = FakeFunction(lambda *a: 0)
            setattr(self, name, fn)
            return fn

    rec = Recorder()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(llama_cpp.ctypes, "CDLL", lambda path: rec)
        with pytest.raises(RuntimeError):  # sd_load gives NULL here
            LlamaCppBackend("lib.so", "m.gguf")
    for name, fn in vars(rec).items():
        bound[name] = len(fn.argtypes)
    assert bound and set(bound) <= set(defined)
    assert {k: defined[k] for k in bound} == bound


def test_ctypes_pointer_types_match_numpy():
    assert ctypes.sizeof(ctypes.c_float) == np.dtype(np.float32).itemsize
    assert ctypes.sizeof(ctypes.c_int) == np.dtype(np.intc).itemsize
