"""Shared by the torch-free core's tests (tests/test_core_*.py).

The render goldens and properties run against Gemma 4's real tokenizer. It is 32 MB, so it
is not committed: point STRANDS_DECIDER_GEMMA4_TOKENIZER at a copy of `tokenizer.json`, and
the tests check its sha256 before they trust it. The goldens were made with the pinned file
(google/gemma-4-E2B-it at GEMMA4_REVISION, which 26B-A4B and 31B also ship) and need it; the
property tests accept any of the Gemma 4 files docs/spec.md lists (GEMMA4_TOKENIZERS), since
the E2B, E4B and 12B repositories now ship slightly different ones. Without a tokenizer those
tests skip; the tests that need only text, numbers or a toy tokenizer always run.

    STRANDS_DECIDER_GEMMA4_TOKENIZER=/path/to/tokenizer.json pytest tests/test_core_*.py

CI's `core tokenizer (gemma 4)` job (.github/workflows/tests.yml) fetches that file and the
E2B chat template at GEMMA4_REVISION, checks both sha256s, and runs these tests without torch.

STRANDS_DECIDER_GEMMA4_TEMPLATES lists the Gemma 4 base models' chat_template.jinja files
(os.pathsep between them) for the tests that compare strands-chat with them, which also
need jinja2 (and transformers, for the apply_chat_template comparison). A decision checkpoint
ships no chat template, and its prompts are never rendered with one.

The test images are small synthetic files in tests/fixtures/core/images, checked in, so that
the core's header reader and the render goldens need no imaging library. `make_image`
draws each one deterministically with Pillow; with STRANDS_DECIDER_REGEN_IMAGES=1 the files
are rewritten, and so is tests/fixtures/core/image_soft_tokens.json, the soft-token count
of every listed size by Gemma 4's real image processors (which needs transformers, Pillow
and torchvision); STRANDS_DECIDER_REGEN_SOFT_TOKENS=1 rewrites the table alone. Regenerate
the render goldens after the images.

tests/fixtures/core/image_pixels.json holds the pixel goldens of docs/spec.md (Images, in The
runtime interface): for each test image, the sha256 of its RGB bytes as v1 decodes it and
after the bicubic resize to (W, H), made with PIXELS_PILLOW; STRANDS_DECIDER_REGEN_PIXELS=1
rewrites it (Pillow only). `reference_pixels` is v1's decode, and `spec_resize` the spec's
resize written out in numpy, without Pillow.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import random
from functools import lru_cache
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "core"
GEMMA4_REPO = "google/gemma-4-E2B-it"
GEMMA4_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GEMMA4_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
# Every Gemma 4 tokenizer.json docs/spec.md lists, by sha256: where each is shipped.
GEMMA4_TOKENIZERS = {
    GEMMA4_TOKENIZER_SHA256: "google/gemma-4-E2B-it@3e22461f, 26B-A4B@4d7ae498, 31B@842da379",
    "ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2": "google/gemma-4-E2B-it@4adf1574",
    "91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc": "google/gemma-4-E4B-it@4ecc2ba6",
    "9788bdb6a021b3170357cd3587b5d622b1177561ad3c5bf43d73728fd71aeacb": "google/gemma-4-12B-it@43c9d014",
}
TOKENIZER_ENV = "STRANDS_DECIDER_GEMMA4_TOKENIZER"
TEMPLATES_ENV = "STRANDS_DECIDER_GEMMA4_TEMPLATES"
REGEN_IMAGES_ENV = "STRANDS_DECIDER_REGEN_IMAGES"
REGEN_PIXELS_ENV = "STRANDS_DECIDER_REGEN_PIXELS"
REGEN_SOFT_TOKENS_ENV = "STRANDS_DECIDER_REGEN_SOFT_TOKENS"
IMAGES = FIXTURES / "images"
SOFT_TOKENS = FIXTURES / "image_soft_tokens.json"
PIXELS = FIXTURES / "image_pixels.json"
# The Pillow that made the pixel goldens. Pillow 10.0 to 13.0 give the same bytes for every
# test image, decoded and resized (`reference_resize`).
PIXELS_PILLOW = "12.2.0"


@lru_cache(maxsize=1)
def _gemma4_path() -> tuple[str, str] | None:
    """The tokenizer's path and sha256, if it is one of docs/spec.md's Gemma 4 files."""
    p = os.environ.get(TOKENIZER_ENV)
    if not p:
        return None
    if os.path.isdir(p):
        p = os.path.join(p, "tokenizer.json")
    with open(p, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest not in GEMMA4_TOKENIZERS:
        raise AssertionError(
            f"{p} has sha256 {digest}, which is not one of the Gemma 4 tokenizer.json files "
            f"docs/spec.md lists: {sorted(GEMMA4_TOKENIZERS)}"
        )
    return p, digest


def gemma4_tokenizer_path() -> str:
    found = _gemma4_path()
    if found is None:
        pytest.skip(f"set {TOKENIZER_ENV} to Gemma 4's tokenizer.json ({GEMMA4_REPO})")
    return found[0]


def pinned_gemma4_tokenizer_path() -> str:
    """The pinned tokenizer.json, the one the render goldens were made with, or a skip."""
    p = gemma4_tokenizer_path()
    found = _gemma4_path()
    assert found is not None
    if found[1] != GEMMA4_TOKENIZER_SHA256:
        pytest.skip(
            f"the render goldens are for {GEMMA4_REPO}@{GEMMA4_REVISION}'s tokenizer.json "
            f"({GEMMA4_TOKENIZER_SHA256}); {TOKENIZER_ENV} is {GEMMA4_TOKENIZERS[found[1]]}'s"
        )
    return p


@lru_cache(maxsize=1)
def _load_gemma4(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def gemma4_tokenizer():
    """Gemma 4's tokenizer, any listed file, or a skip. One instance per session: it is 262k
    tokens."""
    return _load_gemma4(gemma4_tokenizer_path())


def pinned_gemma4_tokenizer():
    """The pinned Gemma 4 tokenizer, the render goldens', or a skip."""
    return _load_gemma4(pinned_gemma4_tokenizer_path())


def gemma4_tokenizer_dir() -> str:
    """The directory of Gemma 4's tokenizer.json, if it also holds the base's
    tokenizer_config.json (Hugging Face's processors need its named tokens), or a skip."""
    d = os.path.dirname(gemma4_tokenizer_path())
    if not os.path.exists(os.path.join(d, "tokenizer_config.json")):
        pytest.skip(f"no tokenizer_config.json beside {TOKENIZER_ENV}'s tokenizer.json")
    return d


def decision_config_fixture() -> dict:
    """The E2B checkpoint's decision_config.json reduced to the schema's fields (codes and
    token ids as the checkpoint has them; `prompt` and `max_length_validated` are test values)."""
    return json.loads((FIXTURES / "decision_config.json").read_text(encoding="utf-8"))


def toy_tokenizer():
    """A word-level tokenizer with Gemma 4's control tokens, its image tokens, and every
    code as one token: enough to drive render and evaluate where the real tokenizer is not
    available."""
    from tokenizers import AddedToken, Regex, Tokenizer, models, pre_tokenizers

    from strands_decider.core import CODES

    words = ["<unk>", "\n", " ", ".", "<", ">", "/", '"', "=", "—", "answer", "model", "user"]
    vocab = {w: i for i, w in enumerate(dict.fromkeys(words + list(CODES)))}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(r"\w+|\s|[^\w\s]"), behavior="isolated")
    specials = ("<bos>", "<|turn>", "<turn|>", "<eos>", "<|image>", "<|image|>", "<image|>")
    tok.add_special_tokens([AddedToken(s, special=True) for s in specials])
    return tok


# ---- test images ------------------------------------------------------------------------

# The checked-in test images: name -> (format, width, height), as their headers give them
# and as Pillow opens them (tests/test_core_images.py checks both).
IMAGE_FILES = {
    "checker-64x48.png": ("PNG", 64, 48),
    "stripes-300x40.jpg": ("JPEG", 300, 40),
    "rings-90x120.webp": ("WEBP", 90, 120),  # lossy: a `VP8 ` chunk
    "dots-50x50.gif": ("GIF", 50, 50),
    "lossless-33x17.webp": ("WEBP", 33, 17),  # a `VP8L` chunk
    "alpha-21x34.webp": ("WEBP", 21, 34),  # lossy with alpha: a `VP8X` chunk first
    "rotated-80x40.jpg": ("JPEG", 80, 40),  # progressive; EXIF orientation 6: shown 40 x 80
    "thin-1x300.png": ("PNG", 1, 300),  # greyscale
    "offset-frame.gif": ("GIF", 25, 36),  # a 10 x 10 screen, its first frame 20 x 30 at (5, 6)
}


def image_bytes(name: str) -> bytes:
    return (IMAGES / name).read_bytes()


def image_b64(name: str, *, data_uri: bool = False) -> str:
    """A test image as a request carries it: base64, behind a data: URI if asked."""
    b64 = base64.b64encode(image_bytes(name)).decode("ascii")
    if not data_uri:
        return b64
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "GIF": "gif"}[IMAGE_FILES[name][0]]
    return f"data:image/{mime};base64,{b64}"


def make_image(name: str) -> bytes:
    """Draw and encode one test image with Pillow, from its name alone (no randomness)."""
    import io

    from PIL import Image

    _, w, h = IMAGE_FILES[name]
    if name == "offset-frame.gif":
        w, h = 20, 30
    xy = [(x, y) for y in range(h) for x in range(w)]
    if name.startswith("checker"):
        px = [(230, 80, 40) if (x // 8 + y // 8) % 2 else (30, 60, 200) for x, y in xy]
    elif name.startswith("stripes"):
        px = [(250, 250, 240) if (x // 10) % 2 else (20, 20, 30) for x, y in xy]
    elif name.startswith("rings"):
        px = [(int(math.hypot(x - 45, y - 60)) * 9 % 256, 90, 160) for x, y in xy]
    else:
        px = [((x * 37 + y * 11) % 256, (x * 5 + y * 23) % 256, (x + y) * 3 % 256) for x, y in xy]
    im = Image.new("RGB", (w, h))
    im.putdata(px)
    buf = io.BytesIO()
    if name == "checker-64x48.png":
        im.save(buf, format="PNG", compress_level=9)
    elif name == "thin-1x300.png":
        im.convert("L").save(buf, format="PNG", compress_level=9)
    elif name == "stripes-300x40.jpg":
        im.save(buf, format="JPEG", quality=75)
    elif name == "rotated-80x40.jpg":
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90 degrees clockwise to display
        im.save(buf, format="JPEG", quality=75, progressive=True, exif=exif.tobytes())
    elif name == "rings-90x120.webp":
        im.save(buf, format="WEBP", quality=60, method=4)
    elif name == "lossless-33x17.webp":
        im.save(buf, format="WEBP", lossless=True, method=4)
    elif name == "alpha-21x34.webp":
        rgba = im.convert("RGBA")
        rgba.putalpha(Image.linear_gradient("L").resize((w, h)))
        rgba.save(buf, format="WEBP", quality=60, method=4)
    else:  # the GIFs
        im.convert("P", palette=Image.Palette.ADAPTIVE, colors=16).save(buf, format="GIF")
    data = buf.getvalue()
    if name == "offset-frame.gif":
        data = _offset_first_frame(data, screen=(10, 10), offset=(5, 6))
    return data


def _offset_first_frame(gif: bytes, *, screen: tuple[int, int], offset: tuple[int, int]) -> bytes:
    """A GIF with its logical screen set to `screen` and its first frame moved to `offset`."""
    b = bytearray(gif)
    b[6:10] = screen[0].to_bytes(2, "little") + screen[1].to_bytes(2, "little")
    pos = 13 + (3 << ((b[10] & 7) + 1) if b[10] & 0x80 else 0)
    while b[pos] == 0x21:  # extensions: label, then sub-blocks up to an empty one
        pos += 2
        while b[pos]:
            pos += 1 + b[pos]
        pos += 1
    assert b[pos] == 0x2C
    b[pos + 1 : pos + 5] = offset[0].to_bytes(2, "little") + offset[1].to_bytes(2, "little")
    return bytes(b)


# Aspect ratios w:h where 280 h / w is a perfect square, so that f h / 48 is an integer in
# exact arithmetic and the floor is decided by the double's last bit (docs/spec.md, The state
# block); 280:1 is the k = 1 family above. Each at KNIFE_EDGE_SCALES, both ways round.
KNIFE_EDGE_RATIOS = ((10, 7), (14, 5), (40, 7), (35, 32), (35, 2), (56, 5), (70, 1))
KNIFE_EDGE_SCALES = (*range(1, 41), 49, 50, 64, 77, 100, 128, 200, 300, 500, 1000)


def knife_edge_sizes() -> set[tuple[int, int]]:
    from strands_decider.core import MAX_IMAGE_PIXELS

    out = set()
    for p, q in KNIFE_EDGE_RATIOS:
        for k in KNIFE_EDGE_SCALES:
            if p * q * k * k <= MAX_IMAGE_PIXELS:
                out |= {(p * k, q * k), (q * k, p * k)}
    return out


def soft_token_sizes() -> list[tuple[int, int]]:
    """The (width, height) pairs of tests/fixtures/core/image_soft_tokens.json: a grid of
    common and boundary sides, extreme aspect ratios to the pixel limit (1 x 16,777,216),
    both sides of the 280:1 ratio where one resized side becomes 48 pixels, every size whose
    area makes the processor's scale factor exactly 1 and samples where it is 1/2, 1/3 and
    1/4, log-uniform random sizes, and the knife-edge ratios at many scales."""
    from strands_decider.core import MAX_IMAGE_PIXELS

    grid = [1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 31, 32, 33, 47, 48, 49, 63, 64, 96, 100, 127, 128,
            200, 224, 256, 299, 300, 384, 448, 480, 512, 640, 720, 768, 803, 804, 1000, 1024,
            1080, 1280, 1920, 2048, 3024, 4032, 4096]  # fmt: skip
    out = {(w, h) for w in grid for h in grid}
    for r in (100, 200, 279, 280, 281, 282, 300, 500, 1000, 4096, 13439, 13440, 13441, 65536,
              1_000_000, MAX_IMAGE_PIXELS):  # fmt: skip
        for k in (1, 2, 3, 7):
            if r * k * k <= MAX_IMAGE_PIXELS:
                out |= {(r * k, k), (k, r * k)}
    for k in range(1, 9):
        for d in (-1, 0, 1):
            out |= {(280 * k + d, k), (k, 280 * k + d)}
    for area in (645120, 4 * 645120, 9 * 645120, 16 * 645120):
        divisors = [d for d in range(1, math.isqrt(area) + 1) if area % d == 0]
        if area != 645120:
            divisors = random.Random(area).sample(divisors, min(40, len(divisors)))
        for d in divisors:
            out |= {(d, area // d), (area // d, d)}
    rng = random.Random(14)
    while len(out) < 3200:
        w = round(math.exp(rng.uniform(0, math.log(16384))))
        h = round(math.exp(rng.uniform(0, math.log(16384))))
        if w * h <= MAX_IMAGE_PIXELS:
            out.add((w, h))
    return sorted(out | knife_edge_sizes())  # after the random sizes, which they leave as they were


GEMMA4_PROCESSOR = "Gemma4ImageProcessor"  # E2B, E4B, 26B-A4B, 31B
UNIFIED_PROCESSOR = "Gemma4UnifiedImageProcessor"  # 12B
# The bases' processor_config.json settings, the same for both processors.
PROCESSOR_SETTINGS = {"patch_size": 16, "max_soft_tokens": 280, "pooling_kernel_size": 3}


def image_processors() -> dict[str, object]:
    """Gemma 4's two image processors with the bases' settings: Gemma4ImageProcessor on its
    PIL backend (what v1 pins) and Gemma4UnifiedImageProcessor (torchvision only)."""
    import transformers

    return {
        GEMMA4_PROCESSOR: transformers.Gemma4ImageProcessorPil(**PROCESSOR_SETTINGS),
        UNIFIED_PROCESSOR: transformers.Gemma4UnifiedImageProcessor(**PROCESSOR_SETTINGS),
    }


def processor_soft_tokens(processor, width: int, height: int) -> int:
    """N as the real processor reports it, for a solid image of that size."""
    from PIL import Image

    out = processor(images=[Image.new("RGB", (width, height), (17, 120, 230))])
    return int(out["num_soft_tokens_per_image"][0])


def _write_test_images() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    for name in IMAGE_FILES:
        (IMAGES / name).write_bytes(make_image(name))
    _write_soft_token_table()


def _write_soft_token_table() -> None:
    import PIL
    import torchvision
    import transformers

    procs = image_processors()
    torchvision_backend = transformers.Gemma4ImageProcessor(**PROCESSOR_SETTINGS)
    rows = []
    for w, h in soft_token_sizes():
        n = [processor_soft_tokens(procs[k], w, h) for k in (GEMMA4_PROCESSOR, UNIFIED_PROCESSOR)]
        assert processor_soft_tokens(torchvision_backend, w, h) == n[0], (w, h)
        rows.append([w, h, *n])
    head = {
        "columns": ["width", "height", GEMMA4_PROCESSOR, UNIFIED_PROCESSOR],
        "settings": PROCESSOR_SETTINGS,
        "versions": {
            "transformers": transformers.__version__,
            "pillow": PIL.__version__,
            "torchvision": torchvision.__version__,
        },
        "note": (
            "num_soft_tokens_per_image from transformers' Gemma4ImageProcessorPil and "
            "Gemma4UnifiedImageProcessor, with the Gemma 4 bases' processor_config.json "
            "settings, for a solid RGB image of each size; Gemma4ImageProcessor's torchvision "
            "backend gives the same. Generated by tests/core_support.py."
        ),
    }
    body = json.dumps(head, separators=(",", ":"))[:-1]
    lines = ",\n".join(json.dumps(r, separators=(",", ":")) for r in rows)
    SOFT_TOKENS.write_text(body + ',"rows":[\n' + lines + "\n]}\n", encoding="utf-8")


# ---- pixels: v1's decode and the spec's resize -------------------------------------------


def reference_pixels(data: bytes):
    """An image file as v1 decodes it (`vision.decode_image`), as a PIL RGB image: the first
    frame, its EXIF orientation applied, RGBA, LA and transparent palette images composited
    onto white, then converted to RGB."""
    import io

    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data), formats=("PNG", "JPEG", "WEBP", "GIF"))
    img.load()
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        img = Image.alpha_composite(Image.new("RGBA", rgba.size, (255, 255, 255, 255)), rgba)
    return img.convert("RGB")


def _bicubic(x: float) -> float:
    """The cubic convolution kernel with a = -0.5 (Keys), support 2."""
    a, x = -0.5, abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


PRECISION_BITS = 22  # 32 - 8 - 2: the fixed-point coefficients' fraction bits


def _coefficients(in_size: int, out_size: int) -> list[tuple[int, list[int]]]:
    """For each output index, its first input index and its fixed-point weights, by the
    spec's rule (Pillow's precompute_coeffs and normalize_coeffs_8bpc)."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = 2.0 * filterscale
    out = []
    for i in range(out_size):
        center = (i + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)  # int() truncates, as C's cast does
        hi = min(int(center + support + 0.5), in_size)
        weights = [_bicubic((j + lo - center + 0.5) / filterscale) for j in range(hi - lo)]
        total = 0.0
        for w in weights:
            total += w
        if total != 0.0:
            weights = [w / total for w in weights]
        fixed = [int((-0.5 if w < 0 else 0.5) + w * (1 << PRECISION_BITS)) for w in weights]
        out.append((lo, fixed))
    return out


def _resample(pixels, out_size: int, axis: int):
    import numpy as np

    src = np.moveaxis(pixels, axis, 0).astype(np.int64)
    out = np.empty((out_size, *src.shape[1:]), np.uint8)
    for i, (lo, weights) in enumerate(_coefficients(src.shape[0], out_size)):
        acc = np.full(src.shape[1:], 1 << (PRECISION_BITS - 1), np.int64)
        for j, w in enumerate(weights):
            acc += src[lo + j] * w
        out[i] = np.clip(acc >> PRECISION_BITS, 0, 255)
    return np.moveaxis(out, 0, axis)


def spec_resize(pixels, width: int, height: int):
    """docs/spec.md's bicubic resize of an RGB array [h, w, 3] (uint8) to [height, width, 3],
    written from the spec's text with numpy: the horizontal pass, its 8-bit result, then the
    vertical pass; a pass whose size does not change is left out."""
    out = pixels
    if out.shape[1] != width:
        out = _resample(out, width, axis=1)
    if out.shape[0] != height:
        out = _resample(out, height, axis=0)
    return out


def reference_resize(img, width: int, height: int):
    """docs/spec.md's resize with Pillow: bicubic, the width first and then the height, in two
    calls. Pillow 10 to 12 do the same in one call; Pillow 13 does the height first when it
    shrinks by more than twice as much as the width, which changes the pixels."""
    from PIL import Image

    if img.width != width:
        img = img.resize((width, img.height), Image.Resampling.BICUBIC)
    if img.height != height:
        img = img.resize((width, height), Image.Resampling.BICUBIC)
    return img


def _write_pixel_goldens() -> None:
    import PIL

    from strands_decider.core import resize_target

    assert PIL.__version__ == PIXELS_PILLOW, (
        f"the pixel goldens are made with Pillow {PIXELS_PILLOW}"
    )
    rows = {}
    for name in IMAGE_FILES:
        decoded = reference_pixels(image_bytes(name))
        w, h = resize_target(*decoded.size)
        resized = reference_resize(decoded, w, h)
        rows[name] = {
            "decoded": list(decoded.size),
            "decoded_sha256": hashlib.sha256(decoded.tobytes()).hexdigest(),
            "resized": [w, h],
            "resized_sha256": hashlib.sha256(resized.tobytes()).hexdigest(),
        }
    head = {
        "pillow": PIXELS_PILLOW,
        "note": (
            "For each test image: its size and the sha256 of its RGB bytes (row by row, 3 per "
            "pixel) as v1 decodes it, then resized to (W, H) from the decoded size by the "
            "spec's bicubic resize, width first. Generated by tests/core_support.py with "
            "Pillow " + PIXELS_PILLOW + "."
        ),
    }
    lines = ",\n".join(
        f"{json.dumps(n)}:{json.dumps(r, separators=(',', ':'))}" for n, r in rows.items()
    )
    body = json.dumps(head, separators=(",", ":"))[:-1]
    PIXELS.write_text(body + ',"images":{\n' + lines + "\n}}\n", encoding="utf-8")


if os.environ.get(REGEN_IMAGES_ENV) == "1":
    _write_test_images()
elif os.environ.get(REGEN_SOFT_TOKENS_ENV) == "1":
    _write_soft_token_table()
if os.environ.get(REGEN_PIXELS_ENV) == "1":
    _write_pixel_goldens()
