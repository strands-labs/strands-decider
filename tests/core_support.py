"""Shared by the torch-free core's tests (tests/test_core_*.py).

The render goldens and properties run against Gemma 4's real tokenizer. It is 32 MB, so it is
not committed: point STRANDS_DECIDER_GEMMA4_TOKENIZER at a copy of `tokenizer.json`, and the
tests check its sha256 before they trust it. It may be any of the four known Gemma 4 files
(GEMMA4_TOKENIZERS, which docs/spec.md refers to), since the E2B, E4B and 12B repositories now
ship slightly different ones: the goldens were made with the pinned file
(google/gemma-4-E2B-it at GEMMA4_REVISION, which 26B-A4B and 31B also ship), which
regenerating them needs, and every golden gives the same ids with all four. Without a
tokenizer those tests skip; the tests that need only text, numbers or a toy tokenizer always
run.

    STRANDS_DECIDER_GEMMA4_TOKENIZER=/path/to/tokenizer.json pytest tests/test_core_*.py

CI's `core tokenizer (gemma 4)` job (.github/workflows/tests.yml) fetches the pinned file and the
E2B chat template at GEMMA4_REVISION, checks both sha256s, and runs these tests without torch;
then it fetches the other three listed files and runs tests/test_core_prompt.py and
tests/test_core_images.py with each.

STRANDS_DECIDER_GEMMA4_TEMPLATES lists the Gemma 4 base models' chat_template.jinja files
(os.pathsep between them) for the tests that compare strands-chat with them, which also
need jinja2 (and transformers, for the apply_chat_template comparison). A decision checkpoint
ships no chat template, and its prompts are never rendered with one.

The test images are small synthetic files in tests/fixtures/core/images, checked in.
`make_image` draws each one deterministically with Pillow; with
STRANDS_DECIDER_REGEN_IMAGES=1 the files are rewritten, and so is
tests/fixtures/core/image_soft_tokens.json, the soft-token count of every listed size by
Gemma 4's real image processors (which needs transformers, Pillow and torchvision);
STRANDS_DECIDER_REGEN_SOFT_TOKENS=1 rewrites the table alone. Regenerate the render goldens
after the images.

tests/fixtures/core/image_pixels.json holds the pixel goldens of docs/spec.md (section 10):
for each test image, the sha256 of its RGB bytes as v1 decodes it and
after the bicubic resize to (W, H), made with PIXELS_PILLOW; STRANDS_DECIDER_REGEN_PIXELS=1
rewrites it (Pillow only). `reference_pixels` is v1's decode, `reference_resize` the
spec's resize, the core's `resize_image`, and `spec_resize` that resize, Pillow's bicubic
filter, written out in numpy, without Pillow, as an oracle for both.
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
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures" / "core"
GEMMA4_REPO = "google/gemma-4-E2B-it"
GEMMA4_REVISION = "3e22461f65e89153144f8adb70e3b8c2cc9845a7"
GEMMA4_TOKENIZER_SHA256 = "cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f"
# Every known Gemma 4 tokenizer.json, by sha256: where each is shipped. docs/spec.md refers to
# this list, and every golden gives the same ids with all four.
GEMMA4_TOKENIZERS = {
    GEMMA4_TOKENIZER_SHA256: (
        "google/gemma-4-E2B-it@3e22461f65e89153144f8adb70e3b8c2cc9845a7, "
        "google/gemma-4-26B-A4B-it@4d7ae4984b7db7de8f8457170b3f1a419ee76d52, "
        "google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475"
    ),
    "ab26b2add729e6f645bd869f228f14190a06615538f3be6f7786f33d13d78ad2": (
        "google/gemma-4-E2B-it@4adf15748de9e1a0097b6274cee9c15836c41c37"
    ),
    "91d9cf86906f78e197021fda88214d448e5c120665d0243a8e776e046986c6bc": (
        "google/gemma-4-E4B-it@4ecc2ba63bcac23ba4a767fc87025b1450f84c9c"
    ),
    "9788bdb6a021b3170357cd3587b5d622b1177561ad3c5bf43d73728fd71aeacb": (
        "google/gemma-4-12B-it@43c9d014b231474fa7212dbb48a762566c2b255d"
    ),
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
    """The tokenizer's path and sha256, if it is one of the Gemma 4 files in GEMMA4_TOKENIZERS."""
    p = os.environ.get(TOKENIZER_ENV)
    if not p:
        return None
    if os.path.isdir(p):
        p = os.path.join(p, "tokenizer.json")
    with open(p, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest not in GEMMA4_TOKENIZERS:
        raise AssertionError(
            f"{p} has sha256 {digest}, which is not one of the known Gemma 4 tokenizer.json "
            f"files: {sorted(GEMMA4_TOKENIZERS)}"
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

# The checked-in test images: name -> (format, width, height), as Pillow reads them from
# their headers (tests/test_core_images.py checks it).
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
# Larger test images, for the pixel goldens of a downscale (the resize target is at most 645,120
# pixels, so real photos and screenshots shrink): kept apart from IMAGE_FILES, which the render
# goldens and the random requests draw on.
DOWNSCALE_FILES = {
    # A tall screenshot: its height shrinks by more than twice as much as its width, where one
    # Pillow 13 resize call does the height first.
    "tall-1080x2400.png": ("PNG", 1080, 2400),
    "wide-1600x1000.png": ("PNG", 1600, 1000),
    "photo-1280x960.jpg": ("JPEG", 1280, 960),  # a lossy decode, then a downscale
}
PIXEL_FILES = {**IMAGE_FILES, **DOWNSCALE_FILES}


def image_bytes(name: str) -> bytes:
    return (IMAGES / name).read_bytes()


def image_b64(name: str, *, data_uri: bool = False) -> str:
    """A test image as a request carries it: base64, behind a data: URI if asked."""
    b64 = base64.b64encode(image_bytes(name)).decode("ascii")
    if not data_uri:
        return b64
    mime = {"PNG": "png", "JPEG": "jpeg", "WEBP": "webp", "GIF": "gif"}[PIXEL_FILES[name][0]]
    return f"data:image/{mime};base64,{b64}"


def make_image(name: str) -> bytes:
    """Draw and encode one test image with Pillow, from its name alone (no randomness)."""
    import io

    from PIL import Image

    if name in DOWNSCALE_FILES:
        return _make_downscale_image(name)
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


def _make_downscale_image(name: str) -> bytes:
    """The larger test images, drawn with numpy and encoded with Pillow."""
    import io

    import numpy as np
    from PIL import Image

    fmt, w, h = DOWNSCALE_FILES[name]
    yy, xx = np.mgrid[0:h, 0:w]
    if name.startswith("tall"):  # text-like rows, a gradient and a disc: edges in both axes
        red = ((xx // 6 + yy // 4) % 2) * ((yy // 60) % 3 == 0) * 200 + 30
        green = yy * 255 // h
        blue = (((xx - w / 2) ** 2 + (yy - h / 3) ** 2) < (w / 3) ** 2) * 180 + 40
    elif name.startswith("wide"):
        red, green, blue = xx * 255 // w, ((xx // 3) % 2) * 220 + 10, (yy // 25 % 3) * 100
    else:  # a smooth photo-like scene, saved as JPEG
        red, green = xx * 255 // w, yy * 255 // h
        blue = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) < (h / 3) ** 2) * 150 + 60
    a = np.stack([red, green, blue], axis=-1).astype(np.uint8)
    buf = io.BytesIO()
    if fmt == "PNG":
        Image.fromarray(a).save(buf, format="PNG", compress_level=9)
    else:
        Image.fromarray(a).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


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


# Aspect ratios w:h = 280:m^2, in lowest terms, where 280 h / w = m^2 is a perfect square, so
# that f h / 48 = m in exact arithmetic and the floor is decided by the double's last bit
# (docs/spec.md, section 3.3); where m divides 280, f w / 48 = 280 / m is an integer too.
# m = 1 to 24, each at KNIFE_EDGE_SCALES, both ways round.
KNIFE_EDGE_RATIOS = tuple(
    (280 // math.gcd(280, m * m), m * m // math.gcd(280, m * m)) for m in range(1, 25)
)
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


def reference_resize(img, width: int, height: int):
    """docs/spec.md's resize with Pillow: the core's `resize_image`, the width first and then
    the height, in two single-axis calls. Pillow 10 to 12 do the same in one call; Pillow 13
    does the height first when it shrinks by more than twice as much as the width, which
    changes the pixels."""
    from strands_decider.core import resize_image

    return resize_image(img, width, height)


# ---- the spec's resize rule in numpy, without Pillow: a test oracle --------------------


def _bicubic(x: float) -> float:
    """The cubic convolution kernel with a = -0.5 (Keys), support 2."""
    a, x = -0.5, abs(x)
    if x < 1.0:
        return ((a + 2.0) * x - (a + 3.0)) * x * x + 1
    if x < 2.0:
        return (((x - 5) * x + 8) * x - 4) * a
    return 0.0


_PRECISION_BITS = 22  # 32 - 8 - 2: the fixed-point coefficients' fraction bits


def _coefficients(in_size: int, out_size: int) -> list[tuple[int, list[int]]]:
    """For each output index, its first input index and its fixed-point weights, as Pillow
    computes them (precompute_coeffs and normalize_coeffs_8bpc)."""
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    inverse = 1.0 / filterscale  # multiplied by, as Pillow does, not divided by
    support = 2.0 * filterscale
    out = []
    for i in range(out_size):
        center = (i + 0.5) * scale
        lo = max(int(center - support + 0.5), 0)  # int() truncates, as C's cast does
        hi = min(int(center + support + 0.5), in_size)
        weights = [_bicubic((j + lo - center + 0.5) * inverse) for j in range(hi - lo)]
        total = 0.0
        for w in weights:
            total += w
        if total != 0.0:
            weights = [w / total for w in weights]
        fixed = [int((-0.5 if w < 0 else 0.5) + w * (1 << _PRECISION_BITS)) for w in weights]
        out.append((lo, fixed))
    return out


def _resample(pixels: Any, out_size: int, axis: int) -> Any:
    import numpy as np

    src = np.moveaxis(pixels, axis, 0).astype(np.int64)
    out = np.empty((out_size, *src.shape[1:]), np.uint8)
    for i, (lo, weights) in enumerate(_coefficients(src.shape[0], out_size)):
        acc = np.full(src.shape[1:], 1 << (_PRECISION_BITS - 1), np.int64)
        for j, w in enumerate(weights):
            acc += src[lo + j] * w
        out[i] = np.clip(acc >> _PRECISION_BITS, 0, 255)
    return np.moveaxis(out, 0, axis)


def spec_resize(pixels: Any, width: int, height: int) -> Any:
    """docs/spec.md's bicubic resize (section 3.3) of RGB pixels, an array [h, w, 3] of
    uint8, to [height, width, 3], written with numpy alone: Pillow's bicubic filter, the
    horizontal pass, its 8-bit result, then the vertical pass; a pass whose size does not
    change is left out. A test oracle, which holds `resize_image` and Pillow to the rule the
    spec names, byte for byte, and lets the pixel goldens be checked without Pillow."""
    import numpy as np

    pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"spec_resize takes RGB pixels [h, w, 3], not {pixels.shape}")
    if not (isinstance(width, int) and isinstance(height, int)) or width < 1 or height < 1:
        raise ValueError(f"a resized image is at least 1 x 1 pixels, not {width!r} x {height!r}")
    out = pixels
    if out.shape[1] != width:
        out = _resample(out, width, axis=1)
    if out.shape[0] != height:
        out = _resample(out, height, axis=0)
    return out


PIXELS_DIR = FIXTURES / "pixels"


def write_rgb_png(path, pixels) -> None:
    """RGB pixels [h, w, 3] (uint8) as a PNG with no row filters, so that `read_rgb_png`, and
    any runtime, reads it with zlib alone."""
    import struct
    import zlib

    h, w, _ = pixels.shape
    raw = b"".join(b"\x00" + pixels[y].tobytes() for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data)
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    body = chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + body)


def read_rgb_png(path):
    """The pixels of a `write_rgb_png` file, [h, w, 3] uint8, with zlib and numpy only."""
    import struct
    import zlib

    import numpy as np

    data = path.read_bytes()
    w, h = struct.unpack(">II", data[16:24])
    pos, idat = 8, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        if data[pos + 4 : pos + 8] == b"IDAT":
            idat += data[pos + 8 : pos + 8 + length]
        pos += 12 + length
    rows = np.frombuffer(zlib.decompress(idat), np.uint8).reshape(h, 1 + 3 * w)
    assert not rows[:, 0].any(), "a row filter: not a write_rgb_png file"
    return rows[:, 1:].reshape(h, w, 3)


def _write_pixel_goldens() -> None:
    import PIL

    from strands_decider.core import resized_size

    assert PIL.__version__ == PIXELS_PILLOW, (
        f"the pixel goldens are made with Pillow {PIXELS_PILLOW}"
    )
    import numpy as np

    rows = {}
    PIXELS_DIR.mkdir(exist_ok=True)
    for name in PIXEL_FILES:
        decoded = reference_pixels(image_bytes(name))
        write_rgb_png(PIXELS_DIR / f"{name}.png", np.asarray(decoded))
        w, h = resized_size(*decoded.size)
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
            "pixel) as v1 decodes it, which tests/fixtures/core/pixels/<name>.png holds, then "
            "resized to (W, H) from the decoded size by the spec's bicubic resize, width first. "
            "Generated by tests/core_support.py with Pillow " + PIXELS_PILLOW + "."
        ),
    }
    lines = ",\n".join(
        f"{json.dumps(n)}:{json.dumps(r, separators=(',', ':'))}" for n, r in rows.items()
    )
    body = json.dumps(head, separators=(",", ":"))[:-1]
    PIXELS.write_text(body + ',"images":{\n' + lines + "\n}}\n", encoding="utf-8")


# ---- readout goldens: the readout apart from any decoder ----------------------------------

READOUT_GOLDENS = FIXTURES / "readout_goldens.json"
REGEN_READOUT_ENV = "STRANDS_DECIDER_REGEN_READOUT"


def _bf16_values(a):
    """float32 values rounded to bf16 by the core, as plain floats (exact in JSON)."""
    from strands_decider.core import round_to_bf16

    return [float(x) for x in round_to_bf16(a).ravel()]


def _write_readout_goldens() -> None:
    """Synthetic readout goldens: a bf16 weight [255, 32], scaled so that logits fall where
    bf16's spacing is 0.125 and past the cap, hidden states in bf16 and in float32, and for
    each the bf16 logits and every question size's probabilities by the core's regime."""
    import base64

    import numpy as np

    from strands_decider.core import (
        RAW_LOGITS,
        code_probabilities,
        readout_logits,
        round_to_bf16,
    )

    rng = np.random.default_rng(130)
    hidden, c = 32, 30.0
    scale = np.array([0.5, 1.0, 2.0, 4.0, 8.0])[np.arange(255) % 5, None]
    weight = round_to_bf16(rng.normal(0, 1, (255, hidden)) * scale)
    weight_bits = (weight.view(np.uint32) >> 16).astype("<u2").tobytes()
    cases = []
    for t, h_kind in [
        (1.0, "bf16"),
        (0.7, "bf16"),
        (1.9, "bf16"),
        (0.05, "bf16"),
        (1.0, "float32"),
        (1.3, "float32"),
    ]:
        h = rng.normal(0, 1, hidden).astype(np.float32)
        h = round_to_bf16(h) if h_kind == "bf16" else h
        z = readout_logits(weight, h)
        cases.append(
            {
                "name": f"{h_kind}-h-T{t}",
                "temperature": t,
                "hidden": [float(x) for x in h],
                "z": [float(x) for x in z],
                "probabilities": {
                    str(k): [
                        float(p)
                        for p in code_probabilities(
                            z, k, stage=RAW_LOGITS, softcap_c=c, temperature=t
                        )
                    ]
                    for k in (2, 3, 7, 40, 255)
                },
            }
        )
    head = {
        "note": (
            "The readout of docs/spec.md section 7.1 on a synthetic bf16 weight: for each "
            "hidden state, z = bf16(W bf16(h)) and the probabilities of the first k codes "
            "after the soft-cap and the temperature. weight_bf16 is the weight's bf16 bits, "
            "little-endian, row by row. Generated by tests/core_support.py."
        ),
        "shape": [255, hidden],
        "final_logit_softcapping": c,
        "weight_bf16": base64.b64encode(weight_bits).decode("ascii"),
    }
    READOUT_GOLDENS.write_text(json.dumps({**head, "cases": cases}) + "\n", encoding="utf-8")


def readout_goldens() -> tuple[dict, object]:
    """The readout goldens, and their weight as float32 [255, hidden]."""
    import base64

    import numpy as np

    g = json.loads(READOUT_GOLDENS.read_text(encoding="utf-8"))
    bits = np.frombuffer(base64.b64decode(g["weight_bf16"]), "<u2").astype(np.uint32) << 16
    return g, bits.view(np.float32).reshape(g["shape"])


if os.environ.get(REGEN_IMAGES_ENV) == "1":
    _write_test_images()
elif os.environ.get(REGEN_SOFT_TOKENS_ENV) == "1":
    _write_soft_token_table()
if os.environ.get(REGEN_PIXELS_ENV) == "1":
    _write_pixel_goldens()
if os.environ.get(REGEN_READOUT_ENV) == "1":
    _write_readout_goldens()
