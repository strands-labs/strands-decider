"""Shared by the torch-free core's tests (tests/test_core_*.py).

The render goldens and properties run against Gemma 4's real tokenizer. It is 32 MB, so it
is not committed: point STRANDS_DECIDER_GEMMA4_TOKENIZER at a copy of `tokenizer.json`
(google/gemma-4-E2B-it at GEMMA4_REVISION, the file every Gemma 4 size ships), and the
tests check its sha256 before they trust it. Without it those tests skip; the tests that
need only text, numbers or a toy tokenizer always run.

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
and torchvision). Regenerate the render goldens after the images.
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
TOKENIZER_ENV = "STRANDS_DECIDER_GEMMA4_TOKENIZER"
TEMPLATES_ENV = "STRANDS_DECIDER_GEMMA4_TEMPLATES"
REGEN_IMAGES_ENV = "STRANDS_DECIDER_REGEN_IMAGES"
IMAGES = FIXTURES / "images"
SOFT_TOKENS = FIXTURES / "image_soft_tokens.json"


@lru_cache(maxsize=1)
def _gemma4_path() -> str | None:
    p = os.environ.get(TOKENIZER_ENV)
    if not p:
        return None
    if os.path.isdir(p):
        p = os.path.join(p, "tokenizer.json")
    with open(p, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    if digest != GEMMA4_TOKENIZER_SHA256:
        raise AssertionError(
            f"{p} has sha256 {digest}; the render goldens are for {GEMMA4_REPO}@{GEMMA4_REVISION} "
            f"({GEMMA4_TOKENIZER_SHA256})"
        )
    return p


def gemma4_tokenizer_path() -> str:
    p = _gemma4_path()
    if p is None:
        pytest.skip(f"set {TOKENIZER_ENV} to Gemma 4's tokenizer.json ({GEMMA4_REPO})")
    return p


@lru_cache(maxsize=1)
def _load_gemma4(path: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(path)


def gemma4_tokenizer():
    """Gemma 4's tokenizer, or a skip. One instance per session: it is 262k tokens."""
    return _load_gemma4(gemma4_tokenizer_path())


def gemma4_tokenizer_dir() -> str:
    """The directory of Gemma 4's tokenizer.json, if it also holds the base's
    tokenizer_config.json (Hugging Face's processors need its named tokens), or a skip."""
    d = os.path.dirname(gemma4_tokenizer_path())
    if not os.path.exists(os.path.join(d, "tokenizer_config.json")):
        pytest.skip(f"no tokenizer_config.json beside {TOKENIZER_ENV}'s tokenizer.json")
    return d


def decision_config_fixture() -> dict:
    """The scrubbed E2B decision_config.json: the schema's fields only (codes and token ids
    as the checkpoint has them; `prompt` and `max_length_validated` are test values)."""
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


def soft_token_sizes() -> list[tuple[int, int]]:
    """The (width, height) pairs of tests/fixtures/core/image_soft_tokens.json: a grid of
    common and boundary sides, extreme aspect ratios to the pixel limit (1 x 16,777,216),
    both sides of the 280:1 ratio where one resized side becomes 48 pixels, every size whose
    area makes the processor's scale factor exactly 1 and samples where it is 1/2, 1/3 and
    1/4, and log-uniform random sizes."""
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
    return sorted(out)


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
    import PIL
    import torchvision
    import transformers

    IMAGES.mkdir(parents=True, exist_ok=True)
    for name in IMAGE_FILES:
        (IMAGES / name).write_bytes(make_image(name))
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


if os.environ.get(REGEN_IMAGES_ENV) == "1":
    _write_test_images()
