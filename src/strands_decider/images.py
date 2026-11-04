"""Images for the decider without torch: decoding, sizing, and where they go in the prompt.

vision.py (the torch engine) and hidden_engine.py (any backend that returns hidden states,
such as llama.cpp in deploy/lambda) read images through these functions, so both prepare
the same pixels and the same prompt. Pillow is imported on first use
(`pip install "strands-decider[vision]"`).
"""

from __future__ import annotations

import base64
import binascii
import io
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .prompting import render_state
from .schema import Content

if TYPE_CHECKING:
    from PIL import Image

VISION_START = "<|vision_start|>"
IMAGE_PAD = "<|image_pad|>"
VISION_END = "<|vision_end|>"


# ---- prompt ------------------------------------------------------------------------


def render_image_state(state: Content, n_images: int) -> str:
    """`render_state(state)` with one unexpanded placeholder per image, before the text.

    Images come first so the expensive part of the prefix (the vision tower and the
    image tokens through the decoder) is identical across the questions of a request.
    """
    text = render_state(state)
    if not n_images:
        return text
    placeholders = "\n".join(f"{VISION_START}{IMAGE_PAD}{VISION_END}" for _ in range(n_images))
    body = text[len("<state>\n") : -len("\n</state>\n")]
    inner = f"{placeholders}\n{body}" if body else placeholders
    return f"<state>\n{inner}\n</state>\n"


def expand_image_tokens(prompt: str, tokens_per_image: Sequence[int]) -> str:
    """Replace the i-th `<|image_pad|>` with `tokens_per_image[i]` copies.

    The processor does the same expansion; doing it on the string lets the tokeniser's
    offsets be read against exactly the text the model sees.
    """
    parts = prompt.split(IMAGE_PAD)
    if len(parts) - 1 != len(tokens_per_image):
        raise ValueError(
            f"prompt holds {len(parts) - 1} image placeholders but "
            f"{len(tokens_per_image)} token counts were given"
        )
    out = [parts[0]]
    for n, tail in zip(tokens_per_image, parts[1:], strict=True):
        out.append(IMAGE_PAD * int(n))
        out.append(tail)
    return "".join(out)


def image_tokens_for_grid(grid_thw: Sequence[Sequence[int]], merge_size: int) -> list[int]:
    """Tokens each image contributes after the patch merger: t*h*w // merge_size**2."""
    m = merge_size * merge_size
    return [int(t) * int(h) * int(w) // m for t, h, w in grid_thw]


# ---- images ------------------------------------------------------------------------


def _pil() -> Any:
    try:
        from PIL import Image as PILImage
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError('image input needs Pillow: pip install "strands-decider[vision]"') from e
    return PILImage


# What docs/vision.md promises. Anything else Pillow can open (EPS through Ghostscript, for
# one) is refused before it is parsed further.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")


def decode_image(data: str, max_pixels: int = 4096 * 4096) -> Image.Image:
    """A base64 image (optionally a `data:image/...;base64,` URI) as an upright RGB image.

    Only IMAGE_FORMATS are opened, and the size is checked from the header before any
    pixel is decoded, so a small file that expands to a huge bitmap is refused cheaply.
    """
    payload = re.sub(r"^data:image/[\w.+-]+;base64,", "", data.strip())
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image is not valid base64: {e}") from e
    pil = _pil()
    try:
        img = pil.open(io.BytesIO(raw), formats=IMAGE_FORMATS)
    except Exception as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    w, h = img.size
    if w * h > max_pixels:
        raise ValueError(f"image is {w}x{h}; at most {max_pixels:,} pixels are accepted")
    from PIL import ImageOps

    try:
        img.load()
        # Phone photos store their rotation in EXIF. A malformed EXIF block raises
        # SyntaxError or struct.error here, which would otherwise surface as HTTP 500.
        img = ImageOps.exif_transpose(img)
    except Exception as e:
        raise ValueError(f"image could not be decoded: {e}") from e
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        white = pil.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = pil.alpha_composite(white, rgba)  # transparent areas read as white, not black
    return fit_image(img, 0)


def fit_image(img: Image.Image, long_side: int) -> Image.Image:
    """RGB, scaled down (never up) so the longer side is at most `long_side` (0: as is)."""
    img = img.convert("RGB")
    w, h = img.size
    if long_side and max(w, h) > long_side:
        s = long_side / max(w, h)
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), _pil().BICUBIC)
    return img


# ---- the processor's grid -----------------------------------------------------------
# What Qwen2VLImageProcessorPil does to an image after fit_image, for a backend that is
# handed pixels instead of running the processor: patch 16 and merge 2 make a 32-pixel grid,
# and Qwen3.5-2B-Base's preprocessor_config.json bounds its area.
GRID_FACTOR = 32
GRID_MIN_PIXELS = 65536
GRID_MAX_PIXELS = 16777216


def smart_resize(height: int, width: int) -> tuple[int, int]:
    """The processor's smart_resize: sides that are multiples of 32, the area in its bounds."""
    f = GRID_FACTOR
    h_bar, w_bar = round(height / f) * f, round(width / f) * f
    if h_bar * w_bar > GRID_MAX_PIXELS:
        beta = math.sqrt(height * width / GRID_MAX_PIXELS)
        h_bar = max(f, math.floor(height / beta / f) * f)
        w_bar = max(f, math.floor(width / beta / f) * f)
    elif h_bar * w_bar < GRID_MIN_PIXELS:
        beta = math.sqrt(GRID_MIN_PIXELS / (height * width))
        h_bar, w_bar = math.ceil(height * beta / f) * f, math.ceil(width * beta / f) * f
    return h_bar, w_bar


def to_grid(img: Image.Image) -> tuple[bytes, int, int, int]:
    """The image resized as the processor resizes it (bicubic, to its grid), as RGB bytes,
    with its width, its height and its tokens after the patch merger."""
    w, h = img.size
    gh, gw = smart_resize(h, w)
    if (gw, gh) != (w, h):
        img = img.resize((gw, gh), _pil().BICUBIC)
    return img.tobytes(), gw, gh, (gh // GRID_FACTOR) * (gw // GRID_FACTOR)


def state_segments(state: Content, n_images: int) -> list[str]:
    """render_image_state's text around its images: n_images + 1 segments, with image i
    between segment i and segment i + 1."""
    parts = render_image_state(state, n_images).split(f"{VISION_START}{IMAGE_PAD}{VISION_END}")
    if len(parts) != n_images + 1:
        raise ValueError("the state text holds a vision placeholder of its own")
    return parts
