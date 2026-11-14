"""Image input (docs/spec.md, sections 1.1 and 3.3): the checks, the size and N.

A request's image is a base64 file: PNG, JPEG, WebP or GIF. The core checks it and reads its
format, width and height from its header with Pillow, never decoding its pixels, and from the
size counts the soft tokens N that Gemma 4's image processor gives the image. That is all the
prompt needs: the image becomes `<|image>`, N `<|image|>` placeholders and `<image|>` in the
state block, and the runtime's vision encoder fills the placeholders in. Decoding the pixels
and preparing them for the encoder stay with the runtime.

The checks are v1's (`strands_decider.vision.decode_image`), with its messages, and a cap on
the file's bytes: at most 4 images; standard base64 (RFC 4648, not Python's rule, which varies
by version), after an optional `data:image/...;base64,` prefix; at most 16 MiB of file; one of
the four formats, as Pillow opens it; at most 4096 x 4096 pixels. A file whose header is sound
may still fail to decode, and the runtime refuses it then, with v1's `image could not be
decoded`. Image input needs Pillow, which the core imports only when it reads or resizes one.
"""

from __future__ import annotations

import binascii
import hashlib
import io
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# What v1 accepts (vision.IMAGE_FORMATS), named as Pillow names them. Pillow opens no other.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
# v1's limits (vision.VisionEngineConfig): images per request, and pixels per image.
MAX_IMAGES = 4
MAX_IMAGE_PIXELS = 4096 * 4096
# Bytes per image file, checked before Pillow reads it. 16 MiB holds a photo of MAX_IMAGE_PIXELS
# as JPEG or WebP. Gemma 4's processor shrinks every image to at most 645,120 pixels, so a larger
# file adds nothing the model sees, and the cap bounds the work of reading a header: on Pillow
# 13, under 2 s for the worst files measured. Pillow 12 reads a GIF comment of many sub-blocks
# in quadratic time, so there a 16 MiB GIF can take minutes.
MAX_IMAGE_BYTES = 16 * 2**20
# v1's prefix rule (vision.decode_image), applied after stripping surrounding whitespace, with
# `\w` written out in ASCII so that it does not follow the interpreter's Unicode tables.
DATA_URI = re.compile(r"^data:image/[A-Za-z0-9_.+-]+;base64,")

# Gemma 4's image processor, as every Gemma 4 base's processor_config.json sets it
# (Gemma4ImageProcessor for E2B, E4B, 26B-A4B and 31B; Gemma4UnifiedImageProcessor for 12B).
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
MAX_SOFT_TOKENS = 280
IMAGE_PROCESSOR_SETTINGS = {
    "patch_size": PATCH_SIZE,
    "pooling_kernel_size": POOLING_KERNEL_SIZE,
    "max_soft_tokens": MAX_SOFT_TOKENS,
}
# Where processor_config.json gives `image_seq_length`, under image_processor or at its top
# level, it is max_soft_tokens.
IMAGE_SEQ_LENGTH = MAX_SOFT_TOKENS
PROCESSOR_CONFIG_FILE = "processor_config.json"

# The three control tokens of an image in the prompt, by their strings: begin, one soft-token
# placeholder per N, end (the processor's `boi_token`, `image_token`, `eoi_token`).
IMAGE_OPEN = "<|image>"
IMAGE_SOFT = "<|image|>"
IMAGE_CLOSE = "<image|>"


def resized_size(
    width: int,
    height: int,
    *,
    max_soft_tokens: int = MAX_SOFT_TOKENS,
    patch_size: int = PATCH_SIZE,
    pooling_kernel_size: int = POOLING_KERNEL_SIZE,
) -> tuple[int, int]:
    """(W, H): the size Gemma 4's image processor resizes a `width` x `height` image to,
    keeping its aspect ratio: the largest whose sides are multiples of
    `pooling_kernel_size * patch_size` (48) and whose area is at most
    `max_soft_tokens * pooling_kernel_size**2 * patch_size**2` pixels. This is its
    arithmetic, operation for operation in IEEE double precision
    (`get_aspect_ratio_preserving_size` in transformers' Gemma4ImageProcessor and
    Gemma4UnifiedImageProcessor, which size alike).

    A runtime resizes to it (`resize_image`) before the processor, with the processor's own
    resize off (docs/spec.md, section 3.3), and calls it with the decoded image's width and
    height, after its EXIF orientation, not the header's: swapping them swaps W and H.
    `check_decoded_size` checks the decoded size against the header and calls it so;
    `soft_tokens` counts its soft tokens, which do not swap."""
    if not (isinstance(width, int) and isinstance(height, int)) or width < 1 or height < 1:
        raise ValueError(f"an image is at least 1 x 1 pixels, not {width!r} x {height!r}")
    max_patches = max_soft_tokens * pooling_kernel_size**2
    target_px = max_patches * patch_size**2
    side = pooling_kernel_size * patch_size
    factor = math.sqrt(target_px / (height * width))
    target_height = math.floor(factor * height / side) * side
    target_width = math.floor(factor * width / side) * side
    if target_height == 0:  # wider than 280:1; the width cannot be 0 too
        target_height = side
        target_width = min(math.floor(width / height) * side, max_soft_tokens * side)
    elif target_width == 0:
        target_width = side
        target_height = min(math.floor(height / width) * side, max_soft_tokens * side)
    if target_height * target_width > target_px:  # never, for any size; the processor checks
        raise ValueError(f"{width} x {height} exceeds {max_soft_tokens} soft tokens")
    return target_width, target_height


def soft_tokens(
    width: int,
    height: int,
    *,
    max_soft_tokens: int = MAX_SOFT_TOKENS,
    patch_size: int = PATCH_SIZE,
    pooling_kernel_size: int = POOLING_KERNEL_SIZE,
) -> int:
    """N, the soft tokens Gemma 4's image processor gives a `width` x `height` image: it
    resizes the image to `resized_size`'s W x H, then pools each 3 x 3 group of 16-pixel
    patches into one soft token, so N = (W / 48) (H / 48). N lies in [1, max_soft_tokens],
    and swapping width and height leaves it unchanged."""
    w, h = resized_size(
        width,
        height,
        max_soft_tokens=max_soft_tokens,
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
    )
    side = pooling_kernel_size * patch_size
    return (h // side) * (w // side)


def resize_image(image: Any, width: int, height: int) -> Any:
    """`image`, a decoded PIL RGB image, resized to `width` x `height` by docs/spec.md's
    bicubic resize: the width's pass first, then the height's, each a single-axis
    `Image.Resampling.BICUBIC` call, so that the result is the spec's on every Pillow version.
    (One call, `image.resize((W, H), BICUBIC)`, is the same before Pillow 13; from 13 it does
    the height first when the height shrinks by more than twice as much as the width, and
    its pixels differ.) A pass whose size does not change is left out. A runtime may use this
    or any implementation within 1 per channel of it."""
    from PIL import Image

    if not (isinstance(width, int) and isinstance(height, int)) or width < 1 or height < 1:
        raise ValueError(f"a resized image is at least 1 x 1 pixels, not {width!r} x {height!r}")
    if image.mode != "RGB":  # Pillow resizes palette and 1-bit images by nearest neighbour
        raise ValueError(f"resize_image takes the decoded RGB image, not a {image.mode} one")
    if image.width != width:
        image = image.resize((width, image.height), Image.Resampling.BICUBIC)
    if image.height != height:
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    return image


def check_decoded_size(image: Any, width: int, height: int) -> tuple[int, int]:
    """(W, H), the size to resize a decoded image of `width` x `height` pixels to (after its
    EXIF orientation), for a request `image` (an EncodedImage or a RenderedImage, sized from
    its header). The decoded size must be the header's or, turned a quarter by the
    orientation, its transpose; otherwise ValueError with v1's `image could not be decoded`.
    W and H come from the decoded size, never the header's: for a turned image they swap,
    and N, which does not, cannot show a runtime that resized it to the header's shape."""
    header = (image.width, image.height)
    if (width, height) not in (header, header[::-1]):
        raise ValueError(
            f"image could not be decoded: its pixels are {width}x{height}, and its header "
            f"gives {header[0]}x{header[1]}"
        )
    return resized_size(width, height)


def check_image_processor(processor_config: Mapping[str, Any] | str | os.PathLike[str]) -> None:
    """That a checkpoint's image processor sizes images as the core counts them: its
    `processor_config.json` (a mapping, the file, or the checkpoint directory) gives
    `patch_size` 16, `pooling_kernel_size` 3 and `max_soft_tokens` 280, under
    `image_processor` or at its top level, and `image_seq_length`, wherever it is given,
    280. ValueError otherwise: a runtime refuses image requests for such a checkpoint
    (docs/spec.md, section 9)."""
    if not isinstance(processor_config, Mapping):
        path = os.fspath(processor_config)
        if os.path.isdir(path):
            path = os.path.join(path, PROCESSOR_CONFIG_FILE)
        with open(path, encoding="utf-8") as f:
            processor_config = json.load(f)
        if not isinstance(processor_config, Mapping):
            raise ValueError(f"{PROCESSOR_CONFIG_FILE} must hold a JSON object")
    settings = processor_config.get("image_processor", processor_config)
    if not isinstance(settings, Mapping):
        raise ValueError(f"{PROCESSOR_CONFIG_FILE}'s image_processor must be an object")
    wrong = {
        k: settings.get(k, "<missing>")
        for k, v in IMAGE_PROCESSOR_SETTINGS.items()
        if isinstance(settings.get(k), bool) or settings.get(k) != v
    }
    for where in (settings, processor_config):
        n = where.get("image_seq_length", IMAGE_SEQ_LENGTH)
        if isinstance(n, bool) or n != IMAGE_SEQ_LENGTH:
            wrong["image_seq_length"] = n
    if wrong:
        raise ValueError(
            f"the checkpoint's image processor has {wrong}; the interface counts soft tokens "
            f"for {IMAGE_PROCESSOR_SETTINGS}, so this checkpoint cannot take images"
        )


@dataclass(frozen=True)
class EncodedImage:
    """An image as the request sends it: the file's bytes, its format, and its width and
    height as its header gives them. `sha256` is the hex digest of `data`."""

    data: bytes = field(repr=False)
    format: str
    width: int
    height: int
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", hashlib.sha256(self.data).hexdigest())

    @property
    def tokens(self) -> int:
        """N: its soft tokens in the prompt."""
        return soft_tokens(self.width, self.height)


_B64_OTHER = re.compile(r"[^A-Za-z0-9+/=]")


def _strict_b64decode(payload: str) -> bytes:
    """Standard base64 and nothing else, by the spec's rule rather than Python's: what
    `base64.b64decode(validate=True)` accepts, and the messages it gives, change between Python
    3.11, 3.12 and 3.13. Non-zero bits before the padding are accepted. RFC 4648 section 4:
    groups of four, the last padded with one or two "=" when short. In linear time and constant
    memory, without a regular expression's repetition."""
    if not payload.isascii():
        raise ValueError("a character that is not ASCII")
    if _B64_OTHER.search(payload):
        raise ValueError("a character outside the standard base64 alphabet")
    data = payload.rstrip("=")
    pad = len(payload) - len(data)
    if len(payload) % 4 or pad > 2 or "=" in data:
        raise ValueError("its length or padding is wrong")
    # The string is now valid by the rule, so the decoder only maps it to bytes: binascii's,
    # which does not depend on what the interpreter's base64 module accepts.
    return binascii.a2b_base64(payload)


def _pillow() -> Any:
    """Pillow's Image module, which image input needs, imported when first used: the core
    imports without it."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError('image input needs Pillow: pip install "strands-decider[vision]"') from e
    return Image


def decode_image(
    data: str, max_pixels: int = MAX_IMAGE_PIXELS, max_bytes: int = MAX_IMAGE_BYTES
) -> EncodedImage:
    """One request image, checked as v1 checks it, with a cap on its bytes: base64, at most
    `max_bytes` bytes of file, a PNG, JPEG, WebP or GIF file as Pillow opens it, and at most
    `max_pixels` pixels.

    `data` is base64, optionally behind a `data:image/...;base64,` prefix. Pillow reads only
    the header; the pixels are not decoded. ImportError without Pillow."""
    payload = DATA_URI.sub("", data.strip())
    try:
        raw = _strict_b64decode(payload)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image is not valid base64: {e}") from e
    if len(raw) > max_bytes:
        raise ValueError(f"image is {len(raw)} bytes; at most {max_bytes:,} bytes are accepted")
    pil = _pillow()
    unreadable = f"image could not be read as {'/'.join(IMAGE_FORMATS)}"
    try:
        with pil.open(io.BytesIO(raw), formats=IMAGE_FORMATS) as im:
            fmt, (width, height) = im.format, im.size
    except Exception as e:  # as v1 catches it: Pillow raises more than OSError for a bad file
        raise ValueError(f"{unreadable}: {e}") from e
    if width < 1 or height < 1:  # Pillow opens a PNG whose header says 0, and cannot decode it
        raise ValueError(f"{unreadable}: its header gives a size of {width}x{height}")
    if width * height > max_pixels:
        raise ValueError(f"image is {width}x{height}; at most {max_pixels:,} pixels are accepted")
    # A JPEG with more pictures after the first (MPO, as some cameras write) is a JPEG here.
    return EncodedImage(raw, "JPEG" if fmt == "MPO" else fmt, width, height)


def decode_images(images: Sequence[str]) -> tuple[EncodedImage, ...]:
    """A request's images, checked as v1 checks them: at most MAX_IMAGES, counted before any is
    decoded, then each in turn, every check of one before the next's (`decode_image`)."""
    if len(images) > MAX_IMAGES:
        raise ValueError(f"{len(images)} images; this server takes at most {MAX_IMAGES}")
    return tuple(decode_image(i) for i in images)
