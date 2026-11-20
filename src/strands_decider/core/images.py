"""Image input: each image's checks, its resized size and its placeholder count N.

The core reads only an image's header (format and size). The backend decodes the pixels, resizes
them with `resize_image` and runs its vision encoder into the image's N placeholders. Pillow is
imported only when an image is read, so the core imports without it.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

if TYPE_CHECKING:
    from PIL.Image import Image

    from .prompt import RenderedImage

# The formats and pixel limit v1 accepts, so a client can send v2 what it sends v1.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
MAX_IMAGE_PIXELS = 4096 * 4096
# Bytes per image file, checked before Pillow reads the header. 16 MiB holds a JPEG or WebP of
# MAX_IMAGE_PIXELS, and bounds the time Pillow spends on a header.
MAX_IMAGE_BYTES = 16 * 2**20
DATA_URI = re.compile(r"^data:image/[A-Za-z0-9_.+-]+;base64,")

# Gemma 4's image processor, as every base's processor_config.json sets it: 16-pixel patches,
# pooled 3 x 3 into one soft token, at most 280 soft tokens per image.
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
MAX_SOFT_TOKENS = 280
PROCESSOR_CONFIG_FILE = "processor_config.json"

# An image in the prompt: begin, one placeholder per soft token, end.
IMAGE_OPEN = "<|image>"
IMAGE_SOFT = "<|image|>"
IMAGE_CLOSE = "<image|>"


def resized_size(width: int, height: int) -> tuple[int, int]:
    """(W, H) Gemma 4's processor resizes an image to: the aspect ratio kept, each side a multiple
    of 48, and at most MAX_SOFT_TOKENS soft tokens. The processor's arithmetic, operation for
    operation (`get_aspect_ratio_preserving_size` in transformers' Gemma4ImageProcessor). Call it
    with the decoded size, after EXIF orientation."""
    side = POOLING_KERNEL_SIZE * PATCH_SIZE
    factor = math.sqrt(MAX_SOFT_TOKENS * side * side / (height * width))
    target_height = math.floor(factor * height / side) * side
    target_width = math.floor(factor * width / side) * side
    if target_height == 0:  # wider than 280:1
        target_height = side
        target_width = min(math.floor(width / height) * side, MAX_SOFT_TOKENS * side)
    elif target_width == 0:
        target_width = side
        target_height = min(math.floor(height / width) * side, MAX_SOFT_TOKENS * side)
    return target_width, target_height


def soft_tokens(width: int, height: int) -> int:
    """N, the image's placeholders in the prompt: one per 48 x 48 block of its resized size."""
    w, h = resized_size(width, height)
    side = POOLING_KERNEL_SIZE * PATCH_SIZE
    return (h // side) * (w // side)


def resize_image(image: Image, width: int, height: int) -> Image:
    """A decoded RGB PIL image resized to W x H with Pillow's bicubic filter, width first, then
    height. Every backend resizes with this and runs its processor with its own resize off, so
    they all give the encoder the same pixels (one two-axis `resize` changes pass order in Pillow 13)."""
    pil = _pillow()
    if image.mode != "RGB":  # Pillow resizes palette and 1-bit images by nearest neighbour
        raise ValueError(f"resize_image takes the decoded RGB image, not a {image.mode} one")
    if image.width != width:
        image = image.resize((width, image.height), pil.Resampling.BICUBIC)
    if image.height != height:
        image = image.resize((width, height), pil.Resampling.BICUBIC)
    return image


def check_decoded_size(
    image: EncodedImage | RenderedImage, width: int, height: int
) -> tuple[int, int]:
    """(W, H) to resize a decoded image of `width` x `height` to. The decoded size must be the
    header's, or its transpose after EXIF orientation; W and H come from the decoded size."""
    header = (image.width, image.height)
    if (width, height) not in (header, header[::-1]):
        raise ValueError(
            f"image could not be decoded: its pixels are {width}x{height}, and its header "
            f"gives {header[0]}x{header[1]}"
        )
    return resized_size(width, height)


class _ProcessorSettings(BaseModel):
    """The image processor settings the core's placeholder count depends on."""

    model_config = ConfigDict(extra="allow")

    patch_size: Literal[16]
    pooling_kernel_size: Literal[3]
    max_soft_tokens: Literal[280]


def check_image_processor(config: dict[str, Any] | str | os.PathLike[str]) -> None:
    """That a checkpoint's processor_config.json (read, or its path or directory) sizes images as
    the core counts their placeholders; a mismatch would give the encoder the wrong number of slots."""
    if isinstance(config, dict):
        settings = config
    else:
        path = Path(config)
        settings = json.loads(
            (path / PROCESSOR_CONFIG_FILE if path.is_dir() else path).read_text("utf-8")
        )
    try:
        _ProcessorSettings.model_validate(settings.get("image_processor", settings))
    except ValidationError as e:
        raise ValueError(f"the checkpoint's image processor can't take images here: {e}") from None


@dataclass(frozen=True)
class EncodedImage:
    """An image as the request sends it: the file, its format, and its size from the header."""

    data: bytes = field(repr=False)
    format: str
    width: int
    height: int
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sha256", hashlib.sha256(self.data).hexdigest())

    @property
    def tokens(self) -> int:
        return soft_tokens(self.width, self.height)


def _pillow() -> Any:  # PIL.Image, typed only where Pillow is installed
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError('image input needs Pillow: pip install "strands-decider[vision]"') from e
    return Image


def decode_image(data: str) -> EncodedImage:
    """One request image, checked from its header: base64 (optionally a data URI), at most
    MAX_IMAGE_BYTES, one of IMAGE_FORMATS, at most MAX_IMAGE_PIXELS. The pixels aren't decoded."""
    try:
        raw = base64.b64decode(DATA_URI.sub("", data.strip()), validate=True)
    except ValueError as e:  # binascii.Error, or a string that isn't ASCII
        raise ValueError(f"image is not valid base64: {e}") from e
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"image is {len(raw)} bytes; at most {MAX_IMAGE_BYTES:,} bytes are accepted"
        )
    pil = _pillow()
    unreadable = f"image could not be read as {'/'.join(IMAGE_FORMATS)}"
    try:
        with pil.open(io.BytesIO(raw), formats=IMAGE_FORMATS) as im:
            fmt, (width, height) = im.format, im.size
    except Exception as e:  # Pillow raises more than OSError for a bad file
        raise ValueError(f"{unreadable}: {e}") from e
    if width < 1 or height < 1:  # Pillow opens a PNG whose header says 0
        raise ValueError(f"{unreadable}: its header gives a size of {width}x{height}")
    if width * height > MAX_IMAGE_PIXELS:
        raise ValueError(
            f"image is {width}x{height}; at most {MAX_IMAGE_PIXELS:,} pixels are accepted"
        )
    # A JPEG with more pictures after the first (MPO, as some cameras write) is a JPEG.
    return EncodedImage(raw, "JPEG" if fmt == "MPO" else fmt, width, height)
