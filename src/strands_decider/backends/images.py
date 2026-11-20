"""Image input for the decision backends, without torch: the pixels every backend gives its image
processor, and the loading of that processor."""

from __future__ import annotations

import base64
import warnings
from collections.abc import Callable
from typing import TypeVar

from PIL import Image

from ..core import (
    DecisionConfig,
    RenderedImage,
    check_decoded_size,
    check_image_processor,
    resize_image,
)
from ..core.config import IMAGE
from ..images import decode_image
from ..refusals import RequestRefused

_Processor = TypeVar("_Processor")


def image_pixels(image: RenderedImage) -> Image.Image:
    """The image decoded by v1's decoder, so a file gives the same pixels on v1 and here (the
    first frame, upright by its EXIF orientation, transparent areas on white, RGB), then resized
    by the core's `resize_image` from its decoded, upright size. A file whose pixels don't decode
    is refused."""
    try:
        img = decode_image(base64.b64encode(image.data).decode("ascii"))
        return resize_image(img, *check_decoded_size(image, *img.size))
    except ValueError as e:
        raise RequestRefused(str(e)) from e


def load_image_processor(path: str, build: Callable[[], _Processor]) -> _Processor | None:
    """`build()`, the checkpoint's image processor, when its decision_config.json takes images and
    the core accepts its processor_config.json; otherwise None, and image requests are refused.
    Why the processor failed may name local paths, so it is warned, never sent to a client."""
    if IMAGE not in DecisionConfig.load(path).inputs:
        return None
    try:
        check_image_processor(path)
        return build()
    except (OSError, ValueError) as e:
        warnings.warn(f"{path}: image requests are refused: {e}", stacklevel=3)
        return None
