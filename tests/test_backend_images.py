"""The pixels every decision backend gives its image processor (strands_decider.backends.images)."""

from __future__ import annotations

import base64

import numpy as np
import pytest
from core_support import PIXEL_FILES, image_bytes, reference_pixels

from strands_decider.backends.images import image_pixels
from strands_decider.core import RenderedImage, decode_image, resize_image, resized_size


@pytest.mark.parametrize("name", sorted(PIXEL_FILES))
def test_image_pixels_are_the_cores_reference(name):
    """v1's decode, resized by the core's `resize_image` from the decoded, upright size:
    rotated-80x40.jpg, whose EXIF turns it, to 528 x 1104, never its header's 1104 x 528."""
    data = image_bytes(name)
    im = decode_image(base64.b64encode(data).decode())
    got = image_pixels(
        RenderedImage(im.data, im.format, im.width, im.height, im.sha256, 0, im.tokens)
    )
    decoded = reference_pixels(data)
    want = resize_image(decoded, *resized_size(*decoded.size))
    assert got.size == want.size and np.array_equal(np.asarray(got), np.asarray(want))
