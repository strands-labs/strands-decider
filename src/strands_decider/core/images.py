"""Image input (docs/spec.md, The request and The state block), without Pillow or torch.

A request's image is a base64 file: PNG, JPEG, WebP or GIF. The core checks it and reads its
width and height from the file header, never from its pixels, and from them counts the soft
tokens N that Gemma 4's image processor gives the image. That is all the prompt needs: the
image becomes `<|image>`, N `<|image|>` placeholders and `<image|>` in the state block, and
the runtime's vision encoder fills the placeholders in. Decoding the pixels and preparing
them for the encoder stay with the runtime.

Validation follows v1's (`strands_decider.vision.decode_image`), with its messages: at most
4 images; standard base64 by the spec's rule (not Python's, which varies by version), after an optional
`data:image/...;base64,` prefix; one of the four formats, by its signature; at most
4096 x 4096 pixels, from the header. v1 opens the file with Pillow, which reads the same
header fields; where the core finds the header malformed it gives its own reason after v1's
prefix. A file whose header is sound may still fail to decode, and the runtime refuses it
then, with v1's `image could not be decoded`.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any

from .text import strip

# What v1 accepts (vision.IMAGE_FORMATS), named as Pillow names them.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
# v1's limits (vision.VisionEngineConfig): images per request, and pixels per image.
MAX_IMAGES = 4
MAX_IMAGE_PIXELS = 4096 * 4096
# v1's prefix rule (vision.decode_image), applied after stripping surrounding whitespace, with
# `\w` written out in ASCII so that it does not follow the interpreter's Unicode tables.
DATA_URI = re.compile(r"^data:image/[A-Za-z0-9_.+-]+;base64,")

# Gemma 4's image processor, as every Gemma 4 base's processor_config.json sets it
# (Gemma4ImageProcessor for E2B, E4B, 26B-A4B and 31B; Gemma4UnifiedImageProcessor for 12B).
PATCH_SIZE = 16
POOLING_KERNEL_SIZE = 3
MAX_SOFT_TOKENS = 280

# The three control tokens of an image in the prompt, by their strings: begin, one soft-token
# placeholder per N, end (the processor's `boi_token`, `image_token`, `eoi_token`).
IMAGE_OPEN = "<|image>"
IMAGE_SOFT = "<|image|>"
IMAGE_CLOSE = "<image|>"


def resize_target(
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

    The width and height are the decoded image's, after its EXIF orientation: swapping them
    swaps W and H (`check_decoded_size`)."""
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
    resizes the image to `resize_target`'s W x H, then pools each 3 x 3 group of 16-pixel
    patches into one soft token, so N = (W / 48) (H / 48). N lies in [1, max_soft_tokens],
    and swapping width and height leaves it unchanged."""
    w, h = resize_target(
        width,
        height,
        max_soft_tokens=max_soft_tokens,
        patch_size=patch_size,
        pooling_kernel_size=pooling_kernel_size,
    )
    side = pooling_kernel_size * patch_size
    return (h // side) * (w // side)


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
    return resize_target(width, height)


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


_B64_DATA = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/")
# RFC 4648 section 4: groups of four, the last one padded with one or two "=" when short.
_B64_CANONICAL = re.compile(r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")


def _strict_b64decode(payload: str) -> bytes:
    """Standard base64 and nothing else, by the spec's rule rather than Python's: what
    `base64.b64decode(validate=True)` accepts, and the messages it gives, change between Python
    3.10, 3.11, 3.12 and 3.13. Non-zero bits before the padding are accepted."""
    if not payload.isascii():
        raise ValueError("a character that is not ASCII")
    if not _B64_DATA.issuperset(payload.replace("=", "")):
        raise ValueError("a character outside the standard base64 alphabet")
    if not _B64_CANONICAL.fullmatch(payload):
        raise ValueError("its length or padding is wrong")
    return base64.b64decode(payload, validate=True)


def decode_image(data: str, max_pixels: int = MAX_IMAGE_PIXELS) -> EncodedImage:
    """One request image, checked as v1 checks it: the file, its format and its header size.

    `data` is base64, optionally behind a `data:image/...;base64,` prefix. The pixels are
    not decoded."""
    payload = DATA_URI.sub("", strip(data))
    try:
        raw = _strict_b64decode(payload)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image is not valid base64: {e}") from e
    try:
        fmt, width, height = image_size(raw)
    except ValueError as e:
        raise ValueError(f"image could not be read as {'/'.join(IMAGE_FORMATS)}: {e}") from e
    if width * height > max_pixels:
        raise ValueError(f"image is {width}x{height}; at most {max_pixels:,} pixels are accepted")
    return EncodedImage(raw, fmt, width, height)


# ---- headers --------------------------------------------------------------------------------


def image_size(data: bytes) -> tuple[str, int, int]:
    """(format, width, height) from an image file's header: PNG, JPEG, WebP or GIF, told
    apart by their signatures (Pillow's). ValueError for any other file, a malformed or
    truncated header, or a width or height of 0."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        fmt, (w, h) = "PNG", _png_size(data)
    elif data.startswith(b"\xff\xd8\xff"):
        fmt, (w, h) = "JPEG", _jpeg_size(data)
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP" and data[12:16] in _WEBP_CHUNKS:
        fmt, (w, h) = "WEBP", _webp_size(data)
    elif data.startswith((b"GIF87a", b"GIF89a")):
        fmt, (w, h) = "GIF", _gif_size(data)
    else:
        raise ValueError("not a PNG, JPEG, WebP or GIF file")
    if w < 1 or h < 1:
        raise ValueError(f"its header gives a size of {w}x{h}")
    return fmt, w, h


def _png_size(b: bytes) -> tuple[int, int]:
    """The IHDR chunk's width and height, 4-byte big-endian integers at the start of its
    data. As Pillow opens a PNG, the chunks are walked up to the first IDAT, which must be
    reached: each needs a type of four ASCII letters, its whole length within the file, and
    a matching CRC. The IHDR may come anywhere before the IDAT (the PNG specification puts
    it first; Pillow does not insist), with at least 13 bytes; a later one replaces an
    earlier one."""
    size: tuple[int, int] | None = None
    pos = 8
    while True:
        if pos + 8 > len(b):
            raise ValueError("truncated PNG file: no IDAT chunk")
        length, ctype = struct.unpack(">I4s", b[pos : pos + 8])
        if not ctype.isalpha():  # bytes: ASCII letters only
            raise ValueError(f"broken PNG file (chunk {ctype!r})")
        if ctype == b"IDAT":
            break
        end = pos + 8 + length
        if end + 4 > len(b):
            raise ValueError(f"truncated PNG chunk {ctype.decode()}")
        if zlib.crc32(b[pos + 4 : end]) != struct.unpack(">I", b[end : end + 4])[0]:
            raise ValueError(f"broken PNG file (bad header checksum in {ctype.decode()})")
        if ctype == b"IHDR":
            if length < 13:
                raise ValueError("truncated IHDR chunk")
            w, h = struct.unpack(">II", b[pos + 8 : pos + 16])
            size = (int(w), int(h))
        pos = end + 4
    if size is None:
        raise ValueError("the PNG file has no IHDR chunk before its image data")
    return size


# JPEG markers as Pillow's JpegImagePlugin reads them before the first scan: the frame
# headers (SOF0-3, 5-7, 9-11, 13-15, and DHP), the other segments that carry a length, and
# the markers without one. Any other marker is refused ("no marker found").
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF, 0xDE}
_SOS = 0xDA
_SEGMENTS = {0xC4, 0xCC, 0xDA, 0xDB, 0xDC, 0xDD, 0xDF, 0xFE, *range(0xE0, 0xF0), *_SOF}
_STANDALONE = {0xC8, *range(0xD0, 0xDA), *range(0xF0, 0xFE)}


def _jpeg_size(b: bytes) -> tuple[int, int]:
    """The frame header (SOFn) before the first scan: height and width as 2-byte big-endian
    integers after the 1-byte sample precision, which must be 8, and then the number of
    components, which must be 1, 3 or 4. Segments are walked as Pillow walks them: bytes
    other than 0xFF between segments are skipped, 0xFF 0xFF is fill, 0xFF 0x00 is skipped,
    and a later frame header replaces an earlier one."""
    n, pos = len(b), 3  # after SOI; the signature's third byte is the next marker's 0xFF
    size: tuple[int, int] | None = None
    prefix = True  # the byte before `pos` was 0xFF
    while True:
        if not prefix:
            if pos >= n:
                break
            prefix, pos = b[pos] == 0xFF, pos + 1  # junk between segments is skipped
            continue
        if pos >= n:
            break
        code, pos = b[pos], pos + 1
        if code == 0xFF:  # fill: this 0xFF starts the marker
            continue
        if code == 0x00:  # an escaped 0xFF: skipped
            prefix = False
            continue
        if code in _SEGMENTS:
            if pos + 2 > n:
                raise ValueError("truncated JPEG segment")
            (length,) = struct.unpack(">H", b[pos : pos + 2])
            body = b[pos + 2 : pos + length]
            if len(body) < length - 2:
                raise ValueError("truncated JPEG segment")
            pos += max(length, 2)
            if code in _SOF:
                if len(body) < 6:
                    raise ValueError("truncated JPEG frame header")
                if body[0] != 8:
                    raise ValueError(f"cannot handle {body[0]}-bit layers")
                if body[5] not in (1, 3, 4):
                    raise ValueError(f"cannot handle {body[5]}-layer images")
                h, w = struct.unpack(">HH", body[1:5])
                size = (int(w), int(h))
            if code == _SOS:
                break
        elif code not in _STANDALONE:
            raise ValueError("no marker found")
        prefix = False
    if size is None:
        raise ValueError("the JPEG file has no frame header before its first scan")
    return size


_WEBP_CHUNKS = (b"VP8 ", b"VP8L", b"VP8X")


def _webp_size(b: bytes) -> tuple[int, int]:
    """The first chunk's size fields, as Pillow's WebP plugin reads them, with the checks
    libwebp makes of the same header when Pillow opens the file:

    `VP8 ` (lossy): a key frame shown, profile at most 3, the first partition shorter than
    the chunk, the start code 9d 01 2a, then width and height in the low 14 bits of two
    2-byte little-endian integers. `VP8L` (lossless): the signature 0x2f, then width - 1
    and height - 1 in 14 bits each and a version of 0. `VP8X` (extended): at least 10 bytes,
    the canvas width - 1 and height - 1 in 3 bytes each. The RIFF size must cover the first
    chunk's header and lie within the file, and the first chunk must lie within the RIFF."""
    if len(b) < 20:
        raise ValueError("truncated WebP header")
    riff_size, chunk_size = struct.unpack("<I", b[4:8])[0], struct.unpack("<I", b[16:20])[0]
    if riff_size < 12 or riff_size + 8 > len(b) or chunk_size > riff_size - 12:
        raise ValueError("truncated or malformed WebP file (RIFF or chunk size)")
    chunk = b[12:16]
    if chunk == b"VP8 " and len(b) >= 30 and b[23:26] == b"\x9d\x01\x2a":
        tag = int.from_bytes(b[20:23], "little")
        key_frame, profile, shown, first_partition = tag & 1, (tag >> 1) & 7, tag >> 4 & 1, tag >> 5
        if key_frame == 0 and profile <= 3 and shown and first_partition < chunk_size:
            w, h = struct.unpack("<HH", b[26:30])
            return w & 0x3FFF, h & 0x3FFF
    if chunk == b"VP8L" and len(b) >= 25 and b[20] == 0x2F:
        (bits,) = struct.unpack("<I", b[21:25])
        if bits >> 29 == 0:
            return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if chunk == b"VP8X" and len(b) >= 30 and chunk_size >= 10:
        return int.from_bytes(b[24:27], "little") + 1, int.from_bytes(b[27:30], "little") + 1
    raise ValueError(f"malformed WebP {chunk.decode('ascii').strip()} header")


def _gif_size(b: bytes) -> tuple[int, int]:
    """The logical screen's width and height (2-byte little-endian), enlarged to cover the
    first image's extent (its left and top offsets plus its width and height), as Pillow
    sizes a GIF: the global colour table and any extension blocks before the first image
    descriptor are skipped, and so is any other byte between blocks."""
    if len(b) < 13:
        raise ValueError("truncated GIF header")
    sw, sh = struct.unpack("<HH", b[6:10])
    flags, n = b[10], len(b)
    pos = 13 + (3 << ((flags & 7) + 1) if flags & 0x80 else 0)
    while pos < n and b[pos] != 0x3B:  # 0x3B: the trailer
        c, pos = b[pos], pos + 1
        if c == 0x21:  # an extension: its label, then data sub-blocks up to an empty one
            pos += 1
            while pos < n and b[pos]:
                pos += 1 + b[pos]
            pos += 1
        elif c == 0x2C:  # the first image descriptor, its colour table, its LZW code size
            local = b[pos + 8] if pos + 9 <= n else 0
            if pos + 9 + (3 << ((local & 7) + 1) if local & 0x80 else 0) >= n:
                raise ValueError("truncated GIF image descriptor")
            x0, y0, w, h = struct.unpack("<HHHH", b[pos : pos + 8])
            return max(sw, x0 + w), max(sh, y0 + h)
    raise ValueError("the GIF file holds no image")
