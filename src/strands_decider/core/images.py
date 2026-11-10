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
import json
import math
import os
import re
import struct
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .text import strip

# What v1 accepts (vision.IMAGE_FORMATS), named as Pillow names them.
IMAGE_FORMATS = ("PNG", "JPEG", "WEBP", "GIF")
# v1's limits (vision.VisionEngineConfig): images per request, and pixels per image.
MAX_IMAGES = 4
MAX_IMAGE_PIXELS = 4096 * 4096
# Pillow refuses to open an image of more pixels than this (twice its MAX_IMAGE_PIXELS, a
# decompression-bomb guard), so v1 refuses it as unreadable rather than as too large.
PILLOW_PIXEL_LIMIT = 2 * 89_478_485
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
    resize off (docs/spec.md, section 11.1), and calls it with the decoded image's width and
    height,
    after its EXIF orientation, not the header's: swapping them swaps W and H.
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
    its pixels differ.) A pass whose size does not change is left out. Pillow is imported
    only here, so the core still needs no imaging library; a runtime may use this or any
    implementation within 1 per channel of it."""
    from PIL import Image

    if not (isinstance(width, int) and isinstance(height, int)) or width < 1 or height < 1:
        raise ValueError(f"a resized image is at least 1 x 1 pixels, not {width!r} x {height!r}")
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
    `image_processor` or at its top level. ValueError otherwise: a runtime refuses image
    requests for such a checkpoint (docs/spec.md, The checkpoint layout)."""
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
    3.10, 3.11, 3.12 and 3.13. Non-zero bits before the padding are accepted. RFC 4648
    section 4: groups of four, the last padded with one or two "=" when short. In linear
    time and constant memory, without a regular expression's repetition."""
    if not payload.isascii():
        raise ValueError("a character that is not ASCII")
    if _B64_OTHER.search(payload):
        raise ValueError("a character outside the standard base64 alphabet")
    data = payload.rstrip("=")
    pad = len(payload) - len(data)
    if len(payload) % 4 or pad > 2 or "=" in data:
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
        if width * height > PILLOW_PIXEL_LIMIT:
            raise ValueError(
                f"its header gives {width}x{height}, more than the {PILLOW_PIXEL_LIMIT:,} "
                "pixels an image may have to be opened"
            )
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


_PNG_CHUNK_TYPE = re.compile(rb"[A-Za-z0-9_]{4}")  # Pillow's check: four ASCII \w


def _png_size(b: bytes) -> tuple[int, int]:
    """The IHDR chunk's width and height, 4-byte big-endian integers at the start of its
    data. As Pillow opens a PNG, the chunks are walked up to the first IDAT or IEND, which
    must be reached: each chunk before it needs a type of four ASCII letters, digits or
    underscores, its whole length within the file, and a matching CRC; of the IDAT or IEND
    only the length and the type are read. The IHDR may come anywhere before them (the PNG
    specification puts it first; Pillow does not insist), with at least 13 bytes; a later
    one replaces an earlier one."""
    size: tuple[int, int] | None = None
    pos = 8
    while True:
        if pos + 8 > len(b):
            raise ValueError("truncated PNG file: no IDAT chunk")
        length, ctype = struct.unpack(">I4s", b[pos : pos + 8])
        if not _PNG_CHUNK_TYPE.fullmatch(ctype):
            raise ValueError(f"broken PNG file (chunk {ctype!r})")
        if ctype in (b"IDAT", b"IEND"):  # Pillow stops at either; IEND alone fails to decode
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
        raise ValueError("the PNG file has no IHDR chunk before its image data or its end")
    return size


# JPEG markers as Pillow's JpegImagePlugin reads them before the first scan: the frame
# headers (SOF0-3, 5-7, 9-11, 13-15, and DHP), the other segments that carry a length, and
# the markers without one. Any other marker is refused ("no marker found").
_SOF = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF, 0xDE}
_SOS = 0xDA
_SEGMENTS = {0xC4, 0xCC, 0xDA, 0xDB, 0xDC, 0xDD, 0xDF, 0xFE, *range(0xE0, 0xF0), *_SOF}
_STANDALONE = {0xC8, *range(0xD0, 0xDA), *range(0xF0, 0xFE)}


def _jpeg_size(b: bytes) -> tuple[int, int]:
    """The last frame header (SOFn) before the first scan (SOS): height and width as 2-byte
    big-endian integers after the 1-byte sample precision, which must be 8, and then the
    number of components, which must be 1, 3 or 4. Segments are walked as Pillow walks
    them: bytes other than 0xFF between segments are skipped, 0xFF 0xFF is fill, 0xFF 0x00
    is skipped, a segment length below 2 counts as 2, and a later frame header replaces an
    earlier one. The file must reach the end of the first scan's header: one that ends
    before it, between segments or inside one, is refused, as Pillow refuses it."""
    n, pos = len(b), 3  # after SOI; the signature's third byte is the next marker's 0xFF
    size: tuple[int, int] | None = None
    prefix = True  # the byte before `pos` was 0xFF
    while True:
        if not prefix:
            ff = b.find(b"\xff", pos)  # junk between segments is skipped
            if ff < 0:
                raise ValueError("truncated JPEG file: it ends before its first scan")
            prefix, pos = True, ff + 1
            continue
        if pos >= n:
            raise ValueError("truncated JPEG file: it ends before its first scan")
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
                if size is None:
                    raise ValueError("the JPEG file has no frame header before its first scan")
                return size
        elif code not in _STANDALONE:
            raise ValueError("no marker found")
        prefix = False


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


_GIF_BLOCK = re.compile(rb"[\x21\x2c\x3b]")  # an extension, an image descriptor, the trailer


def _gif_size(b: bytes) -> tuple[int, int]:
    """The logical screen's width and height (2-byte little-endian), enlarged to cover the
    first image's extent (its left and top offsets plus its width and height), as Pillow
    sizes a GIF: the global colour table and any extension blocks before the first image
    descriptor are skipped, and so is any other byte between blocks.

    Extensions are read as Pillow reads them: the label, then the first data sub-block; a
    comment (FE) then reads sub-blocks while they are not empty; a graphic control extension
    (F9) needs at least 3 bytes in its first sub-block (4 when it gives a transparent index);
    an application extension (FF) naming NETSCAPE2.0 reads one more sub-block; and then
    every extension but a comment reads sub-blocks up to an empty one. So an extension whose
    first sub-block is already the empty one reads one chain of sub-blocks further than the
    GIF specification does, as Pillow does."""
    if len(b) < 13:
        raise ValueError("truncated GIF header")
    sw, sh = struct.unpack("<HH", b[6:10])
    flags, n = b[10], len(b)
    pos = 13 + (3 << ((flags & 7) + 1) if flags & 0x80 else 0)

    def sub_block(p: int) -> tuple[bytes | None, int]:
        """Pillow's `GifImageFile.data`: a sub-block's bytes, or None for an empty one (or the
        file's end), and where the next starts."""
        if p < n and b[p]:
            return b[p + 1 : p + 1 + b[p]], p + 1 + b[p]
        return None, p + 1

    while True:
        found = _GIF_BLOCK.search(b, pos)
        if found is None or b[found.start()] == 0x3B:  # 0x3B: the trailer
            break
        c, pos = b[found.start()], found.start() + 1
        if c == 0x21:  # an extension
            if pos >= n:
                break
            label, pos = b[pos], pos + 1
            block, pos = sub_block(pos)
            if label == 0xF9 and block is not None:
                if len(block) < 3 or (block[0] & 1 and len(block) < 4):
                    raise ValueError("malformed GIF graphic control extension")
            elif label == 0xFE:
                while block:
                    block, pos = sub_block(pos)
                continue
            elif label == 0xFF and block is not None and block.startswith(b"NETSCAPE2.0"):
                block, pos = sub_block(pos)
            while True:
                block, pos = sub_block(pos)
                if not block:
                    break
        else:  # 0x2C: the first image descriptor, its colour table, its LZW code size
            local = b[pos + 8] if pos + 9 <= n else 0
            if pos + 9 + (3 << ((local & 7) + 1) if local & 0x80 else 0) >= n:
                raise ValueError("truncated GIF image descriptor")
            x0, y0, w, h = struct.unpack("<HHHH", b[pos : pos + 8])
            return max(sw, x0 + w), max(sh, y0 + h)
    raise ValueError("the GIF file holds no image")
