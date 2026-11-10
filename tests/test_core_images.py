"""Image input (docs/spec.md, The request and The state block): validation, header sizes, N.

The core reads an image's format and size from its file header, never its pixels, and
counts its soft tokens N as Gemma 4's image processors do. Without any imaging library,
these tests check the header reader on the checked-in test images of all four formats and
on malformed files made from them, the validation and v1's messages, and N against
tests/fixtures/core/image_soft_tokens.json, computed with transformers'
Gemma4ImageProcessor and Gemma4UnifiedImageProcessor. Where Pillow and transformers are
installed (and torchvision, for the 12B's processor and for Gemma4Processor itself), they
also check the header reader against Pillow, N against the processors on random sizes, the
rendered ids against Gemma4Processor and Gemma4UnifiedProcessor, and the validation
against v1's own `vision.decode_image`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import random
import struct
import zlib

import pytest
from core_support import (
    GEMMA4_PROCESSOR,
    IMAGE_FILES,
    PROCESSOR_SETTINGS,
    SOFT_TOKENS,
    UNIFIED_PROCESSOR,
    gemma4_tokenizer,
    gemma4_tokenizer_dir,
    image_b64,
    image_bytes,
    processor_soft_tokens,
)

from strands_decider.core import (
    IMAGE_FORMATS,
    MAX_IMAGE_PIXELS,
    VARIANTS,
    decode_image,
    image_size,
    parse_request,
    render,
    soft_tokens,
)
from strands_decider.core.images import (
    IMAGE_CLOSE,
    IMAGE_OPEN,
    IMAGE_SOFT,
    MAX_SOFT_TOKENS,
    PATCH_SIZE,
    POOLING_KERNEL_SIZE,
)

COULD_NOT_READ = "image could not be read as PNG/JPEG/WEBP/GIF: "


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---- headers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(IMAGE_FILES))
def test_the_test_images_headers(name):
    """Every format and chunk kind: PNG, baseline and progressive JPEG (one with an EXIF
    orientation), lossy, lossless and extended WebP, and GIF, one whose first frame lies
    beyond its logical screen."""
    data = image_bytes(name)
    assert image_size(data) == IMAGE_FILES[name]
    image = decode_image(image_b64(name))
    assert (image.format, image.width, image.height) == IMAGE_FILES[name]
    assert image.data == data and image.sha256 == hashlib.sha256(data).hexdigest()
    assert image.tokens == soft_tokens(image.width, image.height)
    assert decode_image(image_b64(name, data_uri=True)) == image


def _chunk(ctype: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + ctype + data + struct.pack(">I", zlib.crc32(ctype + data))


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_header(width: int, height: int) -> bytes:
    """A PNG's header with no pixel data: what the core reads, and nothing to decode."""
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    return PNG_SIGNATURE + ihdr + _chunk(b"IDAT", b"") + _chunk(b"IEND", b"")


def _malformed() -> dict[str, bytes]:
    """Files the core must refuse, made from the test images by editing their headers."""
    png = image_bytes("checker-64x48.png")
    jpeg = image_bytes("stripes-300x40.jpg")
    sof = jpeg.index(b"\xff\xc0")
    sof_end = sof + 2 + struct.unpack(">H", jpeg[sof + 2 : sof + 4])[0]
    vp8 = image_bytes("rings-90x120.webp")
    vp8l = image_bytes("lossless-33x17.webp")
    vp8x = image_bytes("alpha-21x34.webp")
    gif = image_bytes("dots-50x50.gif")
    gct = 13 + (3 << ((gif[10] & 7) + 1))
    desc = gif.index(b",\x00\x00\x00\x00")

    def edit(b: bytes, at: int, value: int) -> bytes:
        return b[:at] + bytes([value]) + b[at + 1 :]

    ihdr = png[16:29]
    zero_width = PNG_SIGNATURE + _chunk(b"IHDR", b"\x00\x00\x00\x00" + ihdr[4:]) + png[33:]
    return {
        "empty": b"",
        "text": b"hi",
        "bmp": b"BM" + bytes(60),
        "tiff": b"II*\x00" + bytes(60),
        "avif": b"\x00\x00\x00\x1cftypavif" + bytes(60),
        "svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>',
        "png-truncated-ihdr": png[:20],
        "png-bad-ihdr-crc": edit(png, 19, png[19] ^ 1),
        "png-no-idat": png[: png.index(b"IDAT") - 4],
        "png-no-ihdr": PNG_SIGNATURE + png[33:],
        "png-short-ihdr": PNG_SIGNATURE + _chunk(b"IHDR", ihdr[4:12]) + png[33:],
        "png-zero-width": zero_width,
        "png-bad-chunk-type": png[:33] + struct.pack(">I", 1) + b"I@AT" + png[41:],
        "jpeg-truncated": jpeg[:10],
        "jpeg-12-bit": edit(jpeg, sof + 4, 12),
        "jpeg-2-components": edit(jpeg, sof + 9, 2),
        "jpeg-no-frame-header": jpeg[:sof] + jpeg[sof_end:],
        "jpeg-zero-height": jpeg[: sof + 5] + b"\x00\x00" + jpeg[sof + 7 :],
        "jpeg-unknown-marker": b"\xff\xd8\xff\x01" + jpeg[2:],
        "webp-truncated": vp8[:-4],
        "webp-bad-start-code": edit(vp8, 23, 0x9C),
        "webp-not-a-key-frame": edit(vp8, 20, vp8[20] | 1),
        "webp-chunk-past-riff": vp8[:16] + struct.pack("<I", len(vp8)) + vp8[20:],
        "webp-lossless-version-1": edit(vp8l, 24, vp8l[24] | 0x20),
        "webp-lossless-bad-signature": edit(vp8l, 20, 0x2E),
        "webp-unknown-first-chunk": vp8x[:12] + b"ALPH" + vp8x[16:],
        "gif-truncated-screen": gif[:12],
        "gif-no-image": gif[:gct] + b";",
        "gif-truncated-descriptor": gif[: desc + 6],
        "gif-zero-size": gif[:6] + bytes(4) + gif[10 : desc + 5] + bytes(4) + gif[desc + 9 :],
    }


MALFORMED = _malformed()


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_and_unsupported_files_are_refused(case):
    with pytest.raises(ValueError) as e:
        decode_image(_b64(MALFORMED[case]))
    assert str(e.value).startswith(COULD_NOT_READ), str(e.value)
    with pytest.raises(ValueError):
        image_size(MALFORMED[case])


def test_header_reading_never_decodes_pixels():
    """Only the header is read: a PNG whose header is sound and whose pixel data is absent
    is sized, and so is a JPEG cut off right after its scan header. A PNG whose IHDR comes
    after another chunk is read as Pillow, and so v1, reads it."""
    assert image_size(_png_header(640, 480)) == ("PNG", 640, 480)
    png = image_bytes("checker-64x48.png")
    late_ihdr = PNG_SIGNATURE + _chunk(b"tEXt", b"k\x00v") + png[8:]
    assert image_size(late_ihdr) == ("PNG", 64, 48)
    jpeg = image_bytes("stripes-300x40.jpg")
    sos = jpeg.index(b"\xff\xda")
    assert image_size(jpeg[: sos + 2 + struct.unpack(">H", jpeg[sos + 2 : sos + 4])[0]]) == (
        "JPEG",
        300,
        40,
    )


# ---- validation and v1's messages ------------------------------------------------------------


def test_base64_is_standard_and_nothing_else():
    """Standard base64 with its padding, nothing else; surrounding whitespace is stripped first.
    The reasons are the core's own, the same on every Python version."""
    png = image_b64("checker-64x48.png")
    assert decode_image(f" \n{png}\t\n").data == image_bytes("checker-64x48.png")
    alphabet = "a character outside the standard base64 alphabet"
    padding = "its length or padding is wrong"
    cases = {
        png[:-1]: padding,
        png[:40] + "\n" + png[40:]: alphabet,
        png[:40] + " " + png[40:]: alphabet,
        png + "=": padding,
        png + png: padding,
        "=" + png[1:]: padding,
        "-_-_": alphabet,  # "+/+/" in the URL-safe alphabet
        png[:40] + "é" + png[40:]: "a character that is not ASCII",
    }
    for data, reason in cases.items():
        with pytest.raises(ValueError) as e:
            decode_image(data)
        assert str(e.value) == f"image is not valid base64: {reason}", data[-12:]
    with pytest.raises(ValueError) as e:
        decode_image("")
    assert str(e.value).startswith(COULD_NOT_READ)


def test_base64_rule_is_python_312s_and_never_wider_than_this_pythons():
    """The rule is RFC 4648's, which is what Python 3.12's `b64decode(validate=True)` accepts.
    On any version, whatever the core accepts this interpreter's strict decode accepts too, to
    the same bytes; on 3.12 the two accept exactly the same strings."""
    import binascii
    import sys

    from strands_decider.core.images import _strict_b64decode

    rng = random.Random(0)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    for _ in range(5000):
        s = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 14)))
        s += "=" * rng.choice((0, 0, 1, 2, 3))
        if rng.random() < 0.2:
            i = rng.randrange(len(s) + 1)
            s = s[:i] + rng.choice("=-_ \n.A") + s[i:]
        try:
            got: bytes | None = _strict_b64decode(s)
        except ValueError:
            got = None
        try:
            want: bytes | None = base64.b64decode(s, validate=True)
        except (binascii.Error, ValueError):
            want = None
        if got is not None:
            assert got == want, s
        if sys.version_info[:2] == (3, 12):
            assert (got is None) == (want is None), s


def test_data_uri_prefix_is_v1s():
    """`data:image/<subtype>;base64,` is stripped, whatever the subtype names; any other
    prefix is not, and is then not base64."""
    png = image_b64("checker-64x48.png")
    want = image_bytes("checker-64x48.png")
    for prefix in (
        "data:image/png;base64,",
        "data:image/jpeg;base64,",
        "data:image/svg+xml;base64,",
    ):
        assert decode_image(prefix + png).data == want
        assert decode_image(f"  {prefix}{png}\n").data == want
    for prefix in (
        "data:text/plain;base64,",
        "DATA:image/png;base64,",
        "data:image/png,",
        "data:;base64,",
    ):
        with pytest.raises(ValueError, match="image is not valid base64"):
            decode_image(prefix + png)


def test_the_pixel_limit_is_v1s():
    assert decode_image(_b64(_png_header(4096, 4096))).width == 4096
    assert decode_image(_b64(_png_header(1, MAX_IMAGE_PIXELS))).height == MAX_IMAGE_PIXELS
    for w, h in ((4097, 4096), (1, MAX_IMAGE_PIXELS + 1), (65535, 65535)):
        with pytest.raises(ValueError) as e:
            decode_image(_b64(_png_header(w, h)))
        assert str(e.value) == f"image is {w}x{h}; at most 16,777,216 pixels are accepted"


def test_more_than_four_images_are_refused_before_any_is_decoded():
    req = {"state": "", "questions": {"q": {"instructions": "x"}}}
    with pytest.raises(ValueError) as e:
        parse_request({**req, "images": ["not base64!"] * 5})
    assert str(e.value) == "5 images; this server takes at most 4"
    with pytest.raises(ValueError, match="image is not valid base64"):
        parse_request({**req, "images": ["not base64!"] * 4})
    parsed = parse_request({**req, "images": [image_b64(n) for n in list(IMAGE_FILES)[:4]]})
    assert [i.data for i in parsed.images] == [image_bytes(n) for n in list(IMAGE_FILES)[:4]]


def test_audio_is_refused_rather_than_ignored():
    req = {"state": "a call", "questions": {"q": {"instructions": "x"}}}
    with pytest.raises(ValueError) as e:
        parse_request({**req, "audio": ["UklGRg=="]})
    assert str(e.value) == "audio input is not supported"
    for empty in (None, []):
        assert parse_request({**req, "audio": empty}).images == ()


# ---- N ---------------------------------------------------------------------------------------


def _table() -> dict:
    return json.loads(SOFT_TOKENS.read_text(encoding="utf-8"))


def test_soft_tokens_match_the_processor_table():
    """N for every size of the table, as transformers' Gemma4ImageProcessor (E2B, E4B,
    26B-A4B, 31B) and Gemma4UnifiedImageProcessor (12B) gave it: both count alike."""
    t = _table()
    assert t["columns"] == ["width", "height", GEMMA4_PROCESSOR, UNIFIED_PROCESSOR]
    assert (
        t["settings"]
        == PROCESSOR_SETTINGS
        == {
            "patch_size": PATCH_SIZE,
            "max_soft_tokens": MAX_SOFT_TOKENS,
            "pooling_kernel_size": POOLING_KERNEL_SIZE,
        }
    )
    rows = t["rows"]
    assert len(rows) >= 3000
    for w, h, gemma4, unified in rows:
        assert soft_tokens(w, h) == gemma4 == unified, (w, h)
    sizes = {(w, h) for w, h, *_ in rows}
    # The extremes are in it: the pixel limit's thinnest images, and both sides of 280:1.
    assert {(1, MAX_IMAGE_PIXELS), (MAX_IMAGE_PIXELS, 1), (4096, 4096)} <= sizes
    assert {(279, 1), (280, 1), (281, 1), (1, 279), (1, 280), (1, 281)} <= sizes


def test_soft_token_properties():
    rng = random.Random(5)
    for _ in range(20000):
        w, h = rng.randint(1, 5000), rng.randint(1, 5000)
        n = soft_tokens(w, h)
        assert 1 <= n <= MAX_SOFT_TOKENS and n == soft_tokens(h, w), (w, h)
    for side in (1, 2, 3, 48, 100, 803, 804, 4096):
        assert soft_tokens(side, side) == 256  # a square resizes to 768 x 768: 16 x 16
    for w in (281, 1000, 13441, MAX_IMAGE_PIXELS):
        assert soft_tokens(w, 1) == soft_tokens(1, w) == MAX_SOFT_TOKENS  # one 48-pixel row
    assert soft_tokens(64, 48) == 266 and soft_tokens(300, 40) == 270
    assert soft_tokens(840, 768) == 272  # 645,120 pixels: a scale of exactly 1
    assert soft_tokens(100, 1) == 167 and soft_tokens(1, 32) == 188 and soft_tokens(1, 31) == 279
    for bad in ((0, 5), (5, 0), (-1, 3), (1.5, 2), (2, None)):
        with pytest.raises(ValueError):
            soft_tokens(*bad)  # type: ignore[arg-type]


def test_exif_orientation_leaves_n_unchanged():
    """v1 applies an image's EXIF orientation before its processor sees it, so the runtime's
    image may be the header's turned a quarter. N is the same both ways."""
    _, w, h = IMAGE_FILES["rotated-80x40.jpg"]
    assert (
        soft_tokens(w, h)
        == soft_tokens(h, w)
        == decode_image(image_b64("rotated-80x40.jpg")).tokens
    )
    pil = pytest.importorskip("PIL.Image")
    from PIL import ImageOps

    shown = ImageOps.exif_transpose(pil.open(io.BytesIO(image_bytes("rotated-80x40.jpg"))))
    assert shown.size == (h, w)


def _image_processor(which: str):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    transformers.logging.set_verbosity_error()
    if which == UNIFIED_PROCESSOR:
        pytest.importorskip("torchvision", reason="Gemma4UnifiedImageProcessor needs torchvision")
        return transformers.Gemma4UnifiedImageProcessor(**PROCESSOR_SETTINGS)
    return transformers.Gemma4ImageProcessorPil(**PROCESSOR_SETTINGS)


def _random_sizes(n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    out = [(1, 1), (1, 700), (700, 1), (3, 2000), (13441, 1), (4096, 4096)]
    while len(out) < n:
        w, h = rng.randint(1, 3000), rng.randint(1, 3000)
        out.append((w, h) if rng.random() < 0.8 else (w, rng.randint(1, 12)))
    return out


@pytest.mark.parametrize("which", [GEMMA4_PROCESSOR, UNIFIED_PROCESSOR])
def test_soft_tokens_match_the_image_processors(which):
    processor = _image_processor(which)
    for w, h in _random_sizes(80, seed=12):
        assert soft_tokens(w, h) == processor_soft_tokens(processor, w, h), (which, w, h)


# ---- against Pillow, Gemma4Processor and v1 ---------------------------------------------------


def _pillow_size(data: bytes):
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(data), formats=IMAGE_FORMATS)
        return (im.format, *im.size)
    except Exception:
        return None


def _core_size(data: bytes):
    try:
        return image_size(data)
    except ValueError:
        return None


def _pillow_images(n: int, seed: int) -> list[bytes]:
    from PIL import Image

    rng = random.Random(seed)
    saves = [
        ("PNG", "RGB", {}),
        ("PNG", "RGBA", {}),
        ("PNG", "L", {}),
        ("PNG", "P", {}),
        ("PNG", "1", {}),
        ("JPEG", "RGB", {"quality": 80}),
        ("JPEG", "L", {"progressive": True}),
        ("JPEG", "CMYK", {}),
        ("WEBP", "RGB", {"quality": 70}),
        ("WEBP", "RGBA", {"lossless": True}),
        ("WEBP", "RGBA", {"quality": 50}),
        ("GIF", "P", {}),
        ("GIF", "L", {}),
    ]
    out = []
    for _ in range(n):
        fmt, mode, kw = rng.choice(saves)
        size = (rng.choice([1, 2, 3, 17, 64, 255, 1000]), rng.choice([1, 3, 16, 99, 640]))
        buf = io.BytesIO()
        Image.new(mode, size).save(buf, format=fmt, **kw)
        out.append(buf.getvalue())
    return out


def test_headers_match_pillow():
    """Where both read a file, they read the same format and size; on files cut short they
    accept and refuse alike; and a file Pillow opens is never refused. (Pillow, through
    libwebp and its JPEG plugin, also checks some fields past the size, so it refuses some
    corrupted files the core sizes: the runtime refuses those when it opens them.)"""
    pytest.importorskip("PIL")
    files = [image_bytes(n) for n in IMAGE_FILES] + _pillow_images(150, seed=7)
    for data in files:
        assert _core_size(data) == _pillow_size(data) is not None
    rng = random.Random(8)
    for data in files[:: len(files) // 30]:
        for cut in [*range(min(len(data), 80)), len(data) // 2, len(data) - 1]:
            assert _core_size(data[:cut]) == _pillow_size(data[:cut]), (data[:16], cut)
        for _ in range(30):
            flipped = bytearray(data)
            flipped[rng.randrange(min(len(data), 64))] = rng.randrange(256)
            pillow = _pillow_size(bytes(flipped))
            if pillow is not None:
                assert _core_size(bytes(flipped)) == pillow
    for case, data in MALFORMED.items():
        if case != "png-zero-width":  # Pillow opens it, and fails when it decodes
            assert _pillow_size(data) is None, case


def test_validation_matches_v1():
    """The same requests accepted and refused as by v1's `vision.decode_image`, which opens
    and decodes the image with Pillow, with v1's messages: exactly for base64 and the pixel
    limit, and up to v1's prefix where v1 passes on Pillow's own words."""
    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    try:
        from strands_decider.vision import decode_image as v1_decode
    except ImportError as e:  # pragma: no cover - depends on the environment
        pytest.skip(f"v1's image engine does not import here: {e}")
    png = image_b64("checker-64x48.png")
    cases = [image_b64(n) for n in IMAGE_FILES] + [image_b64(n, data_uri=True) for n in IMAGE_FILES]
    cases += [png[:-1], png + "=", png[:40] + "\n" + png[40:], "", "aGk=", f" {png} "]
    cases += ["data:text/plain;base64," + png, "data:image/svg+xml;base64," + png]
    cases += [_b64(b) for b in _pillow_images(40, seed=9)]
    cases += [_b64(b) for b in MALFORMED.values()]
    cases += [_b64(_png_header(4097, 4096)), _b64(_png_header(1, MAX_IMAGE_PIXELS + 1))]
    for data in cases:
        try:
            v1_decode(data)
            v1_error = None
        except ValueError as e:
            v1_error = str(e)
        try:
            decode_image(data)
            error = None
        except ValueError as e:
            error = str(e)
        if v1_error is None:
            assert error is None, (data[:40], error)
        elif v1_error.startswith("image could not be decoded"):
            pass  # the header is sound and the pixels are not: the runtime finds that out
        elif v1_error.startswith(COULD_NOT_READ):
            assert error is not None and error.startswith(COULD_NOT_READ), (data[:40], error)
        else:
            assert error == v1_error, data[:40]


def _hf_processor(which: str):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision", reason="transformers builds Gemma 4's processors with it")
    transformers.logging.set_verbosity_error()
    tok = transformers.AutoTokenizer.from_pretrained(gemma4_tokenizer_dir())
    if which == "Gemma4Processor":
        return transformers.Gemma4Processor(
            feature_extractor=transformers.Gemma4AudioFeatureExtractor(),
            image_processor=transformers.Gemma4ImageProcessorPil(**PROCESSOR_SETTINGS),
            tokenizer=tok,
            video_processor=transformers.Gemma4VideoProcessor(),
            image_seq_length=280,
        )
    return transformers.Gemma4UnifiedProcessor(
        feature_extractor=transformers.Gemma4UnifiedAudioFeatureExtractor(
            audio_samples_per_token=640, feature_size=640
        ),
        image_processor=transformers.Gemma4UnifiedImageProcessor(**PROCESSOR_SETTINGS),
        tokenizer=tok,
        video_processor=transformers.Gemma4UnifiedVideoProcessor(),
        image_seq_length=280,
    )


@pytest.mark.parametrize("which", ["Gemma4Processor", "Gemma4UnifiedProcessor"])
def test_rendered_ids_match_the_gemma4_processor(which):
    """Random requests with one to four images of random sizes and formats: N is the
    image processor's `num_soft_tokens_per_image`, and the prompt's ids are what the processor
    gives for the same text with one `<|image|>` per image, which it expands to `<|image>`,
    N `<|image|>` and `<image|>` (with no special-token strings in the request text, which
    the processor would match as control tokens)."""
    proc = _hf_processor(which)
    from PIL import Image

    tok = gemma4_tokenizer()
    rng = random.Random(which)
    sizes = _random_sizes(40, seed=11)
    for k in range(10):
        files, pixels = [], []
        for _ in range(rng.randint(1, 4)):
            w, h = rng.choice(sizes)
            buf = io.BytesIO()
            fmt = rng.choice(["PNG", "JPEG", "WEBP", "GIF"])
            Image.new("RGB", (w, h), (rng.randrange(256), 90, 160)).save(buf, format=fmt)
            files.append(_b64(buf.getvalue()))
            pixels.append(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        request = {
            "state": rng.choice(["", "Checkout page after the user tapped Pay.", {"step": k}]),
            "images": files,
            "questions": {
                "paid": {"type": "noul", "instructions": "Did the payment succeed?"},
                "next": {
                    "type": "choice",
                    "instructions": "Next?",
                    "criteria": {"a": None, "b": "x"},
                },
            },
        }
        for variant in VARIANTS:
            r = render(request, tok, variant)
            for q in r.questions.values():
                text = q.text
                for image in r.images:
                    block = IMAGE_OPEN + IMAGE_SOFT * image.tokens + IMAGE_CLOSE
                    assert block in text
                    text = text.replace(block, IMAGE_SOFT, 1)
                out = proc(text=[text], images=[pixels], add_special_tokens=False)
                counts = proc.image_processor(images=pixels)["num_soft_tokens_per_image"]
                assert [int(n) for n in counts] == [i.tokens for i in r.images]
                assert list(out["input_ids"][0]) == list(q.ids), (which, variant, k)
