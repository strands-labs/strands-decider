"""Image input (docs/spec.md, sections 1.1 and 3.3): the checks, the size and N.

Pillow reads an image's format and size from its file header, never its pixels, and the core
counts its soft tokens N as Gemma 4's image processors do. These tests check the checks and
v1's messages on the checked-in test images of all four formats and on files made from them,
and N against tests/fixtures/core/image_soft_tokens.json, computed with transformers'
Gemma4ImageProcessor and Gemma4UnifiedImageProcessor. Where transformers is installed (and
torchvision, for the 12B's processor and for Gemma4Processor itself), they also check N
against the processors on random sizes, the rendered ids against Gemma4Processor and
Gemma4UnifiedProcessor, and the checks against v1's own `vision.decode_image`.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import random
import struct
import zlib

import pytest
from core_support import (
    GEMMA4_PROCESSOR,
    IMAGE_FILES,
    PIXEL_FILES,
    PIXELS,
    PIXELS_DIR,
    PIXELS_PILLOW,
    PROCESSOR_SETTINGS,
    SOFT_TOKENS,
    UNIFIED_PROCESSOR,
    gemma4_tokenizer,
    gemma4_tokenizer_dir,
    image_b64,
    image_bytes,
    knife_edge_sizes,
    processor_soft_tokens,
    read_rgb_png,
    reference_pixels,
    reference_resize,
    spec_resize,
)

from strands_decider.core import (
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    PLAIN,
    VARIANTS,
    check_decoded_size,
    decode_image,
    parse_request,
    render,
    render_text,
    resized_size,
    soft_tokens,
)
from strands_decider.core.images import (
    IMAGE_CLOSE,
    IMAGE_OPEN,
    IMAGE_SOFT,
    MAX_SOFT_TOKENS,
    PATCH_SIZE,
    POOLING_KERNEL_SIZE,
    decode_images,
)

COULD_NOT_READ = "image could not be read as PNG/JPEG/WEBP/GIF: "
NOT_BASE64 = "image is not valid base64: "


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---- headers ---------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(PIXEL_FILES))
def test_the_test_images_headers(name):
    """Every format and chunk kind: PNG, baseline and progressive JPEG (one with an EXIF
    orientation), lossy, lossless and extended WebP, and GIF, one whose first frame lies
    beyond its logical screen."""
    data = image_bytes(name)
    image = decode_image(image_b64(name))
    assert (image.format, image.width, image.height) == PIXEL_FILES[name]
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
    """Files the core must refuse: other formats' signatures, and test images with their
    headers cut or edited."""
    png = image_bytes("checker-64x48.png")
    jpeg = image_bytes("stripes-300x40.jpg")
    vp8 = image_bytes("rings-90x120.webp")
    gif = image_bytes("dots-50x50.gif")
    gct = 13 + (3 << ((gif[10] & 7) + 1))
    ihdr = png[16:29]
    return {
        "empty": b"",
        "text": b"hi",
        "bmp": b"BM" + bytes(60),
        "tiff": b"II*\x00" + bytes(60),
        "avif": b"\x00\x00\x00\x1cftypavif" + bytes(60),
        "svg": b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"/>',
        "png-truncated-ihdr": png[:20],
        "png-bad-ihdr-crc": png[:19] + bytes([png[19] ^ 1]) + png[20:],
        "png-zero-width": PNG_SIGNATURE + _chunk(b"IHDR", bytes(4) + ihdr[4:]) + png[33:],
        "png-zero-height": PNG_SIGNATURE
        + _chunk(b"IHDR", ihdr[:4] + bytes(4) + ihdr[8:])
        + png[33:],
        "jpeg-truncated": jpeg[:10],
        "webp-truncated": vp8[:-4],
        "gif-truncated-screen": gif[:12],
        "gif-no-image": gif[:gct] + b";",
    }


MALFORMED = _malformed()


@pytest.mark.parametrize("case", list(MALFORMED))
def test_malformed_and_unsupported_files_are_refused(case):
    with pytest.raises(ValueError) as e:
        decode_image(_b64(MALFORMED[case]))
    assert str(e.value).startswith(COULD_NOT_READ), str(e.value)


def test_only_the_four_formats_are_opened():
    """Pillow opens only PNG, JPEG, WebP and GIF here, as v1 opens them: a sound file of
    another format it reads is refused. A JPEG that carries more pictures (MPO) is a JPEG."""
    from PIL import Image

    for fmt in ("BMP", "TIFF", "ICO", "PPM", "TGA"):
        buf = io.BytesIO()
        Image.new("RGB", (8, 6), (1, 2, 3)).save(buf, format=fmt)
        with pytest.raises(ValueError) as e:
            decode_image(_b64(buf.getvalue()))
        assert str(e.value).startswith(COULD_NOT_READ), fmt
    buf = io.BytesIO()
    Image.new("RGB", (40, 30)).save(
        buf, "MPO", save_all=True, append_images=[Image.new("RGB", (9, 9))]
    )
    image = decode_image(_b64(buf.getvalue()))
    assert (image.format, image.width, image.height) == ("JPEG", 40, 30)


def test_header_reading_never_decodes_pixels():
    """Only the header is read: a PNG whose header is sound and whose pixel data is absent is
    sized, and so is a JPEG cut off right after its scan header."""
    assert decode_image(_b64(_png_header(640, 480))).width == 640
    jpeg = image_bytes("stripes-300x40.jpg")
    sos = jpeg.index(b"\xff\xda")
    cut = jpeg[: sos + 2 + struct.unpack(">H", jpeg[sos + 2 : sos + 4])[0]]
    assert (decode_image(_b64(cut)).width, decode_image(_b64(cut)).height) == (300, 40)


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


def _rfc4648(s: str) -> bytes | None:
    """RFC 4648 section 4, decoded bit by bit: an independent statement of the spec's rule."""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    if len(s) % 4:
        return None
    body = s.rstrip("=")
    if len(s) - len(body) > 2 or any(c not in alphabet for c in body):
        return None
    if len(body) % 4 == 1:
        return None
    bits = "".join(f"{alphabet.index(c):06b}" for c in body)
    return bytes(int(bits[i : i + 8], 2) for i in range(0, len(bits) - len(bits) % 8, 8))


def test_base64_is_the_rfcs_on_every_python():
    """The core's base64 rule is RFC 4648's on every Python version, checked against an
    implementation of the RFC that uses neither base64 nor binascii."""
    from strands_decider.core.images import _strict_b64decode

    rng = random.Random(1)
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
        assert got == _rfc4648(s), s


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
        "data:image/\u00f1;base64,",  # the subtype is ASCII: [A-Za-z0-9_.+-]
        "data:image/\u0660;base64,",  # an Arabic-Indic digit, which Python's \\w matches
    ):
        with pytest.raises(ValueError, match="image is not valid base64"):
            decode_image(prefix + png)


def test_surrounding_whitespace_is_stripped():
    """The string is stripped of whitespace as `render` strips request text, Unicode's
    included; U+FEFF is not whitespace, so a byte order mark makes it invalid base64."""
    png = image_b64("checker-64x48.png")
    want = image_bytes("checker-64x48.png")
    assert decode_image(f"\u00a0\u2028 {png}\u3000\x1f\n").data == want
    with pytest.raises(ValueError, match="not ASCII"):
        decode_image("\ufeff" + png)


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_the_pixel_limit_is_v1s():
    assert decode_image(_b64(_png_header(4096, 4096))).width == 4096
    assert decode_image(_b64(_png_header(1, MAX_IMAGE_PIXELS))).height == MAX_IMAGE_PIXELS
    for w, h in ((4097, 4096), (1, MAX_IMAGE_PIXELS + 1), (9500, 9500), (13377, 13377)):
        with pytest.raises(ValueError) as e:
            decode_image(_b64(_png_header(w, h)))
        assert str(e.value) == f"image is {w}x{h}; at most 16,777,216 pixels are accepted"
    # Past 178,956,970 pixels Pillow will not open the file, so v1 refuses it as unreadable.
    for w, h in ((13378, 13378), (65535, 65535)):
        with pytest.raises(ValueError) as e:
            decode_image(_b64(_png_header(w, h)))
        assert str(e.value).startswith(COULD_NOT_READ)


def _png_of(size: int) -> bytes:
    """A 2 x 2 PNG header padded with an ancillary chunk to `size` bytes in all."""
    head = _png_header(2, 2)
    filler = _chunk(b"teXt", bytes(size - len(head) - 12))
    return head[:33] + filler + head[33:]


def test_the_byte_cap():
    """At most 16 MiB of file per image, checked once the base64 is read and before Pillow
    opens the file, so a file too large is refused as too large whatever it holds."""
    assert MAX_IMAGE_BYTES == 16 * 2**20
    assert decode_image(_b64(_png_of(MAX_IMAGE_BYTES))).width == 2
    for data in (_png_of(MAX_IMAGE_BYTES + 1), bytes(MAX_IMAGE_BYTES + 3)):
        with pytest.raises(ValueError) as e:
            decode_image(_b64(data))
        assert str(e.value) == f"image is {len(data)} bytes; at most 16,777,216 bytes are accepted"
    with pytest.raises(ValueError, match="image is not valid base64"):
        decode_image("!" + _b64(bytes(MAX_IMAGE_BYTES + 3)))  # base64 is checked first
    assert decode_image(_b64(_png_of(100)), max_bytes=100).width == 2
    with pytest.raises(ValueError, match="image is 101 bytes; at most 100 bytes"):
        decode_image(_b64(_png_of(101)), max_bytes=100)


def test_images_need_pillow_and_the_core_does_not(monkeypatch):
    """Without Pillow, an image is refused with an ImportError saying what to install (the
    server's fault, not the client's), and requests without images render as before."""
    import sys

    monkeypatch.setitem(sys.modules, "PIL", None)
    with pytest.raises(ImportError, match=r"image input needs Pillow: pip install"):
        decode_image(image_b64("checker-64x48.png"))
    req = {"state": "", "questions": {"q": {"instructions": "x"}}}
    with pytest.raises(ImportError, match="needs Pillow"):
        render_text({**req, "images": [_b64(b"x")]}, PLAIN)
    assert render_text(req, PLAIN)


def test_each_image_is_checked_in_turn_after_everything_else():
    """The request's own checks come first, then the count, then every check of the first
    image before any of the second's. `parse_request` checks the request and leaves the
    images as they came."""
    req = {"state": "", "questions": {"q": {"instructions": "x"}}}
    big, bad = _b64(_png_header(4097, 4096)), "not base64!"
    with pytest.raises(ValueError, match="image is 4097x4096"):
        render_text({**req, "images": [big, bad]}, PLAIN)
    with pytest.raises(ValueError, match="image is not valid base64"):
        render_text({**req, "images": [bad, big]}, PLAIN)
    bad_question = {"state": "", "questions": {"q": {"type": "noul"}}, "images": [bad] * 5}
    with pytest.raises(ValueError, match=r"^questions\.q\.noul\.instructions: Field required$"):
        render_text(bad_question, PLAIN)
    assert parse_request({**req, "images": [bad] * 5}).images == [bad] * 5


def test_the_image_processors_settings_are_checked(tmp_path):
    from strands_decider.core import check_image_processor

    good = {"image_processor": {"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280}}
    check_image_processor(good)
    check_image_processor(good["image_processor"])
    (tmp_path / "processor_config.json").write_text(json.dumps(good))
    check_image_processor(tmp_path)
    check_image_processor(tmp_path / "processor_config.json")
    for k, v in (("patch_size", 14), ("max_soft_tokens", 560), ("pooling_kernel_size", True)):
        with pytest.raises(ValueError, match="cannot take images"):
            check_image_processor({"image_processor": {**good["image_processor"], k: v}})
    with pytest.raises(ValueError, match="<missing>"):
        check_image_processor({"image_processor": {"patch_size": 16}})
    # image_seq_length, where given (E2B's both places, the 12B's only at the top), is 280.
    check_image_processor({**good, "image_seq_length": 280})
    for where in ("top", "image_processor"):
        bad = {**good, "image_processor": dict(good["image_processor"])}
        (bad if where == "top" else bad["image_processor"])["image_seq_length"] = 256
        with pytest.raises(ValueError, match="image_seq_length"):
            check_image_processor(bad)


def test_more_than_four_images_are_refused_before_any_is_decoded():
    with pytest.raises(ValueError) as e:
        decode_images(["not base64!"] * 5)
    assert str(e.value) == "5 images; this server takes at most 4"
    with pytest.raises(ValueError, match="image is not valid base64"):
        decode_images(["not base64!"] * 4)
    images = decode_images([image_b64(n) for n in list(IMAGE_FILES)[:4]])
    assert [i.data for i in images] == [image_bytes(n) for n in list(IMAGE_FILES)[:4]]


def test_audio_is_refused_rather_than_ignored():
    """Any `audio` but null is refused, an empty one included; null is as if it were absent."""
    req = {"state": "a call", "questions": {"q": {"instructions": "x"}}}
    for given in (["UklGRg=="], "UklGRg==", {"data": "x"}, True, 1, 0.5, [None], [], False, 0, ""):
        with pytest.raises(ValueError) as e:
            parse_request({**req, "audio": given})
        assert str(e.value) == "audio input is not supported", given
    assert parse_request({**req, "audio": None}) == parse_request(req)


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
    # And the knife edges: ratios where f h / 48 is an integer in exact arithmetic, so the
    # floor is decided by the double's last bit, at many scales and both ways round.
    edges = knife_edge_sizes()
    assert len(edges) > 600 and edges <= sizes
    assert {(1000, 700), (700, 1000), (385, 22), (22, 385), (350, 320), (1400, 320)} <= sizes
    # 35:8, both quotients integers (8 and 35); 280:9 and 56:45, one of them.
    assert {(320, 1400), (2800, 90), (560, 450)} <= sizes
    by_size = {(w, h): n for w, h, n, _ in rows}
    assert by_size[(385, 22)] == by_size[(22, 385)] == 276  # 280 in exact arithmetic


def test_the_table_catches_formulas_that_round_otherwise():
    """Ports that write the arithmetic another way, all equal in exact arithmetic, miscount
    some knife-edge sizes of the table: the order of the operations is the rule."""
    rows = _table()["rows"]

    def count(w, h, f, scale):
        th, tw = math.floor(scale(f, h)) * 48, math.floor(scale(f, w)) * 48
        return None if th == 0 or tw == 0 else (th // 48) * (tw // 48)

    exact = [(r[0], r[1], r[2]) for r in rows]
    variants = {
        "f * (h / 48)": lambda w, h: count(
            w, h, math.sqrt(645120 / (h * w)), lambda f, x: f * (x / 48)
        ),
        "sqrt(645120) / sqrt(h w)": lambda w, h: count(
            w, h, math.sqrt(645120) / math.sqrt(h * w), lambda f, x: f * x / 48
        ),
        "exact": lambda w, h: count(
            w, h, 0, lambda f, x: math.isqrt(645120 * x * x // (w * h)) / 48
        ),
    }
    edges = knife_edge_sizes()
    for name, n_of in variants.items():
        wrong = [(w, h) for w, h, n in exact if (got := n_of(w, h)) is not None and got != n]
        assert wrong and set(wrong) & edges, name


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


# ---- pixels: the decode, the resize target and the resize --------------------------------


def _pixel_goldens() -> dict:
    return json.loads(PIXELS.read_text(encoding="utf-8"))


def test_the_resized_size_comes_from_the_decoded_size():
    """W and H are the decoded, oriented image's, never the header's: the rotated JPEG's
    header says 80 x 40 and its pixels are 40 x 80, so it is resized to 528 x 1104, where the
    header's shape would give 1104 x 528 with the same N."""
    goldens = _pixel_goldens()
    assert goldens["pillow"] == PIXELS_PILLOW and set(goldens["images"]) == set(PIXEL_FILES)
    for name, g in goldens["images"].items():
        image = decode_image(image_b64(name))
        decoded, resized = tuple(g["decoded"]), tuple(g["resized"])
        assert decoded in ((image.width, image.height), (image.height, image.width)), name
        assert resized == resized_size(*decoded) == check_decoded_size(image, *decoded), name
        assert soft_tokens(*decoded) == image.tokens == (resized[0] // 48) * (resized[1] // 48)
    rotated = goldens["images"]["rotated-80x40.jpg"]
    assert rotated["decoded"] == [40, 80] and rotated["resized"] == [528, 1104]
    assert resized_size(80, 40) == (1104, 528) and soft_tokens(80, 40) == soft_tokens(40, 80) == 253
    image = decode_image(image_b64("rotated-80x40.jpg"))
    assert check_decoded_size(image, 80, 40) == (1104, 528)  # as the header says, unturned
    for w, h in ((41, 80), (40, 81), (80, 80), (1, 1)):
        with pytest.raises(ValueError) as e:
            check_decoded_size(image, w, h)
        assert (
            str(e.value)
            == f"image could not be decoded: its pixels are {w}x{h}, and its header gives 80x40"
        )


# Formats whose decoding is exact arithmetic: PNG and GIF, and lossless WebP. JPEG and lossy
# WebP decoders may differ from libjpeg-turbo's and libwebp's, within the spec's tolerance.
EXACT_DECODES = {"checker-64x48.png", "dots-50x50.gif", "lossless-33x17.webp", "thin-1x300.png",
                 "offset-frame.gif", "tall-1080x2400.png", "wide-1600x1000.png"}  # fmt: skip


def test_the_pixel_goldens_hold_without_pillow():
    """The procedure a runtime without Pillow follows: the goldens' decoded pixels, read from
    tests/fixtures/core/pixels with zlib, have the goldens' hash, and the spec's resize of
    them, here the numpy port, has the resized hash. The downscales are among them."""
    np = pytest.importorskip("numpy")
    goldens = _pixel_goldens()["images"]
    assert any(g["resized"][0] < g["decoded"][0] for g in goldens.values())  # downscales
    assert goldens["tall-1080x2400.png"]["resized"] == [528, 1152]
    for name, g in goldens.items():
        decoded = read_rgb_png(PIXELS_DIR / f"{name}.png")
        assert [decoded.shape[1], decoded.shape[0]] == g["decoded"], name
        assert hashlib.sha256(decoded.tobytes()).hexdigest() == g["decoded_sha256"], name
        resized = spec_resize(np.ascontiguousarray(decoded), *g["resized"])
        assert hashlib.sha256(resized.tobytes()).hexdigest() == g["resized_sha256"], name


def test_the_reference_decode_and_resize_give_the_pixel_goldens():
    """v1's decode gives the goldens' decoded pixels, exactly where the format's decoding is
    exact and within 1 per channel otherwise, as the spec's tolerance allows a decoder; and
    resize_image of the goldens' decoded pixels gives the resized hash exactly, as does the
    numpy port of the spec's resize."""
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    from PIL import Image

    for name, g in _pixel_goldens()["images"].items():
        reference = read_rgb_png(PIXELS_DIR / f"{name}.png")
        decoded = reference_pixels(image_bytes(name))
        assert list(decoded.size) == g["decoded"], name
        got = np.asarray(decoded)
        if name in EXACT_DECODES:
            assert np.array_equal(got, reference), name
        else:
            assert np.abs(got.astype(int) - reference).max() <= 1, name
        resized = reference_resize(Image.fromarray(np.ascontiguousarray(reference)), *g["resized"])
        assert hashlib.sha256(resized.tobytes()).hexdigest() == g["resized_sha256"], name
        ours = spec_resize(np.ascontiguousarray(reference), *g["resized"])
        assert ours.tobytes() == resized.tobytes(), name


def test_the_specs_resize_is_pillows_bicubic():
    """The spec's resize (cubic convolution, a = -0.5, its support scaled by the downscale
    factor, fixed-point weights, horizontal then vertical with an 8-bit result between them)
    is Pillow's BICUBIC, byte for byte, up and down, in one dimension or both: in one call
    before Pillow 13, and in two, width then height, on any version."""
    PIL = pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    from PIL import Image

    one_call = int(PIL.__version__.split(".")[0]) < 13

    rng = random.Random(17)
    sizes = [
        ((1, 1), (48, 48)),
        ((300, 1), (48, 48)),
        ((97, 31), (97, 13)),
        ((5, 200), (48, 13440)),
    ]
    sizes += [
        ((rng.randint(1, 160), rng.randint(1, 160)), (rng.randint(1, 400), rng.randint(1, 400)))
        for _ in range(40)
    ]
    for i, ((w, h), (tw, th)) in enumerate(sizes):
        a = np.random.default_rng(i).integers(0, 256, (h, w, 3), dtype=np.uint8)
        want = np.asarray(reference_resize(Image.fromarray(a), tw, th))
        assert np.array_equal(spec_resize(a, tw, th), want), ((w, h), (tw, th))
        if one_call:
            single = Image.fromarray(a).resize((tw, th), Image.Resampling.BICUBIC)
            assert np.array_equal(np.asarray(single), want), ((w, h), (tw, th))


def test_pillow_13_resizes_a_tall_image_height_first():
    """Why the reference resizes in two calls: from Pillow 13, one call runs the height's pass
    first when the height shrinks by more than twice as much as the width, as it does for a
    1,080 x 2,400 screenshot, and its pixels differ from the spec's by several levels."""
    PIL = pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    from PIL import Image

    w, h = 1080, 2400
    tw, th = resized_size(w, h)
    yy, xx = np.mgrid[0:h, 0:w]
    a = np.stack([(xx * 7 + yy * 3) % 256, (xx ^ yy) % 256, (xx // 9 + yy // 13) % 2 * 255], -1)
    img = Image.fromarray(a.astype(np.uint8))
    single = np.asarray(img.resize((tw, th), Image.Resampling.BICUBIC)).astype(int)
    reference = np.asarray(reference_resize(img, tw, th)).astype(int)
    assert np.array_equal(reference, spec_resize(np.asarray(img), tw, th))
    if int(PIL.__version__.split(".")[0]) < 13:
        assert np.array_equal(single, reference)
    # From Pillow 13 the single call may differ (by 12 levels here on Pillow 13.0); nothing
    # is asserted of it, since the spec's resize, resize_image, does not depend on it.


def test_resize_image_is_the_specs_resize_on_any_pillow():
    """The core's resize_image, which every runtime may share, is the spec's resize byte for
    byte, on the pixel goldens and on images shrunk and enlarged either way, whatever the
    Pillow version (its two single-axis calls do not depend on Pillow's pass order)."""
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    from PIL import Image

    from strands_decider.core import resize_image

    for name, g in _pixel_goldens()["images"].items():
        out = resize_image(reference_pixels(image_bytes(name)), *g["resized"])
        assert list(out.size) == g["resized"], name
        assert hashlib.sha256(out.tobytes()).hexdigest() == g["resized_sha256"], name
    rng = np.random.default_rng(23)
    for (w, h), (tw, th) in [
        ((1080, 2400), resized_size(1080, 2400)),  # a tall screenshot, height first on Pillow 13
        ((2400, 1080), resized_size(2400, 1080)),
        ((600, 2000), resized_size(600, 2000)),
        ((90, 7), (48, 48)),
        ((31, 400), (64, 5)),
        ((17, 17), (17, 17)),
    ]:
        a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        out = resize_image(Image.fromarray(a), tw, th)
        assert np.array_equal(np.asarray(out), spec_resize(a, tw, th)), ((w, h), (tw, th))
    for bad in ((0, 5), (5, 0), (1.5, 2)):
        with pytest.raises(ValueError):
            resize_image(Image.new("RGB", (4, 4)), *bad)  # type: ignore[arg-type]
    for mode in (
        "P",
        "1",
        "L",
        "RGBA",
    ):  # decoded images are RGB; Pillow would resize these otherwise
        with pytest.raises(ValueError, match="decoded RGB image"):
            resize_image(Image.new(mode, (4, 4)), 8, 8)


def _png(
    width: int, height: int, depth: int, colour: int, rows: list[bytes], extra: bytes = b""
) -> bytes:
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, colour, 0, 0, 0))
    data = zlib.compress(b"".join(b"\x00" + r for r in rows))
    return PNG_SIGNATURE + ihdr + extra + _chunk(b"IDAT", data) + _chunk(b"IEND", b"")


def _on_white(c, a):
    """Pillow's alpha_composite of a channel c with alpha a onto white, in integers."""
    y = 128 * (c * a + 255 * (255 - a) + 128)
    return (((y >> 8) + y) >> 15) if a else 255


def test_v1_decodes_by_these_rules():
    """The decode cases a runtime's decoder must match, as v1's decode gives them with Pillow
    and the pixel goldens pin them (docs/spec.md, sections 3.3 and 10): bit depths, 16-bit
    samples, transparency (composited onto white, and where it is not), CMYK, and a GIF's
    canvas."""
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    import io

    from PIL import Image

    def rgb(data: bytes):
        return np.asarray(reference_pixels(data)).astype(int)

    vals = [0, 1, 255, 256, 1000, 32768, 65535]
    grey16 = _png(len(vals), 1, 16, 0, [b"".join(struct.pack(">H", v) for v in vals)])
    assert rgb(grey16)[0, :, 0].tolist() == [min(v, 255) for v in vals]  # clipped, not scaled
    rgb16 = _png(len(vals), 1, 16, 2, [b"".join(struct.pack(">HHH", v, v, 0) for v in vals)])
    assert rgb(rgb16)[0, :, 0].tolist() == [v >> 8 for v in vals]  # the high byte
    rgba16 = _png(
        len(vals), 1, 16, 6, [b"".join(struct.pack(">HHHH", v, 0, 0, 0x80FF) for v in vals)]
    )
    assert rgb(rgba16)[0, :, 0].tolist() == [_on_white(v >> 8, 0x80) for v in vals]
    assert rgb(_png(8, 1, 1, 0, [bytes([0b10100000])]))[0, :, 0].tolist() == [
        255,
        0,
        255,
        0,
        0,
        0,
        0,
        0,
    ]
    assert rgb(_png(4, 1, 2, 0, [bytes([0b00011011])]))[0, :, 0].tolist() == [0, 85, 170, 255]
    assert rgb(_png(2, 1, 4, 0, [bytes([0x1E])]))[0, :, 0].tolist() == [17, 238]
    # Transparency: a palette's tRNS and grey-and-alpha are composited onto white...
    plte = _chunk(b"PLTE", bytes([255, 0, 0, 0, 255, 0, 0, 0, 255]))
    trns = _chunk(b"tRNS", bytes([0, 128]))  # index 2 has no entry: opaque
    pal = rgb(_png(3, 1, 8, 3, [bytes([0, 1, 2])], plte + trns))[0].tolist()
    assert pal == [[255] * 3, [_on_white(0, 128), 255, _on_white(0, 128)], [0, 0, 255]]
    la = rgb(_png(3, 1, 8, 4, [bytes([100, 0, 100, 128, 100, 255])]))[0, :, 0].tolist()
    assert la == [_on_white(100, a) for a in (0, 128, 255)]
    # ...and a greyscale or RGB image's tRNS is not.
    grey_trns = _png(2, 1, 8, 0, [bytes([0, 128])], _chunk(b"tRNS", struct.pack(">H", 128)))
    assert rgb(grey_trns)[0, :, 0].tolist() == [0, 128]
    rgb_trns = _png(
        1, 1, 8, 2, [bytes([10, 20, 30])], _chunk(b"tRNS", struct.pack(">HHH", 10, 20, 30))
    )
    assert rgb(rgb_trns)[0, 0].tolist() == [10, 20, 30]
    # The compositing formula is Pillow's alpha_composite onto white, for every channel and alpha.
    c, a = np.meshgrid(np.arange(256), np.arange(256))
    rgba = np.stack([c, c, c, a], axis=-1).astype(np.uint8)
    white = Image.new("RGBA", (256, 256), (255, 255, 255, 255))
    got = np.asarray(Image.alpha_composite(white, Image.fromarray(rgba, "RGBA")).convert("RGB"))[
        ..., 0
    ]
    want = np.vectorize(_on_white)(c, a)
    assert np.array_equal(got, want) and np.max(np.abs(want - (c * a + 255 * (255 - a)) / 255)) <= 1
    # CMYK: Pillow reads a four-component JPEG as inverted CMYK, then each channel is
    # (255 - K) - m(C, 255 - K) with m(x, y) = ((t >> 8) + t) >> 8, t = x y + 128.
    for cmyk in [
        (0, 0, 0, 0),
        (255, 0, 0, 0),
        (128, 64, 32, 16),
        (200, 100, 50, 128),
        (3, 250, 77, 255),
    ]:
        buf = io.BytesIO()
        Image.new("CMYK", (16, 16), cmyk).save(buf, "JPEG", quality=100)
        raw = Image.open(io.BytesIO(buf.getvalue()))
        raw.load()
        cc, mm, yy, kk = (int(v) for v in raw.getpixel((5, 5)))
        nk = 255 - kk

        def m(x, y):
            t = x * y + 128
            return ((t >> 8) + t) >> 8

        assert rgb(buf.getvalue())[5, 5].tolist() == [
            nk - m(cc, nk),
            nk - m(mm, nk),
            nk - m(yy, nk),
        ]
    # A GIF's first frame on its canvas: outside the frame, palette index 0 ...
    gif = image_bytes("offset-frame.gif")
    frame = Image.open(io.BytesIO(gif))
    palette = frame.getpalette()
    assert rgb(gif)[0, 0].tolist() == palette[:3] and rgb(gif)[35, 24].tolist() == palette[:3]


def test_the_processor_with_its_resize_off_takes_the_resized_pixels():
    """Gemma4ImageProcessorPil's own resize is Pillow's bicubic to (W, H): it gives the same
    pixel values as resizing first and calling it with do_resize=False, as docs/spec.md
    section 3.3 has the runtime do."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    np = pytest.importorskip("numpy")
    from PIL import Image

    transformers.logging.set_verbosity_error()
    proc = transformers.Gemma4ImageProcessorPil(**PROCESSOR_SETTINGS)
    for name, g in _pixel_goldens()["images"].items():
        decoded = reference_pixels(image_bytes(name))
        own = proc(images=[decoded], return_tensors="np")
        resized = decoded.resize(tuple(g["resized"]), Image.Resampling.BICUBIC)
        ours = proc(images=[resized], do_resize=False, return_tensors="np")
        assert np.array_equal(own["pixel_values"], ours["pixel_values"]), name
        assert int(own["num_soft_tokens_per_image"][0]) == soft_tokens(*g["decoded"]), name


def test_the_12b_processor_takes_the_resized_pixels():
    """The 12B's processor, which resizes with torchvision, takes the Pillow-resized pixels with
    do_resize=False and counts N."""
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("PIL")
    pytest.importorskip("torchvision", reason="Gemma4UnifiedImageProcessor needs torchvision")
    transformers.logging.set_verbosity_error()
    unified = transformers.Gemma4UnifiedImageProcessor(**PROCESSOR_SETTINGS)
    for name, g in _pixel_goldens()["images"].items():
        resized = reference_resize(reference_pixels(image_bytes(name)), *g["resized"])
        out = unified(images=[resized], do_resize=False, return_tensors="pt")
        assert int(out["num_soft_tokens_per_image"][0]) == soft_tokens(*g["decoded"]), name


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


# ---- against Gemma4Processor and v1 -----------------------------------------------------------


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


def test_validation_matches_v1():
    """The same requests accepted and refused as by v1's `vision.decode_image`, which opens
    and decodes the image with Pillow, with v1's messages: exactly for the pixel limit, and
    up to v1's prefix where Pillow gives the reason.

    Base64 is the spec's rule (RFC 4648), which is what Python 3.12's strict decode accepts;
    Python before 3.12 accepts more. So for base64 the core is held to the rule, refuses
    wherever v1 refuses, and matches v1's acceptance exactly only on 3.12. Every later step
    is compared exactly, whenever both sides got past base64."""
    import sys

    from strands_decider.core.images import DATA_URI

    pytest.importorskip("PIL")
    pytest.importorskip("torch")
    try:
        from strands_decider.vision import decode_image as v1_decode
    except ImportError as e:  # pragma: no cover - depends on the environment
        pytest.skip(f"v1's image engine does not import here: {e}")
    png = image_b64("checker-64x48.png")
    cases = [image_b64(n) for n in IMAGE_FILES] + [image_b64(n, data_uri=True) for n in IMAGE_FILES]
    cases += [png[:-1], png + "=", png[:40] + "\n" + png[40:], "", "aGk=", f" {png} "]
    cases += [png[:-2] + "==", png + "==", png[:-4] + "a===", "aGk", "aGk==", "aG=k"]
    cases += ["data:text/plain;base64," + png, "data:image/svg+xml;base64," + png]
    cases += [_b64(b) for b in _pillow_images(40, seed=9)]
    cases += [_b64(b) for b in MALFORMED.values()]
    cases += [_b64(_png_header(4097, 4096)), _b64(_png_header(1, MAX_IMAGE_PIXELS + 1))]
    cases += [_b64(_png_header(13378, 13378)), _b64(_png_header(65535, 65535))]
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
        by_rule = _rfc4648(DATA_URI.sub("", data.strip()))
        if by_rule is None:  # the spec refuses the base64
            assert error is not None and error.startswith(NOT_BASE64), (data[:40], error)
            if sys.version_info[:2] == (3, 12):
                assert v1_error is not None and v1_error.startswith(NOT_BASE64), data[:40]
            continue
        # The spec accepts the base64; v1 accepts at least as much on every version.
        assert error is None or not error.startswith(NOT_BASE64), (data[:40], error)
        assert v1_error is None or not v1_error.startswith(NOT_BASE64), (data[:40], v1_error)
        if v1_error is None:
            assert error is None, (data[:40], error)
        elif v1_error.startswith("image could not be decoded"):
            pass  # the header is sound and the pixels are not: the runtime finds that out
        elif v1_error.startswith(COULD_NOT_READ):  # Pillow's reason may name an object's address
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
