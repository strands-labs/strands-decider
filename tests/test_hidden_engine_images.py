"""HiddenStateEngine's image path, without torch or transformers.

tests/test_vision.py holds this engine's answers to VisionEngine's over a tiny vision
checkpoint, which needs transformers 5.18. This module needs only what the engine does (numpy,
tokenizers, Pillow), so it runs on every CI path: the processor's grid, the state's layout around
images, the window, the limits and the refusals, over a backend that records what it is sent.
"""

from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from strands_decider.hidden_engine import (  # noqa: E402
    HiddenEngineConfig,
    HiddenStateEngine,
    NumpyPointerHead,
)
from strands_decider.images import (  # noqa: E402
    IMAGE_PAD,
    VISION_END,
    VISION_START,
    smart_resize,
    state_segments,
    to_grid,
)

HIDDEN, POINTER = 8, 4
QUESTIONS = {
    "paid": {"type": "noul", "instructions": "Did the payment succeed?"},
    "next": {
        "type": "choice",
        "instructions": "Which control next?",
        "criteria": {"retry": "the retry button", "cancel": "the cancel link"},
    },
}


def _png(width: int, height: int) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (40, 120, 200)).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


class RecordingBackend:
    """An ImageBackend that records what it is sent, sizes images as the processor's grid
    does, and draws hidden states from the request's shape."""

    def __init__(self, extra_tokens: int = 0):
        self.extra_tokens = extra_tokens  # added to each image's count, to size it otherwise
        self.sent: list[dict] = []

    def forward(self, state, questions, rows):
        return [np.random.default_rng(len(q)).normal(size=(len(r), HIDDEN)) for q, r in zip(questions, rows, strict=True)]  # fmt: skip

    def forward_images(self, segments, images, questions, rows):
        self.sent.append(
            {"segments": segments, "sizes": [(w, h) for _, w, h in images], "rows": rows}
        )
        tokens = [(w // 32) * (h // 32) + 2 + self.extra_tokens for _, w, h in images]
        return self.forward([t for s in segments for t in s], questions, rows), tokens


class TextOnlyBackend:
    def forward(self, state, questions, rows):
        return RecordingBackend().forward(state, questions, rows)


def _engine(backend=None, **config) -> HiddenStateEngine:
    vocab = {"<unk>": 0, VISION_START: 1, IMAGE_PAD: 2, VISION_END: 3}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.add_special_tokens(
        [AddedToken(t, normalized=False) for t in (VISION_START, IMAGE_PAD, VISION_END)]
    )
    rng = np.random.default_rng(0)
    head = NumpyPointerHead(
        {
            "norm.weight": rng.normal(size=HIDDEN),
            "norm.bias": rng.normal(size=HIDDEN),
            "q.weight": rng.normal(size=(POINTER, HIDDEN)),
            "q.bias": rng.normal(size=POINTER),
            "k.weight": rng.normal(size=(POINTER, HIDDEN)),
            "k.bias": rng.normal(size=POINTER),
        },
        POINTER,
        1e-5,
    )
    return HiddenStateEngine(
        tok,
        head,
        HiddenEngineConfig(**{"max_length": 4096, **config}),
        backend or RecordingBackend(),
    )


def _request(
    n_images: int = 1, state: str = "The checkout page after the user tapped Pay.", size=(320, 320)
):
    return {
        "state": state,
        "questions": QUESTIONS,
        "images": [_png(*size) for _ in range(n_images)],
    }


# ---- the processor's grid -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("height", "width", "grid"),
    [(448, 448, (448, 448)), (64, 64, (256, 256)), (299, 448, (288, 448)), (149, 224, (224, 320))],
)
def test_smart_resize_fits_the_grid(height, width, grid):
    """Sizes from transformers' Qwen2-VL smart_resize, as tests/test_vision.py checks it."""
    assert smart_resize(height, width) == grid


def test_smart_resize_keeps_sides_on_the_grid_and_the_area_in_bounds():
    rng = np.random.default_rng(0)
    for _ in range(2000):
        h, w = int(rng.integers(8, 5000)), int(rng.integers(8, 5000))
        if max(h, w) / min(h, w) > 200:
            continue
        gh, gw = smart_resize(h, w)
        assert gh % 32 == 0 and gw % 32 == 0
        assert 65536 <= gh * gw <= 16777216


def test_smart_resize_refuses_an_aspect_ratio_over_200_as_the_processor_does():
    assert smart_resize(1, 200) == (32, 3648)  # exactly 200 is allowed, as transformers allows it
    with pytest.raises(
        ValueError, match=r"^absolute aspect ratio must be smaller than 200, got 224\.0$"
    ):
        smart_resize(2, 448)


def test_to_grid_resizes_and_counts_tokens():
    rgb, w, h, tokens = to_grid(Image.new("RGB", (448, 299)))
    assert (w, h, tokens, len(rgb)) == (448, 288, 14 * 9, 448 * 288 * 3)


# ---- the state's layout ---------------------------------------------------------------------


def test_state_segments_put_each_image_between_two_segments():
    segments = state_segments("The page.", 2)
    assert len(segments) == 3 and segments[-1].endswith("The page.\n</state>\n")


def test_state_segments_refuse_a_state_with_an_image_pad_of_its_own():
    with pytest.raises(
        ValueError, match=r"^prompt holds 2 image placeholders but 1 token counts were given$"
    ):
        state_segments(f"see {IMAGE_PAD} here", 1)


# ---- the engine ----------------------------------------------------------------------------


def test_four_images_are_answered_and_five_refused():
    backend = RecordingBackend()
    response = _engine(backend).evaluate(_request(4))
    assert set(response.answers) == {"paid", "next"}
    assert backend.sent[0]["sizes"] == [(320, 320)] * 4 and len(backend.sent[0]["segments"]) == 5
    with pytest.raises(ValueError, match="5 images; this server takes at most 4"):
        _engine(backend).evaluate(_request(5))
    assert len(backend.sent) == 1


def test_images_are_sized_with_the_engines_settings():
    backend = RecordingBackend()
    _engine(backend, image_long_side=224).evaluate(_request(1, size=(600, 400)))
    # 600 x 400 is 224 x 149 at a long side of 224, which the grid's least area raises to 320 x 224.
    assert backend.sent[0]["sizes"] == [(320, 224)]
    with pytest.raises(ValueError, match="at most 10,000 pixels"):
        _engine(backend, max_image_pixels=10_000).evaluate(_request(1, size=(101, 100)))
    assert len(backend.sent) == 1


def test_the_request_limit_counts_the_images():
    backend = RecordingBackend()
    usage = _engine(backend).evaluate(_request(1)).usage.input_tokens
    image_tokens = 10 * 10 + 2  # 320 px square: 10 x 10 on the grid, with its start and end
    _engine(backend, max_request_tokens=usage).evaluate(_request(1))
    with pytest.raises(ValueError, match=f"request of {usage} tokens .* the limit of {usage - 1}"):
        _engine(backend, max_request_tokens=usage - 1).evaluate(_request(1))
    assert len(backend.sent) == 2
    text_only = _engine(TextOnlyBackend()).evaluate({**_request(0)}).usage.input_tokens
    assert usage - text_only == image_tokens


def test_a_prompt_over_the_window_is_refused_with_its_parts():
    engine = _engine()
    segments, q, _ = engine.fit_images("The page.", [102], ["Did it work ? <answer>"])
    exact = sum(len(s) for s in segments) + 102 + max(len(x) for x in q)
    _engine(max_length=exact).fit_images("The page.", [102], ["Did it work ? <answer>"])
    with pytest.raises(
        ValueError,
        match=rf"^prompt of {exact} tokens \(.* state \+ 100 image \+ .* question\) exceeds the context window of {exact - 1} tokens$",
    ):
        _engine(max_length=exact - 1).fit_images("The page.", [102], ["Did it work ? <answer>"])


@pytest.mark.parametrize(
    ("request_", "message"),
    [
        (_request(1, size=(2000, 8)), "absolute aspect ratio must be smaller than 200"),
        (_request(1, state=f"see {IMAGE_PAD} here"), "prompt holds 2 image placeholders"),
    ],
    ids=["aspect ratio", "image pad in the state"],
)
def test_requests_vision_refuses_are_refused_before_the_backend(request_, message):
    backend = RecordingBackend()
    with pytest.raises(ValueError, match=message):
        _engine(backend).evaluate(request_)
    assert not backend.sent


def test_images_need_a_backend_that_reads_them():
    with pytest.raises(ValueError, match="no vision tower"):
        _engine(TextOnlyBackend()).evaluate(_request(1))


def test_a_backend_that_sizes_images_otherwise_is_named():
    with pytest.raises(RuntimeError, match=r"backend made \[103\] image tokens; expected \[102\]"):
        _engine(RecordingBackend(extra_tokens=1)).evaluate(_request(1))
