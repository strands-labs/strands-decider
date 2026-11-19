"""The transformers backend on tiny random-weight Gemma 4 checkpoints in a decision checkpoint's
layout (tests/decision_support.py): no download, CPU in CI. The CUDA and MPS tests skip without
them; the real-checkpoint tests need STRANDS_DECIDER_TEST_CHECKPOINT, a decision checkpoint."""

from __future__ import annotations

import base64
import contextlib
import copy
import io
import json
import math
import os
import warnings

import numpy as np
import pytest
import torch

from decision_support import (  # isort: skip
    GEMMA4,
    HAS_UNIFIED,
    KINDS,
    UNIFIED,
    decision_checkpoint,
    image_b64,
    request,
)
import transformers
from core_conformance import agreement, read_golden_vectors
from core_support import PIXEL_FILES, image_bytes, reference_pixels

from strands_decider.backends import transformers as tb
from strands_decider.core import (
    DecisionConfig,
    Request,
    attention_mask,
    decode_image,
    evaluate,
    load_readout,
    load_tokenizer,
    question_probabilities,
    render,
    resize_image,
    resized_size,
    soft_tokens,
)
from strands_decider.core.prompt import RenderedImage

CUDA = torch.cuda.is_available()
MPS = torch.backends.mps.is_available()
try:
    import torchvision  # noqa: F401

    TORCHVISION = True
except ImportError:
    TORCHVISION = False
REAL = os.environ.get("STRANDS_DECIDER_TEST_CHECKPOINT")
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
needs_real = pytest.mark.skipif(
    not REAL, reason="set STRANDS_DECIDER_TEST_CHECKPOINT to a decision checkpoint"
)

IMAGES = [image_b64(64, 48), image_b64(50, 90, "JPEG", 1)]
# E2B's attention layout: one key/value head shared by 8 query heads, heads of 256 and 512.
E2B_HEADS = {
    "hidden_size": 256,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
}


def _needs_images(kind: str) -> None:
    if kind == UNIFIED and not TORCHVISION:
        pytest.skip("the 12B's image processor needs torchvision")


@pytest.fixture(scope="module", params=KINDS)
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def ckpt(kind, tmp_path_factory):
    return decision_checkpoint(tmp_path_factory.mktemp(kind) / "ckpt", kind)


@pytest.fixture(scope="module")
def e2b(tmp_path_factory):
    """A text-only checkpoint with E2B's attention layout."""
    path = tmp_path_factory.mktemp("e2b") / "ckpt"
    return decision_checkpoint(path, GEMMA4, text=E2B_HEADS, decision={"inputs": ["text"]})


def _parts(path):
    """The interface files as the decision engine reads them: tokenizer, config, readout."""
    return load_tokenizer(path), DecisionConfig.load(path), load_readout(path)


@pytest.fixture(scope="module")
def parts(ckpt):
    return _parts(ckpt)


def _reusing(backend, **settings):
    """`backend`'s model, with the state encoded once."""
    other = copy.copy(backend)
    other.reuse_state = True
    vars(other).update(settings)
    return other


@pytest.fixture(scope="module")
def backends(ckpt):
    """(full, reuse), float32 on CPU: one full pass per question, and the state once then the
    questions in batches of two."""
    full = tb.load_backend(str(ckpt), reuse_state=False)
    return full, _reusing(full, max_batch=2)


def _rendered(parts, req: dict):
    r = render(Request.model_validate(req), parts[0])
    return r, list(r.state_ids), [list(q.question_ids) for q in r.questions.values()]


def _evaluate(parts, req: dict, backend) -> dict:
    tokenizer, config, weight = parts
    return evaluate(Request.model_validate(req), tokenizer, weight, config, backend)


def _tokens(vocab: int, n: int, seed: int = 1) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(10, vocab - 10, n)]


def _rel(ref, got) -> float:
    """The largest difference, relative to the reference's scale."""
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(np.abs(ref - got).max()) / max(1.0, float(np.abs(ref).max()))


# ---- loading ------------------------------------------------------------------------------


def test_loads_the_multimodal_class_under_prefix_bidir(backends, kind):
    full, _ = backends
    want = (
        "Gemma4UnifiedForConditionalGeneration"
        if kind == UNIFIED
        else "Gemma4ForConditionalGeneration"
    )
    assert type(full.model).__name__ == want
    images = kind != UNIFIED or TORCHVISION  # the 12B's processor needs torchvision
    assert (full.image_processor is not None) == (full.image_problem is None) == images
    # The larger sizes' bidirectional image blocks are off, whatever the base config says.
    assert full.model.config.get_text_config().use_bidirectional_attention is None
    layers = [m for m in full.model.modules() if type(m).__name__.endswith("TextAttention")]
    assert layers and {m.config._attn_implementation for m in layers} == {tb.ATTENTION}
    vision = getattr(full.decoder, "vision_tower", None)  # the 12B has none
    if vision is not None:
        assert vision.config._attn_implementation == "sdpa"


def test_a_model_that_keeps_plain_sdpa_is_refused(e2b, monkeypatch):
    """transformers only warns when a model can't switch its attention, and its SDPA with no
    mask is causal: the load refuses such a model rather than serve wrong answers."""
    monkeypatch.setattr(transformers.PreTrainedModel, "set_attn_implementation", lambda *a: None)
    with pytest.raises(RuntimeError, match="did not set the decoder's attention"):
        tb.load_backend(str(e2b))


# ---- the mask -----------------------------------------------------------------------------


@pytest.fixture
def recorder(monkeypatch):
    """(mask, query tokens, key tokens) of each call to `_sdpa`, which only the decoder makes."""
    calls: list[tuple[torch.Tensor | None, int, int]] = []
    sdpa = tb._sdpa

    def record(query, key, value, mask, scale):
        calls.append((mask, query.shape[2], key.shape[2]))
        return sdpa(query, key, value, mask, scale)

    monkeypatch.setattr(tb, "_sdpa", record)
    return calls


@pytest.mark.parametrize("images", [[], IMAGES[:1]], ids=["text", "image"])
def test_every_layer_attends_without_a_dense_mask(backends, parts, kind, recorder, images):
    """What SDPA gets in each layer: in a full pass, the state's queries over its keys, then the
    question's over every key, unmasked; over a reused state, one row of key columns per question
    where a batch has padding. No n x n mask, which is what lets a longer state fit."""
    if images:
        _needs_images(kind)
    full, reuse = backends
    r, state, questions = _rendered(parts, request(images))
    s, nl = len(state), len(full.model.config.get_text_config().layer_types)
    full.forward(state, questions[:1], r.images)
    n = s + len(questions[0])
    assert len(recorder) == 2 * nl
    for i, (m, q, k) in enumerate(recorder):
        assert m is None and (q, k) == ((s, s) if i % 2 == 0 else (n - s, n))

    recorder.clear()
    reuse.forward(state, questions, r.images)
    state_pass, rest = recorder[:nl], recorder[nl:]
    assert all(m is None and (q, k) == (s, s) for m, q, k in state_pass)
    batches = [questions[:2], questions[2:]]
    assert len(rest) == nl * len(batches)
    for b, qs in enumerate(batches):
        width = max(map(len, qs))
        width += width % 64 == 1  # never a 64k+1-wide pass over the state
        for m, q, k in rest[b * nl : (b + 1) * nl]:
            assert (q, k) == (width, s + width)
            if all(len(x) == width for x in qs):
                assert m is None
                continue
            assert m.dtype == torch.bool and m.shape == (len(qs), 1, 1, s + width)
            for row, x in enumerate(qs):  # the state and its own question, never the padding
                assert m[row, 0, 0, : s + len(x)].all() and not m[row, 0, 0, s + len(x) :].any()


@contextlib.contextmanager
def _stock_sdpa():
    """transformers' own SDPA attention in place of the backend's, for references under the
    core's dense masks: given a mask, it is never causal (transformers 5.15 to 5.23)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[tb.ATTENTION] = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        yield
    finally:
        del ALL_ATTENTION_FUNCTIONS[tb.ATTENTION]  # the local override; the registration stays


def _against_dense_masks(backend, parts, req: dict, bound: float) -> None:
    """A full pass's hidden states, and the row `forward` returns, against transformers' SDPA
    under the core's dense mask, for a question of the request and a 65-token one; the state's
    own pass against an all-True mask. With no mask the state sees the question: `bound` tells."""
    r, state, questions = _rendered(parts, req)
    pixels = backend.prepare_pixels(r.images) if r.images else None

    def run(ids, mask):
        positions = torch.arange(ids.shape[1]).view(1, -1)
        with (
            torch.inference_mode(),
            _stock_sdpa() if isinstance(mask, torch.Tensor) else contextlib.nullcontext(),
        ):
            return backend._decode(ids, mask, positions, None, pixels)[0].float().numpy()

    s = len(state)
    for question in (questions[0], _tokens(parts[0].get_vocab_size(), 65)):
        n = s + len(question)
        prompt = torch.tensor([state + question])
        ref = run(prompt, torch.from_numpy(attention_mask(s, n)).view(1, 1, n, n))
        assert _rel(ref, run(prompt, tb.StateSplit(s))) < bound
        assert _rel(ref[-1], backend.forward(state, [question], r.images)[0]) < bound
        assert _rel(ref, run(prompt, None)) > bound
    alone = torch.tensor([state])
    assert _rel(run(alone, torch.ones(1, 1, s, s, dtype=torch.bool)), run(alone, None)) < bound


@pytest.mark.parametrize(("dtype", "bound"), [("float32", 2e-5), ("bfloat16", 0.3)])
@pytest.mark.parametrize("images", [[], IMAGES[:1]], ids=["text", "image"])
def test_hidden_states_are_the_dense_masks(ckpt, parts, kind, images, dtype, bound):
    """bf16's rounding reaches 0.03 with torch 2.15 and 0.18 with 2.7; a wrong mask moves these
    hidden states by more than 1. The window of 2 is shorter than the prompt: no layer keeps it."""
    if images:
        _needs_images(kind)
    backend = tb.load_backend(str(ckpt), dtype=dtype, reuse_state=False)
    _against_dense_masks(backend, parts, request(images), bound)


def test_hidden_states_are_the_dense_masks_with_e2bs_heads(e2b):
    """Keys and values repeated per query head when unmasked, head_dim 512; float32 (3e-5 at most
    with torch 2.7's CPU kernels)."""
    backend = tb.load_backend(str(e2b), reuse_state=False)
    _against_dense_masks(backend, _parts(e2b), request(), 1e-4)


# ---- state reuse --------------------------------------------------------------------------


@pytest.mark.parametrize("images", [[], IMAGES], ids=["text", "two-images"])
def test_state_reuse_is_the_full_pass(backends, parts, kind, images):
    """float32 on CPU, to float rounding."""
    if images:
        _needs_images(kind)
    full, reuse = backends
    r, state, questions = _rendered(parts, request(images))
    a, b = full.forward(state, questions, r.images), reuse.forward(state, questions, r.images)
    assert a.dtype == np.float32 and a.shape == (3, 64)
    assert _rel(a, b) < 2e-5


def test_no_pass_over_the_state_is_64k_plus_1_wide(backends, parts, recorder):
    """The torch 2.7 CUDA workaround, where CI runs: a batch whose longest question is 65 tokens
    runs 66 wide, one more masked column, and a full pass takes the state's last query too. This
    is its guard: with it removed, the CUDA tests pass on torch 2.7.1+cu126 on an H100."""
    full, reuse = backends
    _, state, _ = _rendered(parts, request())
    s, vocab = len(state), parts[0].get_vocab_size()
    qs = [_tokens(vocab, n, seed) for n, seed in ((65, 1), (40, 2), (65, 3))]
    reuse.forward(state, qs[:2])
    reuse.forward(state, [qs[0], qs[2]])  # no shorter question: the pad column still masked
    full.forward(state, qs[:1])
    assert {q for _, q, k in recorder if k > q} == {66}
    over_state = [m for m, q, k in recorder if (q, k) == (66, s + 66)]
    assert over_state and all(
        m is not None and m[:, 0, 0, : s + 65].all() and not m[:, 0, 0, s + 65 :].any()
        for m in over_state[len(over_state) // 2 :]
    )


def test_question_batches_keep_to_their_limits(backends):
    full, reuse = backends
    assert reuse._batches(10, [[1] * 3] * 5) == [[0, 1], [2, 3], [4]]  # max_batch=2
    budget = _reusing(full, max_batch_tokens=40)
    assert budget._batches(10, [[1] * 3] * 4) == [[0, 1, 2], [3]]  # 3 x (10 + 3) = 39 fit
    alone = _reusing(full, max_batch_tokens=1)
    assert alone._batches(10, [[1] * 3, [1] * 5, [1] * 2]) == [[0], [1], [2]]


# ---- images -------------------------------------------------------------------------------


def test_the_image_decides_the_answer(backends, parts, kind):
    """Two images of the same size give the same ids and different answers."""
    _needs_images(kind)
    full, _ = backends
    a = _evaluate(parts, request([image_b64(64, 48, seed=1)]), full)
    b = _evaluate(parts, request([image_b64(64, 48, seed=2)]), full)
    assert a["usage"] == b["usage"] and a["answers"] != b["answers"]


def _rendered_image(data: bytes) -> RenderedImage:
    im = decode_image(base64.b64encode(data).decode())
    return RenderedImage(im.data, im.format, im.width, im.height, im.sha256, 1, im.tokens)


@pytest.mark.parametrize("name", sorted(PIXEL_FILES))
def test_image_pixels_are_the_cores_reference(name):
    """v1's decode, resized by the core's `resize_image` to the size of the decoded, upright
    image: rotated-80x40.jpg, whose EXIF turns it, to 528 x 1104, never its header's 1104 x 528."""
    data = image_bytes(name)
    decoded = reference_pixels(data)
    want = resize_image(decoded, *resized_size(*decoded.size))
    got = tb.image_pixels(_rendered_image(data))
    assert got.size == want.size and np.array_equal(np.asarray(got), np.asarray(want))


def _photo(w: int, h: int, fmt: str, seed: int) -> bytes:
    """A smooth gradient with noise: a resize filter's errors show on it."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx * 255 // max(w - 1, 1), yy * 255 // max(h - 1, 1), (xx + yy) % 256], -1)
    arr = np.clip(base + rng.integers(-40, 40, size=base.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt)
    return buf.getvalue()


@pytest.mark.skipif(not (HAS_UNIFIED and TORCHVISION), reason="needs the 12B's processor")
@pytest.mark.parametrize(
    ("w", "h", "fmt"), [(640, 480, "JPEG"), (300, 40, "PNG"), (97, 1301, "WEBP")]
)
def test_the_12bs_pixels_are_the_resized_image_in_48_pixel_patches(tmp_path, w, h, fmt):
    """With its resize off, each of the 12B processor's N patches is the resized image's 48 x 48
    block at the patch's position, in [0, 1]; the padding is zero."""
    backend = tb.load_backend(str(decision_checkpoint(tmp_path / "u", UNIFIED)))
    im = _rendered_image(_photo(w, h, fmt, 5))
    px = backend.prepare_pixels([im])
    values, positions = px["pixel_values"][0].numpy(), px["image_position_ids"][0].numpy()
    arr = np.asarray(tb.image_pixels(im), dtype=np.float32) / 255
    valid = (positions != -1).all(-1)
    assert int(valid.sum()) == im.tokens == soft_tokens(w, h)
    for j in np.flatnonzero(valid):
        x, y = positions[j]
        block = arr[y * 48 : (y + 1) * 48, x * 48 : (x + 1) * 48].reshape(-1)
        np.testing.assert_allclose(values[j], block, atol=1e-6)
    assert not values[~valid].any()


# ---- other devices ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cuda", marks=needs_cuda),
        pytest.param("mps", marks=pytest.mark.skipif(not MPS, reason="needs Apple's MPS")),
    ],
)
def test_a_device_agrees_with_the_cpu(ckpt, parts, kind, device):
    """float32 to float rounding, the device's kernels' check, reusing the state as in full
    passes; bf16 to its own error, large on random weights (about 0.1 in probability on CPU)."""
    _needs_images(kind)
    for dtype, bound in (("float32", 5e-3), ("bfloat16", 0.15)):
        cpu = tb.load_backend(str(ckpt), dtype=dtype)
        other = tb.load_backend(str(ckpt), device=device, dtype=dtype)
        for req in (request(), request(IMAGES[:1])):
            a = _evaluate(parts, req, cpu)["answers"]
            b = _evaluate(parts, req, other)["answers"]
            for name, p in a["team"]["probabilities"].items():
                assert abs(p - b["team"]["probabilities"][name]) < bound, (dtype, name)
            assert abs(a["late"]["noul"] - b["late"]["noul"]) < bound, dtype
            if dtype == "float32":
                assert a["team"]["choice"] == b["team"]["choice"]
    full = tb.load_backend(str(ckpt), device=device, dtype="float32", reuse_state=False)
    r, state, questions = _rendered(parts, request(IMAGES[:1]))
    got = _reusing(full).forward(state, questions, r.images)
    assert _rel(full.forward(state, questions, r.images), got) < 1e-4


def _question_batches(vocab: int, widths, state_len: int, seed: int = 0):
    """A `state_len`-token state and, for each width, three questions padded to that width in
    one batch over the state's keys and values."""
    g = np.random.default_rng(seed)
    state = [int(t) for t in g.integers(10, vocab - 10, state_len)]
    return state, [
        (w, [[int(t) for t in g.integers(10, vocab - 10, n)] for n in (w, w - 7, w - 30)])
        for w in widths
    ]


WIDTHS_64K_PLUS_1 = (64, 65, 66, 129, 193)


@needs_cuda
def test_cuda_question_batches_of_64k_plus_1_tokens_over_a_cached_state(e2b):
    """torch 2.7's CUDA SDPA gets E2B's layout wrong in passes of 64k+1 queries (65, 129, ...,
    385): under a dense mask the hidden states were off by up to 0.84 (float32, H100), though an
    isolated call on random inputs is right. Full passes and state reuse against one CPU pass per
    prompt under the core's dense mask: within 1.7e-4, torch 2.7 and 2.14. float32 only: in bf16
    this model lies up to 0.78 from float32 with any mask; the next test checks bf16 per call."""
    cpu = tb.load_backend(str(e2b), dtype="float32", reuse_state=False)
    full = tb.load_backend(str(e2b), device="cuda", dtype="float32", reuse_state=False)
    reuse = _reusing(full)

    def dense(state, q):
        s, n = len(state), len(state) + len(q)
        m = torch.from_numpy(attention_mask(s, n)).view(1, 1, n, n)
        with torch.inference_mode(), _stock_sdpa():
            return cpu._decode(torch.tensor([state + q]), m, torch.arange(n).view(1, -1))[0, -1]

    vocab = _parts(e2b)[0].get_vocab_size()
    for state_len in (300, 321):
        state, cases = _question_batches(vocab, WIDTHS_64K_PLUS_1, state_len)
        for width, qs in cases:
            assert reuse._batches(len(state), qs) == [[0, 1, 2]]  # one pass, `width` wide
            ref = np.stack([dense(state, q).numpy() for q in qs])
            for got in (full.forward(state, qs), reuse.forward(state, qs)):
                assert _rel(ref, got) < 1e-3, (state_len, width)


class _Float64Check:
    """Wraps the backend's SDPA call and measures each call's output against float64 on the
    call's own inputs, relative to the output's scale: the largest error over all calls, and
    the number of calls of several rows with no mask (an unpadded batch over a cached state)."""

    def __init__(self, inner):
        self.inner = inner
        self.calls = self.unpadded_batches = 0
        self.worst = 0.0

    def __call__(self, query, key, value, mask, scale):
        out = self.inner(query, key, value, mask, scale)
        groups = query.shape[1] // key.shape[1]
        k = key.double().repeat_interleave(groups, dim=1)
        v = value.double().repeat_interleave(groups, dim=1)
        scale = query.shape[-1] ** -0.5 if scale is None else scale
        scores = query.double() @ k.transpose(-1, -2) * scale
        if mask is not None:
            scores = scores.masked_fill(~mask, -torch.inf)
        ref = scores.softmax(-1) @ v
        err = float((out.double() - ref).abs().max() / ref.abs().max())
        assert math.isfinite(err), err
        self.calls += 1
        self.unpadded_batches += mask is None and query.shape[0] > 1
        self.worst = max(self.worst, err)
        return out


@needs_cuda
@pytest.mark.parametrize(("dtype", "bound"), [("bfloat16", 0.02), ("float32", 1e-4)])
def test_cuda_every_attention_call_against_float64(e2b, monkeypatch, dtype, bound):
    """Every attention call E2B's layout makes on CUDA (head_dim 256 and 512; full passes, whole
    prompts of 385 included, and batches over a reused state, padded or not) against float64 on
    its own inputs: at worst 0.0038 in bf16 and 9e-6 in float32 on torch 2.7 and 2.14, where
    transformers' SDPA under the dense mask was off by 0.97 and 0.93 at 385 tokens on 2.7."""
    full = tb.load_backend(str(e2b), device="cuda", dtype=dtype, reuse_state=False)
    reuse = _reusing(full)
    check = _Float64Check(tb._sdpa)
    monkeypatch.setattr(tb, "_sdpa", check)
    vocab = _parts(e2b)[0].get_vocab_size()
    for state_len in (300, 321):
        state, cases = _question_batches(vocab, (64, 65, 129), state_len)
        for _, qs in cases:
            full.forward(state, qs)
            reuse.forward(state, qs)
        q = cases[0][1][0]  # 64 tokens, so no 64k+1 column either
        reuse.forward(state, [q, q[::-1]])  # one length: no padding, no mask
    assert check.calls and check.unpadded_batches, (check.calls, check.unpadded_batches)
    assert check.worst < bound, (dtype, check.worst)


# ---- a real checkpoint --------------------------------------------------------------------

# The most one full pass per question and state reuse may differ on the test request's answers,
# on a real checkpoint in bf16 on CUDA. Measured on an H100: 0.032 at 12B with torch 2.14 and
# 0.020 with 2.7.1; 0 at E2B with 2.14 and 0.0067 with 2.7.1; 0 at E4B with 2.14.
REAL_REQUEST_BOUND = 0.1


def _real_backend(**settings):
    from strands_decider.decision import checkpoint_dir

    path = checkpoint_dir(REAL)
    assert path is not None, REAL
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = tb.load_backend(
            path, device="cuda" if CUDA else "cpu", reuse_state=False, **settings
        )
    return path, full


@needs_real
def test_a_real_checkpoint():
    """An image request runs where it takes images; state reuse gives the full passes' choice,
    and every noul, score and probability within REAL_REQUEST_BOUND."""
    path, full = _real_backend()
    images = (
        [base64.b64encode(_photo(640, 480, "JPEG", 0)).decode()] if full.image_processor else []
    )
    p = _parts(path)
    a = _evaluate(p, request(images), full)["answers"]
    b = _evaluate(p, request(images), _reusing(full))["answers"]
    assert a["team"]["choice"] == b["team"]["choice"]
    for name, x in a.items():
        for key in ("noul", "score", "probabilities"):
            if key in x:
                got = list(x[key].values()) if key == "probabilities" else x[key]
                want = list(b[name][key].values()) if key == "probabilities" else b[name][key]
                assert np.abs(np.subtract(got, want)).max() < REAL_REQUEST_BOUND, (name, key)


@needs_real
def test_a_real_checkpoint_reuses_states_as_full_passes_in_float32():
    """Question batches over a reused state at 64k+1 widths against full passes, in float32: in
    bf16 these random-token prompts differ by up to 0.4 in probability on a 12B (torch 2.14),
    dense masks or not. Skips where the model doesn't fit in float32 (an 80 GB GPU: 12B, not 31B)."""
    path, full = _real_backend()
    if full.dtype == torch.bfloat16:
        extra = sum(p.numel() for p in full.model.parameters()) * 2  # bf16 -> float32
        if extra > 0.8 * torch.cuda.mem_get_info()[0]:
            pytest.skip("the model does not fit the GPU in float32")
        full.model.float()
    reuse = _reusing(full)
    vocab = _parts(path)[0].get_vocab_size()
    for state_len in (300, 321):
        state, cases = _question_batches(vocab, WIDTHS_64K_PLUS_1, state_len)
        for width, qs in cases:
            assert _rel(full.forward(state, qs), reuse.forward(state, qs)) < 1e-3, (
                state_len,
                width,
            )


GOLDENS = "golden_vectors.jsonl"


def golden_probabilities(path, parts, backend) -> dict:
    """`backend`'s code probabilities on the goldens at `path`, keyed as `read_golden_vectors`
    keys the reference's, after checking that every line renders to the goldens' ids, image
    spans and hashes, and codes exactly."""
    tokenizer, config, weight = parts
    out = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            g = json.loads(line)
            r = render(Request.model_validate(g["request"]), tokenizer)
            assert list(r.state_ids) == g["state_ids"], line_number
            spans = [{"sha256": im.sha256, "span": [im.start, im.end]} for im in r.images]
            assert spans == g["images"], line_number
            ids = {n: (list(q.question_ids), list(q.codes)) for n, q in r.questions.items()}
            want = {n: (q["question_ids"], q["codes"]) for n, q in g["questions"].items()}
            assert ids == want, line_number
            rows = backend.forward(
                list(r.state_ids), [list(q.question_ids) for q in r.questions.values()], r.images
            )
            probs = question_probabilities(rows, config, r, weight=weight)
            out.update({(line_number, n): [float(x) for x in v] for n, v in probs.items()})
    return out


@needs_real
def test_a_real_checkpoint_meets_its_goldens():
    """The rendered ids are the goldens', and in bf16 the runtime conforms (`Agreement.conforms`)
    with one full pass per prompt, the reference, and with the state reused."""
    path, full = _real_backend(dtype="bfloat16")
    goldens = os.path.join(path, GOLDENS)
    if not os.path.exists(goldens):
        pytest.skip(f"the checkpoint has no {GOLDENS} yet")
    p, reference = _parts(path), read_golden_vectors(goldens)
    for backend in (full, _reusing(full)):
        measured = agreement(golden_probabilities(goldens, p, backend), reference)
        assert measured.conforms(), (backend.reuse_state, measured)
