"""The transformers backend (src/strands_decider/backends/transformers.py) on tiny random-weight
Gemma 4 checkpoints in the spec's layout, generated here: no download, CPU in CI.

- loading: both multimodal classes, the intermediate bare decoder (text only, warnings), and
  the checkpoints it refuses;
- the mask: what every decoder layer receives is docs/spec.md section 6's prefix_bidir mask,
  compared with the core's explicit `attention_mask`, and sliding windows have no effect;
- state reuse: the state once and then the questions is the full pass of each prompt;
- the readout: rows are the decoder's hidden states, and the core's regime turns them into
  the answers;
- images: placement at the spans, the per-layer embeddings there, and the span checks;
- refusals;
- pixels: Pillow's bicubic resize then the processor without its resize is
  `Gemma4ImageProcessorPil`, and the soft-token count is the core's N.

Tests that need CUDA or Apple's MPS skip without them; `test_a_real_checkpoint` needs
STRANDS_DECIDER_TEST_CHECKPOINT set to a real decision checkpoint.
"""

from __future__ import annotations

import base64
import io
import json
import os
import warnings

import numpy as np
import pytest
import torch

from decision_support import (  # isort: skip
    GEMMA4,
    GEMMA4_VISION,
    HAS_UNIFIED,
    KINDS,
    SOFTCAP,
    TEMPERATURE,
    UNIFIED,
    decision_checkpoint,
    image_b64,
    intermediate_checkpoint,
    request,
)
import transformers

from strands_decider.backends import transformers as tb
from strands_decider.core import (
    attention_mask,
    evaluate,
    question_probabilities,
    read,
    readout_logits,
    render,
    resize_image,
    resized_size,
    round_to_bf16,
    soft_tokens,
)
from strands_decider.core.request import NO_VISION

CUDA = torch.cuda.is_available()
MPS = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
TORCHVISION = True
try:
    import torchvision  # noqa: F401
except ImportError:
    TORCHVISION = False


# Whether each kind's image path can run here: the 12B's processor needs torchvision. Without
# it the 12B's class still loads and runs text; its image tests skip.
IMAGING = {GEMMA4: True, GEMMA4_VISION: True, UNIFIED: TORCHVISION}


def _needs_images(kind: str) -> None:
    if not IMAGING[kind]:
        pytest.skip("the 12B's image processor needs torchvision")


@pytest.fixture(scope="module", params=KINDS)
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def ckpt(kind, tmp_path_factory):
    return decision_checkpoint(tmp_path_factory.mktemp(kind) / "ckpt", kind)


@pytest.fixture(scope="module")
def backends(ckpt, kind):
    """(reference, reuse): one full pass per question, and the state once then the questions
    in batches of two, both float32 on CPU."""
    images = IMAGING[kind]
    full = tb.TransformersBackend.load(str(ckpt), reuse_state=False, images=images)
    reuse = tb.TransformersBackend.load(str(ckpt), reuse_state=True, max_batch=2, images=images)
    return full, reuse


def _rendered(backend, req):
    ck = backend.checkpoint
    r = render(req, ck.tokenizer(), ck.config.prompt)
    return r, list(r.state_ids), [list(q.question_ids) for q in r.questions.values()]


IMAGES = [image_b64(64, 48), image_b64(50, 90, "JPEG", 1)]


# ---- loading ------------------------------------------------------------------------------


def test_loads_the_multimodal_layout(backends, kind):
    full, reuse = backends
    inputs = ["text", "image"] if IMAGING[kind] else ["text"]
    assert full.outputs == "hidden"
    assert list(full.inputs) == inputs
    assert full.checkpoint.layout == tb.MULTIMODAL and full.checkpoint.unified == (kind == UNIFIED)
    want = (
        "Gemma4UnifiedForConditionalGeneration"
        if kind == UNIFIED
        else "Gemma4ForConditionalGeneration"
    )
    assert type(full.model).__name__ == want
    assert full.describe() == {
        "backend": "transformers",
        "device": "cpu",
        "dtype": "float32",
        "layout": "multimodal",
        "inputs": inputs,
        "reuse_state": False,
    }
    assert reuse.describe()["reuse_state"] is True
    # The larger sizes' bidirectional image blocks are off, whatever the base config says.
    assert full.decoder.config.get_text_config().use_bidirectional_attention is None
    assert full.checkpoint.readout.softcap == SOFTCAP and full.checkpoint.temperature == TEMPERATURE


def test_the_intermediate_layout_loads_as_text_only_with_warnings(tmp_path):
    path = intermediate_checkpoint(tmp_path / "mid")
    with pytest.warns(UserWarning) as caught:
        backend = tb.TransformersBackend.load(str(path))
    text = " ".join(str(w.message) for w in caught)
    assert 'no "interface"' in text and 'no "inputs"' in text
    assert backend.checkpoint.layout == tb.DECODER and backend.checkpoint.config is None
    assert backend.inputs == ("text",) and type(backend.model).__name__ == "Gemma4TextModel"
    # Its soft-cap comes from config.json, its temperature from decision_config.json.
    assert backend.checkpoint.readout.softcap == SOFTCAP
    assert backend.checkpoint.temperature == TEMPERATURE
    tok = backend.checkpoint.tokenizer()
    state = tok.encode("a b c . a b", add_special_tokens=False).ids
    questions = [tok.encode(q, add_special_tokens=False).ids for q in ("A B", "answer . C")]
    h = backend.forward(state, questions)
    with pytest.warns(UserWarning):
        ref = tb.TransformersBackend.load(str(path), reuse_state=False).forward(state, questions)
    assert h.shape == (2, 64) and np.abs(h - ref).max() < 1e-5


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"inputs": ["text", "image"]}, "no vision encoder"),
        ({"attention_mode": "causal"}, "attention_mode"),
        ({"temperature": 0}, "temperature"),
        ({"final_logit_softcapping": 20.0}, "inconsistent"),
        ({"token_ids": [1, 2]}, "token_ids"),
    ],
)
def test_an_intermediate_checkpoint_is_still_checked(tmp_path, overrides, match):
    path = intermediate_checkpoint(tmp_path / "mid", decision=overrides)
    with pytest.raises(ValueError, match=match):
        tb.Checkpoint.open(str(path))


def test_refuses_what_is_not_a_decision_checkpoint(tmp_path):
    path = decision_checkpoint(tmp_path / "a", GEMMA4)
    dc = json.loads((path / "decision_config.json").read_text())
    del dc["interface"]  # the multimodal layout is the released format: the schema is required
    (path / "decision_config.json").write_text(json.dumps(dc))
    with pytest.raises(ValueError, match="interface"):
        tb.Checkpoint.open(str(path))
    cfg = json.loads((path / "config.json").read_text())
    (path / "config.json").write_text(json.dumps({**cfg, "model_type": "qwen3_5"}))
    with pytest.raises(ValueError, match="not a Gemma 4 decision checkpoint"):
        tb.Checkpoint.open(str(path))
    (path / "config.json").write_text(json.dumps({**cfg, "architectures": ["Gemma4ForCausalLM"]}))
    with pytest.raises(ValueError, match="architectures"):
        tb.Checkpoint.open(str(path))


def test_devices_and_dtypes():
    assert tb.default_dtype("cpu") == torch.float32
    assert tb.default_dtype("cuda:1") == torch.bfloat16
    assert tb.default_dtype("mps") == torch.bfloat16
    with pytest.raises(ValueError, match="cpu, cuda or mps"):
        tb.default_dtype("meta")


# ---- the mask -----------------------------------------------------------------------------


class _Recorder:
    """Wraps transformers' SDPA attention and records the mask each decoder layer gets."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[tuple[int, torch.Tensor]] = []

    def __call__(self, module, query, key, value, attention_mask, **kwargs):
        if type(module).__name__.endswith("TextAttention"):
            self.calls.append((module.layer_idx, attention_mask))
        return self.inner(module, query, key, value, attention_mask, **kwargs)


@pytest.fixture
def recorder(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    rec = _Recorder(ALL_ATTENTION_FUNCTIONS["sdpa"])
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS, "sdpa", rec)
    return rec


@pytest.mark.parametrize("images", [[], IMAGES[:1]], ids=["text", "image"])
def test_every_layer_gets_the_prefix_bidir_mask(backends, kind, recorder, images):
    if images:
        _needs_images(kind)
    full, reuse = backends
    r, state, questions = _rendered(full, request(images))
    s = len(state)
    full.forward(state, questions[:1], r.images)
    n = s + len(questions[0])
    want = torch.from_numpy(attention_mask(s, n))
    layers = [i for i, _ in recorder.calls]
    assert layers == list(range(len(full.decoder.config.get_text_config().layer_types)))
    for _, m in recorder.calls:  # sliding-window layers included: no layer has a window
        assert m.dtype == torch.bool and m.shape == (1, 1, n, n)
        assert torch.equal(m[0, 0].cpu(), want)

    recorder.calls.clear()
    reuse.forward(state, questions, r.images)
    nl = len(layers)
    state_pass, rest = recorder.calls[:nl], recorder.calls[nl:]
    for _, m in state_pass:
        assert torch.equal(m[0, 0].cpu(), torch.ones(s, s, dtype=torch.bool))
    # Then batches of two questions over the state: row i sees the state and its own
    # question, which is row S.. of that prompt's prefix_bidir mask.
    batches = [questions[:2], questions[2:]]
    assert len(rest) == nl * len(batches)
    for b, qs in enumerate(batches):
        width = max(map(len, qs))
        width += width % 64 == 1  # never a 64k+1-wide pass over the state
        for _, m in rest[b * nl : (b + 1) * nl]:
            assert m.shape == (len(qs), 1, width, s + width)
            for row, q in enumerate(qs):
                ref = torch.from_numpy(attention_mask(s, s + len(q)))[s:]
                assert torch.equal(m[row, 0, : len(q), : s + len(q)].cpu(), ref)
                assert not m[row, 0, :, s + len(q) :].any()  # never the padding


def test_sliding_windows_have_no_effect(ckpt, kind, tmp_path):
    wide = tmp_path / "wide"
    import shutil

    shutil.copytree(ckpt, wide)
    cfg = json.loads((wide / "config.json").read_text())
    cfg["text_config"]["sliding_window"] = 100_000
    (wide / "config.json").write_text(json.dumps(cfg))
    a = tb.TransformersBackend.load(str(ckpt), reuse_state=False, images=False)
    b = tb.TransformersBackend.load(str(wide), reuse_state=False, images=False)
    _, state, questions = _rendered(a, request())
    assert len(state) > 2 * 2  # longer than the window
    assert np.array_equal(a.forward(state, questions), b.forward(state, questions))


def test_the_mask_is_not_gemma4s_own(backends):
    """The causal, windowed mask transformers builds by default gives other hidden states, so
    the tests above would notice the backend falling back to it."""
    full, _ = backends
    _, state, questions = _rendered(full, request())
    ids = torch.tensor([state + questions[0]])
    with torch.inference_mode():
        own = full.decoder(input_ids=ids).last_hidden_state[0, -1].numpy()
    assert np.abs(own - full.forward(state, questions[:1])[0]).max() > 1e-2


def test_the_state_never_sees_a_question(backends):
    full, _ = backends
    _, state, questions = _rendered(full, request())
    captured = []
    hook = full.decoder.register_forward_hook(
        lambda mod, args, out: captured.append(out.last_hidden_state[0, : len(state)].clone())
    )
    try:
        full.forward(state, questions[:2])
    finally:
        hook.remove()
    assert torch.equal(captured[0], captured[1])


# ---- state reuse --------------------------------------------------------------------------


@pytest.mark.parametrize("images", [[], IMAGES], ids=["text", "two-images"])
def test_state_reuse_is_the_full_pass(backends, kind, images):
    """float32 on CPU: the hidden states agree to float rounding, and the answers exactly."""
    if images:
        _needs_images(kind)
    full, reuse = backends
    r, state, questions = _rendered(full, request(images))
    a, b = full.forward(state, questions, r.images), reuse.forward(state, questions, r.images)
    assert a.dtype == np.float32 and a.shape == (3, 64)
    assert np.abs(a - b).max() < 2e-5 * max(1.0, float(np.abs(a).max()))
    ck = full.checkpoint
    assert read(a, ck.readout, ck.config, r, stage="hidden") == read(
        b, ck.readout, ck.config, r, stage="hidden"
    )


def test_no_pass_over_the_state_is_64k_plus_1_wide(backends, recorder):
    """A batch whose longest question is 65 tokens runs 66 wide: one more masked column."""
    _, reuse = backends
    _, state, _ = _rendered(reuse, request())
    g = np.random.default_rng(1)
    vocab = reuse.checkpoint.tokenizer().get_vocab_size()
    qs = [[int(t) for t in g.integers(10, vocab - 10, n)] for n in (65, 40)]
    reuse.forward(state, qs)
    widths = {m.shape[2] for _, m in recorder.calls if m.shape[2] != len(state)}
    assert widths == {66}


def test_one_question_is_one_full_pass(backends):
    full, reuse = backends
    _, state, questions = _rendered(full, request())
    assert np.array_equal(full.forward(state, questions[:1]), reuse.forward(state, questions[:1]))


def test_batches_follow_the_limits(backends):
    _, reuse = backends
    reuse_one = tb.TransformersBackend(
        reuse.model, reuse.checkpoint, device="cpu", dtype=torch.float32,
        image_processor=reuse.image_processor, max_batch=32, max_batch_tokens=1,
    )  # fmt: skip
    assert reuse_one._batches(10, [[1] * 3, [1] * 5, [1] * 2]) == [[0], [1], [2]]
    assert reuse._batches(10, [[1] * 3] * 5) == [[0, 1], [2, 3], [4]]
    budget = tb.TransformersBackend(
        reuse.model, reuse.checkpoint, device="cpu", dtype=torch.float32, max_batch_tokens=40
    )
    # 3 rows x (10 + 3) = 39 fit; a fourth would not.
    assert budget._batches(10, [[1] * 3] * 4) == [[0, 1, 2], [3]]
    _, state, questions = _rendered(reuse, request())
    assert (
        np.abs(reuse_one.forward(state, questions) - reuse.forward(state, questions)).max() < 2e-5
    )


def test_bf16_state_reuse_moves_answers_by_rounding_only(ckpt, kind):
    _needs_images(kind)
    full = tb.TransformersBackend.load(str(ckpt), dtype=torch.bfloat16, reuse_state=False)
    reuse = tb.TransformersBackend.load(str(ckpt), dtype=torch.bfloat16)
    ck = full.checkpoint
    r, state, questions = _rendered(full, request(IMAGES[:1]))
    pa = read(full.forward(state, questions, r.images), ck.readout, ck.config, r, stage="hidden")
    pb = read(reuse.forward(state, questions, r.images), ck.readout, ck.config, r, stage="hidden")
    assert pa["team"]["choice"] == pb["team"]["choice"]
    for name in ("billing", "shipping", "sales"):
        assert abs(pa["team"]["probabilities"][name] - pb["team"]["probabilities"][name]) < 0.05


# ---- the readout --------------------------------------------------------------------------


def test_rows_are_the_decoders_hidden_states_and_the_core_reads_them(ckpt, kind):
    backend = tb.TransformersBackend.load(
        str(ckpt), dtype=torch.bfloat16, reuse_state=False, images=IMAGING[kind]
    )
    ck = backend.checkpoint
    r, state, questions = _rendered(backend, request())
    rows = backend.forward(state, questions)
    # A bf16 decoder's values, unrounded and unscaled: float32 holds them exactly.
    assert np.array_equal(round_to_bf16(rows), rows)
    ids = torch.tensor([state + questions[0]])
    s, n = len(state), ids.shape[1]
    m = torch.from_numpy(attention_mask(s, n)).view(1, 1, n, n)
    with torch.inference_mode():
        h = backend.decoder(
            input_ids=ids, attention_mask={t: m for t in backend.layer_types}
        ).last_hidden_state[0, -1]
    assert np.array_equal(h.float().numpy(), rows[0])
    # The answers are the core's precision regime on those rows: the bf16 GEMM, then the cap
    # and the checkpoint's temperature in float32, against torch's own bf16 GEMM.
    got = evaluate(request(), ck.tokenizer(), ck.readout, ck.config, backend)["answers"]
    assert got == read(rows, ck.readout, ck.config, r, stage="hidden")
    w = torch.from_numpy(ck.readout.weight).to(torch.bfloat16)
    z = (w @ torch.from_numpy(rows[1]).to(torch.bfloat16)).float()
    np.testing.assert_allclose(
        z.numpy(), readout_logits(ck.readout.weight, rows[1]), atol=0.0625 + 1e-6
    )
    p = torch.softmax(torch.tanh(z[:3] / SOFTCAP) * SOFTCAP / TEMPERATURE, 0)
    probs = got["team"]["probabilities"]
    np.testing.assert_allclose([probs["billing"], probs["shipping"], probs["sales"]], p, atol=2e-4)


# ---- images -------------------------------------------------------------------------------


def test_image_features_land_at_the_spans(backends, kind, monkeypatch):
    _needs_images(kind)
    full, _ = backends
    r, state, questions = _rendered(full, request(IMAGES))
    encode = full.decoder.get_image_features
    marked = []

    def mark(*args, **kwargs):
        """The vision encoder's output, each image's vectors replaced by its number."""
        out = encode(*args, **kwargs)
        marked[:] = [torch.full_like(t, float(i + 1)) for i, t in enumerate(out.pooler_output)]
        out.pooler_output = tuple(marked)
        return out

    monkeypatch.setattr(full.decoder, "get_image_features", mark)
    seen = {}

    def grab(mod, args, kwargs):
        seen.update(kwargs)

    hook = full.decoder.language_model.register_forward_pre_hook(grab, with_kwargs=True)
    try:
        full.forward(state, questions[:1], r.images)
    finally:
        hook.remove()
    assert [t.shape[0] for t in marked] == [im.tokens for im in r.images]
    emb = seen["inputs_embeds"][0]
    for i, im in enumerate(r.images):
        assert torch.equal(emb[im.start : im.end], marked[i].to(emb.dtype))
    others = torch.ones(emb.shape[0], dtype=torch.bool)
    for im in r.images:
        others[im.start : im.end] = False
    assert not (emb[others] == 1).all(-1).any() and not (emb[others] == 2).all(-1).any()
    if seen.get("per_layer_inputs") is not None:  # E2B / E4B: the pad token's, at the images
        table = full.decoder.language_model.embed_tokens_per_layer
        pad = table(torch.tensor([[full.pad_id]]))[0, 0].view(seen["per_layer_inputs"].shape[2:])
        for im in r.images:
            assert torch.equal(seen["per_layer_inputs"][0, im.start], pad)


def test_the_image_decides_the_answer(backends, kind):
    """Two images of the same size give the same ids and different answers."""
    _needs_images(kind)
    full, _ = backends
    ck = full.checkpoint
    a = evaluate(request([image_b64(64, 48, seed=1)]), ck.tokenizer(), ck.readout, ck.config, full)
    b = evaluate(request([image_b64(64, 48, seed=2)]), ck.tokenizer(), ck.readout, ck.config, full)
    assert a["usage"] == b["usage"] and a["answers"] != b["answers"]


def test_spans_are_checked(backends, kind):
    _needs_images(kind)
    full, _ = backends
    r, state, questions = _rendered(full, request(IMAGES[:1]))
    im = r.images[0]
    import dataclasses

    shifted = dataclasses.replace(im, start=im.start + 1, tokens=im.tokens - 1)
    with pytest.raises(ValueError, match="span"):
        full.forward(state, questions, [shifted])
    with pytest.raises(ValueError, match="placeholders"):
        full.forward(state, questions)  # placeholders, but no image
    soft = full.image_ids[1]
    with pytest.raises(ValueError, match="question 0 holds a media placeholder"):
        full.forward(state, [questions[0] + [soft]], r.images)
    with pytest.raises(ValueError, match="vocabulary"):
        full.forward([10**7], questions)


# ---- refusals -----------------------------------------------------------------------------


def test_a_text_only_checkpoint_refuses_images_before_decoding_them(tmp_path):
    path = decision_checkpoint(tmp_path / "t", GEMMA4, decision={"inputs": ["text"]})
    backend = tb.TransformersBackend.load(str(path))
    assert backend.inputs == ("text",) and backend.decoder.vision_tower is None
    ck = backend.checkpoint
    with pytest.raises(ValueError, match=r'decision_config.json inputs are \["text"\]'):
        evaluate(request(["not even base64"]), ck.tokenizer(), ck.readout, ck.config, backend)
    assert evaluate(request(), ck.tokenizer(), ck.readout, ck.config, backend)["answers"]


def test_a_backend_without_its_vision_encoder_refuses_images(ckpt):
    backend = tb.TransformersBackend.load(str(ckpt), images=False)
    ck = backend.checkpoint
    assert backend.inputs == ("text",)
    with pytest.raises(ValueError, match=NO_VISION):
        evaluate(request(IMAGES[:1]), ck.tokenizer(), ck.readout, ck.config, backend)


@pytest.mark.parametrize(
    ("processor", "match"),
    [
        ({"max_soft_tokens": 560}, "'max_soft_tokens': 560"),
        ({"patch_size": 14}, "patch_size"),
        ({"image_processor_type": "SiglipImageProcessor"}, "names image processor"),
    ],
)
def test_an_image_processor_off_the_spec_refuses_images(tmp_path, processor, match):
    from strands_decider.decision import DecisionEngine

    path = decision_checkpoint(tmp_path / "p", GEMMA4, processor=processor)
    with pytest.warns(UserWarning, match=match):
        backend = tb.TransformersBackend.load(str(path))
    assert backend.inputs == ("text",) and match in backend.image_problem
    ck = backend.checkpoint
    engine = DecisionEngine(
        backend, ck.tokenizer(), ck.config, ck.readout, model_name="t",
        image_problem=backend.image_problem,
    )  # fmt: skip
    with pytest.raises(ValueError, match=match):
        engine.evaluate(request(IMAGES[:1]))
    assert engine.evaluate(request()).answers


def test_a_file_that_does_not_decode_is_refused_with_v1s_message(backends, kind):
    _needs_images(kind)
    full, _ = backends
    ck = full.checkpoint
    good = base64.b64decode(image_b64(64, 48))
    broken = good[:40] + b"\x00" * 64 + good[-12:]  # IHDR intact, image data not
    req = request([base64.b64encode(broken).decode()])
    with pytest.raises(ValueError, match="image could not be"):
        evaluate(req, ck.tokenizer(), ck.readout, ck.config, full)


def test_a_prompt_past_max_length_validated_is_refused(tmp_path):
    path = decision_checkpoint(tmp_path / "m", GEMMA4, decision={"max_length_validated": 64})
    backend = tb.TransformersBackend.load(str(path))
    ck = backend.checkpoint
    with pytest.raises(ValueError, match="exceeds the limit of 64 tokens"):
        evaluate(request(), ck.tokenizer(), ck.readout, ck.config, backend)


def test_config_json_must_name_the_interfaces_image_tokens(tmp_path, monkeypatch):
    """The model replaces config.json's image ids; the prompt's are the interface's. The
    backend compares the two without reading the tokenizer, which the engine checks."""
    path = decision_checkpoint(tmp_path / "k", GEMMA4)
    cfg = json.loads((path / "config.json").read_text())
    (path / "config.json").write_text(json.dumps({**cfg, "image_token_id": 3}))
    monkeypatch.setattr(tb.Checkpoint, "tokenizer", lambda self: pytest.fail("read the tokenizer"))
    with pytest.warns(UserWarning, match=r"image_token_id is 3, not the interface.s id"):
        backend = tb.TransformersBackend.load(str(path))
    assert backend.inputs == ("text",)


def test_a_config_json_without_a_hidden_size_is_named(tmp_path):
    path = decision_checkpoint(tmp_path / "h", GEMMA4)
    cfg = json.loads((path / "config.json").read_text())
    del cfg["text_config"]["hidden_size"]
    (path / "config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match=r"config\.json gives no hidden_size"):
        tb.TransformersBackend.load(str(path))


# ---- pixels -------------------------------------------------------------------------------


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


def _rendered_image(data: bytes):
    from strands_decider.core import decode_image
    from strands_decider.core.prompt import RenderedImage

    im = decode_image(base64.b64encode(data).decode())
    return RenderedImage(im.data, im.format, im.width, im.height, im.sha256, 1, im.tokens)


@pytest.mark.skipif(
    not hasattr(transformers, "Gemma4ImageProcessorPil"), reason="no Gemma4ImageProcessorPil"
)
@pytest.mark.parametrize(
    ("w", "h", "fmt"),
    [(64, 48, "PNG"), (640, 480, "JPEG"), (300, 40, "WEBP"), (1, 100, "PNG"), (1200, 900, "JPEG"),
     (768, 768, "PNG"), (97, 1301, "GIF"), (1080, 2400, "PNG")],
)  # fmt: skip
def test_the_spec_resize_then_the_processor_is_gemma4s_pil_processor(w, h, fmt):
    """docs/spec.md section 11.1, for E2B, E4B, 26B-A4B and 31B: the spec's bicubic resize
    (`resize_image`), then the processor without its own, gives the pixels the PIL processor
    gives with its own, wherever Pillow resizes the width first. Pillow 13 does the height
    first when it shrinks more than twice as much as the width (the 1080 x 2400 case), and its
    pixels then differ by a few levels: the spec's resize is the reference."""
    from PIL import Image

    rendered = _rendered_image(_photo(w, h, fmt, w * h))
    decoded = tb.decode(rendered)
    target = resized_size(*decoded.size)
    proc = transformers.Gemma4ImageProcessorPil()
    ours = proc(images=[tb.resize(rendered, decoded)], do_resize=False, return_tensors="pt")
    theirs = proc(images=[decoded], return_tensors="pt")
    assert torch.equal(ours["image_position_ids"], theirs["image_position_ids"])
    assert int(ours["num_soft_tokens_per_image"][0]) == rendered.tokens == soft_tokens(w, h)
    one_call = decoded.resize(target, Image.Resampling.BICUBIC)
    if np.array_equal(np.asarray(one_call), np.asarray(resize_image(decoded, *target))):
        assert torch.equal(ours["pixel_values"], theirs["pixel_values"])
    else:  # Pillow 13, height first: within a few levels, and ours is the spec's
        assert (ours["pixel_values"] - theirs["pixel_values"]).abs().max() <= 16 / 255


def test_an_image_turned_by_its_exif_is_resized_to_its_decoded_shape(backends, kind):
    """rotated-80x40.jpg: an 80 x 40 header, EXIF orientation 6, pixels 40 x 80, resized to
    528 x 1104, never to the header's 1104 x 528 (both give N = 253)."""
    _needs_images(kind)
    from core_support import image_bytes

    full, _ = backends
    rendered = _rendered_image(image_bytes("rotated-80x40.jpg"))
    assert (rendered.width, rendered.height) == (80, 40)
    assert tb.resize(rendered, tb.decode(rendered)).size == (528, 1104)
    positions = full.prepare_pixels([rendered])["image_position_ids"][0]
    valid = positions[(positions != -1).all(-1)]
    side = 16 if kind != UNIFIED else 48  # the 12B's processor merges 3 x 3 patches
    assert (int(valid[:, 0].max()) + 1, int(valid[:, 1].max()) + 1) == (528 // side, 1104 // side)


def test_a_decoded_size_that_is_not_the_headers_is_refused(backends, kind, monkeypatch):
    _needs_images(kind)
    from PIL import Image

    from strands_decider.decision import RequestRefused

    full, _ = backends
    r, _, _ = _rendered(full, request(IMAGES[:1]))
    monkeypatch.setattr(tb, "decode", lambda image: Image.new("RGB", (65, 48)))
    with pytest.raises(RequestRefused, match="image could not be decoded: its pixels are 65x48"):
        full.prepare_pixels(r.images)


@pytest.mark.skipif(not (HAS_UNIFIED and TORCHVISION), reason="needs the 12B's processor")
@pytest.mark.parametrize(("w", "h"), [(64, 48), (640, 480), (300, 40), (1, 100)])
def test_the_12bs_processor_counts_the_cores_n_from_the_spec_resize(w, h):
    from PIL import Image

    proc = transformers.Gemma4UnifiedImageProcessor()
    img = Image.open(io.BytesIO(_photo(w, h, "PNG", 3))).convert("RGB")
    out = proc(
        images=[resize_image(img, *resized_size(w, h))], do_resize=False, return_tensors="pt"
    )
    n = soft_tokens(w, h)
    assert int(out["num_soft_tokens_per_image"][0]) == n
    assert int((out["image_position_ids"][0] != -1).all(-1).sum()) == n


def test_prepare_pixels_checks_the_count(backends, kind, monkeypatch):
    _needs_images(kind)
    full, _ = backends
    r, _, _ = _rendered(full, request(IMAGES[:1]))
    import dataclasses

    wrong = dataclasses.replace(r.images[0], tokens=r.images[0].tokens + 1)
    with pytest.raises(RuntimeError, match="soft tokens"):
        full.prepare_pixels([wrong])


# ---- other devices ------------------------------------------------------------------------


def _agrees_with_cpu(ckpt, device: str) -> None:
    """The device against the CPU at each precision: float32 to float rounding, which checks
    the device's kernels; and bf16, its default, to bf16's own error, which on random weights
    is large (up to about 0.1 in probability on CPU too)."""
    assert tb.default_dtype(device) == torch.bfloat16
    for dtype, bound in ((torch.float32, 5e-3), (torch.bfloat16, 0.15)):
        cpu = tb.TransformersBackend.load(str(ckpt), dtype=dtype)
        other = tb.TransformersBackend.load(str(ckpt), device=device, dtype=dtype)
        ck = cpu.checkpoint
        for req in (request(), request(IMAGES[:1])):
            a = evaluate(req, ck.tokenizer(), ck.readout, ck.config, cpu)["answers"]
            b = evaluate(req, ck.tokenizer(), ck.readout, ck.config, other)["answers"]
            for name, p in a["team"]["probabilities"].items():
                assert abs(p - b["team"]["probabilities"][name]) < bound, (dtype, name)
            assert abs(a["late"]["noul"] - b["late"]["noul"]) < bound, dtype
            if dtype == torch.float32:
                assert a["team"]["choice"] == b["team"]["choice"]


@pytest.mark.skipif(not MPS, reason="needs Apple's MPS")
def test_mps_agrees_with_cpu(ckpt, kind):
    _needs_images(kind)
    _agrees_with_cpu(ckpt, "mps")


@pytest.mark.skipif(not MPS, reason="needs Apple's MPS")
def test_mps_state_reuse_is_the_full_pass(ckpt, kind):
    _needs_images(kind)
    full = tb.TransformersBackend.load(
        str(ckpt), device="mps", dtype=torch.float32, reuse_state=False
    )
    reuse = tb.TransformersBackend.load(str(ckpt), device="mps", dtype=torch.float32)
    r, state, questions = _rendered(full, request(IMAGES[:1]))
    a, b = full.forward(state, questions, r.images), reuse.forward(state, questions, r.images)
    assert np.abs(a - b).max() < 1e-4 * max(1.0, float(np.abs(a).max()))


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_agrees_with_cpu(ckpt, kind):
    _needs_images(kind)
    _agrees_with_cpu(ckpt, "cuda")


# E2B's attention layout: one key/value head shared by 8 query heads, heads of 256 and 512.
E2B_HEADS = {
    "hidden_size": 256,
    "num_attention_heads": 8,
    "num_key_value_heads": 1,
    "head_dim": 256,
    "global_head_dim": 512,
}


def _question_batches(backend, widths, seed=0):
    """A 300-token state and, for each width, three questions padded to that width in one
    batch over the state's keys and values."""
    g = np.random.default_rng(seed)
    vocab = backend.checkpoint.tokenizer().get_vocab_size()
    toks = lambda n: [int(t) for t in g.integers(10, vocab - 10, n)]  # noqa: E731
    state = toks(300)
    return state, [(w, [toks(w), toks(w - 7), toks(w - 30)]) for w in widths]


WIDTHS_64K_PLUS_1 = (64, 65, 66, 129, 193)


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
@pytest.mark.parametrize(("dtype", "bound"), [(torch.float32, 1e-3), (torch.bfloat16, 0.03)])
def test_cuda_question_batches_of_64k_plus_1_tokens_over_a_cached_state(tmp_path, dtype, bound):
    """torch 2.7's CUDA SDPA gave wrong attention (relative error up to 0.68) for E2B's layout
    in a forward that continues a cached state with 65, 129 or another 64k+1 query tokens;
    v1 serves E2B with eager attention for it. A batch of questions over a reused state is such
    a forward, its width the longest question's. Relative to the hidden states' scale, state
    reuse must stay within float rounding of the full passes."""
    path = decision_checkpoint(
        tmp_path / "e2b", GEMMA4, text=E2B_HEADS, decision={"inputs": ["text"]}
    )
    full = tb.TransformersBackend.load(str(path), device="cuda", dtype=dtype, reuse_state=False)
    reuse = tb.TransformersBackend.load(str(path), device="cuda", dtype=dtype)
    state, cases = _question_batches(reuse, WIDTHS_64K_PLUS_1)
    for width, qs in cases:
        assert reuse._batches(len(state), qs) == [[0, 1, 2]]  # one pass, `width` wide
        a, b = full.forward(state, qs), reuse.forward(state, qs)
        rel = float(np.abs(a - b).max()) / max(1.0, float(np.abs(a).max()))
        assert rel < bound, (width, rel)


# ---- a real checkpoint --------------------------------------------------------------------

REAL = os.environ.get("STRANDS_DECIDER_TEST_CHECKPOINT")


def _code_probabilities(ck, rows, k: int = 4) -> np.ndarray:
    """The first k codes' probabilities of each row, in the core's readout regime."""
    from strands_decider.core import code_probabilities

    z = readout_logits(ck.readout.weight, rows)
    return np.stack([
        code_probabilities(r, k, stage="raw_logits", softcap_c=ck.readout.softcap,
                           temperature=ck.temperature)
        for r in z
    ])  # fmt: skip


def _fits_in_float32(backend) -> bool:
    """Whether the bf16 model, cast to float32 in place, fits the GPU's free memory."""
    extra = sum(p.numel() for p in backend.model.parameters()) * 2  # bf16 -> float32
    return extra < 0.8 * torch.cuda.mem_get_info()[0]


@pytest.mark.skipif(not REAL, reason="set STRANDS_DECIDER_TEST_CHECKPOINT to a decision checkpoint")
def test_a_real_checkpoint():
    """A real decision checkpoint, on CUDA when there is one: it loads in the spec's layout,
    state reuse agrees with the full passes, and an image request runs where it takes images."""
    device = "cuda" if CUDA else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = tb.TransformersBackend.load(REAL, device=device, reuse_state=False)
    reuse = tb.TransformersBackend(
        full.model, full.checkpoint, device=device, dtype=full.dtype,
        image_processor=full.image_processor, image_problem=full.image_problem,
    )  # fmt: skip
    ck = full.checkpoint
    assert ck.config is not None, "a decision checkpoint, not an intermediate one"
    images = (
        [base64.b64encode(_photo(640, 480, "JPEG", 0)).decode()] if "image" in full.inputs else []
    )
    tok = ck.tokenizer()
    a = evaluate(request(images), tok, ck.readout, ck.config, full)
    b = evaluate(request(images), tok, ck.readout, ck.config, reuse)
    assert a["answers"]["team"]["choice"] == b["answers"]["team"]["choice"]
    # Question batches over the reused state at 64k+1 widths, on the real weights. In bf16 the
    # two paths differ by rounding (on an H100, up to 0.06 relative in the hidden state and
    # 0.04 in probability at widths 64 to 257, with no peak at 64k+1), so they are compared
    # through the readout; float32 is the kernel's check, where the model fits the device.
    state, cases = _question_batches(reuse, WIDTHS_64K_PLUS_1)
    for width, qs in cases:
        pa, pb = (_code_probabilities(ck, b.forward(state, qs)) for b in (full, reuse))
        assert np.abs(pa - pb).max() < (0.1 if full.dtype == torch.bfloat16 else 1e-3), width
    if CUDA and _fits_in_float32(full):
        del reuse
        f32 = tb.TransformersBackend(
            full.model.float(), ck, device=device, dtype=torch.float32, reuse_state=False
        )
        f32_reuse = tb.TransformersBackend(f32.model, ck, device=device, dtype=torch.float32)
        for width, qs in cases:
            ra, rb = f32.forward(state, qs), f32_reuse.forward(state, qs)
            rel = float(np.abs(ra - rb).max()) / max(1.0, float(np.abs(ra).max()))
            assert rel < 1e-3, (width, rel)


# ---- model goldens (docs/spec.md, section 10) ---------------------------------------------
#
# A golden_vectors.jsonl line, as section 10 gives it: {"request", "variant", "state_ids",
# "images": [{"sha256", "span"}], "questions": {name: {"question_ids", "codes",
# "probabilities"}}}. The core reads the reference's probabilities (`read_golden_vectors`) and
# measures a runtime against them (`agreement`, `Agreement.conforms`).

GOLDENS = "golden_vectors.jsonl"


def golden_probabilities(path, backend) -> dict:
    """`backend`'s code probabilities on the goldens at `path`, keyed as `read_golden_vectors`
    keys the reference's, after checking that every line renders to the goldens' ids, spans,
    image hashes and codes exactly (section 10, conformance condition 1)."""
    ck = backend.checkpoint
    tok = ck.tokenizer()
    out = {}
    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f):
            if not line.strip():
                continue
            g = json.loads(line)
            assert g["variant"] == ck.config.prompt, f"line {line_number}: another prompt"
            r = render(g["request"], tok, g["variant"])
            assert list(r.state_ids) == g["state_ids"], f"line {line_number}: state ids differ"
            spans = [{"sha256": im.sha256, "span": [im.start, im.end]} for im in r.images]
            assert spans == g["images"], f"line {line_number}: image spans differ"
            for name, q in r.questions.items():
                want = g["questions"][name]
                assert list(q.question_ids) == want["question_ids"], (
                    f"line {line_number}, {name}: rendered ids differ"
                )
                assert list(q.codes) == want["codes"], f"line {line_number}, {name}: codes"
            state, qs = list(r.state_ids), [list(q.question_ids) for q in r.questions.values()]
            rows = backend.forward(state, qs, r.images)
            probs = question_probabilities(rows, ck.readout, ck.config, r, stage="hidden")
            out.update({(line_number, n): [float(x) for x in p] for n, p in probs.items()})
    return out


def write_goldens(path, backend, requests) -> None:
    """Goldens in section 10's format, from `backend` (one full pass per prompt, as the
    reference runtime computes them)."""
    ck = backend.checkpoint
    tok = ck.tokenizer()
    with open(path, "w", encoding="utf-8") as f:
        for req in requests:
            r = render(req, tok, ck.config.prompt)
            state, qs = list(r.state_ids), [list(q.question_ids) for q in r.questions.values()]
            probs = question_probabilities(
                backend.forward(state, qs, r.images), ck.readout, ck.config, r, stage="hidden"
            )
            line = {
                "request": req,
                "variant": r.variant,
                "state_ids": list(r.state_ids),
                "images": [{"sha256": im.sha256, "span": [im.start, im.end]} for im in r.images],
                "questions": {
                    n: {
                        "question_ids": list(q.question_ids),
                        "codes": list(q.codes),
                        "probabilities": [float(x) for x in probs[n]],
                    }
                    for n, q in r.questions.items()
                },
            }
            f.write(json.dumps(line) + "\n")


def test_the_goldens_check_passes_its_own_goldens_and_catches_a_wrong_id(backends, kind, tmp_path):
    from strands_decider.core import agreement, read_golden_vectors

    full, reuse = backends
    reqs = [request(), request(state="")]
    if IMAGING[kind]:
        reqs.append(request(IMAGES))
    path = tmp_path / GOLDENS
    write_goldens(path, full, reqs)
    reference = read_golden_vectors(path)
    assert len(reference) == 3 * len(reqs)
    for backend in (full, reuse):
        measured = agreement(golden_probabilities(path, backend), reference)
        assert measured.questions == len(reference)
        assert measured.top1 == 1.0 and measured.mean_abs_dp < 1e-4
        # bf16's bounds are met, but on too few questions for a conformance claim.
        assert not measured.conforms()
    lines = path.read_text().splitlines()
    g = json.loads(lines[0])
    g["questions"]["late"]["question_ids"][-1] += 1
    path.write_text("\n".join([json.dumps(g), *lines[1:]]) + "\n")
    with pytest.raises(AssertionError, match="rendered ids differ"):
        golden_probabilities(path, full)


@pytest.mark.skipif(not REAL, reason="set STRANDS_DECIDER_TEST_CHECKPOINT to a decision checkpoint")
def test_a_real_checkpoint_meets_its_goldens():
    """Section 10: the rendered ids are the goldens', and in bf16 the runtime conforms (top-1
    agreement at least 98% and mean |dp| at most 0.02, on at least 1,000 questions), both with
    one full pass per prompt (the reference) and with the state reused."""
    from strands_decider.core import agreement, read_golden_vectors

    path = os.path.join(tb.Checkpoint.open(REAL).path, GOLDENS)
    if not os.path.exists(path):
        pytest.skip(f"the checkpoint has no {GOLDENS} yet")
    device = "cuda" if CUDA else "cpu"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = tb.TransformersBackend.load(
            REAL, device=device, dtype=torch.bfloat16, reuse_state=False
        )
    reuse = tb.TransformersBackend(
        full.model, full.checkpoint, device=device, dtype=full.dtype,
        image_processor=full.image_processor, image_problem=full.image_problem,
    )  # fmt: skip
    reference = read_golden_vectors(path)
    for backend in (full, reuse):
        measured = agreement(golden_probabilities(path, backend), reference)
        assert measured.conforms(), (backend.reuse_state, measured)


# ---- the 12B's pixels ---------------------------------------------------------------------


@pytest.mark.skipif(not (HAS_UNIFIED and TORCHVISION), reason="needs the 12B's processor")
@pytest.mark.parametrize(
    ("w", "h", "fmt"), [(640, 480, "JPEG"), (300, 40, "PNG"), (97, 1301, "WEBP")]
)
def test_the_12bs_pixels_are_pillows_resize_in_48_pixel_patches(tmp_path, w, h, fmt):
    """Through `prepare_pixels`: each of the N patches is the Pillow-bicubic-resized image's
    48 x 48 block at the patch's position, in [0, 1], rows then columns then channels; the
    padding is zero."""
    path = decision_checkpoint(tmp_path / "u", UNIFIED)
    backend = tb.TransformersBackend.load(str(path))
    r, _, _ = _rendered(backend, request([base64.b64encode(_photo(w, h, fmt, 5)).decode()]))
    im = r.images[0]
    px = backend.prepare_pixels([im])
    values, positions = px["pixel_values"][0].numpy(), px["image_position_ids"][0].numpy()
    arr = np.asarray(tb.resize(im, tb.decode(im)), dtype=np.float32) / 255
    assert arr.shape[:2] == tuple(reversed(resized_size(w, h)))
    valid = (positions != -1).all(-1)
    assert int(valid.sum()) == im.tokens == soft_tokens(w, h)
    for j in np.flatnonzero(valid):
        x, y = positions[j]
        block = arr[y * 48 : (y + 1) * 48, x * 48 : (x + 1) * 48].reshape(-1)
        np.testing.assert_allclose(values[j], block, atol=1e-6)
    assert not values[~valid].any()


# ---- load-time checks ---------------------------------------------------------------------


def test_an_image_processor_that_cannot_be_built_refuses_images_only(tmp_path, monkeypatch):
    path = decision_checkpoint(tmp_path / "p", GEMMA4)

    def missing(*_a, **_k):
        raise OSError(f"{path}/processor_config.json is unreadable")

    monkeypatch.setattr(tb, "load_image_processor", missing)
    with pytest.warns(UserWarning, match="processor_config.json is unreadable"):
        backend = tb.TransformersBackend.load(str(path))
    # Clients get a fixed reason; the error, with its local path, is only in the warning.
    assert backend.inputs == ("text",) and backend.image_problem == tb.PROCESSOR_FAILED
    assert str(path) not in backend.image_problem
    ck = backend.checkpoint
    assert evaluate(request(), ck.tokenizer(), ck.readout, ck.config, backend)["answers"]


def test_a_readout_of_the_wrong_width_is_refused_before_the_weights_load(tmp_path, monkeypatch):
    from safetensors.torch import save_file

    path = decision_checkpoint(tmp_path / "w", GEMMA4)
    save_file(
        {"weight": torch.zeros(255, 32, dtype=torch.bfloat16)}, str(path / "readout.safetensors")
    )
    monkeypatch.setattr(tb, "_load_model", lambda *a, **k: pytest.fail("loaded the weights"))
    with pytest.raises(ValueError, match=r"not \[255, 64\]"):
        tb.TransformersBackend.load(str(path))


def test_dtypes_are_chosen_and_checked(ckpt, kind):
    backend = tb.TransformersBackend.load(str(ckpt), dtype=torch.bfloat16, images=IMAGING[kind])
    assert backend.describe()["dtype"] == "bfloat16"
    with pytest.raises(ValueError, match="float32 or bfloat16"):
        tb.TransformersBackend.load(str(ckpt), dtype=torch.float16)
    with pytest.raises(ValueError, match="cpu, cuda or mps"):
        tb.TransformersBackend.load(str(ckpt), device="meta", dtype=torch.float32)


# ---- refusals and faults, and concurrency -------------------------------------------------


def _engine(backend, **kwargs):
    from strands_decider.decision import DecisionEngine

    ck = backend.checkpoint
    return DecisionEngine(
        backend, ck.tokenizer(), ck.config, ck.readout, model_name="t",
        image_problem=backend.image_problem, **kwargs,
    )  # fmt: skip


def test_refusals_are_the_callers_and_failures_the_runtimes(backends, kind, monkeypatch):
    from strands_decider.decision import RequestRefused

    full, _ = backends
    engine = _engine(full)
    with pytest.raises(RequestRefused, match="audio input is not supported"):
        engine.evaluate({**request(), "audio": ["UklGRg=="]})
    with pytest.raises(RequestRefused, match="exceeds the limit"):
        engine.evaluate(request(state="word " * 5000))
    if IMAGING[kind]:
        good = base64.b64decode(image_b64(64, 48))
        broken = base64.b64encode(good[:40] + b"\x00" * 64 + good[-12:]).decode()
        with pytest.raises(RequestRefused, match="image could not be"):
            engine.evaluate(request([broken]))

    def fails(*_a, **_k):
        raise ValueError("an internal check")

    monkeypatch.setattr(full, "forward", fails)
    with pytest.raises(RuntimeError, match="an internal check"):
        engine.evaluate(request())


class _Peak:
    """Wraps a callable and records how many calls ran inside it at once."""

    def __init__(self, inner, hold: float = 0.05):
        import threading

        self.inner, self.hold = inner, hold
        self.inside = self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, *args, **kwargs):
        import time

        with self.lock:
            self.inside += 1
            self.peak = max(self.peak, self.inside)
        try:
            time.sleep(self.hold)
            return self.inner(*args, **kwargs)
        finally:
            with self.lock:
                self.inside -= 1


def _threads(fn, n: int = 6) -> list:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n) as pool:
        return list(pool.map(lambda _: fn(), range(n)))


@pytest.mark.parametrize("slots", [1, 3])
def test_the_engine_bounds_concurrent_requests(backends, monkeypatch, slots):
    full, _ = backends
    engine = _engine(full, max_concurrency=slots)
    want = engine.evaluate(request()).model_dump()
    peak = _Peak(full.forward)
    monkeypatch.setattr(full, "forward", peak)
    got = _threads(lambda: engine.evaluate(request()).model_dump())
    assert all(g == want for g in got)
    # One slot: never two at once. More: never past the bound (overlap depends on timing).
    assert peak.peak == 1 if slots == 1 else 1 <= peak.peak <= slots


@pytest.mark.skipif(not MPS, reason="needs Apple's MPS")
def test_mps_calls_take_turns_across_backends(ckpt, kind, monkeypatch):
    """Metal aborts on two threads at once, whichever backend each runs: two backends over
    one model on MPS share one lock."""
    one = tb.TransformersBackend.load(str(ckpt), device="mps", images=IMAGING[kind])
    two = tb.TransformersBackend(
        one.model, one.checkpoint, device="mps", dtype=one.dtype, reuse_state=False
    )
    _, state, questions = _rendered(one, request())
    want = [b.forward(state, questions) for b in (one, two)]
    peak = _Peak(lambda f, *a, **k: f(*a, **k), hold=0.02)
    for b in (one, two):
        monkeypatch.setattr(b, "_decode", lambda *a, _f=b._decode, **k: peak(_f, *a, **k))
    got = _threads(lambda: [b.forward(state, questions) for b in (one, two)], n=4)
    assert peak.peak == 1
    assert all(np.array_equal(g[i], want[i]) for g in got for i in (0, 1))


def test_an_intermediate_checkpoints_float32_readout_is_rounded_to_bf16(tmp_path):
    """Section 7.1: a runtime that reads an f32 readout rounds it to bf16 first."""
    from safetensors.torch import save_file

    path = intermediate_checkpoint(tmp_path / "mid")
    w = torch.randn(255, 64, generator=torch.Generator().manual_seed(3))
    save_file({"weight": w}, str(path / "readout.safetensors"))
    ck = tb.Checkpoint.open(str(path))
    assert ck.config is None and ck.warnings  # an intermediate checkpoint, its notes kept
    assert np.array_equal(ck.readout.weight, round_to_bf16(w.numpy()))
    assert not np.array_equal(ck.readout.weight, w.numpy())
