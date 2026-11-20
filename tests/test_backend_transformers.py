"""The transformers backend on tiny random-weight Gemma 4 checkpoints (tests/decision_support.py):
no download, CPU in CI. The CUDA and MPS tests skip without them; the real-checkpoint tests need
STRANDS_DECIDER_TEST_CHECKPOINT, a model checkpoint."""

from __future__ import annotations

import contextlib
import json
import math
import os
import warnings

import numpy as np
import pytest
import torch
import transformers

from decision_support import GEMMA4, KINDS, model_checkpoint, image_b64, request  # isort: skip

from strands_decider.backends.images import image_pixels
from strands_decider.backends.transformers import TransformersBackend, load_backend
from strands_decider.backends.transformers_attention import ATTENTION, StateCache, StateSplit
from strands_decider.core import (
    DecisionConfig,
    Request,
    attention_mask,
    evaluate,
    load_readout,
    load_tokenizer,
    render,
)
from strands_decider.modeling import checkpoint_dir

CUDA = torch.cuda.is_available()
REAL = os.environ.get("STRANDS_DECIDER_TEST_CHECKPOINT")
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")
needs_real = pytest.mark.skipif(not REAL, reason="set STRANDS_DECIDER_TEST_CHECKPOINT")
# 80 x 40 resizes to 1104 x 528, which the image processor's own resize would take to 1152 x 528,
# so that its vectors would not fill its placeholders: the processor runs with its resize off.
IMAGES = [image_b64(64, 48), image_b64(80, 40, "JPEG", 1)]
# E2B's attention layout: one key/value head shared by 8 query heads, heads of 256 and 512.
E2B_HEADS = {"hidden_size": 256, "num_attention_heads": 8, "head_dim": 256, "global_head_dim": 512}
WIDTHS_64K_PLUS_1 = (64, 65, 66, 129, 193)


@pytest.fixture(scope="module", params=KINDS)
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def ckpt(kind, tmp_path_factory):
    return model_checkpoint(tmp_path_factory.mktemp(kind) / "ckpt", kind)


@pytest.fixture(scope="module")
def e2b(tmp_path_factory):
    """A text-only checkpoint with E2B's attention layout."""
    path = tmp_path_factory.mktemp("e2b") / "ckpt"
    return model_checkpoint(path, GEMMA4, text=E2B_HEADS, decision={"inputs": ["text"]})


def _reusing(backend, **settings):
    """`backend`'s model, with the state encoded once."""
    return TransformersBackend(backend.model, image_processor=backend.image_processor, **settings)


@pytest.fixture(scope="module")
def backends(ckpt):
    """(full, reuse), float32 on CPU: a full pass per question, and the state once with the
    questions in batches of two."""
    full = load_backend(str(ckpt), reuse_state=False)
    return full, _reusing(full, max_batch=2)


@pytest.fixture
def sdpa_calls(monkeypatch):
    """(rows, queries, keys, mask) of each call to torch's SDPA."""
    calls = []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def record(query, key, value, attn_mask=None, **kwargs):
        calls.append((query.shape[0], query.shape[2], key.shape[2], attn_mask))
        return sdpa(query, key, value, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", record)
    return calls


def _prompt(path, req: dict):
    """The state's ids, each question's ids and the images, as the core renders `req`."""
    r = render(Request.model_validate(req), load_tokenizer(path))
    return list(r.state_ids), [list(q.question_ids) for q in r.questions.values()], r.images


def _numbers(path, req: dict, backend) -> list[float]:
    """Every noul, score and probability the core answers `req` with through `backend`."""
    tokenizer, config, weight = load_tokenizer(path), DecisionConfig.load(path), load_readout(path)
    answers = evaluate(Request.model_validate(req), tokenizer, weight, config, backend)["answers"]
    return [
        v
        for a in answers.values()
        for v in (a.get("noul"), a.get("score"), *a.get("probabilities", {}).values())
        if v is not None
    ]


def _random_ids(path, *lengths: int, seed: int = 1) -> list[list[int]]:
    vocab, g = load_tokenizer(path).get_vocab_size(), np.random.default_rng(seed)
    return [[int(t) for t in g.integers(10, vocab - 10, n)] for n in lengths]


def _rel(ref, got) -> float:
    """The largest difference, relative to the reference's scale."""
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(np.abs(ref - got).max()) / max(1.0, float(np.abs(ref).max()))


@contextlib.contextmanager
def _stock_sdpa():
    """transformers' own SDPA attention in place of the backend's; given a mask, it is never
    causal (transformers 5.15 to 5.23)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS[ATTENTION] = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        yield
    finally:
        del ALL_ATTENTION_FUNCTIONS[ATTENTION]  # the override; the registration stays


def _reference(backend, state, question, images=(), mask=None) -> np.ndarray:
    """The final-norm hidden state at the prompt's last token from transformers' own SDPA under
    `mask`, n x n booleans: by default the core's dense prefix_bidir mask."""
    n, dev = len(state) + len(question), backend.device
    dense = torch.from_numpy(attention_mask(len(state), n) if mask is None else mask)
    pixels = {}
    if images:
        out = backend.image_processor(
            images=[image_pixels(im) for im in images], do_resize=False, return_tensors="pt"
        )
        pixels = {
            "pixel_values": out["pixel_values"].to(dev, backend.model.dtype),
            "image_position_ids": out["image_position_ids"].to(dev),
        }
    layer_types = backend.model.config.get_text_config().layer_types
    with torch.inference_mode(), _stock_sdpa():
        hidden = backend.model.model(
            input_ids=torch.tensor([state + question], device=dev),
            attention_mask={t: dense.view(1, 1, n, n).to(dev) for t in layer_types},
            position_ids=torch.arange(n, device=dev).view(1, -1),
            **pixels,
        ).last_hidden_state
    return hidden[0, -1].float().cpu().numpy()


def _check_against_the_dense_mask(path, backend, req: dict, bound: float) -> None:
    """`forward` against the reference, for a question of `req` and a 65-token one; with every
    query seeing every key, the reference moves by more than `bound`."""
    state, questions, images = _prompt(path, req)
    for question in (questions[0], *_random_ids(path, 65)):
        ref = _reference(backend, state, question, images)
        assert _rel(ref, backend.forward(state, [question], images)[0]) < bound
        everything = np.ones((len(state) + len(question),) * 2, dtype=bool)
        assert _rel(ref, _reference(backend, state, question, images, everything)) > bound


@pytest.mark.parametrize(("dtype", "bound"), [("float32", 2e-5), ("bfloat16", 0.2)])
@pytest.mark.parametrize("images", [0, 1], ids=["text", "image"])
def test_full_passes_are_the_cores_dense_mask(ckpt, kind, images, dtype, bound):
    """At the last token on CPU, bf16's rounding reaches 0.039 with torch 2.7 and 0.016 with 2.14;
    a wrong mask moves it by 0.246 at least. The window of 2 is shorter than the prompt: no layer
    keeps it."""
    backend = load_backend(str(ckpt), dtype=dtype, reuse_state=False)
    _check_against_the_dense_mask(ckpt, backend, request(IMAGES[:images]), bound)


def test_full_passes_are_the_cores_dense_mask_with_e2bs_heads(e2b):
    """Keys and values repeated per query head, head_dim 512; float32 (3e-5 at most with torch
    2.7's CPU kernels)."""
    _check_against_the_dense_mask(e2b, load_backend(str(e2b), reuse_state=False), request(), 1e-4)


def test_a_checkpoint_the_backend_cannot_run_is_refused(tmp_path):
    """Another architecture is refused by name, and a missing tensor rather than initialised at
    random, as transformers would."""
    from safetensors.torch import load_file, save_file

    other = model_checkpoint(tmp_path / "other", decision={"inputs": ["text"]})
    (other / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="not a Gemma 4 model checkpoint"):
        load_backend(str(other))
    short = model_checkpoint(tmp_path / "short", decision={"inputs": ["text"]})
    tensors = load_file(short / "model.safetensors")
    del tensors[next(k for k in tensors if k.endswith("norm.weight"))]
    save_file(tensors, short / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="the checkpoint lacks 1 tensors"):
        load_backend(str(short))


def test_a_state_split_and_cache_keep_the_whole_state():
    """transformers 5.23 slices each layer's mask whole, and a slice that dropped tokens would
    shift the split; the cache serves the state's keys and values ahead of each question batch."""
    split = StateSplit(3)
    assert split[:, :, 0:, :] is split
    with pytest.raises(TypeError, match="keeps every token"):
        split[:, :, 1:, :]
    cache, state = StateCache(), torch.randn(1, 2, 3, 4)
    assert cache.update(state, state, 0) == (state, state)
    cache.freeze(3)
    question = torch.randn(5, 2, 1, 4)
    keys, _ = cache.update(question, question, 0)
    assert keys.shape == (5, 2, 4, 4) and torch.equal(keys[:, :, :3], state.expand(5, -1, -1, -1))
    assert cache.get_seq_length() == 3


def test_a_model_that_keeps_transformers_sdpa_is_refused(e2b, monkeypatch):
    """transformers only warns when a model can't switch its attention, and its SDPA with no mask
    is causal: the load refuses such a model rather than serve wrong answers."""
    monkeypatch.setattr(transformers.PreTrainedModel, "set_attn_implementation", lambda *a: None)
    with pytest.raises(RuntimeError, match="did not set the decoder's attention"):
        load_backend(str(e2b))


def test_no_layer_gets_a_dense_mask(backends, ckpt, sdpa_calls):
    """A full pass runs the state's queries over its keys, then the question's over every key,
    unmasked. Over a reused state, a batch with padding masks key columns, a row per question.
    No n x n mask, which is what lets a longer state fit."""
    full, reuse = backends
    state, questions, _ = _prompt(ckpt, request())
    s, layers = len(state), len(full.model.config.get_text_config().layer_types)
    full.forward(state, questions[:1])
    n = s + len(questions[0])
    assert [call[1:] for call in sdpa_calls] == [(s, s, None), (n - s, n, None)] * layers
    sdpa_calls.clear()
    reuse.forward(state, questions)
    assert [call[1:] for call in sdpa_calls[:layers]] == [(s, s, None)] * layers
    for b, batch in enumerate([questions[:2], questions[2:]]):
        width = max(map(len, batch))
        for rows, q, k, mask in sdpa_calls[layers * (b + 1) : layers * (b + 2)]:
            assert (rows, q, k) == (len(batch), width, s + width)
            sees = torch.ones(len(batch), s + width, dtype=torch.bool)
            for row, x in enumerate(batch):  # the state and its own question, never the padding
                sees[row, s + len(x) :] = False
            assert torch.equal(sees, torch.ones_like(sees) if mask is None else mask[:, 0, 0])


@pytest.mark.parametrize("images", [0, 2], ids=["text", "two-images"])
def test_state_reuse_is_the_full_pass(backends, ckpt, kind, images):
    """float32 on CPU, to float rounding."""
    full, reuse = backends
    state, questions, imgs = _prompt(ckpt, request(IMAGES[:images]))
    a, b = full.forward(state, questions, imgs), reuse.forward(state, questions, imgs)
    assert a.dtype == np.float32 and a.shape == (3, 64)
    assert _rel(a, b) < 2e-5


def test_question_batches_keep_to_their_limits(backends, ckpt, sdpa_calls):
    """A pass over the state holds at most `max_batch` questions, and rows x (S + the longest
    question) at most `max_batch_tokens` tokens, one question at least."""
    full, _ = backends
    state, questions, _ = _prompt(ckpt, request())
    s, width = len(state), max(map(len, questions))
    layers = len(full.model.config.get_text_config().layer_types)

    def batches(**limits):
        sdpa_calls.clear()
        _reusing(full, **limits).forward(state, questions * 2)
        return [rows for rows, q, k, _ in sdpa_calls if k > q][::layers]

    assert batches(max_batch=4) == [4, 2]
    assert batches(max_batch_tokens=3 * (s + width)) == [3, 3]
    assert batches(max_batch_tokens=1) == [1] * 6


def test_no_pass_over_the_state_is_64k_plus_1_queries_wide(backends, ckpt, sdpa_calls):
    """The torch 2.7 CUDA workaround, where CI runs: a batch whose longest question is 65 tokens
    runs 66 wide, the extra key column masked, and a full pass takes the state's last query too.
    This is its guard: with it removed, the CUDA tests pass on torch 2.7.1+cu126 on an H100."""
    full, reuse = backends
    state, _, _ = _prompt(ckpt, request())
    long, short, other = _random_ids(ckpt, 65, 40, 65)
    reuse.forward(state, [long, short])
    full.forward(state, [long])
    reuse.forward(state, [long, other])  # no shorter question: the extra column is still masked
    assert {q for _, q, k, _ in sdpa_calls if k > q} == {66}
    s, mask = len(state), sdpa_calls[-1][3]
    assert mask.shape == (2, 1, 1, s + 66)
    assert mask[..., : s + 65].all() and not mask[..., s + 65 :].any()


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cuda", marks=needs_cuda),
        pytest.param(
            "mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no MPS")
        ),
    ],
)
def test_a_device_agrees_with_the_cpu(ckpt, kind, device):
    """float32 to float rounding, the device's kernels' check, reusing the state as in full
    passes; bf16 to its own error, large on random weights (about 0.1 in probability on CPU)."""
    for dtype, bound in (("float32", 5e-3), ("bfloat16", 0.15)):
        cpu = load_backend(str(ckpt), dtype=dtype)
        other = load_backend(str(ckpt), device=device, dtype=dtype)
        for req in (request(), request(IMAGES[:1])):
            assert _rel(_numbers(ckpt, req, cpu), _numbers(ckpt, req, other)) < bound, dtype
    full = load_backend(str(ckpt), device=device, dtype="float32", reuse_state=False)
    state, questions, images = _prompt(ckpt, request(IMAGES[:1]))
    got = _reusing(full).forward(state, questions, images)
    assert _rel(full.forward(state, questions, images), got) < 1e-4


def _question_batches(path, widths, state_len: int):
    """A `state_len`-token state, and for each width three questions that pad to it in a pass."""
    lengths = [n for w in widths for n in (w, w - 7, w - 30)]
    state, *questions = _random_ids(path, state_len, *lengths, seed=0)
    return state, [questions[i : i + 3] for i in range(0, len(questions), 3)]


@needs_cuda
def test_cuda_question_batches_of_64k_plus_1_tokens(e2b):
    """torch 2.7's CUDA SDPA gets E2B's layout wrong in passes of 64k+1 queries (65, 129, ...,
    385): under a dense mask the hidden states were off by up to 0.84 (float32, H100), though an
    isolated call on random inputs is right. Full passes and state reuse against one CPU pass per
    prompt under the core's dense mask: within 1.7e-4, torch 2.7 and 2.14. float32 only: in bf16
    this model lies up to 0.78 from float32 with any mask; the next test checks bf16 per call."""
    cpu = load_backend(str(e2b), reuse_state=False)
    full = load_backend(str(e2b), device="cuda", dtype="float32", reuse_state=False)
    for state_len in (300, 321):
        state, batches = _question_batches(e2b, WIDTHS_64K_PLUS_1, state_len)
        for qs in batches:
            ref = np.stack([_reference(cpu, state, q) for q in qs])
            for got in (full.forward(state, qs), _reusing(full).forward(state, qs)):
                assert _rel(ref, got) < 1e-3, (state_len, len(qs[0]))


@needs_cuda
@pytest.mark.parametrize(("dtype", "bound"), [("bfloat16", 0.02), ("float32", 1e-4)])
def test_cuda_every_attention_call_against_float64(e2b, monkeypatch, dtype, bound):
    """Every attention call E2B's layout makes on CUDA (head_dim 256 and 512; full passes, whole
    prompts of 385 included, and batches over a reused state, padded or not) against float64 on
    its own inputs: at worst 0.0038 in bf16 and 9e-6 in float32 on torch 2.7 and 2.14, where
    transformers' SDPA under the dense mask was off by 0.97 and 0.93 at 385 tokens on 2.7."""
    errors, unpadded_batches = [], []
    sdpa = torch.nn.functional.scaled_dot_product_attention

    def checked(query, key, value, attn_mask=None, scale=None, **kwargs):
        out = sdpa(query, key, value, attn_mask=attn_mask, scale=scale, **kwargs)
        groups = query.shape[1] // key.shape[1]
        k, v = (x.double().repeat_interleave(groups, dim=1) for x in (key, value))
        scale = query.shape[-1] ** -0.5 if scale is None else scale
        scores = query.double() @ k.transpose(-1, -2) * scale
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask, -torch.inf)
        ref = scores.softmax(-1) @ v
        errors.append(float((out.double() - ref).abs().max() / ref.abs().max()))
        unpadded_batches.append(attn_mask is None and query.shape[0] > 1)
        return out

    full = load_backend(str(e2b), device="cuda", dtype=dtype, reuse_state=False)
    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", checked)
    for state_len in (300, 321):
        state, batches = _question_batches(e2b, (64, 65, 129), state_len)
        for qs in batches:
            full.forward(state, qs)
            _reusing(full).forward(state, qs)
        q = batches[0][0]  # 64 tokens, so no 64k+1 column either
        _reusing(full).forward(state, [q, q[::-1]])  # one length: no padding, no mask
    assert any(unpadded_batches) and all(map(math.isfinite, errors))
    assert max(errors) < bound, (dtype, max(errors))


def _real_backend(**settings):
    path = checkpoint_dir(REAL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        backend = load_backend(
            path, device="cuda" if CUDA else "cpu", reuse_state=False, **settings
        )
    return path, backend


@needs_real
def test_a_real_checkpoint_reuses_its_states_as_full_passes():
    """An image request where the checkpoint takes images, in its device's dtype: every noul,
    score and probability within 0.1 of the full passes'. Measured on an H100 in bf16: 0.032 at
    12B with torch 2.14 and 0.020 with 2.7.1; 0 at E2B with 2.14 and 0.0067 with 2.7.1."""
    path, full = _real_backend()
    req = request([image_b64(640, 480, "JPEG")] if full.takes_images else [])
    a, b = _numbers(path, req, full), _numbers(path, req, _reusing(full))
    assert np.abs(np.subtract(a, b)).max() < 0.1


@needs_real
def test_a_real_checkpoint_reuses_its_states_as_full_passes_in_float32():
    """Question batches over a reused state at 64k+1 widths against full passes, in float32: in
    bf16 these random-token prompts differ by up to 0.4 in probability on a 12B (torch 2.14),
    dense masks or not. Skips where the model doesn't fit in float32 (an 80 GB GPU: 12B, not 31B)."""
    path, full = _real_backend()
    if full.model.dtype == torch.bfloat16:
        extra = sum(p.numel() for p in full.model.parameters()) * 2  # bf16 -> float32
        if extra > 0.8 * torch.cuda.mem_get_info()[0]:
            pytest.skip("the model does not fit the GPU in float32")
        full.model.float()
    for state_len in (300, 321):
        state, batches = _question_batches(path, WIDTHS_64K_PLUS_1, state_len)
        for qs in batches:
            got = _reusing(full).forward(state, qs)
            assert _rel(full.forward(state, qs), got) < 1e-3, (state_len, len(qs[0]))
