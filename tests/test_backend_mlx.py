"""The MLX backend for decision checkpoints (strands_decider.backends.mlx and mlx_convert).

Tiny random-weight Gemma 4 models are written inside the test in the spec's multimodal
layout (docs/spec.md section 9), with transformers' own classes: `Gemma4ForConditionalGeneration`
shaped like E2B (sliding and full layers, KV sharing, per-layer inputs, a vision and an audio
encoder) and `Gemma4UnifiedForConditionalGeneration` shaped like the 12B (an encoder-free
patch embedder, k = v full layers, Gemma 4's bidirectional vision mode, an audio projection
only). Each has config.json in the bases' own form, a toy tokenizer, a readout and a
decision_config.json. Sliding windows are 4 tokens, so a window left on would show.

The reference is the spec's: transformers' model in float32, one prompt per pass, the dense
prefix_bidir mask passed for both layer types (with SDPA, which reads a boolean mask as the
keys each token sees), and for E2B the pixels prepared by Pillow's bicubic resize and
transformers' PIL image processor with its resize off. (The 12B's processor in transformers
needs torchvision, so its reference takes the backend's pixels and checks the model alone.)
Through the decision engine (`decision.load_decision_engine(..., device="mlx")`) and `serve`,
requests are checked and refused as on every backend, and the transformers backend's own tiny
checkpoints get the same answers from both backends. Apple silicon with the mlx extra only;
nothing is downloaded. tests/test_backend_mlx_routing.py checks the routing on any platform.
"""

from __future__ import annotations

import base64
import io
import json
import platform
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX runs on Apple silicon only", allow_module_level=True)
mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("PIL")
if not hasattr(transformers, "Gemma4UnifiedForConditionalGeneration"):
    pytest.skip("this transformers has no Gemma 4", allow_module_level=True)

from core_support import (  # noqa: E402
    IMAGE_FILES,
    SOFT_TOKENS,
    decision_config_fixture,
    image_b64,
    image_bytes,
    toy_tokenizer,
)
from PIL import Image  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from strands_decider import cli  # noqa: E402
from strands_decider.backends.mlx import (  # noqa: E402
    decode_image,
    load_backend,
    prepare_image,
    resized_size,
)
from strands_decider.backends.mlx_convert import convert, quantized_parts  # noqa: E402
from strands_decider.core import (  # noqa: E402
    CODES,
    HIDDEN,
    DecisionConfig,
    Readout,
    question_probabilities,
    render,
    round_to_bf16,
    soft_tokens,
)
from strands_decider.decision import DecisionEngine, RequestRefused, load_decision_engine  # noqa: E402
from strands_decider.schema import NoulQuestion, SystemOneRequest, SystemOneResponse  # noqa: E402

V = 512  # vocabulary: the toy tokenizer's ids, and room for the audio and video tokens
TOK = toy_tokenizer()
IMAGE_IDS = {
    "boi_token_id": TOK.token_to_id("<|image>"),
    "image_token_id": TOK.token_to_id("<|image|>"),
    "eoi_token_id": TOK.token_to_id("<image|>"),
    "audio_token_id": V - 1,
    "boa_token_id": V - 2,
    "eoa_token_index": V - 3,
    "video_token_id": V - 4,
}
LAYERS = ["sliding_attention", "sliding_attention", "full_attention"] * 2
ROPE = {
    "full_attention": {
        "partial_rotary_factor": 0.25,
        "rope_theta": 1e6,
        "rope_type": "proportional",
    },
    "sliding_attention": {"rope_theta": 1e4, "rope_type": "default"},
}
IMAGE_PROCESSOR = {
    "do_convert_rgb": True,
    "do_normalize": False,
    "do_rescale": True,
    "do_resize": True,
    "image_mean": [0.0, 0.0, 0.0],
    "image_std": [1.0, 1.0, 1.0],
    "image_seq_length": 280,
    "max_soft_tokens": 280,
    "patch_size": 16,
    "pooling_kernel_size": 3,
    "resample": 3,
    "rescale_factor": 1 / 255,
}
KINDS = ("e2b", "unified")


def _base_style(config: dict, unified: bool) -> dict:
    """config.json as the Gemma 4 bases write it, which mlx-vlm reads: transformers 5.22
    writes the full-attention layers' head size and KV heads under `per_layer_config`; the
    bases give `global_head_dim` and `num_global_key_value_heads`."""
    text = config["text_config"]
    for layer in (text.pop("per_layer_config", None) or {}).values():
        if "head_dim" in layer:
            text["global_head_dim"] = layer["head_dim"]
        if "num_key_value_heads" in layer:
            text["num_global_key_value_heads"] = layer["num_key_value_heads"]
    if unified:
        text.setdefault("hidden_size_per_layer_input", 0)
        text.setdefault("vocab_size_per_layer_input", V)
        config["vision_config"].update(model_patch_size=48, num_soft_tokens=280)
        config["audio_config"].update(hidden_size=64, output_proj_dims=64)
    return config


def build(kind: str, path: Path, *, dtype=None, seed: int = 0) -> Path:
    """A tiny decision checkpoint of `kind` ("e2b" or "unified") at `path`, float32 unless
    `dtype` says otherwise."""
    torch.manual_seed(seed)
    common = dict(
        vocab_size=V,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=2,
        head_dim=32,
        global_head_dim=64,
        sliding_window=4,
        layer_types=LAYERS,
        max_position_embeddings=4096,
        final_logit_softcapping=30.0,
        rope_parameters=ROPE,
    )
    if kind == "e2b":
        text = transformers.Gemma4TextConfig(
            **common,
            num_key_value_heads=1,
            vocab_size_per_layer_input=V,
            hidden_size_per_layer_input=16,
            num_kv_shared_layers=3,
            use_double_wide_mlp=True,
        )
        vision = transformers.Gemma4VisionConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            patch_size=16,
            pooling_kernel_size=3,
            position_embedding_size=1024,
        )
        audio = transformers.Gemma4AudioConfig(
            hidden_size=64, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=64
        )
        config = transformers.Gemma4Config(
            text_config=text, vision_config=vision, audio_config=audio, **IMAGE_IDS
        )
        model = transformers.Gemma4ForConditionalGeneration(config)
        processor_type = "Gemma4ImageProcessor"
    else:
        text = transformers.Gemma4UnifiedTextConfig(
            **common,
            num_key_value_heads=2,
            num_global_key_value_heads=1,
            attention_k_eq_v=True,
            use_bidirectional_attention="vision",
        )
        vision = transformers.Gemma4UnifiedVisionConfig(
            patch_size=16,
            pooling_kernel_size=3,
            mm_embed_dim=64,
            mm_posemb_size=320,
            output_proj_dims=64,
        )
        audio = transformers.Gemma4UnifiedAudioConfig(audio_embed_dim=64)
        config = transformers.Gemma4UnifiedConfig(
            text_config=text, vision_config=vision, audio_config=audio, **IMAGE_IDS
        )
        model = transformers.Gemma4UnifiedForConditionalGeneration(config)
        processor_type = "Gemma4UnifiedImageProcessor"
    with torch.no_grad():
        for p in model.parameters():  # norms start at one and position tables at zero
            p.add_(torch.randn_like(p) * 0.05)
    shutil.rmtree(path, ignore_errors=True)
    model.to(dtype or torch.float32).save_pretrained(path)
    (path / "generation_config.json").unlink(missing_ok=True)
    cfg = _base_style(json.loads((path / "config.json").read_text()), kind == "unified")
    (path / "config.json").write_text(json.dumps(cfg, indent=2))
    TOK.save(str(path / "tokenizer.json"))
    (path / "tokenizer_config.json").write_text(json.dumps({"bos_token": "<bos>"}))
    processor = {"image_processor": {**IMAGE_PROCESSOR, "image_processor_type": processor_type}}
    (path / "processor_config.json").write_text(json.dumps(processor))
    from safetensors.torch import save_file

    save_file(
        {"weight": (torch.randn(len(CODES), 64) * 0.5).to(torch.bfloat16)},
        str(path / "readout.safetensors"),
    )
    decision = decision_config_fixture()
    decision["token_ids"] = [TOK.token_to_id(c) for c in CODES]
    (path / "decision_config.json").write_text(json.dumps(decision))
    return path


def edit_json(path: Path, name: str, **changes) -> None:
    p = path / name
    d = json.loads(p.read_text())
    for key, value in changes.items():
        *parents, leaf = key.split(".")
        node = d
        for k in parents:
            node = node[k]
        node[leaf] = value
    p.write_text(json.dumps(d))


CKPT_DIRS: dict[str, Path] = {}


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlx-ckpts")
    CKPT_DIRS.update({kind: build(kind, root / kind) for kind in KINDS})
    return CKPT_DIRS


@pytest.fixture(scope="module")
def backends(ckpts):
    out = {kind: load_backend(ckpts[kind]) for kind in KINDS}
    yield out
    for b in out.values():
        b.close()


def mlx_engine(path: Path, **kw) -> DecisionEngine:
    """The decision engine `ask` and `serve` build for `--device mlx`."""
    return load_decision_engine(str(path), device="mlx", **kw)


def copy_of(ckpt: Path, tmp_path: Path, name: str = "copy") -> Path:
    dst = tmp_path / name
    shutil.copytree(ckpt, dst)
    return dst


# ---- the reference ---------------------------------------------------------------------------


@lru_cache(maxsize=4)
def _torch_model(path: str):
    cfg = json.loads((Path(path) / "config.json").read_text())
    cls = (
        transformers.Gemma4UnifiedForConditionalGeneration
        if cfg["model_type"] == "gemma4_unified"
        else transformers.Gemma4ForConditionalGeneration
    )
    conf = transformers.AutoConfig.from_pretrained(path)
    if cfg["model_type"] == "gemma4":
        conf.audio_config = None  # transformers' audio tower has no SDPA path; it is not run
    return cls.from_pretrained(
        path, config=conf, dtype=torch.float32, attn_implementation="sdpa"
    ).eval()


def _unified_reference_pixels(pil, settings: dict) -> tuple[np.ndarray, np.ndarray]:
    """The 12B's pixels as transformers' `Gemma4UnifiedImageProcessor` makes them, in numpy,
    from an image already resized to multiples of 48: scaled to [0, 1], cut into 16-pixel
    patches with their (x, y) positions, then each 3 x 3 group merged into one 48-pixel patch
    (`patches_merge`: the groups in row-major order, each a 48 x 48 x 3 block), padded to
    `max_soft_tokens` with position -1. Written from transformers' code, not mlx-vlm's."""
    p, k, n = settings["patch_size"], settings["pooling_kernel_size"], settings["max_soft_tokens"]
    img = np.asarray(pil, dtype=np.float32).transpose(2, 0, 1) * np.float32(
        settings["rescale_factor"]
    )
    c, h, w = img.shape
    small = img.reshape(c, h // p, p, w // p, p).transpose(1, 3, 2, 4, 0).reshape(-1, p * p * c)
    gx, gy = np.meshgrid(np.arange(w // p), np.arange(h // p), indexing="xy")
    pos = np.stack([gx, gy], -1).reshape(-1, 2)
    max_x = pos[:, 0].max() + 1
    kernel, within = pos // k, pos % k
    order = (k * k * kernel[:, 0] + k * max_x * kernel[:, 1]) + (within[:, 0] + within[:, 1] * k)
    perm = np.argsort(order, kind="stable")
    length = small.shape[0] // (k * k)
    grouped = small[perm].reshape(length, k, k, p, p, c).transpose(0, 1, 3, 2, 4, 5)
    merged = grouped.reshape(length, k * p * k * p * c)
    merged_pos = (pos[perm] // k).reshape(length, k * k, 2).min(axis=1)
    pad = n - length
    merged = np.concatenate([merged, np.zeros((pad, merged.shape[1]), np.float32)])
    merged_pos = np.concatenate([merged_pos, -np.ones((pad, 2), merged_pos.dtype)])
    return merged, merged_pos


def _torch_pixels(path: Path, images, backend) -> dict:
    settings = json.loads((path / "processor_config.json").read_text())["image_processor"]
    settings = {k: v for k, v in settings.items() if k != "image_processor_type"}
    pils = []
    for im in images:
        pil = decode_image(im.data)
        pils.append(pil.resize(resized_size(*pil.size), Image.Resampling.BICUBIC))
    if backend.unified:
        ref = [_unified_reference_pixels(pil, settings) for pil in pils]
        # The backend's pixels must be these, before the model sees either.
        for im, (pv, pos) in zip(images, ref, strict=True):
            ours = prepare_image(im, backend.processor)
            np.testing.assert_allclose(ours["pixel_values"][0], pv, atol=1e-6)
            np.testing.assert_array_equal(ours["image_position_ids"][0], pos)
        try:  # transformers' own 12B processor needs torchvision
            import torchvision  # noqa: F401
        except ImportError:
            return {
                "pixel_values": torch.tensor(np.stack([pv for pv, _ in ref])),
                "image_position_ids": torch.tensor(np.stack([pos for _, pos in ref])),
            }
        processor = transformers.Gemma4UnifiedImageProcessor(**{**settings, "do_resize": False})
        out = processor(images=pils, return_tensors="pt")
        for (pv, pos), tpv, tpos in zip(
            ref, out["pixel_values"], out["image_position_ids"], strict=True
        ):
            np.testing.assert_allclose(tpv.numpy(), pv, atol=1e-6)
            np.testing.assert_array_equal(tpos.numpy(), pos)
        return {k: v for k, v in out.items() if k != "num_soft_tokens_per_image"}
    processor = transformers.Gemma4ImageProcessorPil(**{**settings, "do_resize": False})
    out = processor(images=pils, return_tensors="pt")
    assert [int(n) for n in out["num_soft_tokens_per_image"]] == [im.tokens for im in images]
    return {k: v for k, v in out.items() if k != "num_soft_tokens_per_image"}


def reference(path: Path, state, question, images=(), backend=None) -> np.ndarray:
    """The final-norm hidden state at the last token: one masked full pass in transformers."""
    model = _torch_model(str(path))
    ids = list(state) + list(question)
    n, s = len(ids), len(state)
    m = torch.zeros(n, n, dtype=torch.bool)
    m[:s, :s] = True
    m[s:, :] = True
    m = m.view(1, 1, n, n)
    kw = _torch_pixels(path, images, backend) if images else {}
    with torch.no_grad():
        out = model.model(
            input_ids=torch.tensor([ids]),
            attention_mask={"full_attention": m, "sliding_attention": m},
            use_cache=False,
            **kw,
        )
    return out.last_hidden_state[0, -1].float().numpy()


def ids(seed: int, n: int) -> list[int]:
    """`n` token ids of ordinary text (the toy tokenizer's words and codes, no control token)."""
    return [int(x) for x in np.random.default_rng(seed).integers(1, 260, n)]


STATE = [TOK.token_to_id("<bos>"), *ids(1, 21)]
QUESTIONS = [ids(2, 9), ids(3, 4), ids(4, 13)]

REQUEST = {
    "state": "Help ! my payouts have failed for 3 days .",
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is this urgent ?"},
        "team": {
            "type": "choice",
            "instructions": "Which team ?",
            "criteria": {"billing": "payouts", "technical": "bugs", "sales": None},
        },
        "mood": {
            "type": "score",
            "instructions": "How angry ?",
            "criteria": ["calm", "upset", "angry"],
        },
    },
}


def png(color, size=(64, 48)) -> str:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---- images: size and decoding ------------------------------------------------------------------


def test_resized_size_gives_the_processors_soft_tokens():
    from mlx_vlm.models.gemma4.processing_gemma4 import get_aspect_ratio_preserving_size

    table = json.loads(SOFT_TOKENS.read_text())
    col = table["columns"].index("Gemma4ImageProcessor")
    for row in table["rows"]:
        w, h, n = row[0], row[1], row[col]
        rw, rh = resized_size(w, h)
        assert rw % 48 == 0 and rh % 48 == 0
        assert (rh // 48) * (rw // 48) == n, (w, h)
        assert get_aspect_ratio_preserving_size(h, w, 16, 280 * 9, 3) == (rh, rw), (w, h)


@pytest.mark.parametrize("name", sorted(IMAGE_FILES))
def test_images_decode_as_v1_decodes_them(name):
    from strands_decider.vision import decode_image as v1_decode

    ours = decode_image(image_bytes(name))
    theirs = v1_decode(image_b64(name))
    assert ours.mode == theirs.mode == "RGB"
    assert ours.size == theirs.size
    assert np.array_equal(np.asarray(ours), np.asarray(theirs))


# ---- the mask --------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("share_state", [True, False])
def test_text_matches_the_reference_mask(kind, share_state, ckpts, backends):
    """Every question, over the shared state or as its own full pass, gives the hidden state
    transformers gives under the explicit dense prefix_bidir mask."""
    b = backends[kind]
    b.share_state = share_state
    try:
        rows = b.forward(STATE, QUESTIONS)
    finally:
        b.share_state = True
    assert b.outputs == HIDDEN and rows.shape == (len(QUESTIONS), 64)
    for q, row in zip(QUESTIONS, rows, strict=True):
        ref = reference(ckpts[kind], STATE, q)
        assert np.abs(row - ref).max() < 2e-4 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("kind", KINDS)
def test_stock_masks_differ_from_the_reference(kind, ckpts, backends):
    """The check above can fail: mlx-vlm's own masks (causal, a 4-token window on sliding
    layers) give another answer."""
    b = backends[kind]
    full = STATE + QUESTIONS[0]
    stock = b._executor.submit(
        lambda: np.array(b._decoder(mx.array([full]))[0, -1].astype(mx.float32))
    ).result()
    ref = reference(ckpts[kind], STATE, QUESTIONS[0])
    assert np.abs(stock - ref).max() > 1e-2


@pytest.mark.parametrize("kind", KINDS)
def test_no_window_anywhere(kind, ckpts, backends, tmp_path):
    """The same weights with a 1,000-token window answer exactly as with a 4-token one."""
    wide = copy_of(ckpts[kind], tmp_path)
    edit_json(wide, "config.json", **{"text_config.sliding_window": 1000})
    b = load_backend(wide)
    try:
        for share_state in (True, False):
            b.share_state = backends[kind].share_state = share_state
            assert (
                np.abs(b.forward(STATE, QUESTIONS) - backends[kind].forward(STATE, QUESTIONS)).max()
                < 1e-6
            )
    finally:
        backends[kind].share_state = True
        b.close()


def test_kv_shared_layers_carry_only_the_last_token_exactly(backends):
    """E2B's KV-shared layers read no other token's hidden state, so running them over the
    last token alone (`logits_to_keep=1`) gives the full pass's last row."""
    b = backends["e2b"]
    assert b._owners == 3 < len(b._decoder.layers)
    full = STATE + QUESTIONS[0]
    from strands_decider.core import attention_mask

    def both():
        mask = mx.array(attention_mask(len(STATE), len(full)))
        x = mx.array([full])
        a = b._decoder(x, mask=mask)[0, -1]
        c = b._decoder(x, mask=mask, logits_to_keep=1)
        assert c.shape[1] == 1
        return np.array(a), np.array(c[0, -1])

    a, c = b._executor.submit(both).result()
    assert np.abs(a - c).max() < 1e-5


# ---- state reuse -------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_the_state_is_encoded_once_and_shared(kind, backends):
    b = backends[kind]
    calls = []
    decoder = b._decoder

    def counting(*args, **kwargs):
        calls.append(kwargs.get("inputs_embeds") is not None)
        return decoder(*args, **kwargs)

    b._decoder = counting
    try:
        together = b.forward(STATE, QUESTIONS)
    finally:
        b._decoder = decoder
    assert calls == [True] + [False] * len(QUESTIONS)  # the state once, then each question
    alone = np.stack([b.forward(STATE, [q])[0] for q in QUESTIONS])
    reverse = b.forward(STATE, QUESTIONS[::-1])[::-1]
    repeated = b.forward(STATE, [QUESTIONS[0]] * 3)
    np.testing.assert_array_equal(together, alone)
    np.testing.assert_array_equal(together, reverse)
    assert (repeated == repeated[0]).all()  # the cache is back to the state after each question
    b.share_state = False
    try:
        assert np.abs(b.forward(STATE, QUESTIONS) - together).max() < 1e-4
    finally:
        b.share_state = True


# ---- images ------------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("state", ["After the user tapped Pay .", ""])
def test_images_match_the_reference(kind, state, ckpts, backends):
    """Two images of different sizes and formats, at their spans in the state, as
    transformers places them, with its per-layer inputs at the placeholders."""
    request = {
        "state": state,
        "images": [image_b64("checker-64x48.png"), image_b64("rings-90x120.webp", data_uri=True)],
        "questions": REQUEST["questions"],
    }
    rendered = render(request, TOK, "strands-plain")
    images = rendered.images
    assert [im.tokens for im in images] == [soft_tokens(64, 48), soft_tokens(90, 120)]
    state_ids = list(rendered.state_ids)
    questions = [list(q.question_ids) for q in rendered.questions.values()]
    b = backends[kind]
    rows = b.forward(state_ids, questions, images)
    for q, row in zip(questions, rows, strict=True):
        ref = reference(ckpts[kind], state_ids, q, images, b)
        assert np.abs(row - ref).max() < 1e-3 * max(1.0, np.abs(ref).max())
    assert np.abs(rows - b.forward(state_ids, questions)).max() > 1e-2  # placeholders alone


@pytest.mark.parametrize("kind", KINDS)
def test_the_image_not_its_ids_decides(kind, backends):
    """Two images of the same size give the same ids; their answers differ."""
    b = backends[kind]
    red = render({**REQUEST, "images": [png((220, 20, 20))]}, TOK, "strands-plain")
    blue = render({**REQUEST, "images": [png((20, 40, 220))]}, TOK, "strands-plain")
    assert red.state_ids == blue.state_ids
    qs = [list(q.question_ids) for q in red.questions.values()]
    a = b.forward(list(red.state_ids), qs, red.images)
    c = b.forward(list(blue.state_ids), qs, blue.images)
    assert np.abs(a - c).max() > 1e-2


# ---- the readout -------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", KINDS)
def test_answers_are_the_cores_readout_of_the_reference(kind, ckpts):
    """End to end: render, the backend, and the core's readout give the probabilities the
    core reads from the reference's hidden states."""
    engine = mlx_engine(ckpts[kind])
    try:
        request = {**REQUEST, "images": [image_b64("stripes-300x40.jpg")]}
        out = engine.evaluate(request).model_dump()
        rendered = render(request, TOK, engine.config.prompt)
        refs = [
            reference(
                ckpts[kind], rendered.state_ids, q.question_ids, rendered.images, engine.backend
            )
            for q in rendered.questions.values()
        ]
        expect = question_probabilities(
            np.stack(refs), engine.readout, engine.config, rendered, inputs=HIDDEN
        )
        team = out["answers"]["team"]["probabilities"]
        for name, p in zip(["billing", "technical", "sales"], expect["team"], strict=True):
            assert abs(team[name] - p) < 2e-3
        assert abs(out["answers"]["urgent"]["noul"] - expect["urgent"][1]) < 2e-3
        assert out["usage"]["input_tokens"] == rendered.input_tokens
    finally:
        engine.close()


def test_a_bf16_checkpoint_hands_over_bf16_hidden_states(tmp_path):
    """h is the bf16 decoder's output, which float32 holds exactly; the core rounds W h to
    bf16 (docs/spec.md section 7.1)."""
    path = build("e2b", tmp_path / "bf16", dtype=torch.bfloat16)
    b = load_backend(path)
    try:
        rows = b.forward(STATE, QUESTIONS)
        assert rows.dtype == np.float32
        np.testing.assert_array_equal(round_to_bf16(rows), rows)
        readout = Readout.load(path)
        assert readout.weight.shape == (len(CODES), 64)
    finally:
        b.close()


# ---- refusals ----------------------------------------------------------------------------------


def test_a_text_only_checkpoint_refuses_images_before_decoding_them(ckpts, tmp_path):
    path = copy_of(ckpts["e2b"], tmp_path)
    edit_json(path, "decision_config.json", inputs=["text"])
    engine = mlx_engine(path)
    try:
        assert engine.backend.inputs == ("text",) and not engine.takes_images
        assert engine.backend.model.vision_tower is None  # not loaded
        with pytest.raises(RequestRefused, match=r'does not take images: .* \["text"\]'):
            engine.evaluate({**REQUEST, "images": ["not base64 at all!"]})
        assert engine.evaluate(REQUEST).answers["urgent"].type == "noul"
    finally:
        engine.close()


def test_other_processor_settings_leave_no_image_path(ckpts, tmp_path):
    path = copy_of(ckpts["e2b"], tmp_path)
    edit_json(path, "processor_config.json", **{"image_processor.max_soft_tokens": 70})
    engine = mlx_engine(path)
    try:
        assert engine.backend.inputs == ("text",) and not engine.takes_images
        assert engine.backend.image_problem.startswith("images are not supported: its image")
        with pytest.raises(RequestRefused, match="images are not supported: its image processor"):
            engine.evaluate({**REQUEST, "images": [png("red")]})
        with pytest.raises(ValueError, match="images are not supported"):  # the backend alone
            r = render({**REQUEST, "images": [png("red")]}, TOK, "strands-plain")
            engine.backend.forward(
                list(r.state_ids), [list(r.questions["urgent"].question_ids)], r.images
            )
    finally:
        engine.close()


@pytest.mark.parametrize(
    ("name", "changes", "message"),
    [
        (
            "decision_config.json",
            {"interface": "pplx-decider"},
            "not a Strands Decider decision config",
        ),
        ("config.json", {"text_config.final_logit_softcapping": 50.0}, "inconsistent"),
        ("config.json", {"model_type": "gemma4_text"}, "multimodal checkpoint"),
    ],
)
def test_refuses_checkpoints_it_cannot_serve(ckpts, tmp_path, name, changes, message):
    path = copy_of(ckpts["e2b"], tmp_path)
    edit_json(path, name, **changes)
    with pytest.raises(ValueError, match=message):
        mlx_engine(path)


def test_refuses_an_image_it_cannot_decode(ckpts, backends):
    """A sound header over broken pixel data passes the core's checks and fails to decode."""
    good = bytearray(base64.b64decode(png("red")))
    idat = good.index(b"IDAT")
    good[idat + 8 : idat + 20] = b"\x00" * 12  # the compressed pixels, not the header
    request = {**REQUEST, "images": [base64.b64encode(bytes(good)).decode("ascii")]}
    rendered = render(request, TOK, "strands-plain")
    qs = [list(q.question_ids) for q in rendered.questions.values()]
    with pytest.raises(ValueError, match="image could not be decoded"):
        backends["e2b"].forward(list(rendered.state_ids), qs, rendered.images)


def test_refuses_a_span_that_is_not_placeholders(backends):
    rendered = render({**REQUEST, "images": [png("red")]}, TOK, "strands-plain")
    qs = [list(q.question_ids) for q in rendered.questions.values()]
    state = list(rendered.state_ids)
    state[rendered.images[0].start] = 5
    with pytest.raises(ValueError, match="not an image's placeholders"):
        backends["e2b"].forward(state, qs, rendered.images)


def test_refuses_a_prompt_past_the_validated_length(ckpts, tmp_path):
    path = copy_of(ckpts["e2b"], tmp_path)
    edit_json(path, "decision_config.json", max_length_validated=40)
    engine = mlx_engine(path)
    try:
        with pytest.raises(RequestRefused, match="exceeds the limit of 40 tokens"):
            engine.evaluate(REQUEST)
    finally:
        engine.close()


# ---- MLX artifacts -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bf16_ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlx-bf16")
    return {kind: build(kind, root / kind, dtype=torch.bfloat16) for kind in KINDS}


def _module(model, path: str):
    m = model
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


@pytest.mark.parametrize("kind", KINDS)
def test_a_bf16_artifact_answers_as_its_checkpoint(kind, bf16_ckpts, tmp_path):
    src = bf16_ckpts[kind]
    summary = convert(src, tmp_path / "mlx")
    assert summary["quantization"] is None and summary["quantized_modules"] == {}
    art = Path(summary["artifact"])
    for name in (
        "readout.safetensors",
        "decision_config.json",
        "processor_config.json",
        "tokenizer.json",
    ):
        assert (art / name).read_bytes() == (src / name).read_bytes()
    assert (art / "model.safetensors.index.json").exists()
    assert not (art / "chat_template.jinja").exists()
    a, c = load_backend(src), load_backend(art)
    try:
        np.testing.assert_array_equal(a.forward(STATE, QUESTIONS), c.forward(STATE, QUESTIONS))
        r = render({**REQUEST, "images": [png("red")]}, TOK, "strands-plain")
        qs = [list(q.question_ids) for q in r.questions.values()]
        np.testing.assert_array_equal(
            a.forward(list(r.state_ids), qs, r.images), c.forward(list(r.state_ids), qs, r.images)
        )
    finally:
        a.close()
        c.close()


@pytest.mark.parametrize("kind", KINDS)
def test_an_8_bit_artifact_quantizes_the_decoder_only(kind, bf16_ckpts, tmp_path):
    src = bf16_ckpts[kind]
    summary = convert(src, tmp_path / "mlx8", bits=8)
    assert summary["quantization"] == {"bits": 8, "group_size": 64, "mode": "affine"}
    assert set(summary["quantized_modules"]) == {"language_model"}
    art = Path(summary["artifact"])
    assert (art / "readout.safetensors").read_bytes() == (src / "readout.safetensors").read_bytes()
    from strands_decider.backends.mlx import AUDIO_MODULES, VISION_MODULES, load_mlx_model

    model = load_mlx_model(art, keep=AUDIO_MODULES + VISION_MODULES)
    kept = ["embed_audio.embedding_projection", "embed_vision.embedding_projection"]
    kept += ["vision_tower.encoder.layers.0.mlp.gate_proj.linear"] if kind == "e2b" else []
    kept += ["vision_embedder.patch_dense"] if kind == "unified" else []
    for path in kept:
        module = _module(model, path)
        assert "scales" not in module and module.weight.dtype == mx.bfloat16, path
    assert quantized_parts(model) == summary["quantized_modules"]
    a, c = load_backend(src), load_backend(art)
    try:
        # A tiny random decoder amplifies rounding (bf16 alone moves it by about 10%), so
        # this checks only that the 8-bit decoder is the same network; E2B's parity run on
        # real weights measures 8-bit's accuracy.
        x, y = a.forward(STATE, QUESTIONS), c.forward(STATE, QUESTIONS)
        cosine = (x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1)
        assert np.isfinite(y).all() and (cosine > 0.8).all()
    finally:
        a.close()
        c.close()


def test_convert_refuses_what_it_cannot_write(bf16_ckpts, tmp_path):
    src = bf16_ckpts["e2b"]
    with pytest.raises(ValueError, match="bits must be one of"):
        convert(src, tmp_path / "a", bits=3)
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "x").write_text("x")
    with pytest.raises(FileExistsError):
        convert(src, tmp_path / "full")
    q = convert(src, tmp_path / "q8", bits=8)["artifact"]
    with pytest.raises(ValueError, match="already a quantized MLX artifact"):
        convert(q, tmp_path / "again", bits=8)


# ---- serve and ask -------------------------------------------------------------------------------


def test_the_decision_engine_serves_v1s_request_and_response_on_mlx(ckpts):
    engine = mlx_engine(ckpts["e2b"], model_name="tiny")
    try:
        assert isinstance(engine, DecisionEngine) and engine.takes_images
        engine.warm_up()
        response = engine.evaluate(
            SystemOneRequest(
                state="Is it red ?",
                images=[png("red")],
                questions={"red": NoulQuestion(instructions="Is the image red ?")},
            )
        )
        assert isinstance(response, SystemOneResponse)
        assert response.model == "tiny" and response.usage.output_tokens == 1
        info = engine.describe()
        assert (info["backend"], info["device"], info["images"]) == ("mlx-vlm", "mlx", True)
        assert info["base_model"] == "google/gemma-4-E2B-it"
        assert info["max_length"] == DecisionConfig.load(ckpts["e2b"]).max_length_validated
    finally:
        engine.close()


def test_serve_on_mlx_over_http(ckpts, tmp_path):
    """Through create_app and HTTP on --device mlx: the health check, an image request, and
    the refusals with 422 and the interface's messages: audio, five images, and images on a
    text-only checkpoint. The worker thread stops with the app."""
    from fastapi.testclient import TestClient

    from strands_decider import server

    img = png("red")
    app = server.create_app(str(ckpts["e2b"]), device="mlx", model_name="tiny")  # warm-up too
    with TestClient(app) as client:
        health = client.get("/health").json()
        assert (health["backend"], health["device"], health["images"]) == ("mlx-vlm", "mlx", True)
        r = client.post("/v1/systemone", json={**REQUEST, "images": [img]})
        assert r.status_code == 200, r.text
        assert set(r.json()["answers"]) == {"urgent", "team", "mood"}
        r = client.post("/v1/systemone", json={**REQUEST, "audio": ["UklGRiQAAABXQVZF"]})
        assert r.status_code == 422 and r.json()["detail"] == "audio input is not supported"
        r = client.post("/v1/systemone", json={**REQUEST, "images": [img] * 5})
        assert r.status_code == 422
        assert r.json()["detail"] == "5 images; this server takes at most 4"
        assert (
            client.post("/v1/systemone", json={**REQUEST, "images": [img] * 4}).status_code == 200
        )
        backend = app.state.engine.backend
    assert backend._executor._shutdown  # closed with the app
    text_only = copy_of(ckpts["e2b"], tmp_path)
    edit_json(text_only, "decision_config.json", inputs=["text"])
    with TestClient(server.create_app(str(text_only), device="mlx", warmup=False)) as client:
        r = client.post("/v1/systemone", json={**REQUEST, "images": [img]})
        assert r.status_code == 422
        assert r.json()["detail"].startswith("this checkpoint does not take images")


def test_cli_ask_and_convert(bf16_ckpts, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)
    runner = CliRunner()
    image = tmp_path / "red.png"
    image.write_bytes(base64.b64decode(png("red")))
    args = ["ask", str(bf16_ckpts["e2b"]), "--device", "mlx", "--state", "a photo", "--json"]
    result = runner.invoke(cli.app, [*args, "--noul", "Is it red ?", "--image", str(image)])
    assert result.exit_code == 0, result.output
    answer = json.loads(result.stdout)["answers"]["noul_0"]
    assert answer["type"] == "noul" and 0 <= answer["noul"] <= 1
    out = tmp_path / "mlx8"
    result = runner.invoke(
        cli.app, ["convert-mlx", str(bf16_ckpts["e2b"]), str(out), "--bits", "8"]
    )
    assert result.exit_code == 0, result.output
    assert "decoder at 8 bits" in result.output
    args[1] = str(out)  # the artifact, through the CLI
    result = runner.invoke(cli.app, [*args, "--choice", "Which colour ?=red,blue"])
    assert result.exit_code == 0, result.output
    assert set(json.loads(result.stdout)["answers"]["choice_0"]["probabilities"]) == {"red", "blue"}


# ---- paths, conversion failures, bounds, notices -------------------------------------------------


def test_a_relative_path_loads_and_converts(bf16_ckpts, tmp_path, monkeypatch):
    """E2B, E4B and the 12B ship one model.safetensors and no index, so they load through a
    directory of links; a relative path must still point the links at the files."""
    src = bf16_ckpts["e2b"]
    assert not (src / "model.safetensors.index.json").exists()
    monkeypatch.chdir(src.parent)
    b = load_backend(src.name)
    try:
        rows = b.forward(STATE, QUESTIONS[:1])
    finally:
        b.close()
    monkeypatch.chdir(tmp_path)
    shutil.copytree(src, tmp_path / "ckpt")
    summary = convert("ckpt", "mlx")
    assert summary["artifact"] == str(tmp_path / "mlx")
    c = load_backend("mlx")
    try:
        np.testing.assert_array_equal(c.forward(STATE, QUESTIONS[:1]), rows)
    finally:
        c.close()


def test_a_failed_conversion_leaves_nothing(bf16_ckpts, tmp_path, monkeypatch):
    from strands_decider.backends import mlx_convert

    def reject(source, artifact):
        raise AssertionError("rejected")

    monkeypatch.setattr(mlx_convert, "verify", reject)
    with pytest.raises(AssertionError, match="rejected"):
        convert(bf16_ckpts["e2b"], tmp_path / "out", bits=8)
    assert list(tmp_path.iterdir()) == []  # no artifact, and no partial directory beside it


def test_4_bit_says_it_makes_no_conformance_claim(bf16_ckpts, tmp_path):
    with pytest.warns(UserWarning, match="4-bit MLX artifact makes no conformance claim"):
        summary = convert(bf16_ckpts["e2b"], tmp_path / "q4", bits=4)
    assert summary["quantization"]["bits"] == 4 and summary["notice"]
    with pytest.warns(UserWarning, match="no conformance claim"):
        load_backend(tmp_path / "q4").close()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # 8-bit is measured: no notice
        q8 = convert(bf16_ckpts["e2b"], tmp_path / "q8", bits=8)
        assert q8["notice"] is None
        load_backend(tmp_path / "q8").close()


@pytest.mark.parametrize("kind", KINDS)
def test_the_processor_hands_over_numpy(kind, backends):
    """Images are prepared on the caller's thread, so their pixels must not be MLX arrays."""
    rendered = render({**REQUEST, "images": [png("red")]}, TOK, "strands-plain")
    px = prepare_image(rendered.images[0], backends[kind].processor)
    assert px and all(isinstance(v, np.ndarray) for v in px.values())


@pytest.mark.parametrize("kind", KINDS)
def test_the_cache_grows_past_its_first_block_and_trims_back(kind, ckpts, backends):
    """A 250-token state fills most of the cache's first 256-token block; a 20-token question
    makes it grow, and the 5-token question after it must see the state alone."""
    b = backends[kind]
    state = [TOK.token_to_id("<bos>"), *ids(11, 249)]
    long, short = ids(12, 20), ids(13, 5)
    together = b.forward(state, [long, short, long])
    np.testing.assert_array_equal(together[1], b.forward(state, [short])[0])
    np.testing.assert_array_equal(together[0], together[2])
    for q, row in zip([long, short], together, strict=False):
        ref = reference(CKPT_DIRS[kind], state, q)
        assert np.abs(row - ref).max() < 2e-4 * max(1.0, np.abs(ref).max())


@pytest.mark.parametrize("kind", KINDS)
def test_forward_from_several_threads(kind, backends):
    """A server calls from a pool: requests from eight threads, with and without images,
    give what they give one at a time."""
    from concurrent.futures import ThreadPoolExecutor

    b = backends[kind]
    jobs = []
    for i, color in enumerate([(200, 30, 30), (30, 200, 30), None, (30, 30, 200)] * 2):
        req = {**REQUEST, "images": [png(color)]} if color else REQUEST
        r = render({**req, "state": f"Request {i} ."}, TOK, "strands-plain")
        jobs.append(
            (list(r.state_ids), [list(q.question_ids) for q in r.questions.values()], r.images)
        )
    alone = [b.forward(*job) for job in jobs]
    with ThreadPoolExecutor(8) as pool:
        together = list(pool.map(lambda job: b.forward(*job), jobs))
    for x, y in zip(alone, together, strict=True):
        np.testing.assert_array_equal(x, y)


def test_image_decoding_and_the_queue_are_bounded(backends, monkeypatch):
    import threading
    import time

    from strands_decider.backends import mlx as backend_mlx

    b = backends["e2b"]
    live, peak, lock = [0], [0], threading.Lock()
    real = backend_mlx.prepare_image

    def counting(image, processor):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        try:
            return real(image, processor)
        finally:
            with lock:
                live[0] -= 1

    monkeypatch.setattr(backend_mlx, "prepare_image", counting)
    r = render({**REQUEST, "images": [png("red")]}, TOK, "strands-plain")
    job = (list(r.state_ids), [list(r.questions["urgent"].question_ids)], r.images)
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda _: b.forward(*job), range(6)))
    assert peak[0] == backend_mlx.DECODE_SLOTS

    # One place in the queue, held by a request the worker is still running: the next waits
    # its timeout, then is refused.
    gate = threading.Event()
    monkeypatch.setattr(b, "_pending", threading.BoundedSemaphore(1))
    monkeypatch.setattr(b, "_max_pending", 1)
    monkeypatch.setattr(b, "_queue_timeout", 0.2)
    forward = b._forward
    monkeypatch.setattr(b, "_forward", lambda *a: (gate.wait(10), forward(*a))[1])
    first = threading.Thread(target=b.forward, args=(STATE, QUESTIONS[:1]))
    first.start()
    time.sleep(0.1)
    with pytest.raises(backend_mlx.BackendBusy, match="1 requests waiting or running"):
        b.forward(STATE, QUESTIONS[:1])
    gate.set()
    first.join()


# ---- against the transformers backend ---------------------------------------------------------


@pytest.mark.parametrize("kind", ["gemma4", "gemma4-vision", "unified"])
def test_answers_as_the_transformers_backend_does(kind, tmp_path):
    """The transformers backend's own tiny checkpoints (tests/decision_support.py: E2B's,
    the 31B's and the 12B's designs, a 2-token window, config.json as transformers 5.2x
    writes it) give the same answers on both backends in float32, with images."""
    import decision_support as ds

    path = ds.decision_checkpoint(tmp_path / kind, kind)
    req = ds.request([ds.image_b64(64, 48), ds.image_b64(90, 120, "JPEG")])
    out = {}
    for device in ("cpu", "mlx"):
        engine = load_decision_engine(str(path), device=device, dtype="float32")
        try:
            out[device] = engine.evaluate(req).model_dump()
        finally:
            engine.close()
    assert out["cpu"]["usage"] == out["mlx"]["usage"]
    for name, a in out["cpu"]["answers"].items():
        b = out["mlx"]["answers"][name]
        pa = a.get("probabilities") or {"p": a["noul"]}
        pb = b.get("probabilities") or {"p": b["noul"]}
        assert pa.keys() == pb.keys()
        assert max(abs(pa[k] - pb[k]) for k in pa) < 2e-3, (name, pa, pb)
