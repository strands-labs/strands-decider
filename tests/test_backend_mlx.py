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
Apple silicon with the mlx extra only; nothing is downloaded.
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
from strands_decider.backends import is_decision_checkpoint  # noqa: E402
from strands_decider.backends.mlx import (  # noqa: E402
    MLXEngine,
    decode_image,
    load_backend,
    load_engine,
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
from strands_decider.core.request import NO_VISION  # noqa: E402
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


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlx-ckpts")
    return {kind: build(kind, root / kind) for kind in KINDS}


@pytest.fixture(scope="module")
def backends(ckpts):
    out = {kind: load_backend(ckpts[kind]) for kind in KINDS}
    yield out
    for b in out.values():
        b.close()


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


def _torch_pixels(path: Path, images, backend) -> dict:
    if backend.unified:  # transformers' 12B processor needs torchvision: the backend's pixels
        px = [prepare_image(im, backend.processor) for im in images]
        return {
            "pixel_values": torch.tensor(np.concatenate([p["pixel_values"] for p in px])),
            "image_position_ids": torch.tensor(
                np.concatenate([p["image_position_ids"] for p in px])
            ),
        }
    settings = json.loads((path / "processor_config.json").read_text())["image_processor"]
    settings = {k: v for k, v in settings.items() if k != "image_processor_type"}
    processor = transformers.Gemma4ImageProcessorPil(**{**settings, "do_resize": False})
    pils = []
    for im in images:
        pil = decode_image(im.data)
        pils.append(pil.resize(resized_size(*pil.size), Image.Resampling.BICUBIC))
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
    engine = load_engine(ckpts[kind])
    try:
        request = {**REQUEST, "images": [image_b64("stripes-300x40.jpg")]}
        out = engine.answer(request)
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
    engine = load_engine(path)
    try:
        assert engine.backend.inputs == ("text",) and not engine.takes_images
        assert engine.backend.model.vision_tower is None  # not loaded
        with pytest.raises(ValueError, match=r'does not take images: .* \["text"\]'):
            engine.answer({**REQUEST, "images": ["not base64 at all!"]})
        assert engine.answer(REQUEST)["answers"]["urgent"]["type"] == "noul"
    finally:
        engine.close()


def test_other_processor_settings_leave_no_image_path(ckpts, tmp_path):
    path = copy_of(ckpts["e2b"], tmp_path)
    edit_json(path, "processor_config.json", **{"image_processor.max_soft_tokens": 70})
    with pytest.warns(UserWarning, match="images are not supported"):
        engine = load_engine(path)
    try:
        assert engine.backend.inputs == ("text",)
        with pytest.raises(ValueError, match=NO_VISION):
            engine.answer({**REQUEST, "images": [png("red")]})
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
        load_engine(path)


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
    engine = load_engine(path)
    try:
        with pytest.raises(ValueError, match="exceeds the limit of 40 tokens"):
            engine.answer(REQUEST)
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


def test_the_engine_serves_v1s_request_and_response(ckpts):
    from strands_decider.infer import EngineConfig, load_mlx

    assert is_decision_checkpoint(ckpts["e2b"])
    assert not is_decision_checkpoint(ckpts["e2b"] / "config.json")
    assert not is_decision_checkpoint("/no/such/checkpoint")
    engine = load_mlx(str(ckpts["e2b"]), EngineConfig(device="mlx", model_name="tiny"))
    try:
        assert isinstance(engine, MLXEngine) and engine.takes_images
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
        assert engine.model.config.base_model == "google/gemma-4-E2B-it"
        assert (
            engine.model.config.max_length == DecisionConfig.load(ckpts["e2b"]).max_length_validated
        )
    finally:
        engine.close()


def test_cli_ask_and_convert(bf16_ckpts, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)
    runner = CliRunner()
    image = tmp_path / "red.png"
    image.write_bytes(base64.b64decode(png("red")))
    args = ["ask", str(bf16_ckpts["e2b"]), "--device", "mlx", "--state", "a photo", "--json"]
    result = runner.invoke(cli.app, [*args, "--noul", "Is it red ?", "--image", str(image)])
    assert result.exit_code == 0, result.output
    answer = json.loads(result.output)["answers"]["noul_0"]
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
    assert set(json.loads(result.output)["answers"]["choice_0"]["probabilities"]) == {"red", "blue"}
