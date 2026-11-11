"""Image input (src/strands_decider/vision.py), on a tiny random-weight Qwen3.5 built here.

Nothing is downloaded: the multimodal checkpoint, the tokeniser and the image processor
are all made in a temp dir. The tokeniser is a real `Qwen3_5Tokenizer` (byte-level BPE
over the 256 byte symbols, no merges, plus the Qwen vision tokens), so it reloads as
itself next to a qwen3_5 config and tokenises every character of a state or question.
The tests pin what the feature promises:

* a text checkpoint loads onto the multimodal torso and answers text exactly as the
  text engine does (the adapter lands on the same decoder);
* answers over images through the shared-prefix path equal a plain full forward, which
  fails without the explicit M-RoPE positions;
* no position state leaks from an image forward into the next text request;
* the window rules match the text path: a prompt that does not fit the window is refused;
* only the promised image formats are opened, sizes are checked before decoding, and
  rotation and transparency are handled;
* a text-only engine refuses images instead of ignoring them, at the engine and the API.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import replace
from functools import partial

import numpy as np
import pytest

transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from strands_decider.hidden_engine import HiddenStateEngine, save_assets  # noqa: E402
from strands_decider.images import (  # noqa: E402
    IMAGE_PAD,
    VISION_END,
    VISION_START,
    decode_image,
    expand_image_tokens,
    fit_image,
    image_tokens_for_grid,
    render_image_state,
    smart_resize,
    to_grid,
)
from strands_decider.infer import EngineConfig, SystemOneEngine  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    StrandsDeciderConfig,
    StrandsDeciderModel,
    masked_log_softmax,
)
from strands_decider.prompting import option_token_index, render_question  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from strands_decider.vision import (  # noqa: E402
    VisionDeciderModel,
    VisionEngine,
    VisionEngineConfig,
    load_vision_engine,
)

TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
]
STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "signed": NoulQuestion(instructions="Is the form signed?"),
    "button": ChoiceQuestion(
        instructions="Which control next?",
        criteria={"submit": "the blue submit button", "cancel": "the grey cancel link"},
    ),
    "sharp": ScoreQuestion(instructions="How sharp?", criteria=["blurred", "soft", "sharp"]),
}


def _tokenizer():
    from tokenizers import pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, **{c: i + 1 for i, c in enumerate(alphabet)}}
    tok = transformers.Qwen3_5Tokenizer(vocab=vocab, merges=[])
    tok.add_special_tokens(
        {"additional_special_tokens": ["<|vision_start|>", IMAGE_PAD, VISION_END]}
    )
    return tok


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny Qwen3.5 multimodal checkpoint: 3 Gated DeltaNet + 1 attention layer, 2-layer ViT."""
    d = tmp_path_factory.mktemp("tiny-qwen35")
    tok = _tokenizer()
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", IMAGE_PAD, VISION_END)}
    cfg = transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "intermediate_size": 128,
            "head_dim": 64,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "vocab_size": len(tok),
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "attn_output_gate": True,
            "tie_word_embeddings": True,
            "mtp_num_hidden_layers": 0,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 1e7,
                "partial_rotary_factor": 0.25,
                "mrope_section": [3, 3, 2],
                "mrope_interleaved": True,
            },
        },
        vision_config={
            "depth": 2,
            "hidden_size": 64,
            "num_heads": 4,
            "intermediate_size": 128,
            "out_hidden_size": 64,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 2304,
        },
        image_token_id=ids[IMAGE_PAD],
        vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids[VISION_END],
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    transformers.Qwen2VLImageProcessorPil(
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        size={"shortest_edge": 65536, "longest_edge": 16777216},
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
    ).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def ckpt(base_dir, tmp_path_factory):
    """A text Strands Decider checkpoint on that base, with a non-trivial adapter."""
    cfg = StrandsDeciderConfig(
        base_model=base_dir,
        pointer_dim=32,
        torch_dtype="float32",
        max_length=1024,
        lora_targets=TARGETS,
        temperature_by_kind={"noul": 0.9, "choice": 0.7},
    )
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    with torch.no_grad():
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    d = tmp_path_factory.mktemp("ckpt")
    model.save_pretrained(str(d))
    return str(d)


@pytest.fixture(scope="module")
def engines(ckpt):
    text = SystemOneEngine(StrandsDeciderModel.load(ckpt), EngineConfig(device="cpu"))
    vision = VisionEngine(VisionDeciderModel.load(ckpt), VisionEngineConfig(device="cpu"))
    return text, vision


def _png(img: Image.Image, fmt: str = "PNG", **save) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, **save)
    return base64.b64encode(buf.getvalue()).decode()


def _b64(w: int, h: int, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    return _png(Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()))


# ---- prompt helpers and tokeniser ---------------------------------------------------


def test_image_state_layout():
    two = render_image_state("two pages", 2)
    assert two == (
        "<state>\n<|vision_start|><|image_pad|><|vision_end|>\n"
        "<|vision_start|><|image_pad|><|vision_end|>\ntwo pages\n</state>\n"
    )
    assert (
        render_image_state("", 1)
        == "<state>\n<|vision_start|><|image_pad|><|vision_end|>\n</state>\n"
    )
    assert render_image_state("text", 0) == "<state>\ntext\n</state>\n"


def test_expansion_and_grid():
    assert image_tokens_for_grid([[1, 28, 28], [1, 20, 28]], merge_size=2) == [196, 140]
    assert expand_image_tokens("a<|image_pad|>b", [3]) == "a" + IMAGE_PAD * 3 + "b"
    with pytest.raises(ValueError):
        expand_image_tokens("a<|image_pad|>b", [1, 2])
    assert fit_image(Image.new("RGB", (900, 300)), 448).size == (448, 149)
    assert fit_image(Image.new("RGB", (100, 50)), 448).size == (100, 50)  # never upscaled


def test_checkpoint_tokenizer_reads_text(engines):
    """The tokeniser that comes back from the checkpoint keeps every character, so the
    tests below exercise real state and question text."""
    _, vision = engines
    assert type(vision.tok).__name__ == "Qwen3_5Tokenizer"
    assert vision.tok.decode(vision.tok(STATE)["input_ids"]) == STATE


# ---- images --------------------------------------------------------------------------


def test_decode_accepts_only_promised_formats_and_sizes():
    assert decode_image(_png(Image.new("RGB", (32, 16)), "JPEG")).size == (32, 16)
    eps = base64.b64encode(
        b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n"
    ).decode()
    with pytest.raises(ValueError, match="could not be read"):
        decode_image(eps)
    with pytest.raises(ValueError, match="pixels"):
        decode_image(_png(Image.new("RGB", (64, 64))), max_pixels=1000)  # refused from the header
    with pytest.raises(ValueError, match="base64"):
        decode_image("not base64!")


def test_decode_rotation_and_transparency():
    exif = Image.Exif()
    exif[0x0112] = 6  # rotated 90 degrees, as phones store it
    assert decode_image(_png(Image.new("RGB", (40, 20)), "JPEG", exif=exif)).size == (20, 40)
    clear = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
    assert decode_image(_png(clear)).getpixel((0, 0)) == (255, 255, 255)
    # Malformed EXIF is a caller error (422), not a server error.
    for fmt in ("PNG", "WEBP"):
        with pytest.raises(ValueError, match="could not be decoded"):
            decode_image(_png(Image.new("RGB", (8, 8)), fmt, exif=b"garbage!"))


def test_processor_is_pinned_to_pil(engines):
    _, vision = engines
    assert type(vision.image_processor).__name__ == "Qwen2VLImageProcessorPil"


# ---- model and answers ------------------------------------------------------------------


def test_adapter_lands_on_the_decoder_and_vision_is_frozen(engines):
    _, vision = engines
    lora = [n for n, _ in vision.model.torso.named_modules() if n.endswith("lora_A")]
    assert lora and all(".language_model." in n for n in lora)
    assert not any(p.requires_grad for p in vision.model.torso.base_model.model.visual.parameters())


def test_text_requests_unchanged(engines):
    text, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    assert (
        text.evaluate(req).model_dump()["answers"] == vision.evaluate(req).model_dump()["answers"]
    )


def _reference(vision: VisionEngine, state: str, images: list[str], q) -> list[float]:
    """Plain full forward of state + one question, positions left to transformers."""
    rq = render_question(q)
    pil = [fit_image(decode_image(b), 448) for b in images]
    proc = vision.image_processor(images=pil, return_tensors="pt")
    counts = image_tokens_for_grid(
        proc["image_grid_thw"].tolist(), vision.image_processor.merge_size
    )
    text = expand_image_tokens(render_image_state(state, len(images)), counts) + rq.text
    enc = vision.tok(text, return_offsets_mapping=True)
    opt = option_token_index(enc["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
    ids = torch.tensor([enc["input_ids"]])
    with torch.no_grad():
        out = vision.model(
            ids,
            torch.ones_like(ids),
            torch.tensor([rq.n_slots]),
            opt_idx=torch.tensor([opt]),
            temperature=vision._temperatures([rq.kind]),
            pixel_values=proc["pixel_values"],
            image_grid_thw=proc["image_grid_thw"],
        )
    return (
        masked_log_softmax(out["logits"], torch.tensor([rq.n_slots]))
        .exp()[0, : rq.n_slots]
        .tolist()
    )


@pytest.mark.parametrize("sizes", [[(600, 800)], [(448, 448), (1000, 200)]])
def test_image_answers_match_full_forward(engines, sizes):
    _, vision = engines
    images = [_b64(w, h, i) for i, (w, h) in enumerate(sizes)]
    rendered = [render_question(q) for q in QUESTIONS.values()]
    pil = [fit_image(decode_image(b), 448) for b in images]
    probs, _ = vision._image_probs("Checkout page after the user tapped pay.", pil, rendered)
    for i, q in enumerate(QUESTIONS.values()):
        ref = _reference(vision, "Checkout page after the user tapped pay.", images, q)
        got = probs[i, : len(ref)].tolist()
        assert max(abs(x - y) for x, y in zip(got, ref, strict=True)) < 1e-5


def test_image_forward_leaves_no_position_state(engines):
    """A plain image forward on the model (as in training or evaluation) stores
    `rope_deltas` on the torso; the next text request must not read it."""
    _, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    before = vision.evaluate(req).model_dump()["answers"]
    _reference(vision, "Checkout page.", [_b64(600, 800, 1)], QUESTIONS["signed"])
    vision.evaluate(SystemOneRequest(state="", questions=QUESTIONS, images=[_b64(600, 800, 1)]))
    assert vision.evaluate(req).model_dump()["answers"] == before


# ---- the window ----------------------------------------------------------------------------


def _small_window(vision: VisionEngine, max_length: int) -> VisionEngine:
    model = vision.model
    eng = VisionEngine.__new__(VisionEngine)
    eng.__dict__.update(vision.__dict__)
    eng.cfg = eng.vcfg = replace(vision.vcfg)
    eng.model = model
    model.config.max_length = max_length
    return eng


def test_long_prompt_with_images_is_refused(engines):
    _, vision = engines
    original = vision.model.config.max_length
    try:
        state = "First words stay. " + "filler " * 200 + "Last words go."
        img = [fit_image(decode_image(_b64(320, 320, 4)), 448)]
        with pytest.raises(ValueError, match="context window"):
            _small_window(vision, 700)._fit_images(
                state, img, [render_question(QUESTIONS["signed"]).text]
            )
    finally:
        vision.model.config.max_length = original


def test_concurrent_image_and_text_requests_match_serial(engines):
    """Vision engine serves mixed image+text requests through FastAPI's threadpool.
    `_fit_images` once wrote offsets onto the engine too; a concurrent request would
    overwrite them before `_image_probs` read them back, so the pointer head scored
    the wrong tokens. Pins that the image path threads offsets per request as the
    text path does."""
    import threading
    import time
    from concurrent.futures import ThreadPoolExecutor

    _, vision = engines
    img = _b64(160, 160, 7)
    reqs = []
    for i in range(8):
        state = f"{STATE} ({i})"
        if i % 2:
            reqs.append(SystemOneRequest(state=state, questions=QUESTIONS, images=[img]))
        else:
            reqs.append(SystemOneRequest(state=state, questions=QUESTIONS))

    def answers(resp):
        return {n: a.model_dump() for n, a in resp.answers.items()}

    with torch.inference_mode():
        serial = [answers(vision.evaluate(r)) for r in reqs]

    # Hold every thread between tokenising and pointing, so they interleave at the
    # race-prone spot in both the text (_fit) and image (_fit_images) paths.
    workers = 4
    barrier = threading.Barrier(workers, timeout=60)
    broke = threading.Event()
    # The HF fast tokenizer (Rust) raises "Already borrowed" if two threads reconfigure
    # truncation on it at once -- orthogonal to the race under test. Serialise just
    # the tokeniser calls; the barrier still interleaves threads at the race-prone spot.
    fit_lock = threading.Lock()

    def _trip():
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            broke.set()
        time.sleep(0.01)

    fit = type(vision)._fit
    fit_images = type(vision)._fit_images

    def slow_fit(self, *a, **kw):
        with fit_lock:
            out = fit(self, *a, **kw)
        _trip()
        return out

    def slow_fit_images(self, *a, **kw):
        with fit_lock:
            out = fit_images(self, *a, **kw)
        _trip()
        return out

    try:
        VisionEngine._fit = slow_fit  # type: ignore[method-assign]
        VisionEngine._fit_images = slow_fit_images  # type: ignore[method-assign]

        def run(r):
            with torch.inference_mode():
                return answers(vision.evaluate(r))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            concurrent = list(pool.map(run, reqs))
    finally:
        VisionEngine._fit = fit  # type: ignore[method-assign]
        VisionEngine._fit_images = fit_images  # type: ignore[method-assign]

    assert not broke.is_set(), "barrier broke; threads did not interleave at _fit/_fit_images"
    assert concurrent == serial


# ---- engine and API ------------------------------------------------------------------------


def test_image_only_request_and_refusals(engines):
    text, vision = engines
    req = SystemOneRequest(
        state="",
        questions={"signed": QUESTIONS["signed"]},
        images=["data:image/png;base64," + _b64(320, 320, 2)],
    )
    assert 0.0 <= vision.evaluate(req).answers["signed"].noul <= 1.0
    with pytest.raises(ValueError):
        text.evaluate(req)  # a text-only engine refuses images rather than ignore them
    with pytest.raises(ValueError):
        vision.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, images=["not base64!"]))


def test_load_vision_engine_keeps_the_server_settings(ckpt):
    eng = load_vision_engine(ckpt, EngineConfig(device="cpu", max_batch=3, model_name="vd-test"))
    assert (eng.cfg.max_batch, eng.cfg.model_name) == (3, "vd-test")


def test_server_routes_images(ckpt):
    from fastapi.testclient import TestClient

    from strands_decider import server

    body = {
        "state": "Checkout page.",
        "images": [_b64(320, 320, 3)],
        "questions": {"signed": {"type": "noul", "instructions": "Is the form signed?"}},
    }
    vision_app = TestClient(server.create_app(ckpt, device="cpu", vision=True))
    r = vision_app.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["answers"]["signed"]["noul"] <= 1.0
    assert vision_app.get("/health").json()["vision"] is True
    text_app = TestClient(server.create_app(ckpt, device="cpu"))
    assert text_app.post("/v1/systemone", json=body).status_code == 422


# ---- the hidden-state engine (hidden_engine.py, without torch) ------------------------------
# deploy/lambda runs it over llama.cpp. Here its backend is this module's vision model, one
# plain full forward per question with positions left to transformers, so any difference from
# VisionEngine is the hidden-state engine's: the grid sizing, the state layout, the window,
# the head or the readouts.


class TorchImageBackend:
    """The vision model as a hidden_engine.ImageBackend."""

    def __init__(self, vision: VisionEngine):
        self.vision = vision
        self.calls: list[str] = []

    @torch.no_grad()
    def _hidden(self, prefix: list[int], q: list[int], r: list[int], **images) -> np.ndarray:
        ids = torch.tensor([prefix + q])
        hidden = self.vision.model.encode(ids, torch.ones_like(ids), **images)
        return hidden[0, len(prefix) :][r].float().numpy()

    def forward(self, state, questions, rows):
        self.calls.append("forward")
        return [self._hidden(state, q, r) for q, r in zip(questions, rows, strict=True)]

    def forward_images(self, segments, images, questions, rows):
        self.calls.append("forward_images")
        pil = [Image.frombytes("RGB", (w, h), rgb) for rgb, w, h in images]
        proc = self.vision.image_processor(images=pil, return_tensors="pt")
        merge = self.vision.image_processor.merge_size
        counts = image_tokens_for_grid(proc["image_grid_thw"].tolist(), merge)
        start, pad, end = (
            self.vision.tok.convert_tokens_to_ids(t) for t in (VISION_START, IMAGE_PAD, VISION_END)
        )
        prefix: list[int] = []
        for i, segment in enumerate(segments):
            prefix += segment
            if i < len(images):
                prefix += [start] + [pad] * counts[i] + [end]
        mm = {"pixel_values": proc["pixel_values"], "image_grid_thw": proc["image_grid_thw"]}
        hidden = [self._hidden(prefix, q, r, **mm) for q, r in zip(questions, rows, strict=True)]
        self.vision._reset_positions()  # the plain forward stores rope_deltas on the torso
        return hidden, [c + 2 for c in counts]


class TextOnlyBackend(TorchImageBackend):
    forward_images = None  # type: ignore[assignment]


class MisSizingBackend(TorchImageBackend):
    def forward_images(self, segments, images, questions, rows):
        hidden, made = super().forward_images(segments, images, questions, rows)
        return hidden, [n + 1 for n in made]


def _hidden_engine(vision: VisionEngine, tmp_path, backend=None, **config) -> HiddenStateEngine:
    save_assets(vision.model, str(tmp_path / "assets"), vision.cfg.model_name)
    return HiddenStateEngine.from_assets(
        str(tmp_path / "assets"), backend or TorchImageBackend(vision), **config
    )


def _probabilities(response):
    return {
        name: [a.noul] if a.type == "noul" else list(a.probabilities.values())
        for name, a in response.answers.items()
    }


def test_grid_is_the_processors(engines):
    """images.to_grid sizes and resamples as the processor does, so a backend handed its
    pixels sees what VisionEngine's processor sees."""
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import smart_resize as hf

    _, vision = engines
    rng = np.random.default_rng(0)
    for _ in range(2000):
        h, w = int(rng.integers(8, 5000)), int(rng.integers(8, 5000))
        try:
            want = hf(h, w, factor=32, min_pixels=65536, max_pixels=16777216)
        except ValueError as refused:  # an aspect ratio over 200: refused alike
            with pytest.raises(ValueError) as got:
                smart_resize(h, w)
            assert str(got.value) == str(refused)
            continue
        assert smart_resize(h, w) == want
    for i, (w, h) in enumerate([(448, 448), (448, 301), (300, 448), (120, 90), (1000, 200)]):
        img = fit_image(decode_image(_b64(w, h, i)), 448)
        rgb, gw, gh, n = to_grid(img)
        out = vision.image_processor(
            images=[img], return_tensors="np", do_rescale=False, do_normalize=False
        )
        t, gy, gx = (int(x) for x in out["image_grid_thw"][0])
        assert (gh, gw, n) == (gy * 16, gx * 16, image_tokens_for_grid([[t, gy, gx]], 2)[0])
        # The processor's patches, put back together as the image: its pixels are to_grid's.
        pv = out["pixel_values"].reshape(gy // 2, gx // 2, 2, 2, 3, 2, 16, 16)[:, :, :, :, :, 0]
        pv = pv.transpose(0, 2, 5, 1, 3, 6, 4).reshape(gy * 16, gx * 16, 3)
        assert np.array_equal(np.frombuffer(rgb, np.uint8).reshape(gh, gw, 3), pv.astype(np.uint8))


def test_hidden_engine_fits_images_as_vision_does(engines, tmp_path):
    _, vision = engines
    state = "First words. " + "filler " * 20 + "Last words."
    imgs = [
        fit_image(decode_image(_b64(w, h, i)), 448)
        for i, (w, h) in enumerate([(320, 320), (448, 200)])
    ]
    texts = [render_question(q).text for q in QUESTIONS.values()]
    s, q, _, offs = vision._fit_images(state, imgs, texts)
    tokens = [to_grid(img)[3] + 2 for img in imgs]
    segments, hq, hoffs = _hidden_engine(vision, tmp_path).fit_images(state, tokens, texts)
    start, pad, end = (
        vision.tok.convert_tokens_to_ids(t) for t in (VISION_START, IMAGE_PAD, VISION_END)
    )
    flat: list[int] = []
    for i, segment in enumerate(segments):
        flat += segment + ([start] + [pad] * (tokens[i] - 2) + [end] if i < len(tokens) else [])
    assert (flat, hq) == (s, q)
    assert hoffs == [[tuple(o) for o in x] for x in offs]


def test_hidden_engine_refuses_a_prompt_over_the_window_as_vision_does(engines, tmp_path):
    _, vision = engines
    original = vision.model.config.max_length
    state = "First words. " + "filler " * 200 + "Last words."
    imgs = [fit_image(decode_image(_b64(320, 320, 4)), 448)]
    texts = [render_question(QUESTIONS["signed"]).text]
    try:
        with pytest.raises(ValueError, match="context window of 700 tokens") as want:
            _small_window(vision, 700)._fit_images(state, imgs, texts)
    finally:
        vision.model.config.max_length = original
    hidden = _hidden_engine(vision, tmp_path, max_length=700)
    with pytest.raises(ValueError) as got:
        hidden.fit_images(state, [to_grid(imgs[0])[3] + 2], texts)
    assert str(got.value) == str(want.value)


@pytest.mark.parametrize(
    "sizes",
    [[(600, 800)], [(448, 448), (1000, 200)], [(320, 320), (448, 200), (64, 64), (300, 448)]],
    ids=["one image", "two images", "four images"],
)
def test_hidden_engine_answers_over_images_as_vision_does(engines, tmp_path, sizes):
    _, vision = engines
    req = SystemOneRequest(
        state="Checkout page after the user tapped pay.",
        questions=QUESTIONS,
        images=[_b64(w, h, i) for i, (w, h) in enumerate(sizes)],
    )
    want = vision.evaluate(req)
    backend = TorchImageBackend(vision)
    got = _hidden_engine(vision, tmp_path, backend).evaluate(req)
    assert backend.calls == ["forward_images"]
    assert got.usage == want.usage
    have, expected = _probabilities(got), _probabilities(want)
    for name in expected:
        assert have[name] == pytest.approx(expected[name], abs=1.5e-4), name  # both round to 4


def test_hidden_engine_sends_text_requests_to_forward(engines, tmp_path):
    _, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    backend = TorchImageBackend(vision)
    got = _hidden_engine(vision, tmp_path, backend).evaluate(req)
    assert backend.calls == ["forward"]
    have, expected = _probabilities(got), _probabilities(vision.evaluate(req))
    for name in expected:
        assert have[name] == pytest.approx(expected[name], abs=1.5e-4), name


@pytest.mark.parametrize(
    ("state", "size"),
    [("Checkout.", (2000, 8)), (f"see {IMAGE_PAD} here", (320, 320))],
    ids=["aspect ratio over 200", "an image pad in the state"],
)
def test_hidden_engine_refuses_what_vision_refuses_with_its_message(engines, tmp_path, state, size):
    _, vision = engines
    req = SystemOneRequest(state=state, questions=QUESTIONS, images=[_b64(*size, 3)])
    with pytest.raises(ValueError) as want:
        vision.evaluate(req)
    with pytest.raises(ValueError) as got:
        _hidden_engine(vision, tmp_path).evaluate(req)
    assert str(got.value) == str(want.value)


def test_hidden_engine_refuses_images_it_cannot_read(engines, tmp_path):
    _, vision = engines
    one = SystemOneRequest(state="Checkout.", questions=QUESTIONS, images=[_b64(160, 160, 1)])
    with pytest.raises(ValueError, match="no vision tower"):
        _hidden_engine(vision, tmp_path, TextOnlyBackend(vision)).evaluate(one)
    five = one.model_copy(update={"images": [_b64(64, 64, i) for i in range(5)]})
    for engine in (vision, _hidden_engine(vision, tmp_path)):
        with pytest.raises(ValueError, match="5 images; this server takes at most 4"):
            engine.evaluate(five)


def test_a_backend_that_sizes_images_otherwise_is_named(engines, tmp_path):
    _, vision = engines
    req = SystemOneRequest(state="Checkout.", questions=QUESTIONS, images=[_b64(320, 320, 2)])
    engine = _hidden_engine(vision, tmp_path, MisSizingBackend(vision))
    with pytest.raises(RuntimeError, match=r"backend made \[103\] image tokens; expected \[102\]"):
        engine.evaluate(req)


def test_hidden_engine_fits_exactly_the_window_vision_does(engines, tmp_path):
    """At the window's edge: a prompt of exactly max_length tokens fits both engines, and
    one token more is refused by both."""
    _, vision = engines
    state = "First words. " + "filler " * 20 + "Last words."
    imgs = [fit_image(decode_image(_b64(320, 320, 6)), 448)]
    texts = [render_question(q).text for q in QUESTIONS.values()]
    s, q, _, _ = vision._fit_images(state, imgs, texts)
    exact = len(s) + max(len(x) for x in q)
    tokens = [to_grid(imgs[0])[3] + 2]
    original = vision.model.config.max_length
    try:
        for window, fits in ((exact, True), (exact - 1, False)):
            torch_engine = _small_window(vision, window)
            hidden = _hidden_engine(vision, tmp_path / str(window), max_length=window)
            for fit in (
                partial(torch_engine._fit_images, state, imgs, texts),
                partial(hidden.fit_images, state, tokens, texts),
            ):
                if fits:
                    fit()
                else:
                    with pytest.raises(ValueError, match=f"context window of {window} tokens"):
                        fit()
    finally:
        vision.model.config.max_length = original


def test_hidden_engine_adds_special_tokens_once_before_the_state(engines, tmp_path):
    """The state is tokenized in segments around its images. With a tokenizer that adds a
    BOS, the segments together must be the whole state's tokens: one BOS, at the start."""
    from tokenizers.processors import TemplateProcessing

    _, vision = engines
    engine = _hidden_engine(vision, tmp_path)
    bos = engine.tok.token_to_id("<|endoftext|>")
    engine.tok.post_processor = TemplateProcessing(
        single="<|endoftext|> $A", special_tokens=[("<|endoftext|>", bos)]
    )
    state = "Before the images. And after them."
    texts = [render_question(QUESTIONS["signed"]).text]
    segments, _, _ = engine.fit_images(state, [7, 9], texts)
    whole = engine.tok.encode(
        render_image_state(state, 2).replace(
            f"{VISION_START}{IMAGE_PAD}{VISION_END}", f"{VISION_START}{VISION_END}"
        )
    ).ids
    start, end = (engine.tok.token_to_id(t) for t in (VISION_START, VISION_END))
    flat: list[int] = []
    for i, segment in enumerate(segments):
        flat += segment + ([start, end] if i < 2 else [])
    assert flat == whole
    assert flat.count(bos) == 1 and flat[0] == bos
