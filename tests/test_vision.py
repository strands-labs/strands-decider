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
* the window rules match the text path: `strict_window` refuses, otherwise the state text
  loses its end and the images are never cut;
* only the promised image formats are opened, sizes are checked before decoding, and
  rotation and transparency are handled;
* a text-only engine refuses images instead of ignoring them, at the engine and the API.
"""

from __future__ import annotations

import base64
import io
import re
from dataclasses import replace

import pytest

transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    StrandsDeciderConfig,
    StrandsDeciderModel,
    masked_log_softmax,
)
from strands_decider.prompting import render_question  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from strands_decider.vision import (  # noqa: E402
    IMAGE_PAD,
    VISION_END,
    VisionDeciderModel,
    VisionEngine,
    VisionEngineConfig,
    decode_image,
    expand_image_tokens,
    fit_image,
    image_tokens_for_grid,
    load_vision_engine,
    render_image_state,
)

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
           "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]
STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "signed": NoulQuestion(instructions="Is the form signed?"),
    "button": ChoiceQuestion(instructions="Which control next?",
                             criteria={"submit": "the blue submit button", "cancel": "the grey cancel link"}),
    "sharp": ScoreQuestion(instructions="How sharp?", criteria=["blurred", "soft", "sharp"]),
}


def _tokenizer():
    from tokenizers import pre_tokenizers

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {"<|endoftext|>": 0, **{c: i + 1 for i, c in enumerate(alphabet)}}
    tok = transformers.Qwen3_5Tokenizer(vocab=vocab, merges=[])
    tok.add_special_tokens({"additional_special_tokens": ["<|vision_start|>", IMAGE_PAD, VISION_END]})
    return tok


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    """A tiny Qwen3.5 multimodal checkpoint: 3 Gated DeltaNet + 1 attention layer, 2-layer ViT."""
    d = tmp_path_factory.mktemp("tiny-qwen35")
    tok = _tokenizer()
    ids = {t: tok.convert_tokens_to_ids(t) for t in ("<|vision_start|>", IMAGE_PAD, VISION_END)}
    cfg = transformers.Qwen3_5Config(
        text_config={
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128, "head_dim": 64,
            "num_attention_heads": 2, "num_key_value_heads": 1, "vocab_size": len(tok),
            "layer_types": ["linear_attention"] * 3 + ["full_attention"],
            "linear_num_key_heads": 4, "linear_num_value_heads": 4,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16, "linear_conv_kernel_dim": 4,
            "attn_output_gate": True, "tie_word_embeddings": True, "mtp_num_hidden_layers": 0,
            "rope_parameters": {"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25,
                                "mrope_section": [3, 3, 2], "mrope_interleaved": True},
        },
        vision_config={"depth": 2, "hidden_size": 64, "num_heads": 4, "intermediate_size": 128,
                       "out_hidden_size": 64, "patch_size": 16, "spatial_merge_size": 2,
                       "temporal_patch_size": 2, "num_position_embeddings": 2304},
        image_token_id=ids[IMAGE_PAD], vision_start_token_id=ids["<|vision_start|>"],
        vision_end_token_id=ids[VISION_END], tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    transformers.Qwen3_5ForConditionalGeneration(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    transformers.Qwen2VLImageProcessorPil(
        patch_size=16, temporal_patch_size=2, merge_size=2,
        size={"shortest_edge": 65536, "longest_edge": 16777216},
        image_mean=[0.5] * 3, image_std=[0.5] * 3,
    ).save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def ckpt(base_dir, tmp_path_factory):
    """A text Strands Decider checkpoint on that base, with a non-trivial adapter."""
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="pointer", pointer_dim=32,
                               torch_dtype="float32", max_length=1024, lora_targets=TARGETS,
                               temperature_by_kind={"noul": 0.9, "choice": 0.7})
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
    assert two == ("<state>\n<|vision_start|><|image_pad|><|vision_end|>\n"
                   "<|vision_start|><|image_pad|><|vision_end|>\ntwo pages\n</state>\n")
    assert render_image_state("", 1) == "<state>\n<|vision_start|><|image_pad|><|vision_end|>\n</state>\n"
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
    eps = base64.b64encode(b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 10 10\nshowpage\n").decode()
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
    assert text.evaluate(req).model_dump()["answers"] == vision.evaluate(req).model_dump()["answers"]


def _reference(vision: VisionEngine, state: str, images: list[str], q) -> list[float]:
    """Plain full forward of state + one question, positions left to transformers."""
    rq = render_question(q)
    pil = [fit_image(decode_image(b), 448) for b in images]
    proc = vision.image_processor(images=pil, return_tensors="pt")
    counts = image_tokens_for_grid(proc["image_grid_thw"].tolist(), vision.image_processor.merge_size)
    text = expand_image_tokens(render_image_state(state, len(images)), counts) + rq.text
    enc = vision.tok(text, return_offsets_mapping=True)
    opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
    ids = torch.tensor([enc["input_ids"]])
    with torch.no_grad():
        out = vision.model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots]), opt_idx=torch.tensor([opt]),
                           temperature=vision._temperatures([rq.kind]), pixel_values=proc["pixel_values"],
                           image_grid_thw=proc["image_grid_thw"])
    return masked_log_softmax(out["logits"], torch.tensor([rq.n_slots])).exp()[0, : rq.n_slots].tolist()


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


def _small_window(vision: VisionEngine, max_length: int, strict: bool) -> VisionEngine:
    model = vision.model
    eng = VisionEngine.__new__(VisionEngine)
    eng.__dict__.update(vision.__dict__)
    eng.cfg = eng.vcfg = replace(vision.vcfg, strict_window=strict)
    eng.model = model
    model.config.max_length = max_length
    return eng


def test_long_state_keeps_its_head_and_the_images(engines):
    _, vision = engines
    original = vision.model.config.max_length
    try:
        eng = _small_window(vision, 700, strict=False)
        state = "First words stay. " + "filler " * 200 + "Last words go."
        img = [fit_image(decode_image(_b64(320, 320, 4)), 448)]
        s, _, _ = eng._fit_images(state, img, [render_question(QUESTIONS["signed"]).text])
        assert s.count(eng.tok.convert_tokens_to_ids(IMAGE_PAD)) == 100  # the image is whole
        kept = eng.tok.decode(s)
        assert "First words stay." in kept and "Last words go." not in kept
        with pytest.raises(ValueError, match="context window"):
            _small_window(vision, 700, strict=True)._fit_images(
                state, img, [render_question(QUESTIONS["signed"]).text])
    finally:
        vision.model.config.max_length = original


# ---- engine and API ------------------------------------------------------------------------


def test_image_only_request_and_refusals(engines):
    text, vision = engines
    req = SystemOneRequest(state="", questions={"signed": QUESTIONS["signed"]},
                           images=["data:image/png;base64," + _b64(320, 320, 2)])
    assert 0.0 <= vision.evaluate(req).answers["signed"].noul <= 1.0
    with pytest.raises(ValueError):
        text.evaluate(req)  # a text-only engine refuses images rather than ignore them
    with pytest.raises(ValueError):
        vision.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, images=["not base64!"]))


def test_load_vision_engine_keeps_the_server_settings(ckpt):
    eng = load_vision_engine(ckpt, EngineConfig(device="cpu", strict_window=True, max_batch=3,
                                                model_name="vd-test"))
    assert (eng.cfg.strict_window, eng.cfg.max_batch, eng.cfg.model_name) == (True, 3, "vd-test")


def test_server_routes_images(ckpt):
    from fastapi.testclient import TestClient

    from strands_decider import server

    body = {"state": "Checkout page.", "images": [_b64(320, 320, 3)],
            "questions": {"signed": {"type": "noul", "instructions": "Is the form signed?"}}}
    vision_app = TestClient(server.create_app(ckpt, device="cpu", vision=True))
    r = vision_app.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["answers"]["signed"]["noul"] <= 1.0
    assert vision_app.get("/health").json()["vision"] is True
    text_app = TestClient(server.create_app(ckpt, device="cpu"))
    assert text_app.post("/v1/systemone", json=body).status_code == 422
