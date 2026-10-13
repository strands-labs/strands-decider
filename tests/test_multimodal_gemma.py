"""Image and audio input on a Gemma 4 torso, on a tiny random-weight Gemma 4 built here.

Nothing is downloaded. The base is a `Gemma4ForConditionalGeneration` with a 4-layer
decoder (sliding, full and KV-shared layers, an 8-token sliding window), a 1-layer vision
encoder and a 1-layer audio encoder, saved with a real `GemmaTokenizer` that holds Gemma 4's
image and audio tokens, and a `processor_config.json` nesting a `Gemma4ImageProcessorPil`
(70 soft tokens per image; E2B and E4B ship 280) and a `Gemma4AudioFeatureExtractor`, as
google/gemma-4-E4B-it ships them. The tests pin what the feature promises:

* a TEXT Gemma checkpoint loads onto the multimodal torso: encoders kept and frozen, the
  audio encoder only with `audio=True`, weights exactly the base's, the adapter remapped
  onto the same decoder modules;
* text requests to an image or audio engine get the text engine's answers;
* an image is `<|image>` + one `<|image|>` per soft token + `<image|>`, an audio clip
  `<|audio>` + one `<|audio|>` per 40 ms + `<audio|>`, images first, then clips, then text;
  the counts are what the encoders emit;
* over images, audio, and both (prefixes far past the sliding window, forked across three
  questions) the shared-prefix path answers as a plain full forward of each prompt;
* clips are decoded to 16 kHz mono, and long, malformed or unsupported clips are refused;
* engines and servers without an audio encoder refuse audio instead of ignoring it.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re

import pytest

transformers = pytest.importorskip("transformers")
PIL = pytest.importorskip("PIL")
sf = pytest.importorskip("soundfile")
pytest.importorskip("scipy")
if tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2]) < (5, 18):
    pytest.skip("image input needs transformers >= 5.18", allow_module_level=True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from strands_decider.audio import (  # noqa: E402
    SAMPLE_RATE,
    audio_token_counts,
    decode_audio,
    load_feature_extractor,
    process_audio,
)
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
    GEMMA,
    VisionDeciderModel,
    VisionEngine,
    VisionEngineConfig,
    decode_image,
    expand_image_tokens,
    load_image_processor,
    load_vision_engine,
    process_images,
    render_image_state,
)

IMAGE_TOKENS = ("<|image>", "<|image|>", "<image|>")
AUDIO_TOKENS = ("<|audio>", "<|audio|>", "<audio|>")
ATTENTION_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# The tiny Gemma's last two layers share earlier layers' keys and values: no k_proj / v_proj.
LORA = {(i, m) for i in range(4) for m in ATTENTION_MLP if i < 2 or m not in ("k_proj", "v_proj")}
STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "signed": NoulQuestion(instructions="Is the form signed?"),
    "button": ChoiceQuestion(instructions="Which control next?",
                             criteria={"submit": "the blue submit button", "cancel": "the grey cancel link"}),
    "sharp": ScoreQuestion(instructions="How sharp?", criteria=["blurred", "soft", "sharp"]),
}


# ---- the tiny base ------------------------------------------------------------------------


def _tokenizer():
    specials = ["<pad>", "<eos>", "<bos>", "<unk>", "<mask>"]
    pieces = specials + [f"<0x{b:02X}>" for b in range(256)] + ["▁"]
    pieces += [chr(c) for c in range(33, 127)] + ["\n"]
    tok = transformers.GemmaTokenizer(vocab={p: i for i, p in enumerate(pieces)}, merges=[],
                                      add_bos_token=True)
    tok.add_special_tokens({"additional_special_tokens": list(IMAGE_TOKENS + AUDIO_TOKENS)})
    return tok


def save_base(d: str) -> str:
    tok = _tokenizer()
    names = ("boi_token_id", "image_token_id", "eoi_token_id", "boa_token_id", "audio_token_id", "eoa_token_id")
    ids = dict(zip(names, tok.convert_tokens_to_ids(list(IMAGE_TOKENS + AUDIO_TOKENS)), strict=True))
    cfg = transformers.Gemma4Config(
        text_config={
            "hidden_size": 64, "num_hidden_layers": 4, "intermediate_size": 128, "head_dim": 16,
            "global_head_dim": 32, "num_attention_heads": 2, "num_key_value_heads": 2,
            "vocab_size": len(tok), "vocab_size_per_layer_input": len(tok),
            "hidden_size_per_layer_input": 8, "num_kv_shared_layers": 2, "sliding_window": 8,
            "layer_types": ["sliding_attention", "full_attention"] * 2,
            "tie_word_embeddings": True, "pad_token_id": 0, "eos_token_id": 1, "bos_token_id": 2,
            "final_logit_softcapping": 30.0,
        },
        vision_config={"hidden_size": 32, "num_hidden_layers": 1, "intermediate_size": 64,
                       "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 16,
                       "global_head_dim": 16},
        audio_config={"hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                      "subsampling_conv_channels": [128, 4], "output_proj_dims": 32},
        **ids,
    )
    torch.manual_seed(0)
    transformers.Gemma4ForConditionalGeneration(cfg).save_pretrained(d)
    tok.save_pretrained(d)
    # One processor_config.json nesting both, as google/gemma-4-E4B-it ships them.
    nested = {"processor_class": "Gemma4Processor",
              "image_processor": transformers.Gemma4ImageProcessorPil(max_soft_tokens=70).to_dict(),
              "feature_extractor": transformers.Gemma4AudioFeatureExtractor().to_dict()}
    with open(os.path.join(d, "processor_config.json"), "w") as fh:
        json.dump(nested, fh, default=str)
    return d


@pytest.fixture(scope="module")
def base_dir(tmp_path_factory):
    return save_base(str(tmp_path_factory.mktemp("tiny-gemma4-mm")))


@pytest.fixture(scope="module")
def ckpt(base_dir, tmp_path_factory):
    """A TEXT Strands Decider checkpoint on the tiny Gemma 4, with a non-trivial adapter."""
    cfg = StrandsDeciderConfig(base_model=base_dir, head_type="pointer", pointer_dim=16,
                               torch_dtype="float32", max_length=2048, lora_targets=ATTENTION_MLP,
                               temperature_by_kind={"noul": 0.9, "choice": 0.7}, force_bos=True)
    model = StrandsDeciderModel.from_pretrained_base(cfg)
    with torch.no_grad():
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
    d = str(tmp_path_factory.mktemp("gemma-text-ckpt"))
    model.save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def engines(ckpt):
    text = SystemOneEngine(StrandsDeciderModel.load(ckpt), EngineConfig(device="cpu"))
    vision = VisionEngine(VisionDeciderModel.load(ckpt), VisionEngineConfig(device="cpu"))
    audio = VisionEngine(VisionDeciderModel.load(ckpt, audio=True), VisionEngineConfig(device="cpu"))
    return text, vision, audio


def _png(w: int, h: int, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    buf = io.BytesIO()
    Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _wav(seconds: float, seed: int, rate: int = SAMPLE_RATE, channels: int = 1, fmt: str = "WAV") -> str:
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * rate)) / rate
    tone = 0.3 * np.sin(2 * np.pi * (220 + 110 * seed) * t) + 0.05 * rng.standard_normal(t.size)
    data = np.stack([tone] * channels, axis=1) if channels > 1 else tone
    buf = io.BytesIO()
    sf.write(buf, data.astype(np.float32), rate, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ---- the torso ----------------------------------------------------------------------------


def test_torso_keeps_audio_only_when_asked(engines):
    _, vision, audio = engines
    for engine, has_audio in ((vision, False), (audio, True)):
        model = engine.model
        base = model.torso.base_model.model
        assert model.family == GEMMA and type(base).__name__ == "Gemma4Model"
        assert model.has_audio is has_audio
        assert (base.audio_tower is not None) is has_audio and (base.embed_audio is not None) is has_audio
        frozen = [n for n, p in model.torso.named_parameters() if not n.startswith("base_model.model.language_model.")]
        assert frozen and not any(p.requires_grad for n, p in model.torso.named_parameters() if n in frozen)
        assert not any(p.requires_grad for p in model.torso.parameters())  # served: nothing trains
        assert (engine.feature_extractor is not None) is has_audio


def test_torso_weights_are_the_bases(base_dir):
    cfg = StrandsDeciderConfig(base_model=base_dir, torch_dtype="float32", use_lora=False)
    want = transformers.Gemma4ForConditionalGeneration.from_pretrained(base_dir).model.state_dict()
    for audio in (False, True):
        torso = VisionDeciderModel._load_torso(cfg, None, None, audio=audio)
        got = torso.state_dict()
        expect = {k: v for k, v in want.items() if audio or not k.startswith(("audio_tower.", "embed_audio."))}
        assert got.keys() == expect.keys()
        for k in expect:
            torch.testing.assert_close(got[k], expect[k], rtol=0, atol=0, msg=k)
        assert not any(p.requires_grad for n, p in torso.named_parameters() if not n.startswith("language_model."))


def test_adapter_is_the_text_checkpoints_on_the_same_modules(ckpt, engines):
    _, _, audio = engines
    found = set()
    for name, _ in audio.model.torso.named_modules():
        m = re.search(r"language_model\.layers\.(\d+)\..*\.(\w+)\.lora_A$", name)
        if m:
            found.add((int(m.group(1)), m.group(2)))
    assert found == LORA
    assert not any("lora_" in n and ".language_model." not in n for n, _ in audio.model.torso.named_parameters())
    saved = load_file(os.path.join(ckpt, "lora", "adapter_model.safetensors"))
    assert saved and not any(".language_model." in k for k in saved)
    params = dict(audio.model.torso.named_parameters())
    for k, v in saved.items():
        name = k.replace("base_model.model.", "base_model.model.language_model.", 1)
        name = name.replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight")
        torch.testing.assert_close(params[name], v, rtol=0, atol=0, msg=k)


def test_processors_read_from_the_base(engines, base_dir):
    _, vision, audio = engines
    assert type(vision.image_processor).__name__ == "Gemma4ImageProcessorPil"
    assert vision.image_processor.max_soft_tokens == 70  # the base's, not the default 280
    assert type(audio.feature_extractor).__name__ == "Gemma4AudioFeatureExtractor"
    assert vision.long_side == 0  # Gemma's processor fixes the budget; no pre-cap
    assert type(load_image_processor(StrandsDeciderConfig(base_model=base_dir))).__name__ == "Gemma4ImageProcessorPil"
    assert load_feature_extractor(base_dir).sampling_rate == SAMPLE_RATE


# ---- the prompt ---------------------------------------------------------------------------


def test_state_layout_images_then_audio_then_text():
    boi, img, eoi = IMAGE_TOKENS
    boa, aud, eoa = AUDIO_TOKENS
    assert render_image_state("hi", 1, GEMMA, n_audio=2) == (
        f"<state>\n{boi}{img}{eoi}\n{boa}{aud}{eoa}\n{boa}{aud}{eoa}\nhi\n</state>\n")
    assert render_image_state("", 0, GEMMA, n_audio=1) == f"<state>\n{boa}{aud}{eoa}\n</state>\n"
    assert render_image_state("text", 0, GEMMA) == "<state>\ntext\n</state>\n"
    with pytest.raises(ValueError):
        render_image_state("x", 0, "qwen3_5", n_audio=1)  # Qwen3.5 has no audio encoder


def test_audio_token_counts_are_40ms_each_and_what_the_encoder_emits(engines):
    _, _, audio = engines
    fe = audio.feature_extractor
    clips = [decode_audio(_wav(s, i)) for i, s in enumerate((1.0, 2.5, 0.3))]
    counts, mm = process_audio(fe, clips)
    assert counts == [25, 63, 8]  # ceil(ms / 40), as Gemma4Processor computes them
    with torch.no_grad():
        out = audio.model.torso.base_model.model.get_audio_features(
            mm["input_features"], mm["input_features_mask"], return_dict=True)
    assert out.attention_mask.sum(-1).tolist() == counts
    assert audio_token_counts(mm["input_features_mask"]) == counts


@pytest.mark.parametrize("sizes", [[(200, 100)], [(64, 48), (50, 300)]])
def test_image_token_counts_are_what_the_vision_encoder_emits(engines, sizes):
    _, vision, _ = engines
    cfg = vision.model.torso.config
    images = [Image.new("RGB", s, (30 * i, 90, 160)) for i, s in enumerate(sizes)]
    counts, mm = process_images(vision.image_processor, images)
    assert all(0 < c <= 70 for c in counts)
    ids = vision.tok(expand_image_tokens(render_image_state("x", len(images), GEMMA), counts, GEMMA))["input_ids"]
    assert ids.count(cfg.image_token_id) == sum(counts)
    with torch.no_grad():
        out = vision.model.torso(input_ids=torch.tensor([ids]), **mm)
    assert out.image_hidden_states.shape[0] == sum(counts)


# ---- answers ------------------------------------------------------------------------------


def test_text_requests_unchanged(engines):
    text, vision, audio = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    want = text.evaluate(req).model_dump()["answers"]
    assert vision.evaluate(req).model_dump()["answers"] == want
    assert audio.evaluate(req).model_dump()["answers"] == want


def _reference(engine: VisionEngine, state: str, images: list[str], clips: list[str], q) -> list[float]:
    """Plain full forward of state + one question, positions left to transformers."""
    rq = render_question(q)
    text, mm = engine.media_prompt(state, [decode_image(b) for b in images], [decode_audio(c) for c in clips])
    text += rq.text
    enc = engine.tok(text, return_offsets_mapping=True)
    opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
    ids = torch.tensor([enc["input_ids"]])
    with torch.no_grad():
        out = engine.model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots]), opt_idx=torch.tensor([opt]),
                           temperature=engine._temperatures([rq.kind]), **mm)
    return masked_log_softmax(out["logits"], torch.tensor([rq.n_slots])).exp()[0, : rq.n_slots].tolist()


@pytest.mark.parametrize(("sizes", "seconds"), [
    ([], [1.0]),                         # audio only
    ([], [0.6, 1.4]),                    # two clips of different lengths (padded batch)
    ([(600, 800)], [0.8]),               # an image and a clip
    ([(448, 448), (1000, 200)], []),     # images only
])
def test_answers_match_full_forward(engines, sizes, seconds):
    """The cached prefix (media + state, far past the 8-token sliding window) forked across
    three questions answers as each full prompt does."""
    _, _, audio = engines
    images = [_png(w, h, i) for i, (w, h) in enumerate(sizes)]
    clips = [_wav(s, i) for i, s in enumerate(seconds)]
    rendered = [render_question(q) for q in QUESTIONS.values()]
    state = "Checkout page after the user tapped pay."
    probs, ntok = audio._image_probs(state, [decode_image(b) for b in images], rendered,
                                     [decode_audio(c) for c in clips])
    assert ntok > 8 * (len(images) + len(clips))
    for i, q in enumerate(QUESTIONS.values()):
        ref = _reference(audio, state, images, clips, q)
        got = probs[i, : len(ref)].tolist()
        assert max(abs(x - y) for x, y in zip(got, ref, strict=True)) < 1e-5, (i, got, ref)


def test_audio_request_answers_through_the_engine(engines):
    _, _, audio = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS, audio=["data:audio/wav;base64," + _wav(1.2, 1)])
    out = audio.evaluate(req)
    assert 0.0 <= out.answers["signed"].noul <= 1.0
    assert sum(out.answers["button"].probabilities.values()) == pytest.approx(1.0, abs=1e-5)
    other = audio.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS, audio=[_wav(1.2, 3)]))
    blind = audio.evaluate(SystemOneRequest(state=STATE, questions=QUESTIONS))
    assert out.answers["signed"].noul != pytest.approx(blind.answers["signed"].noul, abs=1e-6)  # the clip counts
    assert out.answers["signed"].noul != pytest.approx(other.answers["signed"].noul, abs=1e-6)  # and which clip
    # the cached prefix is counted once: 30 audio tokens, <|audio>, <audio|> and a newline
    assert out.usage.input_tokens == blind.usage.input_tokens + 30 + 3


# ---- decoding and refusals ----------------------------------------------------------------


def test_decode_mixes_to_mono_and_resamples():
    wave = decode_audio(_wav(1.5, 0, rate=44_100, channels=2))
    assert wave.dtype == np.float32 and wave.ndim == 1
    assert abs(len(wave) - int(1.5 * SAMPLE_RATE)) <= 1
    assert len(decode_audio(_wav(0.5, 0, rate=8_000, fmt="FLAC"))) == SAMPLE_RATE // 2


def test_refusals(engines):
    text, vision, audio = engines
    clip = _wav(0.5, 0)
    req = SystemOneRequest(state="x", questions={"signed": QUESTIONS["signed"]}, audio=[clip])
    for engine in (text, vision):  # no audio encoder: refused, never silently ignored
        with pytest.raises(ValueError, match="--audio"):
            engine.evaluate(req)
    with pytest.raises(ValueError, match="at most 30 s"):
        decode_audio(_wav(30.5, 0))
    with pytest.raises(ValueError, match="base64"):
        decode_audio("not base64!")
    with pytest.raises(ValueError, match="could not be read"):
        decode_audio(base64.b64encode(b"RIFF....WAVEjunk").decode())
    with pytest.raises(ValueError, match="at most 2"):
        audio.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, audio=[clip] * 3))
    big = VisionEngineConfig(device="cpu", strict_window=True)
    big.max_audio_seconds = 31.0
    narrow = VisionEngine(audio.model, big, audio.image_processor, audio.feature_extractor)
    narrow.model.config.max_length, old = 300, narrow.model.config.max_length
    try:  # 750 audio tokens do not fit a 300-token window
        with pytest.raises(ValueError, match="exceeds the context window"):
            narrow.evaluate(SystemOneRequest(state="x", questions=QUESTIONS, audio=[_wav(30.0, 0)]))
    finally:
        narrow.model.config.max_length = old


def test_qwen_has_no_audio(base_dir, tmp_path):
    cfg = StrandsDeciderConfig(base_model=base_dir, torch_dtype="float32", use_lora=False)
    with pytest.raises(ValueError, match="no audio encoder"):
        no_audio = os.path.join(tmp_path, "noaudio")
        os.makedirs(no_audio)
        c = json.load(open(os.path.join(base_dir, "config.json")))
        c["audio_config"] = None
        for f in os.listdir(base_dir):
            if f != "config.json":
                os.symlink(os.path.join(base_dir, f), os.path.join(no_audio, f))
        json.dump(c, open(os.path.join(no_audio, "config.json"), "w"))
        cfg.base_model = no_audio
        VisionDeciderModel._load_torso(cfg, None, None, audio=True)


# ---- serving ------------------------------------------------------------------------------


def test_load_vision_engine_with_audio(ckpt):
    eng = load_vision_engine(ckpt, EngineConfig(device="cpu", max_batch=2, model_name="mm"), audio=True)
    assert eng.model.has_audio and (eng.cfg.max_batch, eng.cfg.model_name) == (2, "mm")


def test_server_routes_audio(ckpt):
    from fastapi.testclient import TestClient

    from strands_decider import server

    body = {"state": "Support call.", "audio": [_wav(0.7, 2)], "images": [_png(120, 90, 1)],
            "questions": {"angry": {"type": "noul", "instructions": "Is the caller angry?"}}}
    app = TestClient(server.create_app(ckpt, device="cpu", audio=True))
    r = app.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["answers"]["angry"]["noul"] <= 1.0
    health = app.get("/health").json()
    assert health["audio"] is True and health["vision"] is True
    assert app.post("/v1/systemone", json={**body, "audio": [_wav(31, 0)]}).status_code == 422
    vision_only = TestClient(server.create_app(ckpt, device="cpu", vision=True))
    assert vision_only.post("/v1/systemone", json=body).status_code == 422
    text_only = TestClient(server.create_app(ckpt, device="cpu"))
    assert text_only.post("/v1/systemone", json=body).status_code == 422
