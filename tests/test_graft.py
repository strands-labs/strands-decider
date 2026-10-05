"""Grafted eyes for a text-only torso (src/strands_decider/graft.py), on the tiny SigLIP and
Llama of tests/tiny_graft.py; nothing is downloaded. The tests pin what the feature
promises (its training: test_graft_train.py):

* the projector's shapes, and that its pixel-unshuffle groups neighbouring patches;
* each image becomes `tokens_per_image` slots between plain-text markers, 144 for the
  real SigLIP2 so400m at 384 px;
* the projector's outputs land exactly at the image slots, in order, and nowhere else;
* text-only requests answer exactly as the text engine does;
* the shared-prefix path over 1 and 2 images equals a plain full forward;
* the window cuts the state text, never an image;
* a checkpoint saves and loads whole, projector included, and the server and the
  evaluation load it.
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
from tiny_graft import (  # noqa: E402
    SLOT,
    projector_config,
    save_minicpm_like,
    save_siglip,
    save_stage1,
    save_text_checkpoint,
)

from strands_decider.graft import (  # noqa: E402
    GraftedDeciderModel,
    GraftedVisionEngine,
    Projector,
    ProjectorConfig,
    encode_images,
    load_projector_state,
    place_images,
)
from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index  # noqa: E402
from strands_decider.modeling import StrandsDeciderModel, masked_log_softmax  # noqa: E402
from strands_decider.prompting import render_question  # noqa: E402
from strands_decider.schema import (  # noqa: E402
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
    SystemOneRequest,
)
from strands_decider.vision import (  # noqa: E402
    decode_image,
    fit_image,
    is_grafted,
    load_vision_engine,
    mm_base,
)

STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "signed": NoulQuestion(instructions="Is the form signed?"),
    "button": ChoiceQuestion(instructions="Which control next?",
                             criteria={"submit": "the blue submit button", "cancel": "the grey cancel link"}),
    "sharp": ScoreQuestion(instructions="How sharp?", criteria=["blurred", "soft", "sharp"]),
}


@pytest.fixture(scope="module")
def enc(tmp_path_factory):
    return save_siglip(str(tmp_path_factory.mktemp("siglip")))


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    return save_minicpm_like(str(tmp_path_factory.mktemp("minicpm-like")))


@pytest.fixture(scope="module")
def text_ckpt(base, tmp_path_factory):
    return save_text_checkpoint(base, str(tmp_path_factory.mktemp("text-ckpt")))


@pytest.fixture(scope="module")
def stage1(enc, tmp_path_factory):
    return save_stage1(enc, str(tmp_path_factory.mktemp("stage1")))


@pytest.fixture(scope="module")
def ckpt(text_ckpt, stage1, tmp_path_factory):
    """A grafted checkpoint: the text checkpoint with the stage-1 projector."""
    d = str(tmp_path_factory.mktemp("grafted"))
    GraftedDeciderModel.load(text_ckpt, projector_from=stage1).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def engines(text_ckpt, ckpt):
    text = SystemOneEngine(StrandsDeciderModel.load(text_ckpt), EngineConfig(device="cpu"))
    vision = load_vision_engine(ckpt, EngineConfig(device="cpu"))
    return text, vision


def _b64(w: int, h: int, seed: int) -> str:
    g = torch.Generator().manual_seed(seed)
    buf = io.BytesIO()
    Image.fromarray((torch.rand(h, w, 3, generator=g) * 255).to(torch.uint8).numpy()).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---- projector and prompt ------------------------------------------------------------------


def test_projector_shapes_and_unshuffle(enc):
    cfg = projector_config(enc)
    assert (cfg.grid, cfg.tokens_per_image) == (8, 16)
    proj = Projector(cfg)
    assert proj(torch.randn(3, 64, 32)).shape == (3, 16, 64)
    assert [type(m).__name__ for m in proj.mlp] == ["Linear", "GELU", "Linear"]
    assert (proj.mlp[0].in_features, proj.mlp[2].out_features) == (4 * 32, 64)
    # patch (r, c) carries r * 8 + c: slot 0 holds patches (0,0), (0,1), (1,0), (1,1); slot 1 (0,2)...
    feats = torch.arange(64.0).view(1, 64, 1).expand(1, 64, 32)
    groups = proj.unshuffle(feats)[0, :, ::32].tolist()
    assert groups[0] == [0, 1, 8, 9] and groups[1] == [2, 3, 10, 11] and groups[4] == [16, 17, 24, 25]
    with pytest.raises(ValueError, match="patch features"):
        proj(torch.randn(1, 63, 32))
    with pytest.raises(ValueError, match="unshuffle"):
        ProjectorConfig(image_size=384, patch_size=16, unshuffle=5)


def test_real_encoder_budget():
    """SigLIP2 so400m at 384 px in 16 px patches: 576 patches, 144 slots after 2 x 2."""
    cfg = ProjectorConfig()
    assert (cfg.encoder, cfg.vision_hidden, cfg.grid, cfg.tokens_per_image) == (
        "google/siglip2-so400m-patch16-384", 1152, 24, 144)
    assert len(cfg.encoder_revision or "") == 40


def test_image_slots_and_expansion(engines):
    _, vision = engines
    assert isinstance(vision, GraftedVisionEngine)
    assert type(vision.image_processor).__name__ == "SiglipImageProcessorPil"  # pinned to PIL
    imgs = [fit_image(decode_image(_b64(w, h, i)), 448) for i, (w, h) in enumerate([(50, 20), (32, 90)])]
    counts, mm = vision.prompter.process(imgs)
    assert counts == [16, 16] and tuple(mm["pixel_values"].shape) == (2, 3, 32, 32)
    text = vision.prompter.state("two pages", counts)
    assert text == f"<state>\n<image>{SLOT * 16}</image>\n<image>{SLOT * 16}</image>\ntwo pages\n</state>\n"
    ids = vision.tok(text)["input_ids"]
    assert ids.count(vision.model.slot_id) == 32
    assert vision.prompter.state("text", []) == "<state>\ntext\n</state>\n"


def test_projector_outputs_land_exactly_at_the_slots(engines):
    _, vision = engines
    model = vision.model
    imgs = [fit_image(decode_image(_b64(40, 40, s)), 448) for s in (1, 2)]
    counts, mm = vision.prompter.process(imgs)
    rq = render_question(QUESTIONS["signed"])
    text = vision.prompter.state("A state.", counts) + rq.text
    enc_ = vision.tok(text, return_offsets_mapping=True)
    ids = torch.tensor([enc_["input_ids"]])
    opt = _option_token_index(enc_["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
    seen = {}

    def grab(_mod, _args, kwargs):
        seen["embeds"] = kwargs["inputs_embeds"].detach().clone()

    handle = mm_base(model.torso).register_forward_pre_hook(grab, with_kwargs=True)
    try:
        with torch.no_grad():
            model(ids, torch.ones_like(ids), torch.tensor([2]), opt_idx=torch.tensor([opt]),
                  pixel_values=mm["pixel_values"])
    finally:
        handle.remove()
    with torch.no_grad():
        want = encode_images(model.encoder, model.projector, mm["pixel_values"]).reshape(-1, 64)
        table = model.torso.get_input_embeddings()(ids)[0]
    got, slots = seen["embeds"][0], ids[0] == model.slot_id
    assert int(slots.sum()) == 32
    assert torch.equal(got[slots], want)  # image 0's 16 slots, then image 1's
    assert torch.equal(got[~slots], table[~slots])  # every other position is its own token
    with pytest.raises(ValueError, match="image slots"):
        place_images(table[None], ids, model.slot_id, want[None, :16])  # 32 slots, 16 vectors
    with pytest.raises(ValueError, match="image slots"):
        place_images(table[None], ids, model.slot_id, None)


def test_text_requests_unchanged(engines):
    text, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    _reference(vision, "Checkout page.", [_b64(64, 64, 9)], QUESTIONS["signed"])  # an image forward first
    assert text.evaluate(req).model_dump()["answers"] == vision.evaluate(req).model_dump()["answers"]
    one = SystemOneRequest(state=STATE, questions={"signed": QUESTIONS["signed"]})  # the batched path
    assert text.evaluate(one).model_dump()["answers"] == vision.evaluate(one).model_dump()["answers"]


def test_the_text_loader_reads_a_grafted_checkpoint(engines, ckpt):
    """`strands-decider calibrate` loads checkpoints with StrandsDeciderModel.load: on a
    grafted one that is the text model `serve --vision` answers text with."""
    _, vision = engines
    req = SystemOneRequest(state=STATE, questions=QUESTIONS)
    text = SystemOneEngine(StrandsDeciderModel.load(ckpt), EngineConfig(device="cpu"))
    assert text.evaluate(req).model_dump()["answers"] == vision.evaluate(req).model_dump()["answers"]


def _reference(vision, state: str, images: list[str], q) -> list[float]:
    """Plain full forward of state + one question, images placed by `forward`."""
    rq = render_question(q)
    pil = [fit_image(decode_image(b), vision.vcfg.image_long_side) for b in images]
    counts, mm = vision.prompter.process(pil)
    text = vision.prompter.state(state, counts) + rq.text
    enc_ = vision.tok(text, return_offsets_mapping=True)
    opt = _option_token_index(enc_["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
    ids = torch.tensor([enc_["input_ids"]])
    with torch.no_grad():
        out = vision.model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots]), opt_idx=torch.tensor([opt]),
                           temperature=vision._image_temperatures([rq.kind]), pixel_values=mm["pixel_values"])
    return masked_log_softmax(out["logits"], torch.tensor([rq.n_slots])).exp()[0, : rq.n_slots].tolist()


def test_window_keeps_the_images_whole(engines):
    """The state text loses its end to the question's reserve; the images never do, and a
    window too small for them (through the closing marker) is refused."""
    _, vision = engines
    img = [fit_image(decode_image(_b64(40, 40, 4)), 448)]
    q = [render_question(QUESTIONS["signed"]).text]
    state = "First words stay. " + "filler " * 50
    counts, _ = vision.prompter.process(img)
    text = vision.prompter.state(state, counts)
    keep = len(vision.tok(text[: text.index("</image>") + len("</image>")])["input_ids"])
    longest = len(vision.tok(q, add_special_tokens=False)["input_ids"][0])
    original, cfg = vision.model.config.max_length, vision.cfg
    try:
        vision.cfg = replace(cfg, max_question_fraction=1.0)  # the question reserves all it needs
        vision.model.config.max_length = keep + longest
        s, _, _ = vision._fit_images(state, img, q)
        assert len(s) == keep and s.count(vision.model.slot_id) == 16
        assert vision.tok.decode(s).endswith("</image>")
        vision.model.config.max_length = keep + longest - 1
        with pytest.raises(ValueError, match="images take"):
            vision._fit_images(state, img, q)
    finally:
        vision.model.config.max_length, vision.cfg = original, cfg


@pytest.mark.parametrize("sizes", [[(600, 800)], [(448, 448), (1000, 200)]])
def test_shared_prefix_equals_full_forward(engines, sizes):
    _, vision = engines
    images = [_b64(w, h, i) for i, (w, h) in enumerate(sizes)]
    rendered = [render_question(q) for q in QUESTIONS.values()]
    pil = [fit_image(decode_image(b), 448) for b in images]
    probs, _ = vision._image_probs("Checkout page after the user tapped pay.", pil, rendered)
    for i, q in enumerate(QUESTIONS.values()):
        ref = _reference(vision, "Checkout page after the user tapped pay.", images, q)
        got = probs[i, : len(ref)].tolist()
        assert max(abs(x - y) for x, y in zip(got, ref, strict=True)) < 1e-5


# ---- checkpoints ------------------------------------------------------------------------------


def test_save_load_round_trip(text_ckpt, stage1, ckpt, tmp_path):
    assert is_grafted(ckpt) and not is_grafted(text_ckpt)
    with pytest.raises(FileNotFoundError, match="projector"):
        GraftedDeciderModel.load(text_ckpt)  # a text checkpoint alone has no eyes
    a = GraftedDeciderModel.load(ckpt)
    a.save_pretrained(str(tmp_path / "again"))
    b = GraftedDeciderModel.load(str(tmp_path / "again"))
    assert b.pcfg == a.pcfg
    for x, y in ((a.projector, b.projector), (a.head, b.head)):
        sa, sb = x.state_dict(), y.state_dict()
        assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)
    la = {n: p for n, p in a.torso.named_parameters() if "lora_" in n}
    lb = dict(b.torso.named_parameters())
    assert la and all(torch.equal(p, lb[n]) for n, p in la.items())
    s1 = load_projector_state(stage1)
    assert all(torch.equal(s1[k], v) for k, v in b.projector.state_dict().items())
    frozen = [*b.projector.parameters(), *b.encoder.parameters(), *la.values()]
    assert not any(p.requires_grad for p in frozen) and not any(p.requires_grad for p in lb.values())
    t = GraftedDeciderModel.load(ckpt, trainable=True)
    assert all(p.requires_grad for p in t.projector.parameters())
    assert not any(p.requires_grad for p in t.encoder.parameters())


# ---- serving and evaluation ------------------------------------------------------------------


def test_server_and_evaluation_take_the_grafted_checkpoint(ckpt):
    from fastapi.testclient import TestClient

    from strands_decider import server

    body = {"state": "Checkout page.", "images": [_b64(320, 320, 3)],
            "questions": {"signed": {"type": "noul", "instructions": "Is the form signed?"}}}
    app = TestClient(server.create_app(ckpt, device="cpu", vision=True))
    r = app.post("/v1/systemone", json=body)
    assert r.status_code == 200, r.text
    assert 0.0 <= r.json()["answers"]["signed"]["noul"] <= 1.0
    assert app.get("/health").json()["vision"] is True

    from vision.run import Strands

    item = {"kind": "noul", "question": "Is the form signed?", "options": [("no", ""), ("yes", "")],
            "gold": 1, "image": base64.b64decode(_b64(320, 320, 6))}
    system = Strands(ckpt)
    seen, blind = system.probs(item, blind=False), system.probs(item, blind=True)
    assert sum(seen) == pytest.approx(1.0, abs=1e-5) and sum(blind) == pytest.approx(1.0, abs=1e-5)
    assert seen != pytest.approx(blind)
