"""The MLX backend and convert-mlx, on Apple silicon with the mlx extra. The checkpoints are
tests/decision_support.py's tiny random-weight ones (E2B's, the 31B's and the 12B's designs, a
2-token window, config.json as transformers 5.2x writes it), and the reference is the transformers
backend, one full pass per question in float32. Nothing is downloaded; tests/test_backend_mlx_format.py
covers what needs no MLX."""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib.util
import io
import json
import os
import platform
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

if platform.system() != "Darwin" or platform.machine() != "arm64":
    pytest.skip("MLX runs on Apple silicon only", allow_module_level=True)
pytest.importorskip("mlx.core")
pytest.importorskip("mlx_vlm")

from core_support import FIXTURES
from decision_support import GEMMA4, KINDS, UNIFIED, model_checkpoint, image_b64, request
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from strands_decider import cli, server
from strands_decider.backends import mlx
from strands_decider.backends import transformers as tb
from strands_decider.backends.images import image_pixels
from strands_decider.backends.mlx_convert import COPIED, convert
from strands_decider.backends.mlx_format import is_mlx_artifact
from strands_decider.core import Request, load_tokenizer, render, soft_tokens
from strands_decider.core.prompt import RenderedImage
from strands_decider.decision import NO_IMAGE_PROCESSOR, load_decision_engine
from strands_decider.refusals import RequestRefused

TORCHVISION = importlib.util.find_spec("torchvision") is not None
IMAGES = [image_b64(64, 48), image_b64(90, 120, "JPEG", 1)]


@pytest.fixture(scope="module", params=KINDS)
def kind(request):
    return request.param


@pytest.fixture(scope="module")
def ckpt(kind, tmp_path_factory):
    return model_checkpoint(tmp_path_factory.mktemp(kind) / "ckpt", kind)


@pytest.fixture(scope="module")
def pair(ckpt):
    """The checkpoint on the transformers backend, one full pass per question, and on MLX, both
    in float32."""
    ours = mlx.load_backend(str(ckpt), dtype="float32")
    yield tb.load_backend(str(ckpt), reuse_state=False), ours
    ours.close()


@pytest.fixture(scope="module")
def e2b(tmp_path_factory):
    return model_checkpoint(tmp_path_factory.mktemp("e2b") / "ckpt", GEMMA4)


@pytest.fixture(scope="module")
def e2b_mlx(e2b):
    backend = mlx.load_backend(str(e2b))
    yield backend
    backend.close()


@pytest.fixture(autouse=True)
def _no_exit_fast(monkeypatch):
    monkeypatch.setattr(cli, "_exit_fast", False)  # ask sets it; main() is not run here


def _parts(ckpt, images=(), state="The order shipped late, twice."):
    r = render(Request.model_validate(request(list(images), state)), load_tokenizer(ckpt))
    return list(r.state_ids), [list(q.question_ids) for q in r.questions.values()], r.images


def _tokens(n: int, seed: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(10, 250_000, n)]


def _rel(ref, got) -> float:
    """The largest difference, relative to the reference's scale."""
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(np.abs(ref - got).max()) / max(1.0, float(np.abs(ref).max()))


# ---- against the reference ------------------------------------------------------------------


@pytest.mark.parametrize("reuse_state", [True, False])
@pytest.mark.parametrize("images", [[], IMAGES], ids=["text", "two-images"])
def test_hidden_states_are_the_transformers_backends(ckpt, kind, pair, reuse_state, images):
    """Every question, over the state or in its own pass, gets the reference's hidden state: no
    window on the sliding layers, the KV-shared layers exact, and each image's vectors at its
    placeholders."""
    if images and kind == UNIFIED and not TORCHVISION:
        pytest.skip("the transformers backend's 12B image processor needs torchvision")
    ref, ours = pair
    ours = copy.copy(ours)
    ours.reuse_state = reuse_state
    state, questions, rendered = _parts(ckpt, images)
    got = ours.forward(state, questions, rendered)
    assert _rel(ref.forward(state, questions, rendered), got) < 2e-4
    if images:  # the pixels, not the placeholders alone, decide
        assert _rel(ours.forward(state, questions), got) > 1e-2


def test_questions_over_a_long_state_see_the_state_alone(pair):
    """A 250-token state fills most of the KV cache's first 256-token block; a 20-token question
    grows it, and the 5-token question after it must see the state alone."""
    ref, ours = pair
    state, long, short = _tokens(250, 1), _tokens(20, 2), _tokens(5, 3)
    rows = ours.forward(state, [long, short, long])
    np.testing.assert_array_equal(rows[1], ours.forward(state, [short])[0])
    np.testing.assert_array_equal(rows[0], rows[2])
    assert _rel(ref.forward(state, [long, short]), rows[:2]) < 2e-4


def test_forward_from_several_threads(e2b, e2b_mlx):
    """A server calls from a pool: eight requests at once, with images and without, get what they
    get one at a time."""
    jobs = [_parts(e2b, [IMAGES[i % 2]] if i % 3 else [], f"request {i}") for i in range(8)]
    alone = [e2b_mlx.forward(*job) for job in jobs]
    with ThreadPoolExecutor(8) as pool:
        together = list(pool.map(lambda job: e2b_mlx.forward(*job), jobs))
    for x, y in zip(alone, together, strict=True):
        np.testing.assert_array_equal(x, y)


def test_the_encoder_gets_the_cores_golden_pixels(e2b_mlx):
    """Each pixel golden's decoded image, through the backend's resize and its image processor
    with the resize off, comes out as the golden's resized pixels, byte for byte."""
    goldens = json.loads((FIXTURES / "image_pixels.json").read_text())["images"]
    for name, golden in goldens.items():
        decoded = (FIXTURES / "pixels" / f"{name}.png").read_bytes()
        with Image.open(io.BytesIO(decoded)) as img:
            w, h = img.size
        im = RenderedImage(decoded, "PNG", w, h, "", 0, soft_tokens(w, h))
        px = e2b_mlx.processor(image_pixels(im))[0]["pixel_values"][0]  # [3, H, W] in [0, 1]
        rgb = np.rint(px.transpose(1, 2, 0) * 255).astype(np.uint8)
        assert [rgb.shape[1], rgb.shape[0]] == golden["resized"], name
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == golden["resized_sha256"], name


# ---- serve and ask --------------------------------------------------------------------------


def test_serve_on_mlx(e2b, tmp_path):
    """Over HTTP, warmed up: the health check, image requests, the interface's refusals with 422,
    images on a text-only checkpoint refused, and the worker thread stopped with the app."""
    good = base64.b64decode(IMAGES[0])
    broken = base64.b64encode(good[:41] + b"\x00" * 64 + good[105:]).decode()  # pixel data only
    with TestClient(server.create_app(str(e2b), device="mlx", model_name="tiny")) as client:
        health = client.get("/health").json()
        assert (health["backend"], health["device"], health["dtype"]) == (
            "mlx-vlm",
            "mlx",
            "bfloat16",
        )
        assert health["images"] and health["mlx_vlm"] == mlx.MLX_VLM
        r = client.post("/v1/systemone", json=request(IMAGES[:1]))
        assert r.status_code == 200, r.text
        assert set(r.json()["answers"]) == {"late", "team", "mood"}
        assert client.post("/v1/systemone", json=request(IMAGES * 3)).status_code == 200
        r = client.post("/v1/systemone", json={**request(), "audio": ["UklGRiQAAABXQVZF"]})
        assert r.status_code == 422 and "audio input is not supported" in r.json()["detail"]
        r = client.post("/v1/systemone", json=request([broken]))
        assert r.status_code == 422 and "image could not be" in r.json()["detail"]
        backend = client.app.state.engine.backend
    with pytest.raises(RuntimeError, match="shutdown"):
        backend.forward(*_parts(e2b))
    text_only = model_checkpoint(tmp_path / "text", GEMMA4, decision={"inputs": ["text"]})
    app = server.create_app(str(text_only), device="mlx", warmup=False)
    with TestClient(app) as client:
        r = client.post("/v1/systemone", json=request(IMAGES[:1]))
        assert r.status_code == 422
        assert r.json()["detail"].startswith("this checkpoint does not take images")
        assert client.get("/health").json()["images"] is False
        assert app.state.engine.backend.model.vision_tower is None  # not loaded


def test_a_processor_the_core_refuses_refuses_images_only(tmp_path):
    path = model_checkpoint(tmp_path / "ckpt", GEMMA4, processor={"max_soft_tokens": 560})
    with pytest.warns(UserWarning, match="max_soft_tokens"):
        engine = load_decision_engine(str(path), device="mlx")
    try:
        with pytest.raises(RequestRefused, match=NO_IMAGE_PROCESSOR):
            engine.evaluate(Request.model_validate(request(IMAGES[:1])))
        assert engine.evaluate(Request.model_validate(request())).usage.output_tokens == 3
    finally:
        engine.close()


def test_ask_and_convert_from_the_cli(e2b, tmp_path):
    image = tmp_path / "x.png"
    image.write_bytes(base64.b64decode(IMAGES[0]))
    ask = ["ask", str(e2b), "--device", "mlx", "--state", "a b", "--json", "--image", str(image)]
    out = CliRunner().invoke(cli.app, [*ask, "--noul", "a b?"])
    assert out.exit_code == 0, out.output
    assert 0 <= json.loads(out.stdout)["answers"]["noul_0"]["noul"] <= 1
    artifact = tmp_path / "mlx8"
    out = CliRunner().invoke(cli.app, ["convert-mlx", str(e2b), str(artifact), "--bits", "8"])
    assert out.exit_code == 0 and "decoder at 8 bits (group 64)" in " ".join(out.output.split())
    ask[1] = str(artifact)
    out = CliRunner().invoke(cli.app, [*ask, "--choice", "which?=billing,sales"])
    assert out.exit_code == 0, out.output
    assert set(json.loads(out.stdout)["answers"]["choice_0"]["probabilities"]) == {
        "billing",
        "sales",
    }
    out = CliRunner().invoke(cli.app, ["convert-mlx", str(e2b), str(artifact)])
    assert out.exit_code == 1 and "is not an empty directory" in out.output


# ---- MLX artifacts --------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [None, 8])
def test_an_artifact_answers_as_its_checkpoint(ckpt, bits, tmp_path):
    """bf16 gives the checkpoint's answers exactly. 8-bit quantizes the decoder and nothing else;
    a tiny random decoder amplifies rounding, so this checks only that it is the same network
    (E2B's parity run on real weights measured 8-bit's accuracy)."""
    artifact = convert(ckpt, tmp_path / "mlx", bits=bits)
    assert artifact == tmp_path / "mlx" and is_mlx_artifact(artifact)
    for name in (n for n in COPIED if (ckpt / n).exists()):
        assert (artifact / name).read_bytes() == (ckpt / name).read_bytes(), name
    model = mlx.load_mlx_model(artifact)
    quantized = {path.split(".")[0] for path, m in model.named_modules() if "scales" in m}
    assert quantized == ({"language_model"} if bits else set())
    a, c = mlx.load_backend(str(ckpt)), mlx.load_backend(str(artifact))
    try:
        parts = _parts(ckpt, IMAGES[:1])
        x, y = a.forward(*parts), c.forward(*parts)
    finally:
        a.close()
        c.close()
    if bits is None:
        np.testing.assert_array_equal(x, y)
    else:
        assert ((x * y).sum(1) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1) > 0.8).all()


def test_relative_paths_load_and_convert(e2b, e2b_mlx, monkeypatch):
    """E2B, E4B and the 12B ship one model.safetensors and no index, so they load through a
    directory of links; from a relative path, the links must still reach the files."""
    assert not (e2b / "model.safetensors.index.json").exists()
    monkeypatch.chdir(e2b.parent)
    parts = _parts(e2b)
    want = e2b_mlx.forward(*parts)
    convert(e2b.name, "relative-mlx")
    for path in (e2b.name, "relative-mlx"):
        backend = mlx.load_backend(path)
        try:
            np.testing.assert_array_equal(backend.forward(*parts), want)
        finally:
            backend.close()


def test_a_conversion_leaves_an_artifact_or_nothing(e2b, tmp_path, monkeypatch):
    """A failed conversion leaves nothing beside its destination. An empty destination is
    replaced, with the permissions any new directory gets (not a temporary one's 0700)."""

    def crash(path, model, **kw):
        (Path(path) / "model.safetensors").write_bytes(b"half")
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr("mlx_vlm.utils.save_weights", crash)
        with pytest.raises(OSError, match="disk full"):
            convert(e2b, tmp_path / "crashed", bits=8)
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "empty").mkdir()
    old = os.umask(0o022)
    try:
        artifact = convert(e2b, tmp_path / "empty")
    finally:
        os.umask(old)
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o755
    assert [p.name for p in tmp_path.iterdir()] == ["empty"]
    with pytest.raises(ValueError, match="already an MLX artifact"):
        convert(artifact, tmp_path / "again")
    with pytest.raises(ValueError, match="bits must be one of"):
        convert(e2b, tmp_path / "three", bits=3)
    for device in ("cpu", "mps"):
        with pytest.raises(ValueError, match="is an MLX artifact: serve it with --device mlx"):
            load_decision_engine(str(artifact), device=device)


# ---- what is refused at load ----------------------------------------------------------------


def test_another_mlx_vlm_is_refused(e2b, monkeypatch):
    monkeypatch.setattr(mlx, "version", lambda name: "0.7.13")
    with pytest.raises(RuntimeError, match=r"tested on mlx-vlm 0\.7\.12, and mlx-vlm 0\.7\.13"):
        mlx.load_backend(str(e2b))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_type", "gemma4_text", "a Gemma 4 base's multimodal checkpoint"),
        ("hidden_size", 96, "mlx-vlm cannot load this checkpoint"),
    ],
)
def test_a_checkpoint_mlx_cannot_load_is_refused_by_its_path(tmp_path, field, value, message):
    path = model_checkpoint(tmp_path / "ckpt", GEMMA4)
    config = json.loads((path / "config.json").read_text())
    (config if field == "model_type" else config["text_config"])[field] = value
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match=message) as e:
        mlx.load_backend(str(path))
    assert str(path) in str(e.value) and "strands-decider-mlx-" not in str(e.value)
