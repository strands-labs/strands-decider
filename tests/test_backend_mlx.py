"""The MLX backend and convert-mlx, on Apple silicon with the mlx extra. The checkpoints are
tests/decision_support.py's tiny random-weight ones (E2B's, the 31B's and the 12B's designs, a
2-token window, config.json as transformers 5.2x writes it), and the reference is the transformers
backend, one full pass per question in float32. Nothing is downloaded."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
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

from core_support import PIXELS, PIXELS_DIR
from decision_support import GEMMA4, KINDS, UNIFIED, model_checkpoint, image_b64, request
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from strands_decider import cli, server
from strands_decider.backends import mlx, mlx_format
from strands_decider.backends import transformers as tb
from strands_decider.backends.images import image_pixels
from strands_decider.core import RenderedImage, Request, load_tokenizer, render, soft_tokens
from strands_decider.decision import load_decision_engine
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
def reference(ckpt):
    return tb.load_backend(str(ckpt), reuse_state=False)


@pytest.fixture(scope="module", params=[True, False], ids=["reuse-state", "full-passes"])
def ours(ckpt, request):
    """The checkpoint on MLX in float32, the state encoded once or a full pass per question."""
    backend = mlx.load_backend(str(ckpt), dtype="float32", reuse_state=request.param)
    yield backend
    backend.close()


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


def _prompt(ckpt, images=(), state="The order shipped late, twice."):
    r = render(Request.model_validate(request(list(images), state)), load_tokenizer(ckpt))
    return list(r.state_ids), [list(q.question_ids) for q in r.questions.values()], r.images


def _tokens(n: int, seed: int) -> list[int]:
    return [int(t) for t in np.random.default_rng(seed).integers(10, 250_000, n)]


def _rel(ref, got) -> float:
    """The largest difference, relative to the reference's scale."""
    ref, got = np.asarray(ref, np.float64), np.asarray(got, np.float64)
    return float(np.abs(ref - got).max()) / max(1.0, float(np.abs(ref).max()))


@pytest.mark.parametrize("images", [[], IMAGES], ids=["text", "two-images"])
def test_hidden_states_are_the_transformers_backends(ckpt, kind, reference, ours, images):
    """Every question, over the state or in its own pass, gets the reference's hidden state: no
    window on the sliding layers, the KV-shared layers exact, and each image's vectors at its
    placeholders."""
    if images and kind == UNIFIED and not TORCHVISION:
        pytest.skip("the transformers backend's 12B image processor needs torchvision")
    prompt = _prompt(ckpt, images)
    got = ours.forward(*prompt)
    assert _rel(reference.forward(*prompt), got) < 2e-4
    if images:  # the pixels, not the placeholders alone, decide
        assert _rel(ours.forward(*prompt[:2]), got) > 1e-2


def test_questions_over_a_long_state_see_the_state_alone(reference, ours):
    """A 250-token state fills most of the KV cache's first 256-token block; a 20-token question
    grows it, and the 5-token question after it must see the state alone."""
    state, long, short = _tokens(250, 1), _tokens(20, 2), _tokens(5, 3)
    rows = ours.forward(state, [long, short, long])
    np.testing.assert_array_equal(rows[1], ours.forward(state, [short])[0])
    np.testing.assert_array_equal(rows[0], rows[2])
    assert _rel(reference.forward(state, [long, short]), rows[:2]) < 2e-4


def test_forward_from_several_threads(e2b, e2b_mlx):
    """A server calls from a pool: eight requests at once, with images and without, get what they
    get one at a time."""
    jobs = [_prompt(e2b, [IMAGES[i % 2]] if i % 3 else [], f"request {i}") for i in range(8)]
    alone = [e2b_mlx.forward(*job) for job in jobs]
    with ThreadPoolExecutor(8) as pool:
        together = list(pool.map(lambda job: e2b_mlx.forward(*job), jobs))
    for x, y in zip(alone, together, strict=True):
        np.testing.assert_array_equal(x, y)


def test_the_encoder_gets_the_cores_golden_pixels(e2b_mlx):
    """Each pixel golden's decoded image, resized by `image_pixels` and through the image
    processor with its resize off, comes out as the golden's resized pixels, byte for byte."""
    for name, golden in json.loads(PIXELS.read_text())["images"].items():
        w, h = golden["decoded"]
        png = (PIXELS_DIR / f"{name}.png").read_bytes()
        im = RenderedImage(png, "PNG", w, h, "", 0, soft_tokens(w, h))
        px = e2b_mlx.image_processor(image_pixels(im))[0]["pixel_values"][0]  # [3, H, W], 0..1
        rgb = np.rint(px.transpose(1, 2, 0) * 255).astype(np.uint8)
        assert [rgb.shape[1], rgb.shape[0]] == golden["resized"], name
        assert hashlib.sha256(rgb.tobytes()).hexdigest() == golden["resized_sha256"], name


def test_serve_on_mlx(e2b):
    """Over HTTP, warmed up: the health check and image requests, and the worker thread stopped
    when the app shuts down."""
    with TestClient(server.create_app(str(e2b), device="mlx", model_name="tiny")) as client:
        health = client.get("/health").json()
        assert [health[k] for k in ("backend", "device", "dtype", "images")] == [
            "mlx-vlm",
            "mlx",
            "bfloat16",
            True,
        ]
        for images in (IMAGES[:1], IMAGES * 3):
            r = client.post("/v1/systemone", json=request(images))
            assert r.status_code == 200, r.text
            assert set(r.json()["answers"]) == {"late", "team", "mood"}
        backend = client.app.state.engine.backend
    with pytest.raises(RuntimeError, match="shutdown"):
        backend.forward(*_prompt(e2b))


def test_a_processor_the_core_refuses_leaves_the_vision_encoder_unloaded(tmp_path):
    """Image requests are refused and text ones answered; the encoder's weights are never read."""
    path = model_checkpoint(tmp_path / "ckpt", GEMMA4, processor={"max_soft_tokens": 560})
    with pytest.warns(UserWarning, match="max_soft_tokens"):
        engine = load_decision_engine(str(path), device="mlx")
    try:
        assert engine.backend.model.vision_tower is None
        with pytest.raises(RequestRefused):
            engine.evaluate(Request.model_validate(request(IMAGES[:1])))
        assert engine.evaluate(Request.model_validate(request())).usage.output_tokens == 3
    finally:
        engine.close()


def test_convert_and_ask_from_the_cli(e2b, tmp_path):
    """convert-mlx writes an 8-bit artifact, which `ask` answers on mlx with an image, and
    refuses a destination that isn't empty."""
    artifact, image = tmp_path / "mlx8", tmp_path / "x.png"
    image.write_bytes(base64.b64decode(IMAGES[0]))
    out = CliRunner().invoke(cli.app, ["convert-mlx", str(e2b), str(artifact), "--bits", "8"])
    assert out.exit_code == 0 and "decoder at 8 bits (group 64)" in " ".join(out.output.split())
    ask = ["ask", str(artifact), "--device", "mlx", "--state", "a b", "--image", str(image)]
    out = CliRunner().invoke(cli.app, [*ask, "--choice", "which?=billing,sales", "--json"])
    assert out.exit_code == 0, out.output
    probabilities = json.loads(out.stdout)["answers"]["choice_0"]["probabilities"]
    assert set(probabilities) == {"billing", "sales"}
    out = CliRunner().invoke(cli.app, ["convert-mlx", str(e2b), str(artifact)])
    assert out.exit_code == 1 and "is not an empty directory" in out.output


@pytest.mark.parametrize("bits", [None, 8])
def test_an_artifact_answers_as_its_checkpoint(ckpt, bits, tmp_path):
    """bf16 gives the checkpoint's answers exactly. 8-bit quantizes the decoder and nothing else;
    a tiny random decoder amplifies rounding, so this checks only that it is the same network
    (E2B's parity run on real weights measured 8-bit's accuracy)."""
    artifact = mlx_format.convert(ckpt, tmp_path / "mlx", bits=bits)
    assert artifact == tmp_path / "mlx" and mlx_format.is_mlx_artifact(artifact)
    for name in (
        "decision_config.json",
        "readout.safetensors",
        "processor_config.json",
        "tokenizer.json",
    ):
        assert (artifact / name).read_bytes() == (ckpt / name).read_bytes(), name
    model = mlx_format.load_mlx_model(artifact)
    quantized = {path.split(".")[0] for path, m in model.named_modules() if "scales" in m}
    assert quantized == ({"language_model"} if bits else set())
    a, c = mlx.load_backend(str(ckpt)), mlx.load_backend(str(artifact))
    try:
        prompt = _prompt(ckpt, IMAGES[:1])
        x, y = a.forward(*prompt), c.forward(*prompt)
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
    prompt = _prompt(e2b)
    want = e2b_mlx.forward(*prompt)
    mlx_format.convert(e2b.name, "relative-mlx")
    for path in (e2b.name, "relative-mlx"):
        backend = mlx.load_backend(path)
        try:
            np.testing.assert_array_equal(backend.forward(*prompt), want)
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
            mlx_format.convert(e2b, tmp_path / "crashed", bits=8)
    assert list(tmp_path.iterdir()) == []
    (tmp_path / "empty").mkdir()
    old = os.umask(0o022)
    try:
        artifact = mlx_format.convert(e2b, tmp_path / "empty")
    finally:
        os.umask(old)
    assert stat.S_IMODE(artifact.stat().st_mode) == 0o755
    assert [p.name for p in tmp_path.iterdir()] == ["empty"]
    with pytest.raises(ValueError, match="already an MLX artifact"):
        mlx_format.convert(artifact, tmp_path / "again")


def test_another_mlx_vlm_is_refused(e2b, monkeypatch):
    monkeypatch.setattr(mlx_format, "version", lambda name: "0.7.13")
    with pytest.raises(RuntimeError, match=r"tested on mlx-vlm 0\.7\.12, and mlx-vlm 0\.7\.13"):
        mlx.load_backend(str(e2b))


def test_a_checkpoint_mlx_vlm_cannot_load_is_refused_by_its_path(tmp_path):
    path = model_checkpoint(tmp_path / "ckpt", GEMMA4)
    config = json.loads((path / "config.json").read_text())
    config["text_config"]["hidden_size"] = 96
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mlx-vlm cannot load this checkpoint") as e:
        mlx.load_backend(str(path))
    assert str(path) in str(e.value) and "strands-decider-mlx-" not in str(e.value)
