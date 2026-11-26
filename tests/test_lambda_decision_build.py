"""The Lambda image build for a model checkpoint with a decision_config.json
(deploy/lambda/decision/scripts/build_artifact.py): what it fetches, and the checkpoints it refuses before converting anything."""

from __future__ import annotations

import hashlib
import importlib
import sys
import types
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy" / "lambda" / "decision" / "scripts"


@pytest.fixture
def build_artifact(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPTS))
    return importlib.import_module("build_artifact")


def test_an_s3_checkpoint_is_the_files_directly_under_its_prefix(
    build_artifact, tmp_path, monkeypatch
):
    objects = {
        "ckpt/e2b/config.json": b"{}",
        "ckpt/e2b/model.safetensors": b"w",
        "ckpt/e2b/sub/x": b"no",
    }

    class Client:
        def get_paginator(self, name):
            listing = [{"Key": k, "Size": len(v)} for k, v in objects.items()]
            return types.SimpleNamespace(paginate=lambda Bucket, Prefix: [{"Contents": listing}])

        def download_file(self, bucket, key, dest, Config):
            Path(dest).write_bytes(objects[key])

    transfer = types.ModuleType("boto3.s3.transfer")
    transfer.TransferConfig = lambda **kw: None  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda name: Client()))
    monkeypatch.setitem(sys.modules, "boto3.s3", types.ModuleType("boto3.s3"))
    monkeypatch.setitem(sys.modules, "boto3.s3.transfer", transfer)
    d, commit = build_artifact.fetch("s3://b/ckpt/e2b", None, tmp_path)
    assert commit is None
    assert sorted(p.name for p in d.iterdir()) == ["config.json", "model.safetensors"]


def test_a_hub_checkpoint_is_pinned_to_its_commit_and_checked_against_its_hashes(
    build_artifact, tmp_path, monkeypatch
):
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    lfs = types.SimpleNamespace(sha256=hashlib.sha256(files["model.safetensors"]).hexdigest())
    siblings = [types.SimpleNamespace(rfilename="config.json", lfs=None),
                types.SimpleNamespace(rfilename="model.safetensors", lfs=lfs)]  # fmt: skip
    downloaded = []

    def snapshot_download(repo, revision, local_dir):
        downloaded.append(revision)
        for n, b in files.items():
            (Path(local_dir) / n).write_bytes(b)

    info = types.SimpleNamespace(sha="c" * 40, siblings=siblings)
    hub = types.SimpleNamespace(
        HfApi=lambda: types.SimpleNamespace(model_info=lambda repo, revision, files_metadata: info),
        snapshot_download=snapshot_download,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    assert build_artifact.fetch("org/model", "main", tmp_path / "a")[1] == "c" * 40
    assert downloaded == ["c" * 40]  # the resolved commit, not the branch
    lfs.sha256 = "0" * 64
    with pytest.raises(SystemExit, match="does not have the sha256 the Hub records"):
        build_artifact.fetch("org/model", "main", tmp_path / "b")


def test_a_checkpoint_with_another_digest_is_refused_before_conversion(
    build_artifact, tmp_path, monkeypatch
):
    """The digest is `sha256sum * | sort -k 2 | sha256sum` in the checkpoint's directory."""
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    for name, data in {"b.json": b"{}", "a.safetensors": b"weights"}.items():
        (ckpt / name).write_bytes(data)
    lines = "".join(
        f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(ckpt.iterdir())
    )
    assert build_artifact.manifest_digest(ckpt) == hashlib.sha256(lines.encode()).hexdigest()
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "LLAMA_CPP_REV").write_text("f" * 40 + "\n")
    argv = ["build_artifact.py", str(ckpt), str(tmp_path / "out"), "--llama-cpp", str(llama),
            "--quantize", "/bin/false", "--expect-digest", "0" * 64, "--work", str(tmp_path)]  # fmt: skip
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="manifest digest is"):
        build_artifact.main()
    assert not (tmp_path / "out").exists()  # refused before anything was converted


def test_the_request_limit_follows_the_resident_weights(build_artifact):
    # E2B holds 2.7 GB once its per-layer embeddings are left on disk, E4B 5.3 GB.
    assert build_artifact.max_request_tokens(2_699_086_016) == 16384
    assert build_artifact.max_request_tokens(5_264_544_960) == 12288


def test_the_variants_image_keeps_the_libraries_versioned_names():
    """A --lambda-variants libdecision.so links libmtmd.so.1, libllama.so.1 and libggml*.so.1, so
    the image copies every name of each library, its symlinks too, not only libX.so."""
    from pathlib import Path

    dockerfile = (Path(__file__).parents[1] / "deploy/lambda/decision/Dockerfile").read_text()
    assert "cp -a /build/bin/*.so* " in dockerfile
    assert "check_variants.sh /llama /opt/lib" in dockerfile
