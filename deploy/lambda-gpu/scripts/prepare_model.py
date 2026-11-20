"""Download the strands-decider checkpoint AND its base model for the Lambda GPU image.

    python prepare_model.py MODEL REVISION OUT_DIR

MODEL is a Hugging Face Hub id, REVISION is a branch, tag, or commit. Clones MODEL into
`OUT_DIR/checkpoint` and the base model named in its config.json into HF_HOME. The Dockerfile
runs this; the function's handler.py then loads with StrandsDeciderModel.load.

Uses `git clone` with Git-LFS rather than huggingface_hub.snapshot_download because the pytorch
base image's huggingface_hub 1.x uses httpx2 and the bundled Brotli/zstandard decoders reject
httpx2's `process(data, output_buffer_limit=...)` call.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.check_call(cmd, env=env or os.environ)


_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def _git_clone(url: str, revision: str, dst: str) -> None:
    parent = os.path.dirname(dst)
    if parent:
        os.makedirs(parent, exist_ok=True)
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "0", "GIT_TERMINAL_PROMPT": "0"}
    if _SHA_RE.match(revision):
        # `git clone --branch` does not accept commit SHAs. Init, fetch the SHA, checkout.
        _run(["git", "init", "-q", dst], env=env)
        _run(["git", "-C", dst, "remote", "add", "origin", url], env=env)
        _run(["git", "-C", dst, "fetch", "--depth", "1", "origin", revision], env=env)
        _run(["git", "-C", dst, "checkout", "--quiet", "FETCH_HEAD"], env=env)
    else:
        _run(
            [
                "git",
                "clone",
                "--progress",
                "--depth",
                "1",
                "--branch",
                revision,
                url,
                dst,
            ],
            env=env,
        )
    # List files so a smudge failure or empty clone is visible in the build log.
    print(f"-- contents of {dst}", flush=True)
    _run(["ls", "-lah", dst])
    # Force LFS smudge if any pointer files remain (git clone's smudge silently skips files
    # when it hits a transient error).
    if any(
        _looks_like_lfs_pointer(os.path.join(dst, name))
        for name in os.listdir(dst)
        if os.path.isfile(os.path.join(dst, name))
    ):
        print("-- LFS pointer files detected; running `git lfs pull`", flush=True)
        _run(["git", "-C", dst, "lfs", "pull"], env=env)
    shutil.rmtree(os.path.join(dst, ".git"), ignore_errors=True)


def _looks_like_lfs_pointer(path: str) -> bool:
    try:
        if os.path.getsize(path) > 1024:
            return False
        with open(path, "rb") as fh:
            head = fh.read(64)
    except OSError:
        return False
    return head.startswith(b"version https://git-lfs.github.com/spec/")


def _hf_cache_entry(hub_id: str) -> str:
    cache = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    folder = "models--" + re.sub(r"[^A-Za-z0-9._-]", "--", hub_id)
    return os.path.join(cache, "hub", folder)


def _materialize_base(hub_id: str, revision: str) -> None:
    entry = _hf_cache_entry(hub_id)
    snapshot = os.path.join(entry, "snapshots", revision)
    refs = os.path.join(entry, "refs")
    blobs = os.path.join(entry, "blobs")
    os.makedirs(os.path.dirname(snapshot), exist_ok=True)
    os.makedirs(refs, exist_ok=True)
    os.makedirs(blobs, exist_ok=True)
    _git_clone(f"https://huggingface.co/{hub_id}", revision, snapshot)
    with open(os.path.join(refs, revision), "w", encoding="utf-8") as fh:
        fh.write(revision)


def main(model_id: str, revision: str, out: str) -> None:
    dst = os.path.join(out, "checkpoint")
    _git_clone(f"https://huggingface.co/{model_id}", revision, dst)
    # strands-decider's adapter config is `strands_decider_config.json` (not `config.json`), and
    # provenance.json carries the pinned base-model revision the adapter was trained on.
    sd_cfg_path = os.path.join(dst, "strands_decider_config.json")
    prov_path = os.path.join(dst, "provenance.json")
    if not os.path.exists(sd_cfg_path):
        raise FileNotFoundError(f"{sd_cfg_path} missing. Clone contents: {os.listdir(dst)!r}")
    with open(sd_cfg_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    base_model = cfg.get("base_model")
    base_revision = cfg.get("base_revision")
    if not base_revision and os.path.exists(prov_path):
        with open(prov_path, encoding="utf-8") as fh:
            prov = json.load(fh)
        if prov.get("base_model") == base_model:
            base_revision = prov.get("base_model_revision")
    base_revision = base_revision or "main"
    if base_model:
        _materialize_base(base_model, base_revision)


if __name__ == "__main__":
    main(*sys.argv[1:])
