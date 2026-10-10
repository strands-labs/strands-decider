"""Find the release the demo should serve, and keep the Hugging Face repo of converted weights tidy.

    python web/convert/publish.py resolve [--release <org>/<repo>]   # key=value lines for $GITHUB_OUTPUT
    python web/convert/publish.py live <site url>                    # the folder the deployed site serves
    python web/convert/publish.py upload <dir> <folder>              # needs HF_TOKEN
    python web/convert/publish.py prune <folder> [<folder> ...]      # delete the others; needs HF_TOKEN

Converted releases live in WEIGHTS_REPO under `<release>/<inputs id>/<converter id>/`: hashes of
the release files convert.py reads and of the converter itself. The path names its contents
completely, so a folder is written once and never changed; a new release, a changed adapter or
head, or a changed converter gets a new one, and a model-card edit does not. Only
huggingface_hub is needed, so resolving is cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

ORG = "StrandsAgents"
# Releases are `strands-decider-2B-<name>` repos tagged strands-decider (hf_export.py writes the
# tag). A `-parent` checkpoint is not a release, nor is a converted-weights (`-webgpu`) repo.
RELEASE = re.compile(rf"^{ORG}/strands-decider-2B-[\w.-]+(?<!-parent)(?<!-webgpu)$")
WEIGHTS_REPO = os.environ.get("WEIGHTS_REPO") or f"{ORG}/strands-decider-2B-webgpu"
# The decider config, under its name before and after the rename (modeling.CONFIG_NAME).
CONFIG_NAMES = ("strands_decider_config.json", "hobson_config.json")
# Release files convert.py reads, besides the config.
INPUTS = (
    "lora/adapter_config.json",
    "lora/adapter_model.safetensors",
    "head.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "LICENSE.md",
    "provenance.json",
)
BASE_TYPES = ("qwen3_5",)  # base model types convert.py can export
WEIGHTS_CARD = """---
license: apache-2.0
library_name: onnx
tags: [onnxruntime-web, webgpu]
---
# Strands Decider, converted for the browser

Written by the `web` workflow of https://github.com/strands-labs/strands-decider (web/convert/):
each `<release>/<revision>/<converter>/` folder holds one Strands Decider release as an int4 ONNX
graph for ONNX Runtime Web, with its tokenizer, config and pointer head. `manifest.json` names the
source release and base model. Do not edit by hand: folders nothing serves are deleted, and the
history is squashed.
"""


def converter_id() -> str:
    """Hash of everything that decides what convert.py writes: its code and its pinned toolchain."""
    here = os.path.dirname(os.path.abspath(__file__))
    digest = hashlib.sha256()
    for f in ("convert.py", "requirements.txt"):
        with open(os.path.join(here, f), "rb") as fh:
            digest.update(fh.read())
    return digest.hexdigest()[:10]


def inputs(api: Any, release: str) -> tuple[str, str, dict[str, str]]:
    """The release's revision, a hash of the files convert.py reads, and those files' blob ids."""
    info = api.model_info(release, files_metadata=True)
    blobs = {f.rfilename: f.blob_id for f in info.siblings}
    config = next((c for c in CONFIG_NAMES if c in blobs), None)
    used = {f: blobs[f] for f in (*INPUTS, *([config] if config else [])) if f in blobs}
    digest = hashlib.sha256("".join(f"{f}:{b}\n" for f, b in sorted(used.items())).encode())
    return info.sha, digest.hexdigest()[:12], blobs


def unsupported(api: Any, release: str, revision: str, blobs: dict[str, str]) -> str | None:
    """Why convert.py cannot convert this release (None if it can), from its small files only."""
    missing = [f for f in INPUTS if f not in blobs]
    config_name = next((c for c in CONFIG_NAMES if c in blobs), None)
    if missing or not config_name:
        return f"missing {missing or 'a decider config'} (an upload in progress, or not an export)"
    with open(hf_hub_download(release, config_name, revision=revision)) as fh:
        config = json.load(fh)
    if config.get("head_type") != "pointer" or not config.get("use_lora", True):
        return "not a pointer-head LoRA checkpoint"
    base_rev = config.get("base_revision")
    if not base_rev:
        with open(hf_hub_download(release, "provenance.json", revision=revision)) as fh:
            prov = json.load(fh)
        base_rev = (
            prov.get("base_model_revision")
            if prov.get("base_model") == config["base_model"]
            else None
        )
    with open(hf_hub_download(config["base_model"], "config.json", revision=base_rev)) as fh:
        base_type = json.load(fh).get("model_type")
    if base_type not in BASE_TYPES:
        return f"base model type {base_type!r}; convert.py exports {BASE_TYPES}"
    return None


def resolve(api: Any, release: str | None = None) -> dict[str, str]:
    """The release to serve (the newest convertible one unless given) and where its conversion lives."""
    if release:
        candidates = [release]
    else:
        models = [
            m
            for m in api.list_models(author=ORG, filter="strands-decider", full=True)
            if RELEASE.match(m.id) and m.id != WEIGHTS_REPO
        ]
        candidates = [m.id for m in sorted(models, key=lambda m: m.created_at, reverse=True)]
    for candidate in candidates:
        revision, inputs_id, blobs = inputs(api, candidate)
        reason = unsupported(api, candidate, revision, blobs)
        if reason is None:
            break
        print(f"skipping {candidate}: {reason}", file=sys.stderr)
    else:
        sys.exit(f"no convertible release among {candidates or RELEASE.pattern}")
    folder = f"{candidate.split('/')[1]}/{inputs_id}/{converter_id()}"
    exists = api.repo_exists(WEIGHTS_REPO) and api.file_exists(
        WEIGHTS_REPO, f"{folder}/manifest.json"
    )
    return {
        "release": candidate,
        "revision": revision,
        "folder": folder,
        "exists": str(exists).lower(),
        "repo": WEIGHTS_REPO,
        "url": f"https://huggingface.co/{WEIGHTS_REPO}/resolve/main/{folder}/",
    }


def served_folder(html: str) -> str | None:
    """The folder in a built page's decider-weights tag (build.mjs writes it)."""
    m = re.search(r'name="decider-weights" content="[^"]*/resolve/main/([^"]+)/"', html)
    return m.group(1) if m else None


def live(site: str) -> str | None:
    """The folder the deployed site serves; None when there is no site yet."""
    # Past the Pages CDN cache (10 minutes), which would name the folder served before the last deploy.
    url = f"{site}{'&' if '?' in site else '?'}nocache={int(time.time())}"
    req = urllib.request.Request(url, headers={"Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=30) as res:
            return served_folder(res.read().decode())
    except OSError:
        return None


def upload(api: Any, src: str, folder: str) -> None:
    if not api.repo_exists(WEIGHTS_REPO):
        api.create_repo(WEIGHTS_REPO, repo_type="model")
        api.upload_file(
            path_or_fileobj=WEIGHTS_CARD.encode(),
            path_in_repo="README.md",
            repo_id=WEIGHTS_REPO,
            commit_message="Add README.md",
        )
    api.upload_folder(
        repo_id=WEIGHTS_REPO, folder_path=src, path_in_repo=folder, commit_message=f"Add {folder}"
    )


def prune(api: Any, keep: list[str]) -> list[str]:
    """Delete every folder but `keep` (the one being deployed and the one the site served until
    now, for pages still open), then squash the history to one commit so deleted weights stop
    using storage; a squash that failed last time is done now. The site loads through `main`, so
    the squash does not break it. Returns the deleted folders."""
    present = {
        f.removesuffix("/manifest.json")
        for f in api.list_repo_files(WEIGHTS_REPO)
        if f.endswith("/manifest.json")
    }
    if not set(keep) & present:
        sys.exit(f"none of {keep} is in {WEIGHTS_REPO}; refusing to delete anything")
    stale = sorted(present - set(keep))
    for folder in stale:
        api.delete_folder(folder, repo_id=WEIGHTS_REPO, commit_message=f"Remove {folder}")
    if len(api.list_repo_commits(WEIGHTS_REPO)) > 1:
        api.super_squash_history(repo_id=WEIGHTS_REPO, commit_message="Squash history")
    return stale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("resolve").add_argument("--release", help="serve this release, not the newest")
    sub.add_parser("live").add_argument("site")
    u = sub.add_parser("upload")
    u.add_argument("dir")
    u.add_argument("folder")
    sub.add_parser("prune").add_argument("keep", nargs="+")
    a = ap.parse_args()
    if a.cmd == "live":
        print(live(a.site) or "")
    elif a.cmd == "resolve":
        for k, v in resolve(HfApi(), a.release).items():
            print(f"{k}={v}")
    elif a.cmd == "upload":
        upload(HfApi(), a.dir, a.folder)
    else:
        print(f"removed {prune(HfApi(), a.keep) or 'nothing'}")


if __name__ == "__main__":
    main()
