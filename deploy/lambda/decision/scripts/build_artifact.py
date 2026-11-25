"""Build what the Lambda image serves from a model checkpoint with a decision_config.json.

    python build_artifact.py CHECKPOINT OUT_DIR --llama-cpp DIR --quantize PATH [--revision REV]
        [--expect-digest SHA256] [--work DIR] [--gguf-tmp DIR]

CHECKPOINT is a Hugging Face Hub id, an s3:// URI of a checkpoint directory (read with the
environment's AWS credentials), or a local directory. A Hub revision is resolved to its commit, and
each LFS file is checked against the sha256 the Hub records. The checkpoint's manifest digest is the
sha256 of the lines `<sha256>  <file>`, one per file, sorted by name (in its directory,
`sha256sum * | sort -k 2 | sha256sum`); with --expect-digest, any other is refused. The checkpoint
is then loaded as the function loads it, before anything is converted. OUT_DIR gets:

    model.gguf      the decoder: llama.cpp's convert_hf_to_gguf.py to BF16 (lossless from the
                    bf16 checkpoint), then llama-quantize to Q8_0
    mmproj.gguf     for a checkpoint whose inputs hold "image": the vision tower and its
                    projection, convert_hf_to_gguf.py --mmproj at q8_0, without the audio tower
    readout.safetensors, decision_config.json, tokenizer.json, processor_config.json
                    the checkpoint's own
    artifact.json   the model's name; the checkpoint, its Hub commit and its manifest digest; the
                    llama.cpp commit; each file's size and sha256; model.gguf's lazy byte ranges
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from gguf_files import check_mmproj, lazy_ranges, vision_only

from strands_decider.core import DecisionConfig, check_image_processor, load_readout, load_tokenizer
from strands_decider.core.config import IMAGE

# Two settings, not one: llama-quantize quantizes the decoder and takes the type in capitals;
# convert_hf_to_gguf.py writes the vision tower directly (it is not quantized afterwards) and takes
# its --outtype in lower case. On Graviton the q8_0 tower encodes an image about three times as fast
# as the f16 one (README.md, What the build does).
_QUANT = "Q8_0"
_VISION_OUTTYPE = "q8_0"
# What the function holds in memory, the files less the lazily read per-layer embedding table, must
# leave llama.cpp's buffers, Python and a request's state room within Lambda's 10,240 MB: E2B's are
# 2.4 GB of decoder and 0.23 GB of vision tower at q8_0, E4B's 5.0 GB and 0.23 GB.
_MAX_RESIDENT_BYTES = 6_000_000_000
# The files themselves go in the image, which Lambda caps at 10 GB unpacked, beside about 0.6 GB of
# runtime: E2B's are 5.2 GB, E4B's 8.2 GB.
_MAX_FILE_BYTES = 9_000_000_000


def max_request_tokens(resident: int) -> int:
    """The most tokens one request may forward in 10,240 MB, by the weights it holds: llama.cpp's
    batch buffers grow with the state, which is decoded as one batch. E2B (2.7 GB resident) ran
    16,384-token requests; E4B (5.3 GB) ran 13,735 and was killed for memory at 15,894, so it gets
    12,288, which also keeps its slowest requests (about 330 s) well inside the 15-minute timeout."""
    return 16384 if resident <= 3_500_000_000 else 12288


_FILES = ("readout.safetensors", "decision_config.json", "tokenizer.json", "processor_config.json")


def fetch(checkpoint: str, revision: str | None, work: Path) -> tuple[Path, str | None]:
    """The checkpoint as a local directory, and the Hub commit it was read at (None off the Hub)."""
    if os.path.isdir(checkpoint):
        return Path(checkpoint), None
    dest = work / "checkpoint"
    dest.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if checkpoint.startswith("s3://"):
        commit = None
        _fetch_s3(checkpoint, dest)
    else:
        commit = _fetch_hub(checkpoint, revision, dest)
    total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(
        f"fetched {checkpoint} ({commit or 'no commit'}): {total:,} bytes in "
        f"{time.time() - started:.0f} s",
        flush=True,
    )
    return dest, commit


def _fetch_s3(uri: str, dest: Path) -> None:
    """The files directly under an s3:// prefix, not its subdirectories."""
    import boto3
    from boto3.s3.transfer import TransferConfig

    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    s3 = boto3.client("s3")
    config = TransferConfig(
        max_concurrency=32, multipart_chunksize=64 << 20, multipart_threshold=64 << 20
    )
    fetched = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix) :]
            if name and "/" not in name:
                s3.download_file(bucket, obj["Key"], str(dest / name), Config=config)
                fetched += 1
    if not fetched:
        raise SystemExit(f"nothing under {uri}")


def _fetch_hub(repo: str, revision: str | None, dest: Path) -> str:
    """The repository at the commit `revision` resolves to, each LFS file checked against the
    sha256 the Hub records for it; returns the commit."""
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    snapshot_download(repo, revision=info.sha, local_dir=dest)
    for f in info.siblings or []:
        if f.lfs and _sha256(dest / f.rfilename) != f.lfs.sha256:
            raise SystemExit(
                f"{f.rfilename} does not have the sha256 the Hub records, {f.lfs.sha256}"
            )
    return str(info.sha)


def manifest_digest(ckpt: Path) -> str:
    """The sha256 of `<sha256>  <file>` lines, one per file of the checkpoint, sorted by name."""
    files = sorted((p for p in ckpt.iterdir() if p.is_file()), key=lambda p: p.name)
    lines = "".join(f"{_sha256(p)}  {p.name}\n" for p in files)
    return hashlib.sha256(lines.encode()).hexdigest()


def _check(ckpt: Path) -> DecisionConfig:
    """The checkpoint's decision config, with its other files loaded as the function loads them."""
    missing = [name for name in _FILES if not (ckpt / name).is_file()]
    if missing:
        raise SystemExit(f"{ckpt} has no {', '.join(missing)}")
    config = DecisionConfig.load(ckpt)
    load_readout(ckpt)
    load_tokenizer(ckpt)
    if IMAGE in config.inputs:
        check_image_processor(ckpt)
    return config


def _sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _run(*cmd: str | Path) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    started = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
    print(f"  {time.time() - started:.0f} s", flush=True)


def _llama_cpp_rev(llama_cpp: Path) -> str:
    """The commit the tree was built at, which build_native.sh records after checking it against
    the pin."""
    path = llama_cpp / "LLAMA_CPP_REV"
    if not path.is_file():
        raise SystemExit(f"{llama_cpp} has no LLAMA_CPP_REV: build it with scripts/build_native.sh")
    return path.read_text().strip()


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument("out", type=Path)
    ap.add_argument("--revision", default=None, help="the Hub revision (a Hub id only)")
    ap.add_argument(
        "--expect-digest", default="", help="the checkpoint's manifest digest, to refuse any other"
    )
    ap.add_argument("--llama-cpp", type=Path, required=True, help="the patched llama.cpp source")
    ap.add_argument("--quantize", type=Path, required=True, help="llama-quantize")
    ap.add_argument("--work", type=Path, default=Path("/tmp/decision-build"))
    ap.add_argument(
        "--gguf-tmp", type=Path, default=None, help="where the BF16 GGUF goes; --work by default"
    )
    return ap.parse_args()


def main() -> None:
    a = _args()
    a.work.mkdir(parents=True, exist_ok=True)
    tmp = a.gguf_tmp or a.work
    rev = _llama_cpp_rev(a.llama_cpp)
    sys.path.insert(0, str(a.llama_cpp / "gguf-py"))  # the gguf-py the converter writes with
    ckpt, commit = fetch(a.checkpoint, a.revision, a.work)
    digest = manifest_digest(ckpt)
    print(f"checkpoint manifest digest {digest}", flush=True)
    if a.expect_digest and a.expect_digest != digest:
        raise SystemExit(
            f"the checkpoint's manifest digest is {digest}, not the expected {a.expect_digest}"
        )
    config = _check(ckpt)
    name = a.checkpoint.rstrip("/").split("/")[-1]
    a.out.mkdir(parents=True, exist_ok=True)
    convert = (sys.executable, a.llama_cpp / "convert_hf_to_gguf.py", ckpt, "--model-name", name)

    files = {"model.gguf": a.out / "model.gguf"}
    bf16 = tmp / "model-bf16.gguf"
    _run(*convert, "--outtype", "bf16", "--outfile", bf16)
    _run(a.quantize, bf16, files["model.gguf"], _QUANT, str(os.cpu_count() or 1))
    bf16.unlink()
    if IMAGE in config.inputs:
        files["mmproj.gguf"] = a.out / "mmproj.gguf"
        full = tmp / "mmproj-full.gguf"
        _run(*convert, "--mmproj", "--outtype", _VISION_OUTTYPE, "--outfile", full)
        vision_only(full, files["mmproj.gguf"])
        full.unlink()
        check_mmproj(files["mmproj.gguf"], ckpt)
    size = sum(p.stat().st_size for p in files.values())
    lazy = lazy_ranges(files["model.gguf"])
    resident = size - sum(end - start for start, end in lazy)
    if size > _MAX_FILE_BYTES:
        raise SystemExit(
            f"the converted model is {size:,} bytes, over the {_MAX_FILE_BYTES:,} that fit in a "
            "Lambda container image beside the runtime"
        )
    if resident > _MAX_RESIDENT_BYTES:
        raise SystemExit(
            f"the converted model holds {resident:,} bytes in memory (its {size:,} less the lazily "
            f"read per-layer embeddings), over the {_MAX_RESIDENT_BYTES:,} that leave a 10,240 MB "
            "Lambda function room to run it; Lambda serves E2B and E4B"
        )
    for f in _FILES:
        shutil.copyfile(ckpt / f, a.out / f)
        files[f] = a.out / f
    artifact = {
        "model_name": f"{name}-{_QUANT.lower()}",
        "checkpoint": a.checkpoint,
        "revision": commit,
        "checkpoint_digest": digest,
        "llama_cpp": rev,
        "lazy_ranges": lazy,
        "resident_bytes": resident,
        "max_request_tokens": max_request_tokens(resident),
        "files": {k: {"size": p.stat().st_size, "sha256": _sha256(p)} for k, p in files.items()},
    }
    (a.out / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
