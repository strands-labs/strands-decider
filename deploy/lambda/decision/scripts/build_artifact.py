"""Build what the Lambda image serves from a decision checkpoint.

    python build_artifact.py CHECKPOINT OUT_DIR --llama-cpp DIR --quantize PATH [--revision REV]
        [--expect-digest SHA256] [--work DIR] [--gguf-tmp DIR]

CHECKPOINT is a Hugging Face Hub id, an s3:// URI of a checkpoint directory (read with the
environment's AWS credentials), or a local directory. OUT_DIR gets:

    model.gguf             the decoder: llama.cpp's stock convert_hf_to_gguf.py to BF16 (lossless
                           from the bf16 checkpoint), then llama-quantize to Q8_0
    mmproj.gguf            for a checkpoint whose inputs hold "image": the vision tower and its
                           projection, convert_hf_to_gguf.py --mmproj at q8_0, without the audio
                           tower (vision_only_mmproj.py)
    readout.safetensors, decision_config.json, tokenizer.json, processor_config.json
                           the checkpoint's own, unchanged
    artifact.json          the model's name; the checkpoint, its resolved Hub commit and its
                           manifest digest; the llama.cpp commit; each file's size and sha256;
                           and the byte ranges of model.gguf that llama.cpp reads lazily

The checkpoint is pinned: a Hub revision is resolved to its commit, and each of its LFS files is
checked against the sha256 the Hub records. The manifest digest is the sha256 of the lines
`<sha256>  <file>`, one per file of the checkpoint, sorted by name (in a checkpoint directory,
`sha256sum * | sort -k 2 | sha256sum`); with --expect-digest, any other is refused.

Before anything is converted, the checkpoint is loaded as the function loads it: its
decision_config.json, readout, tokenizer and, for images, image processor settings.
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

# From a checkout, the repository's own package; in the image build, the one beside this script.
for _parent in Path(__file__).resolve().parents:
    if (_parent / "src" / "strands_decider").is_dir():
        sys.path.insert(0, str(_parent / "src"))
        break

from strands_decider.core import (  # noqa: E402
    DecisionConfig,
    check_image_processor,
    load_readout,
    load_tokenizer,
)
from strands_decider.core.config import IMAGE  # noqa: E402

HERE = Path(__file__).resolve().parent
QUANT = "Q8_0"
# The decoder and the vision tower must leave the function room for llama.cpp's buffers, Python and
# a request's state within Lambda's 10,240 MB: E2B's are 4.95 GB and, at q8_0, 0.23 GB.
MAX_BYTES = 6_000_000_000
FILES = ("readout.safetensors", "decision_config.json", "tokenizer.json", "processor_config.json")
CHUNK = 16 << 20


def fetch(checkpoint: str, revision: str | None, work: Path) -> tuple[Path, str | None]:
    """The checkpoint as a local directory, and the Hub commit it was read at (None off the Hub).
    A Hub checkpoint's LFS files are checked against the sha256 the Hub records for them."""
    if os.path.isdir(checkpoint):
        return Path(checkpoint), None
    dest = work / "checkpoint"
    started, total = time.time(), 0
    if checkpoint.startswith("s3://"):
        import boto3
        from boto3.s3.transfer import TransferConfig

        bucket, _, prefix = checkpoint.removeprefix("s3://").partition("/")
        prefix = prefix.rstrip("/") + "/" if prefix else ""
        s3 = boto3.client("s3")
        cfg = TransferConfig(
            max_concurrency=32, multipart_chunksize=64 << 20, multipart_threshold=64 << 20
        )
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix) :]
                if not rel or "/" in rel:  # the checkpoint's own files, not subdirectories
                    continue
                (dest / rel).parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(bucket, obj["Key"], str(dest / rel), Config=cfg)
                total += obj["Size"]
        if total == 0:
            raise SystemExit(f"nothing under {checkpoint}")
        commit = None
    else:
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi().model_info(checkpoint, revision=revision, files_metadata=True)
        commit = info.sha
        snapshot_download(checkpoint, revision=commit, local_dir=dest)
        for f in info.siblings or []:
            lfs = getattr(f, "lfs", None)
            want = getattr(lfs, "sha256", None) if lfs is not None else None
            if want and sha256(dest / f.rfilename) != want:
                raise SystemExit(f"{f.rfilename} does not have the sha256 the Hub records, {want}")
        total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(
        f"fetched {checkpoint} ({commit or 'no commit'}): {total:,} bytes in {time.time() - started:.0f} s",
        flush=True,
    )
    return dest, commit


def manifest_digest(ckpt: Path) -> str:
    """The sha256 of `<sha256>  <file>` lines, one per file of the checkpoint, sorted by name."""
    lines = [
        f"{sha256(p)}  {p.name}\n"
        for p in sorted(ckpt.iterdir(), key=lambda p: p.name)
        if p.is_file()
    ]
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def check(ckpt: Path) -> DecisionConfig:
    """The checkpoint's decision config, with its other files loaded as the function loads them."""
    config = DecisionConfig.load(ckpt)
    load_readout(ckpt)
    load_tokenizer(ckpt)
    for name in FILES:
        if not (ckpt / name).is_file():
            raise SystemExit(f"{ckpt} has no {name}")
    if IMAGE in config.inputs:
        check_image_processor(ckpt)
    return config


def run(*cmd: str | Path) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    started = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
    print(f"  {time.time() - started:.0f} s", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK):
            h.update(chunk)
    return h.hexdigest()


def gguf_has_audio(llama_cpp: Path, path: Path) -> bool:
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf

    return any(t.name.startswith(("a.", "mm.a.")) for t in gguf.GGUFReader(path).tensors)


# The tensors llama.cpp reads lazily, a few rows per batch: Gemma 4 E2B's and E4B's per-layer
# embedding table, half of E2B's file.
LAZY_TENSORS = ("per_layer_token_embd.weight",)


def lazy_ranges(llama_cpp: Path, path: Path) -> list[list[int]]:
    """The byte ranges, in the GGUF, of the tensors llama.cpp reads only a few rows of at a time."""
    sys.path.insert(0, str(llama_cpp / "gguf-py"))
    import gguf

    return [
        [int(t.data_offset), int(t.data_offset) + int(t.n_bytes)]
        for t in gguf.GGUFReader(path).tensors
        if t.name in LAZY_TENSORS
    ]


def llama_cpp_rev(llama_cpp: Path) -> str:
    """The llama.cpp commit the tree was built at: the LLAMA_CPP_REV file build_native.sh writes
    after it checks the checkout against the pin."""
    path = llama_cpp / "LLAMA_CPP_REV"
    if not path.is_file():
        raise SystemExit(f"{llama_cpp} has no LLAMA_CPP_REV: build it with scripts/build_native.sh")
    return path.read_text().strip()


def main() -> None:
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
    a = ap.parse_args()

    a.work.mkdir(parents=True, exist_ok=True)
    tmp = a.gguf_tmp or a.work
    rev = llama_cpp_rev(a.llama_cpp)
    ckpt, commit = fetch(a.checkpoint, a.revision, a.work)
    digest = manifest_digest(ckpt)
    print(f"checkpoint manifest digest {digest}", flush=True)
    if a.expect_digest and a.expect_digest != digest:
        raise SystemExit(
            f"the checkpoint's manifest digest is {digest}, not the expected {a.expect_digest}"
        )
    config = check(ckpt)
    name = a.checkpoint.rstrip("/").split("/")[-1]
    a.out.mkdir(parents=True, exist_ok=True)
    convert = a.llama_cpp / "convert_hf_to_gguf.py"

    bf16 = tmp / "model-bf16.gguf"
    run(sys.executable, convert, ckpt, "--outtype", "bf16", "--outfile", bf16, "--model-name", name)
    run(a.quantize, bf16, a.out / "model.gguf", QUANT, str(os.cpu_count() or 1))
    bf16.unlink()
    files = {"model.gguf": a.out / "model.gguf"}
    if IMAGE in config.inputs:
        full = tmp / "mmproj-full.gguf"
        run(sys.executable, convert, ckpt, "--mmproj", "--outtype", "q8_0", "--outfile", full,
            "--model-name", name)  # fmt: skip
        if gguf_has_audio(a.llama_cpp, full):
            run(
                sys.executable,
                HERE / "vision_only_mmproj.py",
                a.llama_cpp,
                full,
                a.out / "mmproj.gguf",
            )
            full.unlink()
        else:
            shutil.move(full, a.out / "mmproj.gguf")
        files["mmproj.gguf"] = a.out / "mmproj.gguf"
    size = sum(p.stat().st_size for p in files.values())
    if size > MAX_BYTES:
        raise SystemExit(
            f"the converted model is {size:,} bytes, over the {MAX_BYTES:,} that leave a 10,240 MB "
            "Lambda function room to run it; Lambda serves E2B"
        )
    for f in FILES:
        shutil.copyfile(ckpt / f, a.out / f)
        files[f] = a.out / f
    artifact = {
        "model_name": f"{name}-{QUANT.lower()}",
        "checkpoint": a.checkpoint,
        "revision": commit,
        "checkpoint_digest": digest,
        "inputs": list(config.inputs),
        "quantization": QUANT,
        "llama_cpp": rev,
        "lazy_ranges": lazy_ranges(a.llama_cpp, a.out / "model.gguf"),
        "files": {k: {"size": p.stat().st_size, "sha256": sha256(p)} for k, p in files.items()},
    }
    (a.out / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
