"""Build what the Lambda image serves from a decision checkpoint (docs/spec.md, The checkpoint layout).

    python build_artifact.py CHECKPOINT OUT_DIR --llama-cpp DIR --quantize PATH [--revision REV]
        [--name NAME] [--mmproj-type f16] [--max-bytes N] [--work DIR] [--gguf-tmp DIR]

CHECKPOINT is a Hugging Face Hub id, an s3:// URI of a checkpoint directory (read with the
environment's AWS credentials), or a local directory. OUT_DIR gets:

    model.gguf             the decoder: llama.cpp's stock convert_hf_to_gguf.py to BF16 (lossless
                           from the bf16 checkpoint), then llama-quantize to Q8_0
    mmproj.gguf            for a checkpoint whose inputs hold "image": the vision tower and its
                           projection, convert_hf_to_gguf.py --mmproj at q8_0 (f16 with
                           --mmproj-type), without the audio tower (vision_only_mmproj.py)
    readout.safetensors, decision_config.json, tokenizer.json, processor_config.json
                           the checkpoint's own, unchanged
    artifact.json          the model's name, the checkpoint, the llama.cpp commit, the files' sha256,
                           and the byte ranges of model.gguf that llama.cpp reads lazily

The checkpoint is checked before anything is converted, as a runtime checks it: decision_config.json
must validate (`"interface": "strands-decider"`, the soft-cap among its required fields),
config.json's `text_config.final_logit_softcapping` must equal it, and an image checkpoint's image
processor must have the settings docs/spec.md counts soft tokens for. The llama.cpp tree must carry
deploy/lambda/decision/patches: without the second, the converter takes the checkpoint for a
pplx-decider one.
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

from strands_decider.backends.llama_cpp import check_processor_config  # noqa: E402
from strands_decider.core import DecisionConfig, Readout  # noqa: E402
from strands_decider.core.config import IMAGE  # noqa: E402

HERE = Path(__file__).resolve().parent
QUANT = "Q8_0"
# The decoder and the vision tower must leave the function room for llama.cpp's buffers, Python and
# a request's state within Lambda's 10,240 MB: E2B's are 4.95 GB and 0.37 GB.
MAX_BYTES = 6_000_000_000
FILES = ("readout.safetensors", "decision_config.json", "tokenizer.json", "processor_config.json")


def fetch(checkpoint: str, revision: str | None, work: Path) -> Path:
    """The checkpoint as a local directory."""
    if os.path.isdir(checkpoint):
        return Path(checkpoint)
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
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(checkpoint, revision=revision, local_dir=dest)
        total = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())
    print(f"fetched {checkpoint}: {total:,} bytes in {time.time() - started:.0f} s", flush=True)
    return dest


def check(ckpt: Path) -> DecisionConfig:
    """The checkpoint's decision config, validated, and its other files checked against it."""
    config = DecisionConfig.load(ckpt)
    Readout.load(ckpt, config)  # the weight's shape, and config.json's soft-cap against config's
    with open(ckpt / "config.json", encoding="utf-8") as f:
        text = json.load(f).get("text_config") or {}
    if text.get("final_logit_softcapping") != config.final_logit_softcapping:
        raise SystemExit(
            f"config.json's text_config.final_logit_softcapping is "
            f"{text.get('final_logit_softcapping')!r}; decision_config.json's is "
            f"{config.final_logit_softcapping!r}"
        )
    for name in FILES:
        if not (ckpt / name).is_file():
            raise SystemExit(f"{ckpt} has no {name}")
    if IMAGE in config.inputs:
        check_processor_config(ckpt / "processor_config.json")
    with open(ckpt / "tokenizer_config.json", encoding="utf-8") as f:
        if "chat_template" in json.load(f) or (ckpt / "chat_template.jinja").exists():
            raise SystemExit(
                "a decision checkpoint carries no chat template (docs/spec.md, section 9)"
            )
    return config


def run(*cmd: str | Path) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    started = time.time()
    subprocess.run([str(c) for c in cmd], check=True)
    print(f"  {time.time() - started:.0f} s", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(16 << 20):
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


def llama_cpp_rev(llama_cpp: Path) -> str | None:
    """The llama.cpp commit: LLAMA_CPP_REV (the image build's), else the tree's git HEAD."""
    if os.environ.get("LLAMA_CPP_REV"):
        return os.environ["LLAMA_CPP_REV"]
    try:
        out = subprocess.run(
            ["git", "-C", str(llama_cpp), "rev-parse", "HEAD"], capture_output=True, text=True
        )
    except OSError:
        return None
    return out.stdout.strip() or None


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoint")
    ap.add_argument("out", type=Path)
    ap.add_argument("--revision", default=None, help="the Hub revision (a Hub id only)")
    ap.add_argument("--name", default=None, help="the model's name; the checkpoint's by default")
    ap.add_argument("--llama-cpp", type=Path, required=True, help="the patched llama.cpp source")
    ap.add_argument("--quantize", type=Path, required=True, help="llama-quantize")
    ap.add_argument("--mmproj-type", choices=("f16", "q8_0"), default="q8_0")
    ap.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    ap.add_argument("--work", type=Path, default=Path("/tmp/decision-build"))
    ap.add_argument(
        "--gguf-tmp", type=Path, default=None, help="where the BF16 GGUF goes; --work by default"
    )
    a = ap.parse_args()

    a.work.mkdir(parents=True, exist_ok=True)
    tmp = a.gguf_tmp or a.work
    ckpt = fetch(a.checkpoint, a.revision, a.work)
    config = check(ckpt)
    name = a.name or config.provenance.release or a.checkpoint.rstrip("/").split("/")[-1]
    name = name.split("/")[-1]
    a.out.mkdir(parents=True, exist_ok=True)
    convert = a.llama_cpp / "convert_hf_to_gguf.py"

    bf16 = tmp / "model-bf16.gguf"
    run(sys.executable, convert, ckpt, "--outtype", "bf16", "--outfile", bf16, "--model-name", name)
    run(a.quantize, bf16, a.out / "model.gguf", QUANT, str(os.cpu_count() or 1))
    bf16.unlink()
    files = {"model.gguf": a.out / "model.gguf"}
    if IMAGE in config.inputs:
        full = tmp / "mmproj-full.gguf"
        run(sys.executable, convert, ckpt, "--mmproj", "--outtype", a.mmproj_type, "--outfile", full,
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
    if size > a.max_bytes:
        raise SystemExit(
            f"the converted model is {size:,} bytes, over the {a.max_bytes:,} that leave a 10,240 MB "
            "Lambda function room to run it; Lambda serves E2B"
        )
    for f in FILES:
        shutil.copyfile(ckpt / f, a.out / f)
    artifact = {
        "model_name": f"{name}-{QUANT.lower()}",
        "checkpoint": a.checkpoint,
        "revision": a.revision
        if not (a.checkpoint.startswith("s3://") or os.path.isdir(a.checkpoint))
        else None,
        "inputs": list(config.inputs),
        "quantization": QUANT,
        "mmproj_type": a.mmproj_type if "mmproj.gguf" in files else None,
        "llama_cpp": llama_cpp_rev(a.llama_cpp),
        "lazy_ranges": lazy_ranges(a.llama_cpp, a.out / "model.gguf"),
        "sha256": {k: sha256(p) for k, p in files.items()},
    }
    (a.out / "artifact.json").write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
