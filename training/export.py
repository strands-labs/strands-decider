"""A merged or averaged checkpoint written as the decision interface's checkpoint (docs/spec.md sections 8 and 9).

    python -m training.export --checkpoint <merged or averaged checkpoint> --out <dir> --release <name> \\
        [--max-length-validated 16384] [--source https://...] [--goldens <jsonl>] [--data-summary <md>] \\
        [--license LICENSE]

The checkpoint is the base Gemma 4 checkpoint with the fine-tuned decoder in it, so every backend that loads the base
(transformers, vLLM, MLX, llama.cpp) loads it as is. The vision and audio encoders are the base's, unchanged, but
decision_config.json declares only the inputs the run trained on (training.json's `inputs`; text for this recipe):
  config.json              the base's, unchanged but for path-valued keys (its text, vision and audio configs, and
                           `use_bidirectional_attention` as the base sets it: a decision runtime replaces the
                           attention mask with the spec's prefix_bidir mask and turns that flag off)
  model*.safetensors       the base's tensors under the base's names and shards, bf16, the language-model tensors the
                           merged ones (no adapter, no LM head: Gemma 4 ties it to the embeddings)
  readout.safetensors      one tensor, `weight`, [255, hidden], bf16
  tokenizer.json           the base's
  tokenizer_config.json    the base's, without `chat_template` and without path-valued keys
  processor_config.json    the base's image and audio preprocessing, without `chat_template`
  preprocessor_config.json the base's, when it has one
  generation_config.json   the base's
  decision_config.json     the spec's schema only, with a public provenance (base_model, base_revision, release,
                           source)
  LICENSE, NOTICE, README.md   Apache-2.0; a NOTICE for this model's base and modifications; a short model card, with
                           --data-summary's text as its training-data section
  golden_vectors.jsonl     only when --goldens is given (the spec's model goldens, made by a reference run)
and nothing else: no code, no chat template, no training record. Before it finishes, the export checks the result with
strands_decider.core (DecisionConfig.load, Readout.load), checks every tensor header, and scans every file but the
tensors for local paths; any failure removes the output and exits non-zero. CPU only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import shutil
import struct
import sys
import textwrap
from pathlib import Path

from .merge import RECORD_FILE, weight_files
from .model import READOUT_FILE
from .prompt import TEMPLATES

WEIGHT_DTYPE = "BF16"
# Strings that mark a local location; none may appear in a released checkpoint's text files or tensor headers.
PRIVATE = ("/fsx/", "/home/", "/Users/", "/opt/", "/dev/shm", "/tmp/", "/mnt/", "s3://", "file://")
FORBIDDEN_NAMES = {
    "source",
    "requirements.txt",
    "chat_template.jinja",
    "__pycache__",
    "uv.lock",
    "pyproject.toml",
    RECORD_FILE,
}
FORBIDDEN_SUFFIXES = (".py", ".pyc", ".pyo", ".so", ".jinja")
GEMMA_LICENSE = "https://ai.google.dev/gemma/docs/gemma_4_license"
DEFAULT_LICENSE = Path(__file__).resolve().parent.parent / "LICENSE"


def header(path: Path) -> dict:
    """A safetensors file's JSON header."""
    with open(path, "rb") as stream:
        (size,) = struct.unpack("<Q", stream.read(8))
        value: dict = json.loads(stream.read(size))
        return value


def place(src: Path, dst: Path) -> None:
    """dst as a hard link to src, or a copy across file systems."""
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def check_weights(checkpoint: Path) -> None:
    """Every tensor bf16, no adapter or LM head tensor, no local path in any header."""
    for name in weight_files(checkpoint):
        for key, info in header(checkpoint / name).items():
            if key == "__metadata__":
                if any(p in json.dumps(info) for p in PRIVATE):
                    raise SystemExit(
                        f"{checkpoint.name}/{name}: its metadata names a local location"
                    )
                continue
            if info["dtype"] != WEIGHT_DTYPE:
                raise SystemExit(f"{key} is {info['dtype']}, not {WEIGHT_DTYPE}")
            if key.startswith("lm_head") or "lora_" in key:
                raise SystemExit(f"{key} does not belong in a decision checkpoint")


def strip_paths(d: dict) -> dict:
    d = {k: v for k, v in d.items() if k not in ("_name_or_path", "name_or_path", "chat_template")}
    return {k: v for k, v in d.items() if not (isinstance(v, str) and any(p in v for p in PRIVATE))}


def notice(release: str, base_model: str, revision: str, prompt: str, averaged: bool) -> str:
    average = (
        (
            " The released weights are the equal-weight average of several such fine-tuned models of the same "
            "base."
        )
        if averaged
        else ""
    )
    paragraphs = [
        release,
        f"Derived from {base_model} (https://huggingface.co/{base_model}), revision {revision}, under the Apache "
        f"License 2.0 ({GEMMA_LICENSE}). The checkpoint is the base model with its language-model decoder weights "
        "replaced by fine-tuned ones; the vision and audio encoders and their embedding projections are the base's, "
        "unchanged.",
        "Modifications: low-rank adapters trained by supervised fine-tuning on decision data are merged into the "
        f"decoder weights.{average} A 255-code decision readout (readout.safetensors) replaces the generation output "
        "layer; it started from the base's output-embedding rows of the answer codes and was trained with the "
        f"adapters. The model reads its prompt ({prompt}) under the prefix-bidirectional attention mask of the Strands "
        "Decider decision interface: the state attends to itself in both directions, in every layer, and each "
        "question attends to the whole prompt. Pooling uses the last prompt token.",
        "Gemma is a trademark of Google LLC. No affiliation with or endorsement by Google is implied.",
    ]
    return (
        "\n\n".join(
            textwrap.fill(t, 100, break_long_words=False, break_on_hyphens=False)
            for t in paragraphs
        )
        + "\n"
    )


def trained_inputs(record: dict) -> list[str]:
    """The input types a checkpoint may declare: those its run trained on (the training record's `inputs`), text when
    the record predates the field (the previous trainer also dropped every row with images)."""
    inputs = list((record.get("finetune") or {}).get("inputs") or ["text"])
    if not set(inputs) <= {"text", "image"} or "text" not in inputs:
        raise SystemExit(f"the training record's inputs {inputs} are not text, or text and image")
    return inputs


def card(
    release: str, base_model: str, revision: str, cfg: dict, inputs: list[str], data_summary: str
) -> str:
    text = f"""# {release}

A Strands Decider decision model. Given a state and one or more questions, each with its options, it returns a
probability for every option in a single forward pass, with no generated text.

| | |
|---|---|
| Base | [{base_model}](https://huggingface.co/{base_model}) @ `{revision[:8]}` (the full model; vision and audio encoders unchanged) |
| Inputs | {", ".join(inputs)} |
| Prompt template | `{cfg["prompt"]}` |
| Attention | `{cfg["attention_mode"]}` |
| Readout | 255 answer codes, soft-cap {cfg["final_logit_softcapping"]}, temperature {cfg["temperature"]} |
| Longest validated prompt | {cfg["max_length_validated"]:,} tokens |
| License | Apache-2.0 (`LICENSE`, `NOTICE`) |

The checkpoint follows the Strands Decider decision interface (`docs/spec.md` of
[strands-labs/strands-decider](https://github.com/strands-labs/strands-decider)): render requests with
`strands_decider.core.render` (never a chat template), run the decoder under the `prefix_bidir` mask, and read the
answer from `readout.safetensors` as the spec's section 7 describes.
"""
    if data_summary:
        text += "\n## Training data\n\n" + data_summary.strip() + "\n"
    return text


def scan(out: Path) -> None:
    for path in sorted(out.rglob("*")):
        rel = path.relative_to(out)
        if any(part in FORBIDDEN_NAMES for part in rel.parts) or path.suffix in FORBIDDEN_SUFFIXES:
            raise SystemExit(
                f"{rel}: not part of a decision checkpoint (no code, no chat template)"
            )
        if path.is_dir():
            raise SystemExit(f"{rel}: a decision checkpoint is one flat directory")
        if path.suffix == ".safetensors":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in PRIVATE:
            if marker in text:
                raise SystemExit(f"{rel}: contains {marker!r} (a local location)")


def main() -> None:
    from strands_decider import core

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="a merged (training.merge) or averaged (training.average) checkpoint",
    )
    ap.add_argument(
        "--out", required=True, help="the checkpoint directory to write (must not exist)"
    )
    ap.add_argument(
        "--release", required=True, help="the released model's name, e.g. Strands-Decider-E2B"
    )
    ap.add_argument(
        "--max-length-validated",
        type=int,
        default=None,
        help="the longest prompt (tokens) the checkpoint was validated at (default: its training max length)",
    )
    ap.add_argument(
        "--source",
        default=None,
        help="an https URL of the public code and recipe (provenance.source)",
    )
    ap.add_argument(
        "--goldens", default=None, help="golden_vectors.jsonl from a reference run, copied in"
    )
    ap.add_argument(
        "--data-summary",
        default=None,
        help="a markdown file (sources, licences, attributions) for the card's training-data section",
    )
    ap.add_argument(
        "--license", default=str(DEFAULT_LICENSE), help="the Apache-2.0 license text to ship"
    )
    a = ap.parse_args()
    checkpoint, out = Path(a.checkpoint).resolve(), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists; export into a new directory")
    if not (checkpoint / RECORD_FILE).is_file():
        raise SystemExit(f"{checkpoint}: no {RECORD_FILE} (export a merged or averaged checkpoint)")
    src = json.loads((checkpoint / RECORD_FILE).read_text())
    if src.get("prompt") not in TEMPLATES:
        raise SystemExit(
            f"{checkpoint}: prompt {src.get('prompt')!r}; the spec defines {TEMPLATES} only"
        )
    if src.get("attention_mode") != "prefix_bidir" or src.get("pooling", "last") != "last":
        raise SystemExit(
            f"{checkpoint}: the spec's checkpoints are prefix_bidir with last-token pooling"
        )
    if not src.get("finetune"):
        raise SystemExit(f"{checkpoint}: no training record; export a trained checkpoint")
    if re.fullmatch(r"[0-9a-f]{40}", src.get("base_revision") or "") is None:
        raise SystemExit("the checkpoint's base revision is not a 40-character commit")
    max_len = a.max_length_validated or src["finetune"].get("max_length")
    if not max_len:
        raise SystemExit(
            "pass --max-length-validated (the checkpoint records no training max length)"
        )

    config = json.loads((checkpoint / "config.json").read_text())
    if "vision_config" not in config or "text_config" not in config:
        raise SystemExit(
            f"{checkpoint}: not a full Gemma 4 checkpoint (no vision_config/text_config)"
        )
    softcap = config["text_config"].get("final_logit_softcapping")
    if softcap != src.get("final_logit_softcapping"):
        raise SystemExit("config.json's final_logit_softcapping differs from the training record's")
    config = strip_paths(config)
    check_weights(checkpoint)
    readout = header(checkpoint / READOUT_FILE)
    shape = (readout.get("weight") or {}).get("shape")
    if (
        set(readout) - {"__metadata__"} != {"weight"}
        or readout["weight"]["dtype"] != WEIGHT_DTYPE
        or shape is None
        or shape[0] != len(core.CODES)
    ):
        raise SystemExit(
            f"{checkpoint}: readout.safetensors must hold one bf16 tensor `weight` [255, hidden]"
        )
    license_text = Path(a.license).read_text()
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise SystemExit(f"{a.license} is not the Apache License 2.0")

    inputs = trained_inputs(src)
    decision = {
        "interface": core.INTERFACE,
        "format_version": core.FORMAT_VERSION,
        "prompt": src["prompt"],
        "codes": list(src["codes"]),
        "token_ids": list(src["token_ids"]),
        "temperature": float(src.get("temperature", 1.0)),
        "final_logit_softcapping": float(softcap) if softcap is not None else None,
        "attention_mode": "prefix_bidir",
        "pooling": "last",
        "max_length_validated": int(max_len),
        "provenance": {
            k: v
            for k, v in {
                "base_model": src.get("base_model"),
                "base_revision": src.get("base_revision"),
                "release": a.release,
                "source": a.source,
            }.items()
            if v
        },
    }
    fields = (
        {f.name for f in dataclasses.fields(core.DecisionConfig)}
        if dataclasses.is_dataclass(core.DecisionConfig)
        else set()
    )
    if "inputs" in fields:
        decision["inputs"] = inputs
    else:
        print(
            json.dumps(
                {
                    "note": "this strands_decider core has no `inputs` field; decision_config.json leaves it "
                    "out"
                }
            ),
            file=sys.stderr,
        )
    # the spec's schema, strictly, before anything is written
    core.DecisionConfig.from_dict(decision)
    data_summary = Path(a.data_summary).read_text() if a.data_summary else ""

    out.mkdir(parents=True)
    try:
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        for name in weight_files(checkpoint):
            place(checkpoint / name, out / name)
        if (checkpoint / "model.safetensors.index.json").exists():
            shutil.copyfile(
                checkpoint / "model.safetensors.index.json", out / "model.safetensors.index.json"
            )
        shutil.copyfile(checkpoint / READOUT_FILE, out / READOUT_FILE)
        place(checkpoint / "tokenizer.json", out / "tokenizer.json")
        tokenizer_config = strip_paths(
            json.loads((checkpoint / "tokenizer_config.json").read_text())
        )
        (out / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n")
        for name in ("processor_config.json", "preprocessor_config.json", "generation_config.json"):
            if (checkpoint / name).exists():
                (out / name).write_text(
                    json.dumps(strip_paths(json.loads((checkpoint / name).read_text())), indent=2)
                    + "\n"
                )
        (out / "decision_config.json").write_text(json.dumps(decision, indent=2) + "\n")
        (out / "LICENSE").write_text(license_text)
        (out / "NOTICE").write_text(
            notice(
                a.release,
                src["base_model"],
                src["base_revision"],
                src["prompt"],
                averaged="average" in src,
            )
        )
        (out / "README.md").write_text(
            card(
                a.release,
                src["base_model"],
                src["base_revision"],
                decision,
                decision.get("inputs", inputs),
                data_summary,
            )
        )
        if a.goldens:
            shutil.copy2(a.goldens, out / "golden_vectors.jsonl")
        scan(out)
        linked = {name for name in weight_files(checkpoint)} | {"tokenizer.json"}
        for path in out.iterdir():
            if (
                path.name not in linked
            ):  # the hard links share the merged checkpoint's files and modes
                os.chmod(path, 0o644)
        loaded = core.DecisionConfig.load(out)
        core.Readout.load(out, loaded)
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "out": str(out),
                "release": a.release,
                "prompt": decision["prompt"],
                "inputs": decision.get("inputs"),
                "files": sorted(p.name for p in out.iterdir()),
                "max_length_validated": decision["max_length_validated"],
                "goldens": bool(a.goldens),
            }
        )
    )


if __name__ == "__main__":
    main()
