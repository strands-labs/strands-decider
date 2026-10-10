"""Convert a published Strands Decider release into the files the browser demo loads.

    python web/convert/convert.py StrandsAgents/strands-decider-2B-hobson-v21 --out dist/weights

Steps: download the release and its pinned base model, fold the LoRA adapter into the base
torso, export the torso to ONNX with the ONNX Runtime GenAI model builder (int4 weights, no
LM head, output `hidden_states`), split the weights into shards the browser can allocate,
rewrite the int4 embedding lookup into standard ops so ONNX Runtime Web's WASM build can
run the graph too, and write `manifest.json`, which the worker reads.

The output directory holds `model/` (tokenizer, decider config, pointer head) and `onnx/`
(graph and weight shards). `manifest.json` records where everything came from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from importlib.metadata import version

import onnx
import torch
from huggingface_hub import HfApi, snapshot_download
from onnx import TensorProto
from onnx import helper as h
from onnx.external_data_helper import ExternalDataInfo
from publish import converter_id  # web/convert/ is the script directory, so on sys.path
from safetensors.torch import save_file

from strands_decider.modeling import (
    StrandsDeciderConfig,
    base_revision,
    checkpoint_dir,
    config_path,
)

SHARD_BYTES = 512 << 20  # Chrome refuses much larger single ArrayBuffers in a worker
MODEL_FILES = ["tokenizer.json", "tokenizer_config.json", "head.safetensors", "LICENSE.md"]


def merge(release_dir: str, config: StrandsDeciderConfig, out: str) -> dict:
    """The base torso with the adapter folded in, saved in the base checkpoint's layout
    (Qwen3_5ForConditionalGeneration names, text weights only, fp16) for the model builder."""
    import transformers
    from peft import PeftModel

    base_cfg = transformers.AutoConfig.from_pretrained(
        config.base_model, revision=config.base_revision
    )
    if base_cfg.model_type != "qwen3_5":
        raise SystemExit(f"base model type {base_cfg.model_type!r}: only qwen3_5 is supported")
    # The same text-tower load as StrandsDeciderModel._load_torso, in fp32 so the merge adds
    # the LoRA delta at full precision before the single rounding to fp16.
    lm = transformers.Qwen3_5ForCausalLM.from_pretrained(
        config.base_model,
        config=base_cfg.get_text_config(),
        revision=config.base_revision,
        dtype=torch.float32,
    )
    torso = PeftModel.from_pretrained(lm.model, os.path.join(release_dir, "lora"))
    torso = torso.merge_and_unload()
    del lm
    # One tensor at a time, so the fp32 and fp16 copies of the whole model never coexist
    # (that peak is too close to a 16 GB CI runner).
    for p in torso.parameters():
        p.data = p.data.half()
    state = {f"model.language_model.{k}": v.contiguous() for k, v in torso.state_dict().items()}
    del torso
    os.makedirs(out, exist_ok=True)
    save_file(state, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    del state
    # The base's own config and tokenizer files, as the builder reads them from the base repo.
    base = snapshot_download(
        config.base_model, revision=config.base_revision, allow_patterns=["*.json", "*.txt"]
    )
    for f in os.listdir(base):
        if not f.startswith("model.safetensors"):
            shutil.copy(os.path.join(base, f), out)
    with open(os.path.join(out, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["torch_dtype"] = cfg["dtype"] = cfg["text_config"]["dtype"] = "float16"
    with open(os.path.join(out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    return cfg["text_config"]


def build_onnx(merged: str, out: str, cache: str) -> None:
    subprocess.run(
        [
            sys.executable, "-m", "onnxruntime_genai.models.builder",
            "-i", merged, "-o", out, "-p", "int4", "-e", "webgpu", "-c", cache,
            "--extra_options", "exclude_lm_head=true", "exclude_mtp=true",
            "op_types_to_quantize=MatMul/Gather",
        ],
        check=True,
    )  # fmt: skip


def shard(model: onnx.ModelProto, data_path: str, out: str) -> list[str]:
    """Rewrite the external data into <= SHARD_BYTES files, 64-byte aligned."""
    shards: list[str] = []
    cur, size = None, 0
    with open(data_path, "rb") as src:
        for t in model.graph.initializer:
            if t.data_location != TensorProto.EXTERNAL:
                continue
            info = ExternalDataInfo(t)
            if cur is None or size + info.length > SHARD_BYTES:
                if cur:
                    cur.close()
                shards.append(f"model.onnx.data.{len(shards)}")
                cur, size = open(os.path.join(out, shards[-1]), "wb"), 0
            pad = (-size) % 64
            cur.write(b"\0" * pad)
            size += pad
            src.seek(info.offset)
            cur.write(src.read(info.length))
            del t.external_data[:]
            for key, value in (("location", shards[-1]), ("offset", size), ("length", info.length)):
                entry = t.external_data.add()
                entry.key, entry.value = key, str(value)
            size += info.length
    if cur:
        cur.close()
    return shards


def portable_embedding(model: onnx.ModelProto) -> None:
    """Replace com.microsoft.GatherBlockQuantized (the int4 embedding) with standard ops.

    ONNX Runtime Web's WASM build has no kernel for it. The int4 table's bytes are read as
    uint8 [V, D/2] (element 2i in the low nibble), so the weight shards stay as they are."""
    g = model.graph
    node = next(n for n in g.node if n.op_type == "GatherBlockQuantized")
    qname, ids, sname = node.input
    block = next(a.i for a in node.attribute if a.name == "block_size")
    table = next(i for i in g.initializer if i.name == qname)
    if table.data_type != TensorProto.INT4:
        raise SystemExit(f"embedding table is {table.data_type}, expected INT4")
    V, D = table.dims
    table.data_type = TensorProto.UINT8
    del table.dims[:]
    table.dims.extend([V, D // 2])
    scale_type = next(i for i in g.initializer if i.name == sname).data_type
    P = "/model/embed_tokens/portable/"
    g.initializer.extend(
        [
            h.make_tensor(P + "15", TensorProto.UINT8, [], [15]),
            h.make_tensor(P + "4", TensorProto.UINT8, [], [4]),
            h.make_tensor(P + "8u", TensorProto.UINT8, [], [8]),
            h.make_tensor(P + "8f", scale_type, [], [8.0]),
            h.make_tensor(P + "ax", TensorProto.INT64, [1], [-1]),
            h.make_tensor(P + "s4", TensorProto.INT64, [4], [0, 0, D // block, block]),
            h.make_tensor(P + "s3", TensorProto.INT64, [3], [0, 0, D]),
        ]
    )
    new = [
        h.make_node("Gather", [qname, ids], [P + "g"], axis=0),
        h.make_node("BitwiseAnd", [P + "g", P + "15"], [P + "lo"]),
        h.make_node("BitShift", [P + "g", P + "4"], [P + "hi"], direction="RIGHT"),
        h.make_node("Unsqueeze", [P + "lo", P + "ax"], [P + "lo1"]),
        h.make_node("Unsqueeze", [P + "hi", P + "ax"], [P + "hi1"]),
        h.make_node("Concat", [P + "lo1", P + "hi1"], [P + "pair"], axis=-1),
        h.make_node("BitwiseXor", [P + "pair", P + "8u"], [P + "x"]),
        h.make_node("Cast", [P + "x"], [P + "xf"], to=scale_type),
        h.make_node("Sub", [P + "xf", P + "8f"], [P + "q"]),  # signed int4 in [-8, 7]
        h.make_node("Reshape", [P + "q", P + "s4"], [P + "qb"]),
        h.make_node("Gather", [sname, ids], [P + "sc"], axis=0),
        h.make_node("Unsqueeze", [P + "sc", P + "ax"], [P + "sc1"]),
        h.make_node("Mul", [P + "qb", P + "sc1"], [P + "deq"]),
        h.make_node("Reshape", [P + "deq", P + "s3"], [node.output[0]]),
    ]
    i = list(g.node).index(node)
    g.node.remove(node)
    for k, n in enumerate(new):
        g.node.insert(i + k, n)


def state_inputs(model: onnx.ModelProto, text_config: dict) -> dict[str, dict]:
    """The recurrent, conv and KV-cache inputs with the shape of an empty state, so the
    worker can feed a fresh forward without knowing the architecture."""
    types = {TensorProto.FLOAT16: "float16", TensorProto.FLOAT: "float32"}
    out = {}
    for i in model.graph.input:
        if i.name in ("input_ids", "attention_mask", "position_ids"):
            continue
        t = i.type.tensor_type
        dims = []
        for d in t.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            elif d.dim_param == "batch_size":
                dims.append(1)
            elif "sequence" in d.dim_param or "past" in d.dim_param:
                dims.append(0)
            elif d.dim_param == "kv_cache_dim":
                dims.append(text_config["head_dim"])
            else:
                raise SystemExit(f"input {i.name}: unknown dimension {d.dim_param!r}")
        out[i.name] = {"dtype": types[t.elem_type], "shape": dims}
    return out


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 24):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("repo", help="Hub model repo of the release")
    ap.add_argument("--revision", help="commit of the release (default: its main branch)")
    ap.add_argument("--out", required=True, help="output directory (replaced)")
    ap.add_argument("--work", help="scratch directory (default: a temporary one)")
    a = ap.parse_args()

    revision = a.revision or HfApi().model_info(a.repo).sha
    release = checkpoint_dir(a.repo, revision)
    config = StrandsDeciderConfig.from_json(config_path(release))
    config.base_revision = base_revision(release, config)
    if config.head_type != "pointer" or not config.use_lora:
        raise SystemExit("the browser demo supports pointer-head LoRA checkpoints only")

    shutil.rmtree(a.out, ignore_errors=True)
    os.makedirs(os.path.join(a.out, "model"))
    os.makedirs(os.path.join(a.out, "onnx"))
    with tempfile.TemporaryDirectory(dir=a.work) as work:
        merged, built = os.path.join(work, "merged"), os.path.join(work, "onnx")
        text_config = merge(release, config, merged)
        build_onnx(merged, built, os.path.join(work, "cache"))
        shutil.rmtree(merged)
        graph = onnx.load(os.path.join(built, "model.onnx"), load_external_data=False)
        shards = shard(graph, os.path.join(built, "model.onnx.data"), os.path.join(a.out, "onnx"))
    portable_embedding(graph)
    onnx.save(graph, os.path.join(a.out, "onnx", "model.onnx"))

    for f in MODEL_FILES:
        shutil.copy(os.path.join(release, f), os.path.join(a.out, "model", f))
    # One name for the decider config whichever name the release uses (hobson_config.json
    # before the rename); base_revision filled in from provenance.json.
    with open(os.path.join(a.out, "model", "decider_config.json"), "w") as fh:
        fh.write(config.to_json())

    files = {}
    for sub in ("model", "onnx"):
        for f in sorted(os.listdir(os.path.join(a.out, sub))):
            p = os.path.join(a.out, sub, f)
            files[f"{sub}/{f}"] = {"bytes": os.path.getsize(p), "sha256": sha256(p)}
    manifest = {
        "converter": converter_id(),
        "source": {"repo": a.repo, "revision": revision},
        "base": {"repo": config.base_model, "revision": config.base_revision},
        "versions": {
            p: version(p) for p in ("onnxruntime-genai", "onnx", "transformers", "peft", "torch")
        },
        "graph": "onnx/model.onnx",
        "shards": [f"onnx/{s}" for s in shards],
        "state_inputs": state_inputs(graph, text_config),
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)
    total = sum(f["bytes"] for f in files.values())
    print(f"wrote {a.out}: {len(files)} files, {total / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
