"""Create a local dtype-override checkpoint without changing cached Hub files.

The output links checkpoint assets and copies only its small configuration file.
Usage: python evaluation/prepare_mlx_dtype.py LOCAL_SNAPSHOT OUTPUT --dtype float16
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], required=True)
    args = ap.parse_args()
    source = args.source.resolve(strict=True)
    config_name = "strands_decider_config.json"
    if not (source / config_name).exists():
        config_name = "hobson_config.json"
    config = json.loads((source / config_name).read_text())
    original_dtype = config["torch_dtype"]
    config["torch_dtype"] = args.dtype
    args.output.mkdir(parents=True, exist_ok=False)
    # Deliberately do not copy the upstream checksum manifest: config has changed.
    for name in ["head.safetensors", "slot_head.pt", "lora", "tokenizer.json",
                 "tokenizer_config.json", "chat_template.jinja", "provenance.json",
                 "special_tokens_map.json", "vocab.json", "merges.txt"]:
        asset = source / name
        if asset.exists():
            (args.output / name).symlink_to(asset.resolve())
    (args.output / config_name).write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "dtype_override.json").write_text(json.dumps({
        "source": str(source), "original_dtype": original_dtype, "dtype": args.dtype
    }, indent=2) + "\n")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
