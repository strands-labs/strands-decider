"""The tiny checkpoint converted as the image build converts it, and the conversion checked: the
converter's pplx-decider loader claims a Qwen3.5 pplx-decider directory and none of the others
(ggml-org/llama.cpp#32156); the vision-only mmproj is the converter's vision tower byte for byte,
without the audio tower or its keys, and preprocesses as the image processor does; lazy_ranges
finds the per-layer embedding table.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from gguf_files import check_mmproj, is_audio_key, is_audio_tensor, lazy_ranges, vision_only

# Which directories the converter's pplx-decider loader claims: a config.json naming the backbone
# and a decision_config.json in each, beside a readout. Run in llama.cpp's tree, as its own tests
# import it.
_MATCHER = """
import json, sys, tempfile
from pathlib import Path
from conversion.pplx_decider import _is_pplx_decider_checkpoint
qwen, gemma = {"model_type": "qwen3_5"}, {"model_type": "gemma4"}
cases = {"pplx-decider": (qwen, {"format_version": 1, "temperature": 1.0}),
         "vllm-d3": (qwen, {"format_id": "d3-code-readout-v1", "format_version": 1}),
         "strands-decider": (gemma, {"interface": "strands-decider", "format_version": 1}),
         "not an object": (gemma, [1, 2])}
out = {}
for name, (config, decision) in cases.items():
    with tempfile.TemporaryDirectory() as d:
        for f, text in (("config.json", json.dumps(config)),
                        ("decision_config.json", json.dumps(decision)), ("readout.safetensors", "")):
            (Path(d) / f).write_text(text)
        try:
            out[name] = _is_pplx_decider_checkpoint(Path(d))
        except Exception as e:
            out[name] = repr(e)
print(json.dumps(out))
"""
_MATCHES = {
    "pplx-decider": True,
    "vllm-d3": False,
    "strands-decider": False,
    "not an object": False,
}


def convert(llama_cpp: Path, ckpt: Path, out: Path) -> tuple[Path, Path, Path]:
    """The stock converter's decoder and full mmproj, both f32, and the image build's vision-only
    copy of the mmproj."""
    out.mkdir(parents=True, exist_ok=True)
    gguf, full, mmproj = out / "model.gguf", out / "mmproj-full.gguf", out / "mmproj.gguf"
    for args in ([ckpt, "--outtype", "f32", "--outfile", gguf],
                 [ckpt, "--mmproj", "--outtype", "f32", "--outfile", full]):  # fmt: skip
        r = subprocess.run(
            [sys.executable, llama_cpp / "convert_hf_to_gguf.py", *map(str, args)],
            capture_output=True, text=True,
        )  # fmt: skip
        if r.returncode != 0:
            raise RuntimeError(
                f"convert_hf_to_gguf.py {' '.join(map(str, args[1:]))} failed:\n{r.stderr[-3000:]}"
            )
    vision_only(full, mmproj)
    return gguf, full, mmproj


def check_conversion(llama_cpp, ckpt, gguf_path, full, mmproj, failures) -> dict:
    """What `convert` made, measured; what misses is added to `failures`."""
    import gguf

    path = os.pathsep.join([str(llama_cpp), str(llama_cpp / "gguf-py")])
    env = {**os.environ, "PYTHONPATH": path}
    r = subprocess.run(
        [sys.executable, "-c", _MATCHER], cwd=llama_cpp, env=env, capture_output=True, text=True
    )
    matches = json.loads(r.stdout) if r.returncode == 0 else {"error": r.stderr[-500:]}
    if matches != _MATCHES:
        failures.append(f"the converter's pplx-decider loader claims {matches}, not {_MATCHES}")
    converted, vision = gguf.GGUFReader(full), gguf.GGUFReader(mmproj)
    if not any(is_audio_tensor(t.name) for t in converted.tensors):
        failures.append("the converter's mmproj has no audio tower to leave out")
    want = {t.name: t.data.tobytes() for t in converted.tensors if not is_audio_tensor(t.name)}
    if {t.name: t.data.tobytes() for t in vision.tensors} != want:
        failures.append("the vision-only mmproj is not the converter's vision tower, byte for byte")
    if any(is_audio_key(k) for k in vision.fields):
        failures.append("the vision-only mmproj keeps audio keys")
    try:
        check_mmproj(mmproj, ckpt)
    except ValueError as e:
        failures.append(str(e))
    ranges, size = lazy_ranges(gguf_path), gguf_path.stat().st_size
    if len(ranges) != 1 or not 0 < ranges[0][0] < ranges[0][1] <= size:
        failures.append(f"lazy_ranges gives {ranges} for a {size}-byte GGUF")
    return {
        "matcher": matches,
        "mmproj_tensors": [len(converted.tensors), len(vision.tensors)],
        "lazy_ranges": ranges,
    }
