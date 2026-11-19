"""The llama.cpp shim, compiled and run, against transformers (deploy/lambda/decision/scripts/shim_check.py).

It runs the real library on a tiny random Gemma 4 decision checkpoint, converted by the stock
converter, and holds it to a transformers reference: the prefix_bidir mask (and not a causal
state, a causal question, a sliding window, or the causal full-attention layers a build without
patches/0001 gives), state reuse bit for bit, image rows at their spans, the vision-only mmproj,
the converter's loader matcher, and the answers through `strands_decider.core.evaluate`.

It needs a llama.cpp tree built by deploy/lambda/decision/scripts/build_native.sh, and torch,
transformers, safetensors, sentencepiece and Pillow; no GPU, no download. Without them it skips,
unless STRANDS_DECIDER_REQUIRE_SHIM=1 (CI) makes that a failure:

    sh deploy/lambda/decision/scripts/build_native.sh /tmp/llama.cpp /tmp/llama-build
    STRANDS_DECIDER_LLAMA_CPP=/tmp/llama.cpp STRANDS_DECIDER_LLAMA_LIB=/tmp/llama-build/libdecision.so \\
        python -m pytest tests/test_llama_cpp_shim.py
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "deploy"
    / "lambda"
    / "decision"
    / "scripts"
    / "shim_check.py"
)
SRC = os.environ.get("STRANDS_DECIDER_LLAMA_CPP")
LIB = os.environ.get("STRANDS_DECIDER_LLAMA_LIB")
REQUIRED = os.environ.get("STRANDS_DECIDER_REQUIRE_SHIM") == "1"


@pytest.mark.skipif(
    not (SRC and LIB) and not REQUIRED, reason="set STRANDS_DECIDER_LLAMA_CPP and _LIB to a build"
)
def test_the_shim_is_the_reference(tmp_path, monkeypatch):
    if not (SRC and LIB):
        pytest.fail("STRANDS_DECIDER_REQUIRE_SHIM=1 but no llama.cpp build was given")
    for module in ("torch", "transformers", "safetensors", "sentencepiece", "PIL"):
        if REQUIRED:
            importlib.import_module(module)
        else:
            pytest.importorskip(module)
    # The check puts the checkout's src/, llama.cpp's gguf-py and its own directory first on
    # sys.path; a copy is put back afterwards, so that later tests read the installed distribution.
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("shim_check", SCRIPT)
    shim_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim_check)
    report = shim_check.run(Path(SRC), Path(LIB), tmp_path)
    print(json.dumps(report, indent=1, default=str))
    assert report["failures"] == []
