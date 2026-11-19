"""The llama.cpp shim, compiled and run, held to transformers (deploy/lambda/decision/scripts/shim_check.py).

It needs a llama.cpp tree built by deploy/lambda/decision/scripts/build_native.sh, and torch,
transformers, safetensors, sentencepiece and Pillow; no GPU, no download. Without them it skips,
unless STRANDS_DECIDER_REQUIRE_SHIM=1 (CI) makes that a failure:

    sh deploy/lambda/decision/scripts/build_native.sh /tmp/llama.cpp /tmp/llama-build
    STRANDS_DECIDER_LLAMA_CPP=/tmp/llama.cpp STRANDS_DECIDER_LLAMA_LIB=/tmp/llama-build/libdecision.so \\
        python -m pytest tests/test_llama_cpp_shim.py
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "deploy" / "lambda" / "decision" / "scripts"
SRC = os.environ.get("STRANDS_DECIDER_LLAMA_CPP")
LIB = os.environ.get("STRANDS_DECIDER_LLAMA_LIB")
REQUIRED = os.environ.get("STRANDS_DECIDER_REQUIRE_SHIM") == "1"


def test_the_shim_is_the_reference(tmp_path, monkeypatch):
    if not (SRC and LIB):
        if REQUIRED:
            pytest.fail("STRANDS_DECIDER_REQUIRE_SHIM=1 but no llama.cpp build was given")
        pytest.skip("set STRANDS_DECIDER_LLAMA_CPP and _LIB to a build")
    for module in ("torch", "transformers", "safetensors", "sentencepiece", "PIL"):
        if REQUIRED:
            importlib.import_module(module)
        else:
            pytest.importorskip(module)
    monkeypatch.syspath_prepend(str(SCRIPTS))
    monkeypatch.syspath_prepend(str(Path(SRC) / "gguf-py"))
    report = importlib.import_module("shim_check").run(Path(SRC), Path(LIB), tmp_path)
    print(json.dumps(report, indent=1, default=str))
    assert report["failures"] == []
