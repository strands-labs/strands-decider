"""The llama.cpp shim, compiled and run, against transformers (deploy/lambda/decision/scripts/shim_check.py).

The other tests of the backend replace the shim with a stand-in; this one runs the real library
on a tiny random Gemma 4 decision checkpoint, converted by the stock converter, and holds it to a
transformers reference: the prefix_bidir mask (and not a causal state, a causal question, a
sliding window, or causal full-attention layers, which a build without patches/0001 gives), state reuse bit for bit, image rows at their spans, the vision-only mmproj, and
the answers through `strands_decider.core.evaluate`.

It needs a llama.cpp tree built by deploy/lambda/decision/scripts/build_native.sh, and torch,
transformers, safetensors, sentencepiece and Pillow; no GPU, no download. Without them it skips:

    sh deploy/lambda/decision/scripts/build_native.sh /tmp/llama.cpp /tmp/llama-build
    STRANDS_DECIDER_LLAMA_CPP=/tmp/llama.cpp STRANDS_DECIDER_LLAMA_LIB=/tmp/llama-build/libdecision.so \\
        python -m pytest tests/test_llama_cpp_shim.py
"""

from __future__ import annotations

import importlib.util
import json
import os
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

pytestmark = pytest.mark.skipif(
    not (SRC and LIB),
    reason="set STRANDS_DECIDER_LLAMA_CPP and STRANDS_DECIDER_LLAMA_LIB to a build",
)


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    for module in ("torch", "transformers", "safetensors", "sentencepiece", "PIL"):
        pytest.importorskip(module)
    spec = importlib.util.spec_from_file_location("shim_check", SCRIPT)
    shim_check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(shim_check)
    res = shim_check.run(Path(SRC), Path(LIB), tmp_path_factory.mktemp("shim"))
    print(
        json.dumps({k: v for k, v in res.items() if k != "golden_answers"}, indent=1, default=str)
    )
    return res


def test_the_conversion(result):
    assert not [f for f in result["failures"] if "mmproj" in f or "lazy_ranges" in f]
    full, vision = result["convert"]["mmproj_tensors"]
    assert vision < full


def test_the_mask_is_prefix_bidir(result):
    mask = result["mask"]  # per question, the shim's distance from each mask's reference
    assert max(mask["prefix_bidir"]) < 0.05
    for other in ("causal_state", "causal_question", "causal", "window_8", "swa_only"):
        for far, near in zip(mask[other], mask["prefix_bidir"], strict=True):
            assert far > 5 * near, other


def test_state_reuse_is_exact(result):
    assert result["reuse_max_abs"] == 0.0


def test_image_rows_go_to_their_spans(result):
    images = result["images"]
    assert images["tokens"] == [266, 253] and images["rel_error"] < 0.05
    assert images["swap_rel_change"] > 5 * images["rel_error"]


def test_the_golden(result):
    for name, g in result["golden"].items():
        ref, shim = g["reference"], g["shim"]
        assert ref.index(max(ref)) == shim.index(max(shim)), name
        assert max(abs(a - b) for a, b in zip(ref, shim, strict=True)) <= 0.01, name


def test_nothing_failed(result):
    assert result["failures"] == []
