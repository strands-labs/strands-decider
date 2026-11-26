"""The llama.cpp shim, built, held to transformers on a tiny random Gemma 4 model checkpoint.

    python shim_check.py --llama-cpp SRC --lib LIB

SRC is the patched llama.cpp source (its convert_hf_to_gguf.py and gguf-py), LIB the shim
(`libdecision.so`) built against it. It needs torch, transformers, safetensors, sentencepiece and
Pillow, and no GPU or download, and takes about a minute. The Lambda image build runs it on the
image's own library; tests/test_llama_cpp_shim.py runs it too.

The checkpoint (tiny_checkpoint.py) is converted as the image build converts it, and the
conversion checked (conversion_check.py). Then LlamaCppBackend runs it, against transformers in
float32 on the same prompts:

  mask      the shim matches the prefix_bidir reference, and for each question is several times
            further from the reference under each other mask (tiny_checkpoint.MASKS);
  reuse     several questions in one call give, bit for bit, what each gives alone, in any order;
  images    with two images in the state, the shim matches the reference, which places each
            image's vision-tower rows at its span; swapping the images changes the rows;
  golden    `strands_decider.core.evaluate` on the shim gives the reference's top answer to each
            question, with probabilities within 0.01.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from conversion_check import check_conversion, convert
from tiny_checkpoint import MASKS, SEED, VOCAB, Reference, make_checkpoint

from strands_decider.backends.llama_cpp import LlamaCppBackend, image_pixels
from strands_decider.core import (
    DecisionConfig,
    Request,
    evaluate,
    load_readout,
    load_tokenizer,
    read,
    render,
)

# A relative error of the shim's hidden state against the prefix_bidir reference within which
# the two agree (the KV cache is f16: 0.007 to 0.026 here), and the factor by which, for each
# question, every other mask's reference must be further away (18 to 41 times for swa_only, the
# nearest; a build without patches/0001 is 27 to 35 times nearer swa_only than prefix_bidir).
_AGREE = 0.05
_MISS_FACTOR = 5.0
_GOLDEN_DP = 0.01  # a variants build's modules get 0.015: docs/deployment/graviton-cpu-variants.md#the-image-builds-tiny-model-check
_RED, _BLUE = ((220, 30, 30), (64, 48)), ((30, 30, 220), (48, 96))  # (colour, size)
_GOLDEN = {
    "state": {"ticket": 4182, "text": "The checkout page fails after the payment step."},
    "questions": {
        "urgent": {"type": "noul", "instructions": "Is it urgent?"},
        "team": {"type": "choice", "instructions": "Which team?",
                 "criteria": {"billing": "payments", "web": "pages", "mobile": None}},
        "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]},
    },
}  # fmt: skip


def _rel(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def _png(colour, size) -> str:
    """A base64 PNG: a rectangle of `colour` on white."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).rectangle(
        (size[0] // 4, size[1] // 4, 3 * size[0] // 4, 3 * size[1] // 4), fill=colour
    )
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _check_mask(ref, state, questions, rows, failures) -> dict:
    """The shim's rows against each mask's reference, per question."""
    errs = {
        mask: [_rel(h, ref.hidden(state, q, mask)) for q, h in zip(questions, rows, strict=True)]
        for mask in MASKS
    }
    bidir = errs["prefix_bidir"]
    if max(bidir) > _AGREE:
        failures.append(f"the shim misses the prefix_bidir reference by {max(bidir):.4f}")
    for mask in MASKS[1:]:
        for i, (other, err) in enumerate(zip(errs[mask], bidir, strict=True)):
            if other < _MISS_FACTOR * max(err, 1e-6):
                failures.append(
                    f"question {i}: the shim is {other:.4f} from the {mask} reference and "
                    f"{err:.4f} from prefix_bidir's (under {_MISS_FACTOR:g} times further)"
                )
    return {k: [round(x, 4) for x in v] for k, v in errs.items()}


def _check_reuse(backend, state, questions, rows, failures) -> float:
    """The largest difference between the questions together and each alone, or reversed."""
    alone = np.stack([backend.forward(state, [q])[0] for q in questions])
    reverse = backend.forward(state, questions[::-1])[::-1]
    worst = float(max(np.abs(rows - alone).max(), np.abs(rows - reverse).max()))
    if worst != 0.0:
        failures.append(f"questions in one call differ from questions alone by {worst}")
    return worst


def _check_images(backend, ref, tok, failures) -> dict:
    """Two images in the state: the shim's row against the reference's, and against its own with
    the images swapped."""
    request = {"state": "two pictures", "images": [_png(*_RED), _png(*_BLUE)],
               "questions": {"which": {"type": "choice", "instructions": "which?",
                                       "criteria": {"red": None, "blue": None}}}}  # fmt: skip
    rendered = render(Request.model_validate(request), tok)
    state, q = list(rendered.state_ids), list(next(iter(rendered.questions.values())).question_ids)
    shim = backend.forward(state, [q], rendered.images)[0]
    want = ref.hidden(state, q, pixels=[image_pixels(im) for im in rendered.images])
    swapped = render(Request.model_validate({**request, "images": request["images"][::-1]}), tok)
    rel = _rel(shim, want)
    change = _rel(backend.forward(list(swapped.state_ids), [q], swapped.images)[0], shim)
    if rel > _AGREE:
        failures.append(f"with images the shim misses the reference by {rel:.4f}")
    if change < _MISS_FACTOR * max(rel, 1e-6):
        failures.append("swapping the two images barely changes the question's row")
    spans = [im.span for im in rendered.images]
    return {"spans": spans, "rel_error": rel, "swap_rel_change": change}


def _probabilities(answer: dict) -> dict[str, float]:
    return answer.get("probabilities") or {"false": 1 - answer["noul"], "true": answer["noul"]}


def _check_golden(backend, ref, ckpt, tok, failures, golden_dp: float = _GOLDEN_DP) -> dict:
    """`evaluate`'s answers on the shim against the core's reading of the reference's rows."""
    config, weight = DecisionConfig.load(ckpt), load_readout(ckpt)
    request = Request.model_validate({**_GOLDEN, "images": [_png(*_RED)]})
    got = evaluate(request, tok, weight, config, backend)["answers"]
    rendered = render(request, tok)
    pixels = [image_pixels(im) for im in rendered.images]
    rows = [
        ref.hidden(rendered.state_ids, q.question_ids, pixels=pixels)
        for q in rendered.questions.values()
    ]
    want = read(rows, config, rendered, weight=weight)
    for name in want:
        g, w = _probabilities(got[name]), _probabilities(want[name])
        if max(g, key=g.__getitem__) != max(w, key=w.__getitem__):
            failures.append(f"golden {name!r}: the top answer is not the reference's, {w}")
        if max(abs(g[k] - w[k]) for k in w) > golden_dp:
            failures.append(f"golden {name!r}: probabilities {g}, the reference's {w}")
    return {name: {"reference": want[name], "shim": got[name]} for name in want}


def run(llama_cpp: Path, lib: Path, work: Path, golden_dp: float = _GOLDEN_DP) -> dict:
    """Every check's measurements, and `failures`, empty when the shim passes. llama.cpp's
    gguf-py must be importable."""
    started = time.time()
    failures: list[str] = []
    ckpt = make_checkpoint(work / "checkpoint")
    gguf, full, mmproj = convert(llama_cpp, ckpt, work / "gguf")
    report: dict = {"convert": check_conversion(llama_cpp, ckpt, gguf, full, mmproj, failures)}
    backend = LlamaCppBackend(lib, gguf, mmproj=mmproj, threads=4)
    try:
        ref, tok = Reference(ckpt), load_tokenizer(ckpt)
        # A 40-token state and three 12- to 20-token questions, all longer than the window.
        rng = np.random.default_rng(SEED)
        state = [2, *rng.integers(1000, VOCAB - 4000, 39).tolist()]
        questions = [rng.integers(1000, VOCAB - 4000, n).tolist() for n in (12, 16, 20)]
        rows = backend.forward(state, questions)
        report["mask"] = _check_mask(ref, state, questions, rows, failures)
        report["reuse_max_abs"] = _check_reuse(backend, state, questions, rows, failures)
        report["images"] = _check_images(backend, ref, tok, failures)
        report["golden"] = _check_golden(backend, ref, ckpt, tok, failures, golden_dp)
    finally:
        backend.close()
    report["seconds"] = round(time.time() - started, 1)
    report["failures"] = failures
    return report


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--llama-cpp", type=Path, required=True)
    ap.add_argument("--lib", type=Path, required=True)
    ap.add_argument(
        "--golden-dp", type=float, default=_GOLDEN_DP, help="each golden probability's tolerance"
    )
    a = ap.parse_args()
    sys.path.insert(0, str(a.llama_cpp / "gguf-py"))
    with tempfile.TemporaryDirectory() as work:
        report = run(a.llama_cpp, a.lib, Path(work), a.golden_dp)
    print(json.dumps(report, indent=1, default=str))
    if report["failures"]:
        raise SystemExit("shim check FAILED:\n  " + "\n  ".join(report["failures"]))
    print("shim check OK")


if __name__ == "__main__":
    main()
