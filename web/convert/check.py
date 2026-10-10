"""Check a converted release against the official PyTorch implementation before it is published,
and write the reference the browser tests compare against.

    python web/convert/check.py dist/weights                  # full check (minutes on CPU)
    python web/convert/check.py dist/weights --prompts-only   # refresh the reference (seconds)

The full check runs every question in web/test/questions.json through `strands_decider` (the
reference) and through the converted ONNX graph on ONNX Runtime's CPU provider with the pointer
head in numpy (the browser's math). It fails when they pick different answers where the
reference is not a near-tie, or when the mean probability error is too large for int4
quantization alone.

Both modes write `<weights>/reference.json`: token ids and option positions from the engine's
tokenization, the reference's probabilities, and the answers the engine builds from them.
--prompts-only rebuilds the ids and answers from the strands_decider checked out now, with only
the tokenizer, and keeps the probabilities of the existing reference.json; the CI test job runs
it, so a change to prompting.py or the engine's tokenization is held against the JavaScript port
even when the published conversion is reused. Questions added since have no probabilities until
the next full check; their ids are still compared.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from pydantic import TypeAdapter
from transformers import AutoTokenizer

from strands_decider.infer import EngineConfig, SystemOneEngine, _to_answer, load_engine
from strands_decider.modeling import StrandsDeciderConfig, checkpoint_dir
from strands_decider.prompting import RenderedQuestion, render_question, render_state
from strands_decider.schema import Question

QUESTIONS = os.path.join(os.path.dirname(__file__), "..", "test", "questions.json")
NEAR_TIE = 0.05  # top-2 margin in the reference under which int4 may flip the answer
MAX_MEAN_ERROR = 0.05  # mean over questions of the max abs probability error; v21: 0.026


def probabilities(answer: dict) -> list[float]:
    if answer["type"] == "noul":  # slots are (false, true)
        return [1 - answer["noul"], answer["noul"]]
    return list(answer["probabilities"].values())


class Prompts:
    """The engine's tokenization (`_fit`, `_option_idx`) and answer building (`_to_answer`)
    without its model: an engine shell holding only the tokenizer and the decider config."""

    def __init__(self, weights: str):
        model_dir = os.path.join(weights, "model")
        self.config = StrandsDeciderConfig.from_json(os.path.join(model_dir, "decider_config.json"))
        self.engine = object.__new__(SystemOneEngine)
        self.engine.tok = AutoTokenizer.from_pretrained(model_dir)
        self.engine.cfg = EngineConfig(device="cpu")
        self.engine.device = "cpu"
        self.engine.model = SimpleNamespace(config=self.config)

    def build(
        self, item: dict
    ) -> tuple[list[int], dict[str, tuple[list[int], list[int], RenderedQuestion]]]:
        """State ids, and per question: its ids, option positions and rendering."""
        parse = TypeAdapter(Question).validate_python
        rendered = {k: render_question(parse(q)) for k, q in item["questions"].items()}
        state_ids, question_ids = self.engine._fit(  # noqa: SLF001
            render_state(item["state"]), [rq.text for rq in rendered.values()]
        )
        opt_idx = self.engine._option_idx(list(rendered.values()), len(state_ids)).tolist()  # noqa: SLF001
        return state_ids, {
            k: (question_ids[i], [j for j in opt_idx[i] if j >= 0], rq)
            for i, (k, rq) in enumerate(rendered.items())
        }

    def answer(self, rq: RenderedQuestion, probs: list[float]) -> dict:
        return _to_answer(rq, probs, ordinal_smoothing=self.config.ordinal_smoothing).model_dump()


class Converted:
    """The converted graph and head, run the way web/site/js/worker.js runs them."""

    def __init__(self, weights: str, manifest: dict):
        import onnxruntime as ort
        from safetensors.numpy import load_file

        self.manifest = manifest
        path = os.path.join(weights, manifest["graph"])
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.head = load_file(os.path.join(weights, "model", "head.safetensors"))
        with open(os.path.join(weights, "model", "decider_config.json")) as fh:
            self.cfg = json.load(fh)

    def probs(self, ids: list[int], opt_idx: list[int], kind: str) -> np.ndarray:
        types = {"tensor(float16)": np.float16, "tensor(float)": np.float32}
        L, feeds = len(ids), {}
        for i in self.session.get_inputs():
            if i.name == "input_ids":
                feeds[i.name] = np.array([ids], np.int64)
            elif i.name == "attention_mask":
                feeds[i.name] = np.ones((1, L), np.int64)
            elif i.name == "position_ids":
                feeds[i.name] = np.tile(np.arange(L, dtype=np.int64), (3, 1, 1))
            else:
                feeds[i.name] = np.zeros(
                    self.manifest["state_inputs"][i.name]["shape"], types[i.type]
                )
        hidden = self.session.run(["hidden_states"], feeds)[0][0].astype(np.float32)
        H = self.head

        def project(name: str, x: np.ndarray) -> np.ndarray:
            m = x.mean(-1, keepdims=True)
            v = ((x - m) ** 2).mean(-1, keepdims=True)
            xn = (x - m) / np.sqrt(v + 1e-5) * H["norm.weight"] + H["norm.bias"]
            return xn @ H[f"{name}.weight"].T + H[f"{name}.bias"]

        q, k = project("q", hidden[-1]), project("k", hidden[opt_idx])
        logits = (k @ q) / np.sqrt(q.shape[-1])
        logits /= self.cfg["temperature_by_kind"].get(kind, self.cfg["temperature"])
        e = np.exp(logits - logits.max())
        return e / e.sum()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("weights", help="convert.py output directory")
    ap.add_argument("--prompts-only", action="store_true", help="refresh reference.json only")
    a = ap.parse_args()

    with open(os.path.join(a.weights, "manifest.json")) as fh:
        manifest = json.load(fh)
    with open(QUESTIONS) as fh:
        sets = json.load(fh)
    prompts, ref_path = Prompts(a.weights), os.path.join(a.weights, "reference.json")
    if a.prompts_only:
        with open(ref_path) as fh:
            old = {
                (r["set"], r["name"], k): q["probabilities"]
                for r in json.load(fh)["results"]
                for k, q in r["questions"].items()
            }
    else:
        src = manifest["source"]
        engine = load_engine(checkpoint_dir(src["repo"], src["revision"]), device="cpu")
        conv = Converted(a.weights, manifest)

    rows, reference = [], []
    for set_name, items in sets.items():
        for item in items:
            state_ids, built = prompts.build(item)
            if not a.prompts_only:
                answers = engine.ask(item["state"], item["questions"]).model_dump()["answers"]
            result = {
                "set": set_name,
                "name": item["name"],
                "state_ids": state_ids,
                "questions": {},
            }
            for key, (ids, opts, rq) in built.items():
                if a.prompts_only:
                    probs = old.get((set_name, item["name"], key))
                else:
                    want = np.array(probabilities(answers[key]))
                    probs = want.tolist()
                    t = time.time()
                    got = conv.probs(state_ids + ids, opts, rq.kind)
                    ms = (time.time() - t) * 1000
                    top2 = np.sort(want)[-2:]
                    name = f"{set_name}/{item['name']}" + (f"/{key}" if len(built) > 1 else "")
                    r = {
                        "name": name,
                        "error": float(np.abs(got - want).max()),
                        "same_top": bool(got.argmax() == want.argmax()),
                        "near_tie": bool(top2[1] - top2[0] < NEAR_TIE),
                    }
                    rows.append(r)
                    print(f"{name:40s} {len(state_ids) + len(ids):5d} tok  err {r['error']:.3f}  "
                          f"{'same' if r['same_top'] else 'DIFFERENT'} top{' (near tie)' if r['near_tie'] else ''}"
                          f"  {ms:.0f} ms", flush=True)  # fmt: skip
                result["questions"][key] = {
                    "ids": ids,
                    "opt_idx": opts,
                    "probabilities": probs,
                    # The engine's own answer in a full check; built from the kept probabilities otherwise.
                    "answer": answers[key]
                    if not a.prompts_only
                    else (prompts.answer(rq, probs) if probs else None),
                }
            reference.append(result)

    with open(ref_path, "w") as fh:
        json.dump({"source": manifest["source"], "results": reference}, fh)
    if a.prompts_only:
        n = sum(len(r["questions"]) for r in reference)
        missing = sum(
            q["probabilities"] is None for r in reference for q in r["questions"].values()
        )
        print(
            f"reference.json: {n} questions, {missing} without probabilities (new since the check)"
        )
        return
    mean = float(np.mean([r["error"] for r in rows]))
    flipped = [r["name"] for r in rows if not r["same_top"] and not r["near_tie"]]
    print(
        f"{len(rows)} questions, mean max-abs error {mean:.4f}, different answers: {flipped or 'none'}"
    )
    if flipped or mean > MAX_MEAN_ERROR:
        sys.exit(f"check failed: mean error {mean:.4f} (limit {MAX_MEAN_ERROR}), flipped {flipped}")


if __name__ == "__main__":
    main()
