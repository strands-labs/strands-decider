"""Image and audio evaluation of a Gemma 4 E2B / E4B Strands Decider (docs/audio.md).

One checkpoint (by default strands-decider-E4B-gemma4-v1-2610, unchanged) is loaded once with
its vision and audio encoders (`VisionDeciderModel.load(..., audio=True)`), and scored three
ways on every item:

  served     the served path: `VisionEngine._image_probs`, the shared-prefix forward the
             server runs, with the checkpoint's per-kind temperatures.
  blind      the same question with the image or clip removed (text only), to measure how
             much the answer depends on the medium.
  untrained  the base model (google/gemma-4-E4B-it) read without the adapter: the frozen
             torso's own option-number logits at <answer> (`frozen_slot_log_probs`), the
             medium in <state>, as the KL reference reads it.

Benchmarks:

  naturalbench      images; 2 images x 2 questions per group (blind models score at chance)
  pope_adversarial  images; object presence on COCO val2014
  esc50_choice      audio; which of 5 sound classes (the clip's and 4 others, seeded)
  esc50_presence    audio; "is the sound of X in this recording?", half true, half false
  speech_choice     audio; which of the 10 Speech Commands words is spoken
  speech_presence   audio; "does the speaker say X?", half true, half false

ESC-50 (CC BY-NC 3.0, evaluation only) is ashraq/esc50 on the Hub, fold 1 (400 clips, 5 s,
44.1 kHz). Speech Commands v0.02 (CC BY 4.0) is its official test-set archive; 40 clips per
word, the first by file name, 1 s, 16 kHz. Results stream to <out>/<system>.jsonl, and
<out>/summary.json is rewritten after each system.

    python evaluation/multimodal/run.py --out results --nb-groups 300 --pope 600 \\
        --esc50 400 --speech 40
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import tarfile
import time
import traceback
from collections.abc import Iterator
from typing import Any

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "vision"))
from metrics import image_dependence, naturalbench_paired, summarise  # noqa: E402


def _vision_run() -> Any:
    """evaluation/vision/run.py (its NaturalBench and POPE loaders), under its own name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("vision_run", os.path.join(HERE, "..", "vision", "run.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


from strands_decider.audio import read_audio  # noqa: E402
from strands_decider.infer import _option_token_index  # noqa: E402
from strands_decider.modeling import masked_log_softmax  # noqa: E402
from strands_decider.prompting import RenderedQuestion, render_question  # noqa: E402
from strands_decider.schema import ChoiceQuestion, NoulQuestion, Question  # noqa: E402
from strands_decider.vision import VisionDeciderModel, VisionEngine, VisionEngineConfig  # noqa: E402

E4B, E4B_REV = "StrandsAgents/strands-decider-E4B-gemma4-v1-2610", "1344d44397c457f3be34b8ff15c593b41248c6bf"
ESC50_REPO, ESC50_REV = "ashraq/esc50", "e3e2a63ffff66b9a9735524551e3818e96af03ee"
ESC50_FILES = ("data/train-00000-of-00002-2f1ab7b824ec751f.parquet",
               "data/train-00001-of-00002-27425e5c1846b494.parquet")
SPEECH_URL = "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz"
SPEECH_WORDS = ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
LETTERS = "ABCDEFGHIJ"


# ---- audio data ------------------------------------------------------------------------


def esc50(n: int, seed: int = 0) -> Iterator[dict[str, Any]]:
    """Fold 1 of ESC-50: a 5-way choice and a presence question per clip."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    df = pd.concat([pd.read_parquet(hf_hub_download(ESC50_REPO, f, repo_type="dataset", revision=ESC50_REV))
                    for f in ESC50_FILES])
    classes = sorted(df["category"].unique())
    df = df[df["fold"] == 1].sort_values("filename").head(n)
    rng = random.Random(seed)
    for k, (_, row) in enumerate(df.iterrows()):
        audio = row["audio"]["bytes"]
        cat = row["category"]
        others = [c for c in classes if c != cat]
        opts = [*rng.sample(others, 4), cat]
        rng.shuffle(opts)
        options = [(LETTERS[i], c.replace("_", " ")) for i, c in enumerate(opts)]
        yield {"id": f"esc50-{row['filename']}-choice", "bench": "esc50_choice", "kind": "choice",
               "question": "Which sound is heard in the recording?", "options": options,
               "gold": opts.index(cat), "audio": audio}
        probe = cat if k % 2 == 0 else rng.choice(others)
        yield {"id": f"esc50-{row['filename']}-presence", "bench": "esc50_presence", "kind": "noul",
               "question": f"Is the sound of {probe.replace('_', ' ')} heard in this recording?",
               "options": [("no", ""), ("yes", "")], "gold": int(probe == cat), "audio": audio}


def speech_commands(per_word: int, cache: str, seed: int = 0) -> Iterator[dict[str, Any]]:
    """The v0.02 test set, the 10 core words: a 10-way choice and a presence question per clip."""
    path = os.path.join(cache, "speech_commands_test_set_v0.02.tar.gz")
    if not os.path.exists(path):
        import urllib.request

        os.makedirs(cache, exist_ok=True)
        urllib.request.urlretrieve(SPEECH_URL, path)
    clips: dict[str, list[tuple[str, bytes]]] = {w: [] for w in SPEECH_WORDS}
    with tarfile.open(path) as tar:
        members = sorted((m for m in tar.getmembers() if m.isfile() and m.name.endswith(".wav")),
                         key=lambda m: m.name)
        for m in members:
            word = m.name.split("/")[-2]
            if word in clips and len(clips[word]) < per_word:
                f = tar.extractfile(m)
                assert f is not None
                clips[word].append((m.name, f.read()))
    rng = random.Random(seed)
    k = 0
    for word in SPEECH_WORDS:
        for name, audio in clips[word]:
            stem = name.split("/")[-1][:-4]
            options = [(LETTERS[i], w) for i, w in enumerate(SPEECH_WORDS)]
            yield {"id": f"speech-{word}-{stem}-choice", "bench": "speech_choice", "kind": "choice",
                   "question": "Which word does the speaker say?", "options": options,
                   "gold": SPEECH_WORDS.index(word), "audio": audio}
            probe = word if k % 2 == 0 else rng.choice([w for w in SPEECH_WORDS if w != word])
            yield {"id": f"speech-{word}-{stem}-presence", "bench": "speech_presence", "kind": "noul",
                   "question": f'Does the speaker say the word "{probe}"?',
                   "options": [("no", ""), ("yes", "")], "gold": int(probe == word), "audio": audio}
            k += 1


# ---- the systems -----------------------------------------------------------------------


class Gemma4Eval:
    """One loaded checkpoint, read three ways (served, blind, untrained)."""

    def __init__(self, checkpoint: str, revision: str | None, device: str) -> None:
        from huggingface_hub import snapshot_download

        if not os.path.isdir(checkpoint):
            checkpoint = snapshot_download(checkpoint, revision=revision)
        self.device = device
        model = VisionDeciderModel.load(checkpoint, audio=True).to(device).eval()
        self.engine = VisionEngine(model, VisionEngineConfig(device=device))
        self.model = model
        cfg = model.config
        self.temps = dict(cfg.temperature_by_kind)
        self.t_default = cfg.temperature

    @staticmethod
    def _question(it: dict[str, Any], explicit_noul: bool) -> Question:
        if it["kind"] == "noul":
            if explicit_noul:  # the untrained reading needs the options spelled out
                return NoulQuestion(instructions=it["question"],
                                    criteria={"false": "the answer is no", "true": "the answer is yes"})
            return NoulQuestion(instructions=it["question"])  # as served: default criteria
        return ChoiceQuestion(instructions=it["question"], criteria=dict(it["options"]))

    @staticmethod
    def _gold_order(rq: RenderedQuestion, it: dict[str, Any]) -> list[int]:
        """Slot order -> the item's option order (noul slots are false, true)."""
        if it["kind"] == "noul":
            return [rq.slot_labels.index("false"), rq.slot_labels.index("true")]
        return [rq.slot_labels.index(n) for n, _ in it["options"]]

    def _media(self, it: dict[str, Any]) -> tuple[list[Any], list[Any]]:
        from PIL import Image

        images = [Image.open(io.BytesIO(it["image"])).convert("RGB")] if "image" in it else []
        clips = [read_audio(it["audio"])] if "audio" in it else []
        return images, clips

    def _whole(self, it: dict[str, Any], rq: RenderedQuestion, blind: bool) -> tuple[Any, ...]:
        images, clips = ([], []) if blind else self._media(it)
        text, mm = (self.engine.media_prompt("", images, clips) if images or clips
                    else (render_question_state(), {}))
        text += rq.text
        enc = self.model.tokenizer(text, return_offsets_mapping=True)
        opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        return ids, torch.tensor([opt], device=self.device), mm

    @torch.no_grad()
    def served(self, it: dict[str, Any]) -> list[float]:
        rq = render_question(self._question(it, False))
        images, clips = self._media(it)
        probs, _ = self.engine._image_probs("", images, [rq], clips)
        p = probs[0].float().cpu()
        return [float(p[s]) for s in self._gold_order(rq, it)]

    @torch.no_grad()
    def blind(self, it: dict[str, Any]) -> list[float]:
        rq = render_question(self._question(it, False))
        ids, opt, mm = self._whole(it, rq, blind=True)
        out = self.model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots], device=self.device), opt_idx=opt,
                         temperature=self.temps.get(it["kind"], self.t_default), **mm)
        p = masked_log_softmax(out["logits"].float().cpu(), torch.tensor([rq.n_slots])).exp()[0]
        return [float(p[s]) for s in self._gold_order(rq, it)]

    @torch.no_grad()
    def untrained(self, it: dict[str, Any], blind: bool = False) -> list[float] | None:
        """None where the reading does not exist: an option past 9 has no single-token number
        ("10"), so the 10-way speech choice cannot be read this way."""
        rq = render_question(self._question(it, True))
        ids, _, mm = self._whole(it, rq, blind=blind)
        lp, eligible = self.model.frozen_slot_log_probs(ids, torch.ones_like(ids),
                                                        torch.tensor([rq.n_slots], device=self.device), **mm)
        if not bool(eligible.all()):
            return None
        return [float(lp[0, s].exp()) for s in self._gold_order(rq, it)]


def render_question_state() -> str:
    from strands_decider.prompting import render_state

    return render_state("")


# ---- driver ------------------------------------------------------------------------------


def run(system: Gemma4Eval, items: list[dict[str, Any]], out: str, readings: list[str]) -> dict[str, Any]:
    fns = {"served": system.served, "blind": system.blind, "untrained": system.untrained,
           "untrained-blind": lambda it: system.untrained(it, blind=True)}
    res: dict[str, list[dict[str, Any]]] = {}
    for tag in readings:
        path = os.path.join(out, f"{tag}.jsonl")
        done: dict[str, dict[str, Any]] = {}
        if os.path.exists(path):  # resume
            with open(path) as fh:
                done = {json.loads(line)["id"]: json.loads(line) for line in fh if line.strip()}
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with open(path, "a") as fh:
            for n, it in enumerate(todo, 1):
                r = {k: it[k] for k in ("id", "bench", "kind", "gold", "group", "q", "i") if k in it}
                r["probs"] = fns[tag](it)
                fh.write(json.dumps(r) + "\n")
                done[it["id"]] = r
                if n % 100 == 0 or n == len(todo):
                    rate = n / max(1e-6, time.time() - t0)
                    print(f"[mm-eval] {tag}: {n}/{len(todo)} {rate:.2f} it/s "
                          f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
        res[tag] = [done[it["id"]] for it in items if it["id"] in done and done[it["id"]]["probs"] is not None]
    return res


def score(all_res: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for tag, rs in all_res.items():
        by: dict[str, list[dict[str, Any]]] = {}
        for r in rs:
            by.setdefault(r["bench"], []).append(r)
        out[tag] = {b: summarise(v) for b, v in by.items()}
        if by.get("naturalbench"):
            out[tag]["naturalbench"]["paired"] = naturalbench_paired(by["naturalbench"])
    for tag, blind in (("served", "blind"), ("untrained", "untrained-blind")):
        if tag in all_res and blind in all_res:
            per: dict[str, Any] = {}
            for b in {r["bench"] for r in all_res[tag]}:
                per[b] = image_dependence([r for r in all_res[tag] if r["bench"] == b],
                                          [r for r in all_res[blind] if r["bench"] == b])
            out[tag]["medium_dependence"] = per
    return out


@torch.no_grad()
def latency(system: Gemma4Eval, items: list[dict[str, Any]], reps: int = 20) -> dict[str, Any]:
    """Wall time of a served request (decode + encoders + decoder), one and three questions."""
    from strands_decider.schema import SystemOneRequest

    def b64(x: bytes) -> str:
        return base64.b64encode(x).decode()

    image = next(it for it in items if "image" in it)["image"] if any("image" in it for it in items) else None
    clip = next(it for it in items if "audio" in it)["audio"] if any("audio" in it for it in items) else None
    qs = {"a": NoulQuestion(instructions="Is the customer angry?"),
          "b": ChoiceQuestion(instructions="What should happen next?",
                              criteria={"refund": "issue a refund", "escalate": "escalate", "close": "close"}),
          "c": NoulQuestion(instructions="Is there anything unsafe?")}
    shapes = {"text": {}, "image": {"images": [b64(image)]} if image else None,
              "audio": {"audio": [b64(clip)]} if clip else None,
              "image+audio": {"images": [b64(image)], "audio": [b64(clip)]} if image and clip else None}
    out: dict[str, Any] = {}
    for name, extra in shapes.items():
        if extra is None:
            continue
        for nq in (1, 3):
            req = SystemOneRequest(state="A support ticket about a late delivery.",
                                   questions=dict(list(qs.items())[:nq]), **extra)
            for _ in range(3):
                system.engine.evaluate(req)
            ts = []
            for _ in range(reps):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t = time.perf_counter()
                r = system.engine.evaluate(req)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                ts.append(time.perf_counter() - t)
            ts.sort()
            out[f"{name}-{nq}q"] = {"p50_ms": round(1000 * ts[len(ts) // 2], 1),
                                    "p90_ms": round(1000 * ts[int(0.9 * (len(ts) - 1))], 1),
                                    "input_tokens": r.usage.input_tokens}
    return out


def _versions(device: str, checkpoint: str, revision: str | None) -> dict[str, Any]:
    import platform

    import transformers

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "gpu": gpu, "device": device,
            "revisions": {checkpoint: revision, ESC50_REPO: ESC50_REV}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=E4B)
    ap.add_argument("--revision", default=None, help=f"default {E4B_REV} for {E4B}")
    ap.add_argument("--readings", default="served,blind,untrained,untrained-blind")
    ap.add_argument("--nb-groups", type=int, default=300)
    ap.add_argument("--pope", type=int, default=600)
    ap.add_argument("--esc50", type=int, default=400)
    ap.add_argument("--speech", type=int, default=40, help="clips per word")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/strands-decider-eval"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--latency-reps", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    revision = a.revision or (E4B_REV if a.checkpoint == E4B else None)
    items: list[dict[str, Any]] = []
    vision = _vision_run()
    items += list(vision.naturalbench(a.nb_groups, None)) if a.nb_groups else []
    items += list(vision.pope(a.pope, None)) if a.pope else []
    items += list(esc50(a.esc50)) if a.esc50 else []
    items += list(speech_commands(a.speech, a.cache)) if a.speech else []
    counts: dict[str, int] = {}
    for it in items:
        counts[it["bench"]] = counts.get(it["bench"], 0) + 1
    print(f"[mm-eval] {len(items)} items {counts}", flush=True)
    t0 = time.time()
    system = Gemma4Eval(a.checkpoint, revision, a.device)
    load_s = time.time() - t0
    errors: dict[str, str] = {}
    all_res: dict[str, Any] = {}
    lat: dict[str, Any] = {}
    try:
        all_res = run(system, items, a.out, a.readings.split(","))
        lat = latency(system, items, a.latency_reps) if a.latency_reps else {}
    except Exception:
        errors["run"] = traceback.format_exc()
        print(f"[mm-eval] FAILED\n{errors['run']}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump({"args": vars(a), "versions": _versions(a.device, a.checkpoint, revision), "items": counts,
                   "load_s": round(load_s, 1), "peak_gpu_gib": peak, "latency": lat, "errors": errors,
                   "scores": score(all_res)}, fh, indent=2)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
