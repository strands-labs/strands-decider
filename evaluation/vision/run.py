"""Image evaluation: Strands Decider (v19) over images, against two baselines.

Systems, each also scored "blind" (the image removed) to measure image dependence:

  strands   Strands Decider (v19) loaded with `--vision` (`VisionDeciderModel.load`):
            its adapter and head unchanged, the vision tower kept. Temperatures per
            question kind applied as the server applies them.
  qwen      Qwen3.5-2B-Base untrained, read the way the KL reference reads it: the
            frozen torso's option-number logits at <answer>, image in <state>.
  mapika    Mapika/decider-2b-vision through its own code (letter logits at an answer
            slot, its own prompt, images <= 768 px), for comparison.

Benchmarks: NaturalBench (2 images x 2 questions per group, built so that blind models
score at chance), POPE adversarial (object presence on COCO val2014), and the Image
JevBench preview items, optional, rebuilt from their source datasets by a separate
builder (README.md; only the 60 "exact" items are faithful, and the official Image
JevBench set is not downloadable).

`--checkpoint` may be any Strands Decider checkpoint that `serve --vision` loads: v19, or an
image-trained one; images are processed by its own pinned PIL processor.

Every item is one question about one image, so each is forwarded whole (state + question);
`VisionEngine`'s shared-prefix path gives the same probabilities to < 1e-5
(tests/test_vision.py). Results stream to <out>/<system>.jsonl; <out>/summary.json is
rewritten after each system.

    python evaluation/vision/run.py --out results --systems strands,qwen,mapika \\
        --nb-groups 300 --pope 600 [--ijb-jsonl ijb_preview.jsonl]

The recorded runs behind docs/vision.md, made with this script, are published as a
release asset; README.md links them and gives the exact command.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from collections.abc import Iterator
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import score_runs

from strands_decider.infer import _option_token_index
from strands_decider.modeling import StrandsDeciderConfig, masked_log_softmax
from strands_decider.prompting import render_question
from strands_decider.schema import ChoiceQuestion, NoulQuestion, Question
from strands_decider.vision import ImagePrompt, VisionDeciderModel, fit_image

# Every download is pinned to the revision the published results were measured on.
V19, V19_REV = "StrandsAgents/strands-decider-2B-hobson-v19", "bb282d786bc251fd4e3068de3ada9ddbb38127cd"
BASE, BASE_REV = "Qwen/Qwen3.5-2B-Base", "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
MAPIKA, MAPIKA_REV = "Mapika/decider-2b-vision", "863e290863655f1d6b69324d77d09ac972d21609"
NB_REPO, NB_FILE, NB_REV = ("BaiqiL/NaturalBench", "data/train-00000-of-00003.parquet",
                            "ba41a7d564877a9b64c094b08015ca493cc3e54b")
POPE_REPO, POPE_FILE, POPE_REV = ("lmms-lab/POPE", "Full/adversarial-00000-of-00001.parquet",
                                  "4db1276663dfa5eb8ad16a52d24c31a09e470896")


# ---- data ------------------------------------------------------------------------------


def _parquet(repo: str, filename: str, revision: str, local: str | None) -> Any:
    import pandas as pd

    if local:
        return pd.read_parquet(local)
    from huggingface_hub import hf_hub_download

    return pd.read_parquet(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))


def _mc(question: str) -> tuple[str, list[tuple[str, str]]]:
    """Options out of a NaturalBench question. Shapes seen:
    'stem?\\nOption: A:To the right; B:To the left;'
    'stem?\\nOption: A: Noticeably curved. B: Slightly curved.'
    'stem? A. Nothing happens. B. Something moves.'
    Markers are taken in sequence (A, then B, ...).
    """
    start = question.find("Option:")
    start = start + len("Option:") if start >= 0 else 0
    marks: list[tuple[str, int, int]] = []
    want = "A"
    for m in re.finditer(r"(?:(?<=\s)|(?<=^)|(?<=;)|(?<=:))([A-H])\s*[:.]\s*", question[start:]):
        if m.group(1) == want:
            marks.append((want, start + m.start(), start + m.end()))
            want = chr(ord(want) + 1)
    if len(marks) < 2:
        raise ValueError(f"cannot parse options from {question!r}")
    stem = question[: question.find("Option:")] if "Option:" in question else question[: marks[0][1]]
    opts = []
    for k, (name, _, end) in enumerate(marks):
        stop = marks[k + 1][1] if k + 1 < len(marks) else len(question)
        opts.append((name, question[end:stop].strip().rstrip(";").strip()))
    return stem.strip(), opts


def naturalbench(n_groups: int, local: str | None, start: int = 0) -> Iterator[dict[str, Any]]:
    # Question k on image j -> column Image_j_Question_k (a fixed pattern in every group).
    # The evaluation set is the first 300 groups; `start=300` gives held-out groups on
    # which to fit image temperatures (temps.py).
    df = _parquet(NB_REPO, NB_FILE, NB_REV, local).iloc[start : start + n_groups]
    for _, row in df.iterrows():
        imgs = [row["Image_0"]["bytes"], row["Image_1"]["bytes"]]
        for k in (0, 1):
            q = row[f"Question_{k}"]
            mc = row["Question_Type"] == "multiple_choice"
            for j in (0, 1):
                gold = str(row[f"Image_{j}_Question_{k}"]).strip()
                if mc:
                    stem, opts = _mc(q)
                    item = {"kind": "choice", "question": stem, "options": opts,
                            "gold": [n for n, _ in opts].index(gold[0])}
                else:
                    item = {"kind": "noul", "question": q.strip(), "options": [("no", ""), ("yes", "")],
                            "gold": 1 if gold.lower().startswith("y") else 0}
                yield {**item, "id": f"nb-{row['Index']}-q{k}-i{j}", "bench": "naturalbench",
                       "group": int(row["Index"]), "q": k, "i": j, "image": imgs[j]}


def pope(n: int, local: str | None) -> Iterator[dict[str, Any]]:
    df = _parquet(POPE_REPO, POPE_FILE, POPE_REV, local).sort_values("question_id").head(n)
    for _, row in df.iterrows():
        yield {"id": f"pope-{row['question_id']}", "bench": "pope_adversarial", "kind": "noul",
               "question": row["question"].strip(), "options": [("no", ""), ("yes", "")],
               "gold": 1 if row["answer"].strip().lower() == "yes" else 0, "image": row["image"]["bytes"]}


def image_jevbench(jsonl: str) -> Iterator[dict[str, Any]]:
    """Image JevBench preview items rebuilt from their source datasets (60 of 128 exactly);
    the builder lives with the research notes (see README.md)."""
    root = os.path.dirname(os.path.abspath(jsonl))
    with open(jsonl, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            path = r["image_path"]
            if not os.path.exists(path):
                path = os.path.join(root, "images", os.path.basename(path))
            with open(path, "rb") as g:
                image = g.read()
            yield {"id": r["id"], "bench": r["bench"], "kind": r["kind"], "question": r["question"],
                   "options": [tuple(o) for o in r["options"]], "gold": r["gold"], "image": image,
                   "dataset": r["dataset"], "exact": r["exact"],
                   "official_mapika_ok": r.get("official_mapika_ok")}


def _pil(b: bytes) -> Any:
    from PIL import Image

    return Image.open(io.BytesIO(b)).convert("RGB")


# ---- the Strands systems -----------------------------------------------------------------


class _DeciderSystem:
    """Shared prompt building: image(s) in <state>, one rendered question, whole forward."""

    model: VisionDeciderModel
    long_side = 448
    max_pixels = 0
    device = "cpu"

    def __init__(self, prompter: ImagePrompt) -> None:
        self.prompter = prompter  # the model's own, its processor pinned as VisionEngine pins it

    def _question(self, it: dict[str, Any]) -> Question:
        raise NotImplementedError

    def _inputs(self, it: dict[str, Any], blind: bool) -> tuple[Any, ...]:
        q = self._question(it)
        rq = render_question(q)
        mm: dict[str, torch.Tensor] = {}
        counts: list[int] = []
        n_img = 0 if blind else 1
        if n_img:
            counts, mm = self.prompter.process([fit_image(_pil(it["image"]), self.long_side, self.max_pixels)])
        text = self.prompter.state("", counts) + rq.text
        enc = self.model.tokenizer(text, return_offsets_mapping=True)
        opt = _option_token_index(enc["offset_mapping"], rq.option_spans, len(text) - len(rq.text))
        ids = torch.tensor([enc["input_ids"]], device=self.device)
        mm = {k: v.to(self.device) for k, v in mm.items()}
        return rq, ids, torch.tensor([opt], device=self.device), mm

    def _gold_slot(self, rq: Any, it: dict[str, Any]) -> list[int]:
        """Slot order -> the item's option order (noul slots are false, true)."""
        if it["kind"] == "noul":
            return [rq.slot_labels.index("false"), rq.slot_labels.index("true")]
        return [rq.slot_labels.index(n) for n, _ in it["options"]]


class Strands(_DeciderSystem):
    name = "strands-v19"

    def __init__(self, checkpoint: str = V19, device: str = "cpu",
                 image_temps: dict[str, float] | None = None) -> None:
        from huggingface_hub import snapshot_download

        if checkpoint == V19:
            checkpoint = snapshot_download(V19, revision=V19_REV)
        self.device = device
        self.model = VisionDeciderModel.load(checkpoint).to(torch.float32).to(device).eval()
        super().__init__(self.model.image_prompt())
        cfg = self.model.config
        # Questions over an image take the image temperatures (the checkpoint's, or
        # `image_temps`), as VisionEngine applies them; the image-removed pass is a
        # text-only request, answered with the text ones.
        self.text_temps = dict(cfg.temperature_by_kind)
        self.temps = {**cfg.temperature_by_kind, **(image_temps or cfg.image_temperature_by_kind)}
        self.t_default = cfg.temperature

    def _question(self, it: dict[str, Any]) -> Question:
        if it["kind"] == "noul":  # as served: default criteria
            return NoulQuestion(instructions=it["question"])
        return ChoiceQuestion(instructions=it["question"], criteria=dict(it["options"]))

    @torch.no_grad()
    def probs(self, it: dict[str, Any], blind: bool) -> list[float]:
        rq, ids, opt, mm = self._inputs(it, blind)
        out = self.model(ids, torch.ones_like(ids), torch.tensor([rq.n_slots], device=self.device), opt_idx=opt,
                         temperature=(self.text_temps if blind else self.temps).get(it["kind"], self.t_default),
                         **mm)
        p = masked_log_softmax(out["logits"].float().cpu(), torch.tensor([rq.n_slots])).exp()[0]
        return [float(p[s]) for s in self._gold_slot(rq, it)]


class QwenUntrained(_DeciderSystem):
    """An untrained multimodal base (Qwen3.5-2B-Base by default), its own option-number readout."""

    name = "qwen-untrained"

    def __init__(self, base: str = BASE, dtype: str = "float32", device: str = "cpu") -> None:
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer

        if base == BASE:
            base = snapshot_download(BASE, revision=BASE_REV)

        cfg = StrandsDeciderConfig(base_model=base, use_lora=False, head_type="pointer",
                                   torch_dtype=dtype, max_length=4096)
        tok = AutoTokenizer.from_pretrained(base)
        self.device = device
        self.model = VisionDeciderModel(cfg, VisionDeciderModel._load_torso(cfg, None, None), tok).to(device).eval()
        super().__init__(self.model.image_prompt())

    def _question(self, it: dict[str, Any]) -> Question:
        if it["kind"] == "noul":
            return NoulQuestion(instructions=it["question"],
                                criteria={"false": "the answer is no", "true": "the answer is yes"})
        return ChoiceQuestion(instructions=it["question"], criteria=dict(it["options"]))

    @torch.no_grad()
    def probs(self, it: dict[str, Any], blind: bool) -> list[float]:
        rq, ids, _, mm = self._inputs(it, blind)
        lp, _ = self.model.frozen_slot_log_probs(ids, torch.ones_like(ids),
                                                 torch.tensor([rq.n_slots], device=self.device), **mm)
        return [float(lp[0, s].exp()) for s in self._gold_slot(rq, it)]


class Mapika:
    """Mapika/decider-2b-vision through its own code (one image, <= 10 options)."""

    name = "mapika"

    def __init__(self, dtype: str = "bfloat16", device: str = "cpu") -> None:
        from huggingface_hub import snapshot_download

        repo = snapshot_download(MAPIKA, revision=MAPIKA_REV)
        sys.path.insert(0, repo)
        from decider.infer import Example, Q
        from decider.vision import VisionDecisionModel

        self.example, self.q = Example, Q
        self.m = VisionDecisionModel(repo, dtype=getattr(torch, dtype), grad_ckpt=False).to(device).eval()

    @torch.no_grad()
    def probs(self, it: dict[str, Any], blind: bool) -> list[float]:
        img = None
        if not blind:
            img = _pil(it["image"])
            img.thumbnail((768, 768))  # its training images were <= 768 px
        opts = ["no", "yes"] if it["kind"] == "noul" else [f"{n}: {d}" for n, d in it["options"]]
        inp = self.m.prepare([(img, self.example("This is a visual question about the image.",
                                                  [self.q(it["question"], opts, 0)]))])
        lg = self.m.slot_logits(inp).float()[0]
        return torch.softmax(lg, -1)[: len(it["options"])].tolist()


# ---- driver ------------------------------------------------------------------------------


def run_system(system: Any, items: list[dict[str, Any]], out: str,
               blind_modes: tuple[bool, ...] = (False, True)) -> dict[str, list[dict[str, Any]]]:
    res: dict[str, list[dict[str, Any]]] = {}
    for blind in blind_modes:
        tag = f"{system.name}{'-blind' if blind else ''}"
        path = os.path.join(out, f"{tag}.jsonl")
        done: dict[str, dict[str, Any]] = {}
        if os.path.exists(path):  # resume
            with open(path) as fh:
                done = {json.loads(line)["id"]: json.loads(line) for line in fh if line.strip()}
        todo = [it for it in items if it["id"] not in done]
        t0 = time.time()
        with open(path, "a") as fh:
            for n, it in enumerate(todo, 1):
                r = {k: it[k] for k in ("id", "bench", "kind", "gold", "group", "q", "i", "dataset",
                                        "exact", "official_mapika_ok") if k in it}
                r["probs"] = system.probs(it, blind)
                fh.write(json.dumps(r) + "\n")
                done[it["id"]] = r
                if n % 50 == 0 or n == len(todo):
                    rate = n / max(1e-6, time.time() - t0)
                    print(f"[vision-eval] {tag}: {n}/{len(todo)} {rate:.2f} it/s "
                          f"eta {(len(todo) - n) / max(rate, 1e-6) / 60:.0f} min", flush=True)
        res[tag] = [done[it["id"]] for it in items if it["id"] in done]
    return res


def _versions() -> dict[str, Any]:
    import platform

    import transformers

    return {"torch": torch.__version__, "transformers": transformers.__version__,
            "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
            "threads": torch.get_num_threads(),
            "revisions": {V19: V19_REV, BASE: BASE_REV, MAPIKA: MAPIKA_REV, NB_REPO: NB_REV,
                          POPE_REPO: POPE_REV}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--systems", default="strands,qwen,mapika")
    ap.add_argument("--nb-groups", type=int, default=300)
    ap.add_argument("--pope", type=int, default=600)
    ap.add_argument("--ijb-jsonl", help="Image JevBench preview items, built as README.md describes")
    ap.add_argument("--nb-local")
    ap.add_argument("--pope-local")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--long-side", type=int, default=448,
                    help="longest image side for strands/qwen (0: no cap)")
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="pixel budget per image for strands/qwen, aspect kept, never upscaled (0: none)")
    ap.add_argument("--nb-start", type=int, default=0,
                    help="first NaturalBench group (300: held-out groups for fitting image temperatures)")
    ap.add_argument("--temps", default="",
                    help='JSON per-kind image temperatures for the strands system, e.g. {"noul": 0.76}')
    ap.add_argument("--no-blind", action="store_true", help="skip the image-removed pass")
    ap.add_argument("--checkpoint", default=V19,
                    help="the strands system's checkpoint: v19 by default, or a fine-tuned one")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _DeciderSystem.long_side = a.long_side
    _DeciderSystem.max_pixels = a.max_pixels
    items = (list(naturalbench(a.nb_groups, a.nb_local, a.nb_start)) if a.nb_groups else []) + \
        (list(pope(a.pope, a.pope_local)) if a.pope else [])
    if a.ijb_jsonl:
        items += list(image_jevbench(a.ijb_jsonl))
    print(f"[vision-eval] {len(items)} items", flush=True)
    image_temps = {k: float(v) for k, v in json.loads(a.temps).items()} if a.temps else None
    systems = {"strands": lambda: Strands(a.checkpoint, device=a.device, image_temps=image_temps),
               "qwen": lambda: QwenUntrained(device=a.device),
               "mapika": lambda: Mapika(device=a.device)}
    all_res: dict[str, list[dict[str, Any]]] = {}
    errors: dict[str, str] = {}
    temperatures: dict[str, Any] = {}  # the strands system's, for rescoring (temps.py)
    for name in a.systems.split(","):
        try:
            system = systems[name]()
            if isinstance(system, Strands):
                temperatures = {"image": system.temps, "text": system.text_temps, "default": system.t_default}
            all_res.update(run_system(system, items, a.out, (False,) if a.no_blind else (False, True)))
        except Exception:
            errors[name] = traceback.format_exc()
            print(f"[vision-eval] {name} FAILED\n{errors[name]}", flush=True)
        with open(os.path.join(a.out, "summary.json"), "w") as fh:
            json.dump({"args": vars(a), "versions": _versions(), "errors": errors,
                       "temperatures": temperatures, "scores": score_runs(all_res)}, fh, indent=2)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
