"""Question sensitivity: does the readout follow the question, or classify the state?

Same state, same options in the same order; only the question changes. A model that
reads the question answers "which option is listed first?" with the first option and
"which option does this text NOT fit?" with something other than the state's class. A
model that learned to classify the state gives the same answer to every question.

Each checkpoint is scored twice on identical prompts: through its trained readout, and
with its adapter disabled through the frozen torso's option-number tokens -- so every
run carries its own untrained baseline. `--torso-tokens` adds a third reading, adapter
on but token readout, to separate the trained torso from the trained head.

First run (v7, v8), on held-out choice tasks:

                                            frozen base    v7     chance
    "listed first?"  accuracy                  0.700      0.080   0.184
    "does NOT fit?"  picks the original class  0.028      0.730   0.184
    any changed question: same answer as real   ~0.4      ~0.95

Training on a corpus where each state is asked one fixed question taught the model
that the question never changes the answer. Lower "same as real" is better; so is
lower capture on NOT and higher accuracy on first / last.

    python evaluation/question_sensitivity.py CHECKPOINT [CHECKPOINT ...] [--n 200]
        [--torso-tokens] [--out results.json]
"""
import argparse
import json
import random
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from strands_decider.data.collate import CollatorConfig, SystemOneCollator  # noqa: E402
from strands_decider.data.format import Example  # noqa: E402
from strands_decider.modeling import (  # noqa: E402
    MASK_VALUE,
    StrandsDeciderModel,
    masked_log_softmax,
    pool_last_token,
)
from strands_decider.prompting import NOUL_DEFAULT_CRITERIA, NOUL_SLOT_LABELS  # noqa: E402
from strands_decider.train import ExampleDataset  # noqa: E402

GENERIC_NOUL = [[lbl, NOUL_DEFAULT_CRITERIA[lbl]] for lbl in NOUL_SLOT_LABELS]

TRAIN, HOLDOUT = "data/train_v5.jsonl", "data/holdout_v5_norule.jsonl"
CHOICE_TASKS = {"ag_news": TRAIN, "banking77": TRAIN, "clinc150": TRAIN,
                "emotion": HOLDOUT, "massive_intent": HOLDOUT}
# Fixed-question tasks, then two whose question differs on every row.
NOUL_TASKS = {"spam": TRAIN, "toxicity": TRAIN, "sarcasm": HOLDOUT,
              "boolq": TRAIN, "mnli_entail": TRAIN}
HELD_OUT = {"emotion", "massive_intent", "sarcasm"}
IRRELEVANT = [  # other tasks' questions, asked over the wrong option set
    "Which language is this text written in?",
    "Which emotion is the author of this message expressing?",
    "Which section of the newspaper does this article belong to?",
    "Which customer-service intent does this message express?",
]
GROUPS = [
    ("choice, tasks in training", lambda t: t in CHOICE_TASKS and t not in HELD_OUT),
    ("choice, held-out tasks", lambda t: t in CHOICE_TASKS and t in HELD_OUT),
    ("noul, fixed-question tasks in training", lambda t: t in ("spam", "toxicity")),
    ("noul, fixed-question held-out task", lambda t: t == "sarcasm"),
    ("noul, per-row-question tasks in training", lambda t: t in ("boolq", "mnli_entail")),
]


def _rows(path, task, kind):
    with open(path, encoding="utf-8") as fh:
        return [d for d in map(json.loads, fh) if d["task"] == task and d["kind"] == kind]


def build_items(n_per_task: int, seed: int = 0):
    """(task, condition, example id, Example, target or None, original gold slot)."""
    rng = random.Random(seed)
    items, eid = [], 0
    for task, path in CHOICE_TASKS.items():
        # <= 9 options: the frozen token readout has single-token numbers only for 1-9.
        rows = [d for d in _rows(path, task, "choice") if 3 <= len(d["options"]) <= 9]
        for d in rng.sample(rows, min(n_per_task, len(rows))):
            n, g = len(d["options"]), d["label"]
            # One fixed order per example with gold neither first nor last, so the
            # position questions have an answer other than the state's class.
            order = list(range(n))
            rng.shuffle(order)
            while order.index(g) in (0, n - 1):
                rng.shuffle(order)
            opts = [d["options"][i] for i in order]
            gold = order.index(g)
            conds = {
                "real": (d["instructions"], gold),
                "first": ("Ignore the text above. Which option is listed first below?", 0),
                "last": ("Ignore the text above. Which option is listed last below?", n - 1),
                "not": ("Which option does this text clearly NOT fit?", None),
                "irrelevant": (rng.choice([q for q in IRRELEVANT if q != d["instructions"]]),
                               None),
            }
            for c, (q, tgt) in conds.items():
                ex = Example(kind="choice", state=d["state"], instructions=q, options=opts,
                             label=gold if tgt is None else tgt, task=task)
                items.append((task, c, eid, ex, tgt, gold))
            eid += 1
    for task, path in NOUL_TASKS.items():
        rows = _rows(path, task, "noul")
        k = n_per_task // 2  # balanced, so majority-class guessing scores 0.5
        pos = [d for d in rows if d["label"] == 1]
        neg = [d for d in rows if d["label"] == 0]
        for d in rng.sample(pos, min(k, len(pos))) + rng.sample(neg, min(k, len(neg))):
            g = d["label"]
            wrapped = f'Is the answer to the following question "no"? {d["instructions"]}'
            # The negated prompt gets the generic true/false descriptions: the task's
            # own ("true -- promotional, scam, or bulk message") would contradict the
            # wrapped question and make "negated" ill-posed.
            for c, (q, tgt, opts) in {"real": (d["instructions"], g, d["options"]),
                                      "negated": (wrapped, 1 - g, GENERIC_NOUL)}.items():
                ex = Example(kind="noul", state=d["state"], instructions=q,
                             options=opts, label=tgt, task=task)
                items.append((task, c, eid, ex, tgt, g))
            eid += 1
    return items


@torch.no_grad()
def readouts(ckpt: str, items, torso_tokens: bool):
    """Argmax per prompt: trained readout, frozen token readout, optionally adapter-on tokens."""
    model = StrandsDeciderModel.load(ckpt).eval().to("cuda")
    coll = SystemOneCollator(
        model.tokenizer,
        CollatorConfig(max_length=model.config.max_length, num_slots=model.config.num_slots,
                       head_type=model.config.head_type),
        train=False,
    )
    loader = DataLoader(ExampleDataset([it[3] for it in items]), batch_size=16,
                        shuffle=False, collate_fn=coll)
    out = defaultdict(list)
    name = ckpt.rstrip("/\\").split("/")[-1].split("\\")[-1]
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        res = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    n_slots=batch["n_slots"], opt_idx=batch.get("opt_idx"), temperature=1.0)
        out[name] += res["log_probs"].argmax(-1).tolist()
        lp, ok = model.frozen_slot_log_probs(batch["input_ids"], batch["attention_mask"],
                                             batch["n_slots"])
        assert bool(ok.all()), "frozen readout needs <= 9 options"
        out[f"{name} frozen"] += lp.argmax(-1).tolist()
        if torso_tokens:
            h = model.encode(batch["input_ids"], batch["attention_mask"])
            logits = model.slot_logits(pool_last_token(h, batch["attention_mask"]).float())
            logits = torch.nn.functional.pad(
                logits, (0, model.config.num_slots - logits.size(-1)), value=MASK_VALUE)
            out[f"{name} torso+tok"] += masked_log_softmax(
                logits, batch["n_slots"]).argmax(-1).tolist()
    del model
    torch.cuda.empty_cache()
    return dict(out)


def summarise(items, preds):
    real = {(m, eid): preds[m][i] for m in preds
            for i, (_, c, eid, *_rest) in enumerate(items) if c == "real"}
    report = {}
    for title, keep in GROUPS:
        stats = defaultdict(lambda: defaultdict(list))
        for m in preds:
            for i, (t, c, eid, ex, tgt, g) in enumerate(items):
                if not keep(t):
                    continue
                p, s = preds[m][i], stats[(m, c)]
                if tgt is not None:
                    s["acc"].append(p == tgt)
                if c != "real":
                    s["capture"].append(p == g)
                    s["same_as_real"].append(p == real[(m, eid)])
                s["chance"].append(1 / ex.n_options)
        rows = report[title] = {}
        for (m, c), d in stats.items():
            for metric, v in d.items():
                rows.setdefault(f"{c}/{metric}", {})[m] = sum(v) / len(v)
    return report


def print_report(report, models):
    for title, rows in report.items():
        print(f"\n### {title}")
        w = max(len(m) for m in models) + 2
        print(f"{'':<24}" + "".join(f"{m:>{w}}" for m in models))
        for key, vals in rows.items():
            if key.endswith("/chance"):
                continue
            chance = rows.get(key.split("/")[0] + "/chance", {})
            ch = next(iter(chance.values()), None)
            shown = key.endswith("/capture") or key in ("first/acc", "last/acc")
            print(f"{key:<24}" + "".join(f"{vals[m]:>{w}.3f}" for m in models)
                  + (f"   chance {ch:.3f}" if shown and ch is not None else ""))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--n", type=int, default=200, help="examples per task")
    ap.add_argument("--torso-tokens", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = build_items(args.n)
    print(f"{len(items)} prompts")
    preds = {}
    for ckpt in args.checkpoints:
        preds.update(readouts(ckpt, items, args.torso_tokens))
    report = summarise(items, preds)
    print_report(report, list(preds))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
