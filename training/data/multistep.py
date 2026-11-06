"""Multi-step documents from public datasets, as typed questions with exact labels.

JevBench's hard tier fails v7 and v13 in three ways (see research/history.md,
"v14: multi-step documents with a teacher"): sequential lookups through a long
procedure, conditions that must all hold, and rules that override other rules. The
training corpus has none of them -- it is short classification. These
converters bring in real or carefully generated documents that do:

  contractnli  full NDAs (CC BY 4.0); each of 17 fixed claims is entailed, contradicted
               or not mentioned. One contract, many questions with different answers.
  hotpotqa     two-hop comparisons over ten Wikipedia paragraphs, eight of them
               distractors (CC BY-SA 4.0). Only questions whose answer is "yes", "no" or
               one of the two compared entities, so the options are exact.
  boardgame    rules that conflict, resolved by stated preferences (BoardgameQA, CC BY
               4.0): proved, disproved, or not settled.
  musique      2-4 hop questions over paragraphs with distractors, half of them made
               unanswerable by removing a supporting paragraph (MuSiQue full, CC BY 4.0).
               Options are the answer, the intermediate answers of earlier hops, a few
               distractor entities, and "cannot be determined" -- so stopping one hop
               short, or answering when a link is missing, is a named wrong option.

Every label comes from the dataset; nothing is estimated. Documents longer than the
character budget are windowed only where that cannot change the label (ContractNLI:
the window keeps every evidence span) or trimmed only of distractors (MuSiQue).
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Iterator
from typing import Any

from .format import Example

CHAR_BUDGET = 9000  # ~2,300 Qwen tokens of English prose, leaving room for the question

NLI_OPTIONS = [
    ["entailed", "the contract states or clearly implies it"],
    ["contradicted", "the contract states or clearly implies the opposite"],
    ["not mentioned", "the contract does not address it"],
]
NLI_LABEL = {"Entailment": 0, "Contradiction": 1, "NotMentioned": 2}
BOARDGAME_OPTIONS = [
    ["proved", "the rules and preferences establish that it is true"],
    ["disproved", "the rules and preferences establish that it is false"],
    ["unknown", "the rules and preferences do not settle it"],
]
YES_NO = [["false", "the answer is no"], ["true", "the answer is yes"]]
CANNOT = "cannot be determined from these paragraphs"


# ---- ContractNLI ------------------------------------------------------------


def _window(text: str, spans: list[list[int]], evidence: list[int], budget: int) -> str | None:
    """A contiguous stretch of at most `budget` characters containing every evidence span.

    None when the evidence itself is spread wider than the budget. Cut at span edges so
    no sentence is split.
    """
    if len(text) <= budget:
        return text
    if not evidence:  # "not mentioned" holds for any part of the document
        lo, hi = 0, 0
    else:
        lo = min(spans[i][0] for i in evidence)
        hi = max(spans[i][1] for i in evidence)
        if hi - lo > budget:
            return None
    # grow outward, alternating sides, by whole spans
    start, end = lo, hi
    edges = sorted({s for s, _ in spans} | {e for _, e in spans})
    left = [x for x in edges if x <= start][::-1]
    right = [x for x in edges if x >= end]
    li = ri = 0
    while True:
        grew = False
        if ri < len(right) and right[ri] - start <= budget:
            end = right[ri]
            ri += 1
            grew = True
        if li < len(left) and end - left[li] <= budget:
            start = left[li]
            li += 1
            grew = True
        if not grew:
            break
    return text[start:end].strip()


def contractnli(path: str, budget: int = CHAR_BUDGET) -> Iterator[Example]:
    data = json.load(open(path, encoding="utf-8"))
    hyps = data["labels"]
    for doc in data["documents"]:
        for key, ann in doc["annotation_sets"][0]["annotations"].items():
            label = NLI_LABEL[ann["choice"]]
            state = _window(doc["text"], doc["spans"], ann["spans"], budget)
            if state is None:
                continue
            q = (
                "Does this contract entail, contradict, or not mention the following "
                f'statement? "{hyps[key]["hypothesis"]}"'
            )
            yield Example(
                kind="choice",
                state=state,
                instructions=q,
                options=NLI_OPTIONS,
                label=label,
                task="contractnli",
            )


# ---- HotpotQA ---------------------------------------------------------------


def hotpotqa(rows: Iterable[dict]) -> Iterator[Example]:
    for r in rows:
        if r["type"] != "comparison":
            continue
        ctx = r["context"]
        state = "\n\n".join(
            f"{t}: {''.join(s)}" for t, s in zip(ctx["title"], ctx["sentences"], strict=True)
        )
        ans = r["answer"].strip()
        titles = list(dict.fromkeys(r["supporting_facts"]["title"]))
        if ans.lower() in ("yes", "no"):
            yield Example(
                kind="noul",
                state=state,
                instructions=r["question"],
                options=YES_NO,
                label=int(ans.lower() == "yes"),
                task="hotpotqa",
            )
        elif len(titles) == 2 and ans in titles:
            yield Example(
                kind="choice",
                state=state,
                instructions=r["question"],
                options=[[t, ""] for t in titles],
                label=titles.index(ans),
                task="hotpotqa",
            )


# ---- BoardgameQA ------------------------------------------------------------

BOARDGAME_Q = "Based on the game state and the rules and preferences, "


def boardgame(rows: Iterable[dict]) -> Iterator[Example]:
    for r in rows:
        ex = r["example"]
        cut = ex.rfind(BOARDGAME_Q)
        if cut < 0:
            continue
        label = {"proved": 0, "disproved": 1, "unknown": 2}[r["label"]]
        yield Example(
            kind="choice",
            state=ex[:cut].strip(),
            instructions=ex[cut:].strip(),
            options=BOARDGAME_OPTIONS,
            label=label,
            task="boardgame",
        )


# ---- MuSiQue ----------------------------------------------------------------


def musique(rows: Iterable[dict], budget: int = CHAR_BUDGET, seed: int = 0) -> Iterator[Example]:
    rng = random.Random(seed)
    for r in rows:
        paras = r["paragraphs"]
        support = [p for p in paras if p["is_supporting"]]
        others = [p for p in paras if not p["is_supporting"]]
        keep, used = list(support), sum(len(p["paragraph_text"]) for p in support)
        for p in others:  # distractors until the budget, in the dataset's own order
            if used + len(p["paragraph_text"]) > budget:
                continue
            keep.append(p)
            used += len(p["paragraph_text"])
        keep.sort(key=lambda p: p["idx"])
        state = "\n\n".join(f"{p['title']}: {p['paragraph_text']}" for p in keep)

        answer = r["answer"].strip()
        hops = [d["answer"].strip() for d in r["question_decomposition"][:-1]]
        cands = [answer] + hops + [p["title"] for p in rng.sample(others, min(2, len(others)))]
        options = [*dict.fromkeys(c for c in cands if c), CANNOT]
        if len(options) < 3:
            continue
        label = options.index(answer) if r["answerable"] else options.index(CANNOT)
        yield Example(
            kind="choice",
            state=state,
            instructions=r["question"],
            options=[[o, ""] for o in options],
            label=label,
            task="musique",
        )


# ---- corpus assembly --------------------------------------------------------


def balance_by_claim(
    examples: list[Example], *, min_count: int = 25, seed: int = 0
) -> list[Example]:
    """Equal counts of each answer per ContractNLI claim, so the claim alone predicts nothing.

    Every contract is asked the same 17 claims, and several are nearly constant ("no
    reverse engineering" is not mentioned in 363 of 423 contracts): answering each claim
    with its usual label scores 0.681 on dev without reading a word. For each claim, keep
    the answers it has at least `min_count` of, subsampled to the rarest of them; what
    remains is contracts asked the same claim with different answers.
    """
    rng = random.Random(seed)
    by: dict = {}
    for ex in examples:
        by.setdefault((ex.instructions, ex.label), []).append(ex)
    claims = {q for q, _ in by}
    out: list[Example] = []
    for q in sorted(claims):
        groups = [by[(q, lab)] for lab in range(3) if len(by.get((q, lab), [])) >= min_count]
        if len(groups) < 2:
            continue
        n = min(len(g) for g in groups)
        for g in groups:
            out += rng.sample(g, n)
    return out


def musique_pairs(rows: Iterable[dict], n_questions: int, seed: int = 0) -> list[dict]:
    """`n_questions` questions with BOTH their answerable and unanswerable versions."""
    by: dict = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r)
    ids = sorted(k for k, v in by.items() if len(v) == 2)
    pick = random.Random(seed).sample(ids, min(n_questions, len(ids)))
    return [r for k in pick for r in by[k]]


def fits(
    examples: list[Example], tokenizer: Any, max_tokens: int, frame: Any = None
) -> list[Example]:
    """Drop rows whose rendered prompt exceeds the window (truncation would cut options).

    With `frame` (prompting.chat_frame), measure the lettered prompt instead, with every
    phrasing of the question, because the collator samples one and refuses a prompt that is
    too long.
    """
    from strands_decider.prompting import build_prompt, lettered_prompt

    if frame is None:
        return [
            e
            for e in examples
            if len(tokenizer(build_prompt(e.state, e.to_question())[0])["input_ids"]) <= max_tokens
        ]

    def longest(e: Example) -> int:
        texts = [
            "".join(lettered_prompt(e.state, e.to_question(i), frame)) for i in e.all_instructions()
        ]
        ids = tokenizer(texts, add_special_tokens=False)["input_ids"]
        return max(len(x) for x in ids)

    return [e for e in examples if longest(e) <= max_tokens]


def main(argv: list[str] | None = None) -> None:
    import argparse

    from datasets import load_dataset
    from transformers import AutoTokenizer

    from .format import write_jsonl

    ap = argparse.ArgumentParser(description="Build the multi-step training and eval sets (v14).")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/multistep_v14.jsonl")
    ap.add_argument("--eval-out", default="data/multistep_v14_eval.jsonl")
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-2B-Base")
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--musique-questions", type=int, default=3000, help="x2 rows (both versions)")
    ap.add_argument("--boardgame", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    def read(path: str) -> list[dict]:
        with open(path, encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    mus_train = read(f"{args.raw}/musique/data/musique_full_v1.0_train.jsonl")
    mus_dev = read(f"{args.raw}/musique/data/musique_full_v1.0_dev.jsonl")
    bg_train = load_dataset("tasksource/Boardgame-QA", split="train")
    bg_valid = load_dataset("tasksource/Boardgame-QA", split="valid")

    train = {
        "contractnli": balance_by_claim(
            list(contractnli(f"{args.raw}/contract-nli/train.json")), seed=args.seed
        ),
        "musique": list(
            musique(musique_pairs(mus_train, args.musique_questions, args.seed), seed=args.seed)
        ),
        "boardgame": list(boardgame(bg_train)),
    }
    evals = {
        # natural label distribution: the claim-only baseline is reported alongside
        "contractnli": list(contractnli(f"{args.raw}/contract-nli/dev.json")),
        "musique": list(musique(musique_pairs(mus_dev, 600, args.seed), seed=args.seed)),
        "boardgame": list(boardgame(bg_valid)),
        # HELD OUT: never in training, the pre-registered transfer test
        "hotpotqa": list(
            hotpotqa(load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation"))
        ),
    }

    def some(xs: list[Example], n: int) -> list[Example]:
        return rng.sample(xs, min(n, len(xs)))

    rows, eval_rows = [], []
    train["boardgame"] = some(train["boardgame"], args.boardgame)
    evals["boardgame"] = some(evals["boardgame"], 900)
    evals["hotpotqa"] = some(evals["hotpotqa"], 1000)
    for name, exs in train.items():
        kept = fits(exs, tok, args.max_tokens)
        print(f"train {name:<12} {len(kept):>6,} rows ({len(exs) - len(kept)} over the window)")
        rows += kept
    for name, exs in evals.items():
        kept = fits(exs, tok, args.max_tokens)
        print(f"eval  {name:<12} {len(kept):>6,} rows ({len(exs) - len(kept)} over the window)")
        eval_rows += kept
    rng.shuffle(rows)
    write_jsonl(args.out, rows)
    write_jsonl(args.eval_out, eval_rows)
    print(f"{args.out}: {len(rows):,}   {args.eval_out}: {len(eval_rows):,}")


if __name__ == "__main__":
    main()
