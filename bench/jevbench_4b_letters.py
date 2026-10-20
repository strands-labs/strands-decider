#!/usr/bin/env python3
"""Frozen Qwen3.5-4B on the JevBench public set, SemIf-style: one forward pass per
task, softmax over the option-letter logits. No generation, no thinking tokens.

This reproduces the reference method in research/generations.md ("frozen models read
through the chat template, option-letter logits") closely enough for a local
comparison with the 2B decider; it is not the author's own code path (that needs the
semif_phase1 package) and score tasks are included, which SemIf excludes.

Writes /tmp/jev_4b_letters.jsonl: one {id, probs, predicted, correct, ms} per task.
"""

import json
import statistics
import time

import mlx.core as mx
from mlx_lm import load

MODEL = "Qwen/Qwen3.5-4B"
LETTERS = [chr(ord("A") + i) for i in range(26)]


def option_text(task, label):
    q = task["question"]
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        key = {"yes": "true", "no": "false", "true": "true", "false": "false"}[label]
        return f"{key}: {crit.get(key, '')}"
    if q["type"] == "choice":
        return f"{label}: {crit.get(label) or label}"
    return f"{label}: {crit[int(label)]}"


def main():
    tasks = [json.loads(line) for line in open("/tmp/jevbench_all.jsonl", encoding="utf-8") if line.strip()]
    model, tokenizer = load(MODEL)
    print(f"{len(tasks)} tasks; model loaded", flush=True)

    # Letter token ids: with and without the leading space, whichever the tokenizer uses.
    letter_ids = {}
    for i, letter in enumerate(LETTERS):
        ids = {tokenizer.encode(letter, add_special_tokens=False)[0]}
        ids.add(tokenizer.encode(" " + letter, add_special_tokens=False)[0])
        letter_ids[i] = sorted(ids)

    system = ("You are answering a multiple-choice question. Reply with the letter of "
              "the best option and nothing else.")
    correct, lat = [], []
    with open("/tmp/jev_4b_letters.jsonl", "w", encoding="utf-8") as out:
        for t in tasks:
            labels = [str(x) for x in t["labels"]]
            options = "\n".join(
                f"{LETTERS[i]}) {option_text(t, lab)}" for i, lab in enumerate(labels)
            )
            user = (f"{t['state']}\n\n{t['question']['instructions']}\n\n"
                    f"Options:\n{options}\n\nAnswer with the letter only.")
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                add_generation_prompt=True,
            )
            # Qwen3.5 is a thinking model: the token after the generation prompt starts a
            # <think> block, where letter logits are meaningless (0.35 accuracy measured).
            # Scoring at the position after an EMPTY think block is the answer position —
            # this doubled accuracy on a 60-task subset.
            if isinstance(prompt, str):
                prompt = tokenizer.encode(prompt)
            prompt = prompt + tokenizer.encode("<think>\n\n</think>\n\n")
            t0 = time.perf_counter()
            ids = mx.array(prompt)
            logits = model(ids[None])[0, -1]
            picked = [max(logits[i].item() for i in letter_ids[j]) for j in range(len(labels))]
            mx.eval(picked)
            ms = (time.perf_counter() - t0) * 1000

            picked = mx.array(picked)
            picked -= picked.max()
            weights = mx.exp(picked)
            probs = (weights / weights.sum()).tolist()
            by_label = dict(zip(labels, probs, strict=True))
            predicted = max(by_label, key=by_label.get)
            ok = predicted == t["expected"]
            correct.append(ok)
            lat.append(ms)
            out.write(json.dumps({"id": t["id"], "probs": by_label, "predicted": predicted,
                                  "correct": ok, "ms": ms}) + "\n")
            out.flush()
            if len(correct) % 25 == 0:
                print(f"{len(correct)}/{len(tasks)} acc so far {sum(correct)/len(correct):.3f}",
                      flush=True)

    s = sorted(lat)
    print(f"\n{MODEL} (frozen, letter logits): accuracy {sum(correct)}/{len(correct)} "
          f"= {sum(correct)/len(correct):.3f}")
    print(f"latency ms: median {statistics.median(s):.0f}, p95 {s[int(len(s)*.95)]:.0f}")


if __name__ == "__main__":
    main()
