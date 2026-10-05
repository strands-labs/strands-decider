"""Teacher distributions from a larger frozen model, read the way SemIf reads it.

SemIf scores a frozen instruct model with one forward pass: the decision goes through
the chat template as JSON (evidence, criterion, lettered options), thinking disabled,
and the answer is the softmax over the option letters' next-token logits. That reading
of Qwen3.5-4B scores 0.810 on JevBench against our 0.667, so a model read this way is a
candidate teacher: its distribution over the options, for each training prompt, becomes
a soft target alongside the gold label.

`render` reproduces SemIf's prompt byte for byte (`tests/test_teacher.py` checks it
against SemIf's own `encode_prompt` when that package is installed), and `to_row`
maps our examples the way JevBench's `semif_direct` adapter maps its tasks -- noul as
("true", "false") with the criteria as descriptions, choice one option per criterion,
score "0".."k-1" -- so the labels mean what the teacher's benchmark score means.

Batched, unlike SemIf's reference path: right-padded, the final hidden state gathered at
each row's last real token, and only the option letters' output rows applied. Under
causal attention the real tokens never see the right padding, so this matches the
batch-of-one path up to bf16 accumulation order (`--check` measures the difference).

    python -m strands_decider.data.teacher --src data/train_v5.jsonl --out data/teacher_v5.jsonl

On N GPUs, run one process per GPU with `--num-shards N --shard-index i`, then the same
command with `--merge` instead of `--shard-index` to write `--out` (and `--shift-by`'s
file) exactly as one process would (see data/shards.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from typing import Any

from . import shards
from .format import Example, read_jsonl

LETTERS = "ABCDEFGHIJKLMNOP"  # SemIf's limit: rows with more options get no teacher
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def to_row(ex: Example) -> tuple[dict, list[int]] | None:
    """SemIf's row for an example, and for each teacher option its canonical index.

    None when the example has more options than there are answer letters.
    """
    if ex.n_options > len(LETTERS):
        return None
    if ex.kind == "noul":
        # Canonical noul order is (false, true); SemIf lists true first.
        desc = {name: d for name, d in ex.options}
        order = [1, 0]
        options = [{"id": k, "description": f"{k}: " + desc.get(k, f"The proposition is {k}.")}
                   for k in ("true", "false")]
    elif ex.kind == "choice":
        order = list(range(ex.n_options))
        options = [{"id": n, "description": f"{n}: " + (d or n)} for n, d in ex.options]
    else:
        order = list(range(ex.n_options))
        options = [{"id": str(i), "description": f"{i}: {d}"} for i, (_, d) in enumerate(ex.options)]
    row = {"id": "x", "state": ex.state, "question": ex.instructions, "options": options}
    return row, order


def render(tokenizer: Any, row: dict) -> str:
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [{"letter": LETTERS[i], "description": o["description"]}
                    for i, o in enumerate(row["options"])],
    }
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]
    return str(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False))


def letter_ids(tokenizer: Any) -> list[int]:
    ids = []
    for letter in LETTERS:
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1 or tokenizer.decode(enc) != letter:
            raise ValueError(f"answer letter {letter!r} is not a single round-trip token")
        ids.append(enc[0])
    return ids


def load(model_id: str, revision: str) -> tuple[Any, Any]:
    import torch
    import transformers

    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    config = transformers.AutoConfig.from_pretrained(model_id, revision=revision)
    cls = transformers.AutoModelForCausalLM
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        # A multimodal checkpoint; load only the text tower, as SemIf's loader does.
        # Right padding stays exact for its recurrent (Gated DeltaNet) layers: the pads
        # come after every real token, so the state at the last real token never sees them.
        cls = transformers.Qwen3_5ForCausalLM
        config = config.get_text_config()
    model = cls.from_pretrained(model_id, revision=revision, config=config,
                                dtype=torch.bfloat16, device_map={"": "cuda"})
    model.eval()
    return model, tok


def prompts(tok: Any, examples: Sequence[Example],
            max_tokens: int) -> list[tuple[int, list[int], list[int]]]:
    """(example index, prompt token ids, option order) for every example the teacher can
    read in `max_tokens` tokens, shortest first."""
    rows = []
    for i, ex in enumerate(examples):
        mapped = to_row(ex)
        if mapped is None:
            continue
        row, order = mapped
        ids = tok.encode(render(tok, row), add_special_tokens=False)
        if len(ids) <= max_tokens:
            rows.append((i, ids, order))
    rows.sort(key=lambda r: len(r[1]))
    return rows


def canonical(p: list[float], order: list[int]) -> list[float]:
    """Probabilities in the teacher's option order -> the example's canonical order."""
    canon = [0.0] * len(order)
    for teacher_pos, canon_idx in enumerate(order):
        canon[canon_idx] = p[teacher_pos]
    return canon


def label(model: Any, tok: Any, examples: Sequence[Example], *, max_batch_tokens: int = 4000,
          max_batch: int = 32, max_tokens: int = 4096, log_every: int = 2000,
          skip: set | None = None,
          sink: Callable[[int, list[float]], None] | None = None, num_shards: int = 1,
          shard_index: int = 0) -> list[list[float] | None]:
    """Teacher probabilities in each example's canonical option order (None if unlabelled).

    `skip` holds example indices already labelled (a resumed run); `sink(i, probs)` is
    called as each row is labelled, so progress survives an interrupted run.

    With `num_shards` > 1 this labels only batches `shard_index`, `shard_index + N`, ...
    of the batching a fresh single-process run makes, so each row is computed in the
    batch it is in there. A resumed shard keeps those batches and leaves out the rows
    already labelled; a resumed single process instead batches its remaining rows afresh.

    The token budget matters more than it looks. Rows run shortest first, so the last
    batches are the widest; on a 24 GiB card a 12,000-token budget pushed the teacher
    past device memory there, and Windows silently spilled to system RAM over PCIe --
    the run fell from ~30 rows/s to under 2 without failing.
    """
    import torch

    letters = letter_ids(tok)
    head = model.get_output_embeddings().weight
    rows = [r for r in prompts(tok, examples, max_tokens)
            if num_shards > 1 or not (skip and r[0] in skip)]

    out: list[list[float] | None] = [None] * len(examples)
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    total = len(rows) // num_shards  # a shard's share, roughly, for the progress line
    done, t0, k, n_batch = 0, time.time(), 0, 0
    while k < len(rows):
        batch: list[tuple[int, list[int], list[int]]] = []
        while k < len(rows) and len(batch) < max_batch and \
                (len(batch) + 1) * len(rows[k][1]) <= max_batch_tokens:
            batch.append(rows[k])
            k += 1
        if not batch:  # a single row longer than the token budget
            batch, k = [rows[k]], k + 1
        n_batch += 1
        batch = [r for r in batch if not (skip and r[0] in skip)]  # a resumed shard
        if (n_batch - 1) % num_shards != shard_index or not batch:
            continue
        width = max(len(ids) for _, ids, _ in batch)
        ids_t = torch.full((len(batch), width), pad, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for j, (_, ids, _) in enumerate(batch):
            ids_t[j, : len(ids)] = torch.tensor(ids)
            mask[j, : len(ids)] = 1
        with torch.inference_mode():
            hidden = model.model(input_ids=ids_t.to(head.device),
                                 attention_mask=mask.to(head.device),
                                 use_cache=False).last_hidden_state
            last = hidden[torch.arange(len(batch)), mask.sum(1).to(head.device) - 1]
            for j, (i, _, order) in enumerate(batch):
                n = len(order)
                logits = last[j].float() @ head[letters[:n]].float().t()
                canon = canonical(torch.softmax(logits, -1).tolist(), order)
                out[i] = canon
                if sink is not None:
                    sink(i, canon)
        done += len(batch)
        if log_every and done // log_every != (done - len(batch)) // log_every:
            rate = done / (time.time() - t0)
            print(f"[teacher] {done:,}/{total:,}  {rate:.1f} rows/s  "
                  f"eta {(total - done) / rate / 60:.0f} min", flush=True)
    return out


def label_vllm(model_id: str, revision: str, examples: Sequence[Example], *,
               max_tokens: int = 4096, skip: set | None = None,
               sink: Callable[[int, list[float]], None] | None = None,
               chunk: int = 4096) -> list[list[float] | None]:
    """`label`'s distributions computed by vLLM, which batches and caches far better.

    The same prompts as `label`, passed as token ids. One greedy token restricted to the
    option letters, with `logprobs_mode="processed_logprobs"`: vLLM then reports the
    log-softmax taken after that restriction, which is the softmax over the letters'
    logits that `label` computes. Agreement is to bf16 kernel rounding, not bit for bit.
    Needs vLLM >= 0.10.2 (`logprobs_mode`); a distribution that does not sum to 1 means
    an engine that reports something else, and stops the run.
    """
    import math

    import transformers
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    tok = transformers.AutoTokenizer.from_pretrained(model_id, revision=revision)
    letters = letter_ids(tok)
    rows = [r for r in prompts(tok, examples, max_tokens) if not (skip and r[0] in skip)]
    llm = LLM(model=model_id, revision=revision, dtype="bfloat16", max_model_len=max_tokens + 1,
              logprobs_mode="processed_logprobs", max_logprobs=len(LETTERS),
              enable_prefix_caching=True)
    out: list[list[float] | None] = [None] * len(examples)
    t0 = time.time()
    for start in range(0, len(rows), chunk):
        part = rows[start:start + chunk]
        params = [SamplingParams(max_tokens=1, temperature=0.0, logprobs=len(order),
                                 allowed_token_ids=letters[:len(order)]) for _, _, order in part]
        results = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids, _ in part], params,
                               use_tqdm=False)
        for (i, _, order), res in zip(part, results, strict=True):
            lp = res.outputs[0].logprobs[0]
            p = [math.exp(lp[t].logprob) for t in letters[:len(order)]]
            if abs(sum(p) - 1) > 1e-3:
                raise RuntimeError(f"vLLM letter probabilities sum to {sum(p):.4f}, not 1: "
                                   "this vLLM does not report processed logprobs")
            canon = out[i] = canonical(p, order)
            if sink is not None:
                sink(i, canon)
        done = start + len(part)
        rate = done / (time.time() - t0)
        print(f"[teacher] {done:,}/{len(rows):,}  {rate:.1f} rows/s  "
              f"eta {(len(rows) - done) / rate / 60:.0f} min", flush=True)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Label a corpus with a frozen teacher's option distribution.")
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--revision", default="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (timing, checks)")
    ap.add_argument("--max-batch-tokens", type=int, default=4000)
    ap.add_argument("--shift-by", nargs="*", default=[],
                    help="train files listed BEFORE --src in train_files; also writes "
                         "<out>_train.jsonl with indices shifted past their rows, which is "
                         "what TrainConfig.teacher_file expects")
    shards.add_args(ap)
    args = ap.parse_args(argv)
    sharded = shards.sharded(ap, args)

    examples = list(read_jsonl(args.src))
    if args.limit:
        examples = examples[: args.limit]
    out = shards.path(args.out, args.shard_index, args.num_shards) if sharded else args.out
    # Rows are appended as they are labelled; rerunning the same command resumes.
    done: dict[int, list[float]] = {}
    t0 = time.time()
    if args.merge:
        done = shards.merge(args.out, args.num_shards)
    elif os.path.exists(out):
        done = shards.read(out)
        print(f"resuming: {len(done):,} rows already labelled")
    if not args.merge:
        model, tok = load(args.model, args.revision)
        with open(out, "a", encoding="utf-8") as fh:
            def sink(i: int, p: list[float]) -> None:
                fh.write(json.dumps({"i": i, "probs": [round(x, 6) for x in p]}) + "\n")
                fh.flush()
                done[i] = p
            label(model, tok, examples, max_batch_tokens=args.max_batch_tokens,
                  skip=set(done), sink=sink, num_shards=args.num_shards,
                  shard_index=args.shard_index or 0)
    secs = time.time() - t0

    # Rewrite in row order, so the file is the same however many resumes it took.
    with open(out, "w", encoding="utf-8") as fh:
        for i in sorted(done):
            fh.write(json.dumps({"i": i, "probs": [round(x, 6) for x in done[i]]}) + "\n")
    n = agree = 0
    by_task: dict[str, list[int]] = {}
    for i, p in done.items():
        hit = int(max(range(len(p)), key=p.__getitem__) == examples[i].label)
        n += 1
        agree += hit
        by_task.setdefault(examples[i].task, []).append(hit)
    print(f"labelled {n:,} of {len(examples):,} rows ({secs / 60:.1f} min this run); "
          f"teacher argmax = gold on {agree / max(n, 1):.3f}")
    for task, hits in sorted(by_task.items()):
        print(f"  {task:<22} {sum(hits) / len(hits):.3f}  (n={len(hits):,})")
    if args.shift_by and not sharded:  # for shards, --merge writes it
        by = 0
        for path in args.shift_by:
            with open(path, encoding="utf-8") as fh:
                by += sum(1 for _ in fh)
        stem = args.out[: -len(".jsonl")] if args.out.endswith(".jsonl") else args.out
        shifted = stem + "_train.jsonl"
        with open(shifted, "w", encoding="utf-8") as fh:
            for i in sorted(done):
                fh.write(json.dumps({"i": i + by, "probs": [round(x, 6) for x in done[i]]}) + "\n")
        print(f"{shifted}: indices shifted by {by:,} (rows in {', '.join(args.shift_by)})")
    print(json.dumps({"model": args.model, "revision": args.revision, "src": args.src}))
    if sharded:
        shards.mark_done(out)


if __name__ == "__main__":
    main()
