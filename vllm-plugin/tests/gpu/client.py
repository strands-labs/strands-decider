"""Checks of a running ``vllm serve`` of the plugin, over /v1/completions. Standard library only.

    python client.py SCENARIO --url URL --rows ROWS.jsonl [--out OUT.json] [tolerances]

``--rows``: rendered requests, one per line: ``token_ids``, ``prefix_end`` (S), ``label_token_ids``
(the question's code ids) and optionally a reference's ``probabilities``.

- ``requests``: malformed requests get HTTP 400 and the server stays healthy and answers as before;
  the prefix cache (an exact repeat, a further question, another S, on exact counts of cached
  tokens) and mixed batches give the cold answers; with reference probabilities, the agreement.
- ``compare``: the same answers as another server's ``requests`` run (``--expect``).
- ``preempt``: concurrent requests on a server with a small KV cache all succeed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
import urllib.error
import urllib.request

FAILURES: list[str] = []
DROP = object()


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        FAILURES.append(what)
    return ok


class Server:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def post(self, payload):
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + "/v1/completions", data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")[:600]
        except Exception as e:  # the server is down
            return None, repr(e)[:300]

    def healthy(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False


def body(row, **over):
    """The README's request for one row."""
    b = {
        "model": "m",
        "prompt": row["token_ids"],
        "max_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "seed": 0,
        "logprobs": len(row["label_token_ids"]),
        "allowed_token_ids": row["label_token_ids"],
        "return_tokens_as_token_ids": True,
        "vllm_xargs": {"prefix_end": row["prefix_end"]},
    }
    b.update(over)
    return {k: v for k, v in b.items() if v is not DROP}


def ask(srv, row, **over):
    """(status, response, the codes' probabilities or None)."""
    st, r = srv.post(body(row, **over))
    if st != 200:
        return st, r, None
    top = r["choices"][0]["logprobs"]["top_logprobs"][0]
    lp = {int(k.split(":")[1]): v for k, v in top.items()}
    e = [math.exp(lp.get(t, -math.inf)) for t in row["label_token_ids"]]
    return st, r, [x / sum(e) for x in e]


def cached(r):
    return ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def dmax(p, q):
    return max(abs(a - b) for a, b in zip(p, q, strict=True))


def concurrently(fn, items):
    results = {}
    threads = [
        threading.Thread(target=lambda i=i, x=x: results.update({i: fn(i, x)}))
        for i, x in enumerate(items)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [results[i] for i in range(len(items))]


def malformed(row):
    """(name, body or raw JSON): one of each refusal, and vLLM's own refusal of prompt embeddings."""
    S, L = row["prefix_end"], len(row["token_ids"])
    return [
        ("prefix_end a string", body(row, vllm_xargs={"prefix_end": str(S)})),
        ("prefix_end 0, as JSON false arrives", body(row, vllm_xargs={"prefix_end": 0})),
        (f"prefix_end = L = {L}", body(row, vllm_xargs={"prefix_end": L})),
        (
            "prefix_end NaN",
            json.dumps(body(row)).replace(f'"prefix_end": {S}', '"prefix_end": NaN'),
        ),
        ("no vllm_xargs", body(row, vllm_xargs=DROP)),
        ("max_tokens omitted (16)", body(row, max_tokens=DROP)),
        ("temperature 0.7", body(row, temperature=0.7)),
        ("top_k 64", body(row, top_k=64)),
        ("logit_bias on a code", body(row, logit_bias={str(row["label_token_ids"][0]): 5.0})),
        ("allowed_token_ids omitted", body(row, allowed_token_ids=DROP)),
        ("prompt_embeds", {**body(row, prompt=DROP), "prompt_embeds": "AAAAAA=="}),
    ]


def run_malformed(srv, row, p0, cases, tol):
    for name, payload in cases:
        st, r = srv.post(payload)
        plugin = name != "prompt_embeds"
        check(
            st == 400 and (not plugin or "strands-decider: " in str(r)),
            f"{name}: HTTP {st} {str(r)[:120]}",
        )
        _, _, p = ask(srv, row)
        check(
            srv.healthy() and p is not None and dmax(p, p0) <= tol,
            f"{name}: the server answers as before",
        )


def scenario_requests(srv, rows, a):
    row = next(x for x in rows if x["prefix_end"] >= 64)  # a state of at least one KV block
    st, _, p0 = ask(srv, row, cache_salt="baseline")
    if not check(st == 200, f"a decision request: HTTP {st}"):
        return {}
    run_malformed(srv, row, p0, malformed(row), a.repeat_tol)
    _, _, p255 = ask(srv, row, allowed_token_ids=DROP, logprobs=255, cache_salt="baseline")
    check(dmax(p255, p0) <= a.repeat_tol, "logprobs 255 without allowed_token_ids: the same answer")

    # vLLM reuses whole blocks of A tokens, A the block size or, when the layers' KV pages differ in
    # size, a multiple of it: an exact repeat all but the last token's, a further question the state's.
    salt = f"cold-{time.time()}"
    _, _, cold = ask(srv, row, cache_salt=salt)
    ask(srv, row)
    _, r, again = ask(srv, row)
    L, B = len(row["token_ids"]), a.block_size
    B = next((A for A in (B, 2 * B, 4 * B) if B and cached(r) == (L - 1) // A * A), B)
    hit = cached(r) == (L - 1) // B * B if B else cached(r) > 0
    check(
        hit and dmax(again, cold) <= a.same_server_tol,
        f"exact repeat: {cached(r)} of {L} tokens cached, |dp| vs cold {dmax(again, cold):.2e}",
    )
    _, r, _ = ask(srv, row, vllm_xargs={"prefix_end": row["prefix_end"] - 1})
    check(cached(r) == 0, f"the same tokens with another S: {cached(r)} tokens cached")
    other = next(x for x in rows if x is not row and x["prefix_end"] < len(x["token_ids"]) - 1)
    S = row["prefix_end"]
    further = {
        **other,
        "token_ids": row["token_ids"][:S] + other["token_ids"][other["prefix_end"] :],
        "prefix_end": S,
    }
    _, r, p = ask(srv, further)
    _, _, p_cold = ask(srv, further, cache_salt=salt + "f")
    hit = cached(r) == S // B * B if B else 0 < cached(r) <= S
    check(
        hit and dmax(p, p_cold) <= a.same_server_tol,
        f"a further question: {cached(r)} of S = {S} tokens cached, |dp| vs cold {dmax(p, p_cold):.2e}",
    )

    group = rows[1:9]
    alone = [ask(srv, x, cache_salt=f"alone{i}")[2] for i, x in enumerate(group)]
    mixed = concurrently(lambda i, x: ask(srv, x, cache_salt=f"mixed{i}")[2], group)
    worst = max(dmax(m, s) for m, s in zip(mixed, alone, strict=True)) if all(mixed) else math.inf
    check(
        worst <= a.same_server_tol,
        f"a mixed batch of {len(group)}: worst |dp| vs alone {worst:.2e}",
    )

    # Every row cold, against the reference where the rows carry its probabilities.
    answers = concurrently(lambda i, x: ask(srv, x, cache_salt=f"answers-{i}")[2], rows[: a.n])
    check(all(answers), f"{sum(map(bool, answers))} of {len(answers)} rows answered")
    out = {"answers": answers}
    if all(answers) and all("probabilities" in x for x in rows[: a.n]):
        dps = [dmax(p, x["probabilities"]) for p, x in zip(answers, rows, strict=False)]
        top1 = sum(
            argmax(p) == argmax(x["probabilities"]) for p, x in zip(answers, rows, strict=False)
        ) / len(dps)
        out["reference"] = {"top1": top1, "mean_dp": sum(dps) / len(dps), "max_dp": max(dps)}
        check(
            top1 >= a.min_top1 and sum(dps) / len(dps) <= a.max_mean_dp and max(dps) <= a.max_dp,
            f"reference: top-1 {top1:.4f}, mean |dp| {sum(dps) / len(dps):.4f}, max |dp| {max(dps):.4f} over {len(dps)} rows",
        )
    check(srv.healthy(), "server healthy at the end")
    return out


def scenario_compare(srv, rows, a):
    expect = json.load(open(a.expect))["answers"]
    row = next(x for x in rows if x["prefix_end"] >= 64)
    _, _, p0 = ask(srv, row, cache_salt="baseline")
    run_malformed(srv, row, p0, malformed(row)[:5], a.repeat_tol)
    worst, flips, near_ties = 0.0, 0, 0
    for i, (x, e) in enumerate(zip(rows, expect, strict=False)):
        st, _, p = ask(srv, x, cache_salt=f"answers-{i}")  # cold, as the other server's
        if not check(st == 200, f"row {i}: HTTP {st}"):
            continue
        worst = max(worst, dmax(p, e))
        if argmax(p) != argmax(e):
            top = sorted(e, reverse=True)
            # A flip where the other server's top two are within --tie-gap is a near-tie.
            if top[0] - top[1] <= a.tie_gap:
                near_ties += 1
            else:
                flips += 1
    check(
        flips == 0 and worst <= a.cross_server_tol,
        f"{len(expect)} rows vs the other server: {flips} top-1 flips ({near_ties} near-ties), worst |dp| {worst:.4f}",
    )
    check(srv.healthy(), "server healthy at the end")
    return {"worst_dp": worst, "flips": flips, "near_ties": near_ties}


def scenario_preempt(srv, rows, a):
    group = [x for x in rows if len(x["token_ids"]) <= 1500][:8]
    statuses = concurrently(lambda i, x: ask(srv, x, cache_salt=f"c{i}-{time.time()}")[0], group)
    check(statuses == [200] * len(group), f"{len(group)} concurrent requests: {statuses}")
    check(srv.healthy(), "server healthy at the end")
    return {}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenario", choices=["requests", "compare", "preempt"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=60, help="rows answered (requests)")
    ap.add_argument("--expect", help="a requests run's --out (compare)")
    ap.add_argument("--block-size", type=int, default=0, help="the KV block size, for exact counts")
    # Tolerances on |dp|, measured on the released checkpoints; tests/gpu/ci.sh sets the tiny ones'.
    ap.add_argument("--repeat-tol", type=float, default=2e-3, help="an answer asked again")
    ap.add_argument(
        "--same-server-tol", type=float, default=0.01, help="cached and batched answers"
    )
    ap.add_argument(
        "--cross-server-tol", type=float, default=0.03, help="two separately started servers"
    )
    ap.add_argument("--tie-gap", type=float, default=0.0)
    ap.add_argument("--min-top1", type=float, default=0.95)
    ap.add_argument("--max-mean-dp", type=float, default=1.0)
    ap.add_argument("--max-dp", type=float, default=1.0)
    a = ap.parse_args()
    rows = [json.loads(line) for line in open(a.rows) if line.strip()]
    scenario = {
        "requests": scenario_requests,
        "compare": scenario_compare,
        "preempt": scenario_preempt,
    }
    out = scenario[a.scenario](Server(a.url), rows, a)
    if a.out:
        json.dump({**out, "failures": FAILURES}, open(a.out, "w"))
    print(f"{a.scenario}: {'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}", flush=True)
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
