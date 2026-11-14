"""GPU integration checks for strands-decider-vllm against a running server. Python standard library only.

Each scenario asserts what it checks and exits non-zero on the first failure; ``--out`` receives a JSON summary.
``--rows`` is a JSON Lines file of rendered requests: ``token_ids``, ``prefix_end`` (S) and ``label_token_ids`` (the
question's code ids), optionally ``probabilities`` (a reference's) and ``n_tokens``.

- ``requests``: every malformed request gets HTTP 400 with the server healthy afterwards and the next valid request
  answered as before; the prefix cache (exact repeat, further question, other S) and mixed batches match cold answers;
  with reference probabilities, top-1 agreement and |dp| are reported and gated.
- ``compare``: the same answers as another server's ``requests`` run (``--expect``), plus a few malformed requests.
- ``preempt``: concurrent requests on a server with a small KV cache all succeed; ``max_tokens`` 300 gets 400.
- ``resume``: for a server whose ``max_tokens`` check is bypassed (``bypass_max_tokens/``): concurrent long
  generations are preempted and resumed without stopping the engine, and their first-token answers match cold ones.

  python client.py requests --url http://127.0.0.1:8100 --rows rows.jsonl --out main.json
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
#: Same-server tolerances on max |dp|: a request answered again (repeat) and a cached or batched answer against a cold
#: one (same). Measured on the released checkpoints; a tiny random-weight model's bf16 noise needs looser ones
#: (tests/gpu/ci.sh passes them).
TOL = {"repeat": 2e-3, "same": 0.01}


def log(*x):
    print(*x, flush=True)


def check(ok, what):
    log(("ok    " if ok else "FAIL  ") + what)
    if not ok:
        FAILURES.append(what)
    return ok


class Server:
    def __init__(self, url):
        self.url = url.rstrip("/")

    def post(self, path, payload, timeout=900):
        data = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
        req = urllib.request.Request(
            self.url + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")[:600]
        except Exception as e:  # connection refused or reset: the server is down
            return None, repr(e)[:300]

    def health(self):
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=10) as r:
                return r.status
        except Exception as e:
            return repr(e)[:120]

    def metric(self, name):
        try:
            text = urllib.request.urlopen(self.url + "/metrics", timeout=10).read().decode()
        except Exception:
            return None
        return sum(
            float(line.split()[-1]) for line in text.splitlines() if line.startswith(name + "{")
        )


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
    for k, v in over.items():
        if v is DROP:
            b.pop(k, None)
        else:
            b[k] = v
    return b


DROP = object()


def probs(resp, codes):
    top = resp["choices"][0]["logprobs"]["top_logprobs"][0]
    lp = {int(k.split(":")[1]): v for k, v in top.items()}
    vals = [lp.get(t, -math.inf) for t in codes]
    m = max(vals)
    e = [math.exp(v - m) for v in vals]
    s = sum(e)
    return [x / s for x in e]


def cached(resp):
    return ((resp.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


def argmax(p):
    return max(range(len(p)), key=p.__getitem__)


def dmax(p, q):
    return max(abs(a - b) for a, b in zip(p, q, strict=True))


def ask(srv, row, **over):
    """(status, response, probabilities or None)."""
    st, r = srv.post("/v1/completions", body(row, **over))
    return st, r, (probs(r, row["label_token_ids"]) if st == 200 else None)


def malformed_cases(row):
    """(name, request body or raw JSON text, expected status, whether the plugin's message is expected)."""
    L, S = len(row["token_ids"]), row["prefix_end"]
    raw_nan = json.dumps(body(row)).replace(f'"prefix_end": {S}', '"prefix_end": NaN')
    return [
        ("prefix_end string '9'", body(row, vllm_xargs={"prefix_end": "9"}), 400, True),
        ("prefix_end false (arrives as 0)", body(row, vllm_xargs={"prefix_end": False}), 400, True),
        ("prefix_end 0", body(row, vllm_xargs={"prefix_end": 0}), 400, True),
        ("prefix_end -3", body(row, vllm_xargs={"prefix_end": -3}), 400, True),
        (f"prefix_end = L = {L}", body(row, vllm_xargs={"prefix_end": L}), 400, True),
        (f"prefix_end = L + 5 = {L + 5}", body(row, vllm_xargs={"prefix_end": L + 5}), 400, True),
        ("prefix_end 9.5", body(row, vllm_xargs={"prefix_end": 9.5}), 400, True),
        (f"prefix_end {S}.0 (a float)", body(row, vllm_xargs={"prefix_end": float(S)}), 400, True),
        ("prefix_end [9]", body(row, vllm_xargs={"prefix_end": [9]}), 400, True),
        ("prefix_end NaN", raw_nan, 400, True),
        ("no vllm_xargs", body(row, vllm_xargs=DROP), 400, True),
        ("vllm_xargs without prefix_end", body(row, vllm_xargs={"other": 1}), 400, True),
        ("max_tokens omitted (16)", body(row, max_tokens=DROP), 400, True),
        ("max_tokens 300", body(row, max_tokens=300), 400, True),
        ("temperature 0.7", body(row, temperature=0.7), 400, True),
        ("temperature 0", body(row, temperature=0.0), 400, True),
        ("top_k 64", body(row, top_k=64), 400, True),
        ("top_p 0.9", body(row, top_p=0.9), 400, True),
        ("min_p 0.1", body(row, min_p=0.1), 400, True),
        ("repetition_penalty 1.2", body(row, repetition_penalty=1.2), 400, True),
        (
            "logit_bias on a code",
            body(row, logit_bias={str(row["label_token_ids"][0]): 5.0}),
            400,
            True,
        ),
        ("allowed_token_ids omitted, logprobs k", body(row, allowed_token_ids=DROP), 400, True),
        # vLLM refuses prompt embeddings itself unless --enable-prompt-embeds, which the plugin refuses at startup.
        (
            "prompt_embeds",
            {**body(row, prompt=DROP), "prompt_embeds": "AAAAAA=="},
            400,
            False,
        ),
    ]


def run_malformed(srv, row, p0, cases):
    for name, payload, expected, plugin_message in cases:
        st, r = srv.post("/v1/completions", payload)
        text = r if isinstance(r, str) else json.dumps(r)
        ok = st == expected and (not plugin_message or "strands-decider: " in text)
        check(ok, f"{name}: HTTP {st} (want {expected}) {text[:160] if st != 200 else ''}")
        check(srv.health() == 200, f"{name}: server healthy afterwards")
        st2, _, p = ask(srv, row)
        d = dmax(p, p0) if st2 == 200 and p is not None else None
        check(
            d is not None and d <= TOL["repeat"],
            f"{name}: the next valid request is answered as before (|dp| {d if d is None else f'{d:.2e}'})",
        )


def base_row(rows):
    """A row whose state fills at least one KV block, so that a further question has something to reuse."""
    return next(x for x in rows if x["prefix_end"] >= 64)


def scenario_requests(srv, rows, a):
    out = {}
    row = base_row(rows)
    st, _, p0 = ask(srv, row, cache_salt="baseline")
    if not check(st == 200, f"baseline decision request: HTTP {st}"):
        return out
    run_malformed(srv, row, p0, malformed_cases(row))

    # JSON true arrives as 1 (vLLM's request model), so it is served as S = 1, as documented.
    st_t, _, p_t = ask(srv, row, vllm_xargs={"prefix_end": True}, cache_salt="true")
    st_1, _, p_1 = ask(srv, row, vllm_xargs={"prefix_end": 1}, cache_salt="one")
    check(
        st_t == 200 and st_1 == 200 and dmax(p_t, p_1) < 1e-6, "prefix_end true is served as S = 1"
    )

    # Without allowed_token_ids, logprobs 255 returns every code; renormalized, the same answer.
    st_a, _, p_a = ask(srv, row, allowed_token_ids=DROP, logprobs=255, cache_salt="baseline")
    d_a = dmax(p_a, p0) if st_a == 200 else float("inf")
    check(
        d_a <= TOL["repeat"],
        f"logprobs 255 without allowed_token_ids gives the same answer (|dp| {d_a:.2e})",
    )

    # The prefix cache: an exact repeat, a further question on the same state, another S, against cold answers.
    salt = f"cold-{time.time()}"
    _, _, cold = ask(srv, row, cache_salt=salt + "a")
    ask(srv, row)
    _, rr, rep = ask(srv, row)
    L, B = len(row["token_ids"]), a.block_size
    # vLLM reuses every whole block of A tokens but the last token's: floor((L - 1) / A) A. A is the block size, or a
    # multiple of it when the layers' KV pages differ in size (vLLM aligns cache hits to their least common multiple).
    if B:
        B = next((A for A in (B, 2 * B, 4 * B) if cached(rr) == (L - 1) // A * A), B)
    want = (L - 1) // B * B if B else None
    check(
        (cached(rr) == want if B else bool(cached(rr)))
        and dmax(rep, cold) <= TOL["same"]
        and argmax(rep) == argmax(cold),
        f"exact repeat: {cached(rr)} of {L} tokens cached (want {want}), max|dp| vs cold {dmax(rep, cold):.2e}",
    )
    _, ro, _ = ask(srv, row, vllm_xargs={"prefix_end": row["prefix_end"] - 1})
    check(cached(ro) == 0, f"same tokens with another S: {cached(ro)} tokens cached (want 0)")
    other = next(x for x in rows if x is not row and x["prefix_end"] < len(x["token_ids"]) - 1)
    further = dict(
        other,
        token_ids=row["token_ids"][: row["prefix_end"]] + other["token_ids"][other["prefix_end"] :],
        prefix_end=row["prefix_end"],
    )
    _, rf, pf = ask(srv, further)
    _, _, pfc = ask(srv, further, cache_salt=salt + "f")
    S = further["prefix_end"]
    want = S // B * B if B else None  # the state's full blocks
    check(
        (cached(rf) == want if B else 0 < (cached(rf) or 0) <= S) and dmax(pf, pfc) <= TOL["same"],
        f"further question: {cached(rf)} of S = {S} tokens cached (want {want}), max|dp| vs cold {dmax(pf, pfc):.2e}",
    )

    # Mixed batches: decision requests of different lengths at once, each against its own answer alone.
    group = rows[1:9]
    solo = [ask(srv, x, cache_salt=f"solo{i}")[2] for i, x in enumerate(group)]
    res = {}

    def one(i, x):
        res[i] = ask(srv, x, cache_salt=f"mix{i}")

    threads = [threading.Thread(target=one, args=(i, x)) for i, x in enumerate(group)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    worst = max(dmax(res[i][2], solo[i]) for i in range(len(group)) if res[i][0] == 200)
    check(
        all(res[i][0] == 200 for i in range(len(group))) and worst <= TOL["same"],
        f"mixed batch of {len(group)}: all answered, worst max|dp| vs alone {worst:.2e}",
    )

    # Answers for every row (and, with reference probabilities, agreement with the reference).
    answers, agree, dps = {}, 0, []
    lock = threading.Lock()
    todo = list(enumerate(rows[: a.n]))

    def worker():
        while True:
            with lock:
                if not todo:
                    return
                i, x = todo.pop()
            st, _, p = ask(srv, x, cache_salt=f"answers-{i}")  # cold: a salt per row
            with lock:
                answers[i] = p if st == 200 else None

    threads = [threading.Thread(target=worker) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    n_ok = sum(p is not None for p in answers.values())
    check(n_ok == min(a.n, len(rows)), f"{n_ok} of {min(a.n, len(rows))} rows answered")
    if all("probabilities" in x for x in rows[: a.n]):
        for i, p in answers.items():
            if p is None:
                continue
            ref = rows[i]["probabilities"]
            agree += argmax(p) == argmax(ref)
            dps.append(dmax(p, ref))
        top1 = agree / max(1, len(dps))
        out["reference"] = {
            "n": len(dps),
            "top1": top1,
            "mean_dp": sum(dps) / len(dps),
            "max_dp": max(dps),
        }
        check(
            top1 >= a.min_top1
            and out["reference"]["mean_dp"] <= a.max_mean_dp
            and out["reference"]["max_dp"] <= a.max_dp,
            f"reference: top-1 {top1:.4f} (gate {a.min_top1}), mean|dp| "
            f"{out['reference']['mean_dp']:.4f} (gate {a.max_mean_dp}), max|dp| {out['reference']['max_dp']:.4f} "
            f"(gate {a.max_dp}) over {len(dps)} rows",
        )
    out["answers"] = [answers.get(i) for i in range(min(a.n, len(rows)))]
    check(srv.health() == 200, "server healthy at the end")
    return out


# Two separately started servers differ by bf16 numerics from run to run: up to 0.023 max|dp| on
# one question between two starts of identical code. Same-server checks keep 0.01.
CROSS_SERVER_MAX_DP = 0.03


def scenario_compare(srv, rows, a):
    expect = json.load(open(a.expect))["answers"]
    row = base_row(rows)
    st, _, p0 = ask(srv, row, cache_salt="baseline")
    check(st == 200, f"baseline decision request: HTTP {st}")
    run_malformed(
        srv,
        row,
        p0,
        [
            c
            for c in malformed_cases(row)
            if c[0].startswith(
                ("prefix_end string", "prefix_end = L ", "max_tokens 300", "no vllm_xargs")
            )
        ],
    )
    worst, flips, tie_flips = 0.0, 0, 0
    for i, x in enumerate(rows[: len(expect)]):
        st, _, p = ask(srv, x, cache_salt=f"answers-{i}")  # cold, as the other server's answers
        if not check(st == 200, f"row {i}: HTTP {st}") or expect[i] is None:
            continue
        worst = max(worst, dmax(p, expect[i]))
        if argmax(p) != argmax(expect[i]):
            top = sorted(expect[i], reverse=True)
            # --tie-gap: a flip where the other server's top two are that close is a near-tie, not a flip.
            gap = top[0] - top[1] if len(top) > 1 else 1.0
            log(f"      row {i}: top-1 flip; the other server's top-2 gap {gap:.4f}")
            if gap <= a.tie_gap:
                tie_flips += 1
            else:
                flips += 1
    check(
        flips == 0 and worst <= CROSS_SERVER_MAX_DP,
        f"{len(expect)} rows vs the other server: {flips} top-1 flips"
        + (f" (and {tie_flips} at near-ties within {a.tie_gap})" if tie_flips else "")
        + f", worst max|dp| {worst:.4f}",
    )
    check(srv.health() == 200, "server healthy at the end")
    return {"worst_dp": worst, "flips": flips, "near_tie_flips": tie_flips}


def concurrent(srv, group, **over):
    res = {}

    def one(i, x):
        res[i] = ask(srv, x, **{"cache_salt": f"c{i}-{time.time()}", **over})

    threads = [threading.Thread(target=one, args=(i, x)) for i, x in enumerate(group)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [res[i] for i in range(len(group))]


def scenario_preempt(srv, rows, a):
    group = [x for x in rows if len(x["token_ids"]) <= 1500][:8]
    p0 = srv.metric("vllm:num_preemptions_total")
    res = concurrent(srv, group)
    statuses = [st for st, _, _ in res]
    check(
        statuses == [200] * len(group),
        f"{len(group)} concurrent decision requests, max_tokens 1: statuses {statuses}",
    )
    st, r, _ = ask(srv, group[0], max_tokens=300)
    check(st == 400 and "max_tokens" in str(r), f"max_tokens 300: HTTP {st}")
    check(srv.health() == 200, "server healthy at the end")
    return {"preemptions": (srv.metric("vllm:num_preemptions_total") or 0) - (p0 or 0)}


def scenario_resume(srv, rows, a):
    group = [x for x in rows if len(x["token_ids"]) <= 1500][:8]
    solo = [ask(srv, x, cache_salt=f"solo{i}")[2] for i, x in enumerate(group)]
    p0 = srv.metric("vllm:num_preemptions_total") or 0
    res = concurrent(srv, group, max_tokens=300, allowed_token_ids=DROP, logprobs=255)
    statuses = [st for st, _, _ in res]
    preempted = (srv.metric("vllm:num_preemptions_total") or 0) - p0
    check(
        statuses == [200] * len(group),
        f"{len(group)} concurrent decision generations of 300 tokens: statuses {statuses}",
    )
    check(
        preempted > 0,
        f"preemptions during the run: {preempted:.0f} (want > 0, so that requests were resumed)",
    )
    worst = 0.0
    for (st, r, _), x, p in zip(res, group, solo, strict=True):
        if st == 200:
            worst = max(worst, dmax(probs(r, x["label_token_ids"]), p))
    check(worst <= TOL["same"], f"first-token answers vs cold: worst max|dp| {worst:.4f}")
    check(srv.health() == 200, "server healthy at the end")
    return {"preemptions": preempted, "worst_dp": worst}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scenario", choices=["requests", "compare", "preempt", "resume"])
    ap.add_argument("--url", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--n", type=int, default=60, help="rows answered (requests)")
    ap.add_argument(
        "--min-top1", type=float, default=0.95, help="gate on reference agreement (requests)"
    )
    ap.add_argument("--expect", help="a requests run's --out (compare)")
    ap.add_argument(
        "--repeat-tol", type=float, default=TOL["repeat"], help="max |dp| of an answer asked again"
    )
    ap.add_argument(
        "--same-server-tol",
        type=float,
        default=TOL["same"],
        help="max |dp| of cached and batched answers",
    )
    ap.add_argument(
        "--cross-server-tol", type=float, default=None, help="compare: max |dp| (default 0.03)"
    )
    ap.add_argument(
        "--tie-gap",
        type=float,
        default=0.0,
        help="compare: a top-1 flip where the other server's top two are within this gap is not counted",
    )
    ap.add_argument(
        "--max-mean-dp", type=float, default=1.0, help="requests: the gate on mean |dp|"
    )
    ap.add_argument("--max-dp", type=float, default=1.0, help="requests: the gate on max |dp|")
    ap.add_argument(
        "--block-size",
        type=int,
        default=0,
        help="the server's KV block size: the cache checks then want exact counts of cached tokens",
    )
    ap.add_argument("--out")
    a = ap.parse_args()
    TOL.update(repeat=a.repeat_tol, same=a.same_server_tol)
    if a.cross_server_tol is not None:
        global CROSS_SERVER_MAX_DP
        CROSS_SERVER_MAX_DP = a.cross_server_tol
    rows = [json.loads(line) for line in open(a.rows) if line.strip()]
    srv = Server(a.url)
    fn = {
        "requests": scenario_requests,
        "compare": scenario_compare,
        "preempt": scenario_preempt,
        "resume": scenario_resume,
    }[a.scenario]
    out = fn(srv, rows, a)
    out["failures"] = FAILURES
    if a.out:
        json.dump(out, open(a.out, "w"))
    log(f"{a.scenario}: {'PASS' if not FAILURES else f'FAIL ({len(FAILURES)})'}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
