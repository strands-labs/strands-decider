#!/usr/bin/env python3
r"""Benchmark Strands deciders: time to first token (TTFT), throughput, JevBench and the Decision Index.

Run it where you want the client to be. It measures whatever --model points at and never starts anything elsewhere.

  # a local checkpoint, served here with `strands-decider serve`
  python decider_bench.py --model ./checkpoints/my-2b --device cuda

  # a deployed Lambda function, at three lengths and three loads, plus JevBench
  python decider_bench.py --model lambda:my-decider:live --metrics ttft,jevbench \
      --lengths 256,1024,4096 --concurrency 1,8,32

  # two models side by side, on your own requests, under Poisson traffic
  python decider_bench.py --model base=StrandsAgents/strands-decider-2B-qwen3.5-v1-2610 --model mine=./ckpt \
      --dataset my_requests.jsonl --rps 1,4,16

  # combine runs made on different hosts into one set of figures
  python decider_bench.py plot runs/g6e runs/p4d runs/lambda

What to benchmark (--model, repeatable; prefix NAME= to name it in tables and plots):
  ./path or ORG/NAME     a checkpoint directory or Hub id, served on this host by `strands-decider serve`
                         (--device picks the device; --serve-cmd runs any other server instead)
  http(s)://HOST:PORT    a server that is already running and speaks POST /v1/systemone
  lambda:FUNCTION[:ALIAS][@REGION]     a Lambda function
  sagemaker:ENDPOINT[@REGION]          a SageMaker endpoint

What to measure (--metrics, default ttft):
  ttft            time to first token. A decider decodes nothing, so this is its whole latency. Measured for every
                  input and every load below, with the decisions per second achieved at that load.
  jevbench        the 231 public JevBench tasks, scored by JevBench's own harness
  decision-index  the public Decision Index, scored by its own kit; needs --di-suite (see the kit's README)
JevBench and the Decision Index always use their own inputs. The input and load flags apply to ttft only.

Inputs for ttft (either or both; each is reported separately; default --lengths 256,1024,4096):
  --lengths 256,1024     synthetic requests trimmed to exactly that many prompt tokens
  --dataset FILE         your own requests, one JSON object per line: {"state": ..., "questions": {...}}

Load for ttft (either or both; default --concurrency 1, one request at a time):
  --concurrency 1,8,32   keep N requests in flight: each simulated user sends its next request when its answer
                         arrives. The highest N shows the maximum throughput.
  --rps 2,8              send N requests per second at random (Poisson) times, whether or not answers keep up,
                         as independent users do. TTFT then includes any queueing.

Each input and load is measured over --num-requests requests (default 50), after a warm-up that is not counted.
Synthetic requests start with a random ticket id, so no prefix cache can hit.

Where each model ran is recorded automatically: this host's EC2 instance type (or CPU and GPU) for a local
checkpoint, and the function's memory or the endpoint's instance type for Lambda and SageMaker.

Output (--out, default runs/<time>): latency.csv and accuracy.csv (one row per measurement, for your own
analysis), summary.json, requests.jsonl (every timed request), and the figures:
  ttft_vs_length.png       TTFT against prompt length, one line per model and load
  ttft_vs_throughput.png   p95 TTFT against decisions per second, one panel per input
  throughput_vs_length.png the highest decisions per second measured against prompt length (per GPU on GPU hosts)
  accuracy.png             JevBench and the Decision Index
`plot RUN_DIR ...` draws the same figures over several runs, with one column per host.
"""

import argparse
import base64
import http.client
import json
import os
import platform
import random
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

JEVBENCH_REPO = "https://github.com/fstandhartinger/jevbench"
JEVBENCH_COMMIT = "1bcc55eb6c8cffde2306b3db03ede39b61c6152a"  # pinned by strands-decider's evaluation/jevbench/jevbench.sh
METRICS = ("ttft", "jevbench", "decision-index")

SECTION = (
    "Section {n}. Refunds and payout disputes. A customer may request a refund within 30 days of the charge "
    "date. Payouts that fail because of an invalid bank account are retried twice, 24 hours apart, before the "
    "account is flagged for manual review. Support agents must confirm the account holder's identity before "
    "changing payout details, and must escalate any dispute above $5,000 to the finance team. Where a policy "
    "in this section conflicts with a regional addendum, the addendum takes precedence for customers "
    "domiciled in that region. "
)
TAIL = "\nCustomer message: Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "route": {
        "type": "choice",
        "instructions": "Which team should handle this ticket?",
        "criteria": {
            "billing": "payments, invoices, payouts",
            "technical": "bugs, outages, API errors",
            "sales": "pricing and upgrades",
            "trust": "fraud, identity, account security",
        },
    }
}
DOC = "".join(
    SECTION.format(n=n) for n in range(1, 400)
)  # about 210k characters, beyond any window


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---------------------------------------------------------------------------------------------------- HTTP client
_tls = threading.local()


def post(url, body):
    """POST body to url/v1/systemone on a per-thread keep-alive connection: (status, parsed JSON or None, client ms)."""
    u = urllib.parse.urlsplit(url)
    conns = _tls.__dict__.setdefault("conns", {})
    for attempt in (0, 1):
        c = conns.get(u.netloc)
        if c is None:
            cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
            c = conns[u.netloc] = cls(u.netloc, timeout=900)
        try:
            t = time.perf_counter()
            c.request(
                "POST",
                u.path.rstrip("/") + "/v1/systemone",
                body,
                {"Content-Type": "application/json"},
            )
            r = c.getresponse()
            data = r.read()
            ms = (time.perf_counter() - t) * 1000
            try:
                return r.status, json.loads(data), ms
            except ValueError:
                return r.status, {"detail": data[:500].decode(errors="replace")}, ms
        except (http.client.HTTPException, OSError):
            c.close()
            conns.pop(u.netloc, None)
            if attempt:
                raise


def get_json(url, timeout=5):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


# ---------------------------------------------------------------------------------------------------- models
class Shim:
    """A local /v1/systemone front for Lambda and SageMaker. `call(body) -> (status, dict)` does the invoke."""

    def __init__(self, call, health):
        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj):
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self):
                self._send(200, health) if self.path.startswith("/health") else self._send(
                    404, {"detail": "not found"}
                )

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                t = time.perf_counter()
                try:
                    code, obj = call(body)
                except (
                    Exception
                ) as e:  # transport failures reach the client as 502s, which the metrics count
                    code, obj = 502, {"detail": f"{type(e).__name__}: {e}"[:2000]}
                if code == 200:
                    obj["x_roundtrip_ms"] = round((time.perf_counter() - t) * 1000, 2)
                self._send(code, obj)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", free_port()), H)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def lambda_call(function, qualifier, region):
    import boto3
    from botocore.config import Config

    lam = boto3.client(
        "lambda",
        region_name=region,
        config=Config(
            read_timeout=900,
            connect_timeout=10,
            retries={"max_attempts": 4, "mode": "standard"},
            max_pool_connections=512,
        ),
    )

    def call(body):
        kw = {
            "FunctionName": function,
            "Payload": body,
            **({"Qualifier": qualifier} if qualifier else {}),
        }
        r = lam.invoke(**kw)
        p = json.loads(r["Payload"].read() or b"null")
        if r.get("FunctionError"):
            err = p.get("errorType") if isinstance(p, dict) else None
            return (422 if err in ("ValueError", "ValidationError") else 502), {
                "detail": str(p.get("errorMessage", p))[:2000]
            }
        if (
            isinstance(p, dict) and "statusCode" in p
        ):  # a handler that answers in the Function URL shape
            b = p.get("body") or "{}"
            return int(p["statusCode"]), json.loads(
                base64.b64decode(b) if p.get("isBase64Encoded") else b
            )
        timing = p.pop("timing", None) or {}
        if "engine_s" in timing:
            p.setdefault("latency_ms", round(1000 * timing["engine_s"], 2))
        if timing.get("cpu"):
            p["x_cpu"] = timing["cpu"]
        return 200, p

    return call


def sagemaker_call(endpoint, region):
    import boto3
    from botocore.config import Config

    sm = boto3.client(
        "sagemaker-runtime",
        region_name=region,
        config=Config(
            read_timeout=900,
            connect_timeout=10,
            retries={"max_attempts": 4, "mode": "standard"},
            max_pool_connections=512,
        ),
    )

    def call(body):
        try:
            r = sm.invoke_endpoint(
                EndpointName=endpoint,
                ContentType="application/json",
                Accept="application/json",
                Body=body,
            )
        except sm.exceptions.ModelError as e:  # the container answered with an error status
            return int(e.response.get("OriginalStatusCode", 502)), {
                "detail": e.response.get("OriginalMessage", str(e))[:2000]
            }
        return 200, json.loads(r["Body"].read())

    return call


class Model:
    def __init__(self, arg, args, out):
        m = re.match(r"^([\w./+-]+)=(.+)$", arg)
        self.label, self.spec = (
            (m.group(1), m.group(2))
            if m and not arg.startswith(("http:", "https:"))
            else (None, arg)
        )
        s = self.spec
        self.kind = (
            "http"
            if s.startswith(("http://", "https://"))
            else "lambda"
            if s.startswith("lambda:")
            else "sagemaker"
            if s.startswith("sagemaker:")
            else "local"
        )
        if not self.label:
            if self.kind in ("lambda", "sagemaker"):
                self.label = s.split(":", 1)[1].split("@")[0].replace(":", "-")
            elif self.kind == "http":
                self.label = urllib.parse.urlsplit(s).netloc
            else:
                self.label = s.rstrip("/").split("/")[-1]
            self.label = re.sub(r"[^\w.+-]+", "_", self.label)
        self.args, self.dir = args, out / self.label.replace("/", "__")
        self.proc = self.shim = None
        self.url = None
        self.health = {}

    def start(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        a = self.args
        if self.kind == "http":
            self.url = re.sub(r"/v1/systemone/?$", "", self.spec.rstrip("/"))
        elif self.kind in ("lambda", "sagemaker"):
            target, _, region = self.spec.split(":", 1)[1].partition("@")
            region = region or None  # None: boto3's own default (AWS_REGION, ~/.aws/config)
            if self.kind == "lambda":
                function, _, qualifier = target.partition(":")
                call = lambda_call(function, qualifier or None, region)
            else:
                call = sagemaker_call(target, region)
            self.shim = Shim(
                call, {"status": "ok", "model": self.label, "device": self.spec, "region": region}
            )
            self.url = self.shim.url
        else:
            port = free_port()
            if a.serve_cmd:
                cmd = shlex.split(
                    a.serve_cmd.format(model=self.spec, port=port, device=a.device or "")
                )
            else:
                exe = shutil.which("strands-decider")
                cmd = ([exe] if exe else [sys.executable, "-m", "strands_decider.cli"]) + [
                    "serve",
                    self.spec,
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ]
                cmd += (["--device", a.device] if a.device else []) + shlex.split(
                    a.serve_args or ""
                )
            log(f"{self.label}: starting {shlex.join(cmd)}")
            self.server_log = open(self.dir / "server.log", "ab")
            self.proc = subprocess.Popen(
                cmd, stdout=self.server_log, stderr=subprocess.STDOUT, start_new_session=True
            )
            self.url = f"http://127.0.0.1:{port}"
            t0 = time.time()
            while True:
                if self.proc.poll() is not None:
                    raise SystemExit(
                        f"{self.label}: server exited with {self.proc.returncode}; see {self.dir / 'server.log'}"
                    )
                try:
                    get_json(self.url + "/health")
                    break
                except OSError:
                    if time.time() - t0 > a.health_timeout:
                        raise SystemExit(
                            f"{self.label}: /health not up after {a.health_timeout} s"
                        ) from None
                    time.sleep(2)
            self.load_s = round(time.time() - t0, 1)
            log(f"{self.label}: up in {self.load_s} s")
        try:
            self.health = get_json(self.url + "/health")
        except (OSError, ValueError):
            self.health = {}
        return self

    def stop(self):
        if self.shim:
            self.shim.close()
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)

    def host(self):
        """Where the model ran, found rather than typed."""
        if self.kind == "local" or (
            self.kind == "http"
            and urllib.parse.urlsplit(self.url or self.spec).hostname in ("127.0.0.1", "localhost")
        ):
            return this_host()
        if self.kind == "http":
            return urllib.parse.urlsplit(self.spec).netloc
        target, _, region = self.spec.split(":", 1)[1].partition("@")
        try:
            import boto3

            if self.kind == "lambda":
                function, _, qualifier = target.partition(":")
                c = boto3.client("lambda", region_name=region or None).get_function_configuration(
                    FunctionName=function, **({"Qualifier": qualifier} if qualifier else {})
                )
                return f"Lambda {c.get('Architectures', ['x86_64'])[0]} {c['MemorySize']:,} MB"
            sm = boto3.client("sagemaker", region_name=region or None)
            cfg = sm.describe_endpoint_config(
                EndpointConfigName=sm.describe_endpoint(EndpointName=target)["EndpointConfigName"]
            )
            return f"SageMaker {cfg['ProductionVariants'][0].get('InstanceType', 'serverless')}"
        except Exception:  # no describe permission: the kind is still informative
            return "Lambda" if self.kind == "lambda" else "SageMaker"

    def gpus(self):
        """GPUs the local server can see (CUDA_VISIBLE_DEVICES, else all), for per-GPU throughput; None if unknown."""
        if self.kind != "local":
            return None
        if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
            return len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
        if shutil.which("nvidia-smi"):
            out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
            return sum(1 for line in out.splitlines() if line.startswith("GPU")) or None
        return None

    def describe(self):
        return {
            "label": self.label,
            "spec": self.spec,
            "kind": self.kind,
            "url": self.url,
            "health": self.health,
            "host": self.host(),
            "gpus": self.gpus(),
            **({"load_s": self.load_s} if hasattr(self, "load_s") else {}),
        }


# ---------------------------------------------------------------------------------------------------- requests
class Prompts:
    """Synthetic latency requests, trimmed until the server reports each target length."""

    def __init__(self, url, seed, max_length=None):
        self.url, self.rng, self.lock, self.max_length = (
            url,
            random.Random(seed),
            threading.Lock(),
            max_length,
        )
        self.chars, self.tokens = {}, {}
        self.ratio = 0.25  # tokens per character, refined by every probe

    def body(self, chars):
        with self.lock:
            nonce = self.rng.getrandbits(64)
        state = f"Ticket {nonce:016x}. " + (DOC[:chars].rsplit(" ", 1)[0] if chars else "") + TAIL
        return json.dumps({"state": state, "questions": QUESTIONS}).encode()

    def probe(self, chars):
        status, p, _ = post(self.url, self.body(chars))
        if status != 200:
            return status, None
        return status, (p.get("usage") or {}).get("input_tokens") or None

    def calibrate(self, n):
        """Find the document length that gives n prompt tokens. The probes also warm the server up at this shape."""
        if self.max_length and n > self.max_length:
            self.chars[n], self.tokens[n] = None, None
            log(f"  {n} tokens: skipped, longer than the model's {self.max_length}-token window")
            return
        if not hasattr(self, "base"):
            status, self.base = self.probe(0)
            if status != 200:
                raise SystemExit(f"the first request failed with HTTP {status}")
            if self.base is None:
                log(
                    "the server reports no usage.input_tokens; lengths are approximate (4 characters a token)"
                )
        if self.base is None:
            self.chars[n], self.tokens[n] = 4 * n, None
            self.probe(self.chars[n])
            return
        chars, got = max(0, int((n - self.base) / self.ratio)), None
        for _ in range(5):
            status, got = self.probe(chars)
            if status == 422:
                self.chars[n], self.tokens[n] = None, None
                log(f"  {n} tokens: refused (HTTP 422), longer than the model's window")
                return
            if status != 200 or got is None:
                raise SystemExit(f"calibration at {n} tokens failed with HTTP {status}")
            if abs(got - n) <= max(2, 0.005 * n) or got <= self.base:
                break
            self.ratio = (got - self.base) / max(chars, 1)
            new = max(0, int((n - self.base) / self.ratio))
            if new == chars:
                break
            chars = new
        self.probe(chars)  # one more untimed request at the final shape
        self.chars[n], self.tokens[n] = chars, got
        if abs(got - n) > max(2, 0.005 * n):
            log(f"  {n} tokens: got {got}; the server may truncate to its window")


def timed(url, body):
    t = time.time()
    try:
        status, p, ms = post(url, body)
    except Exception as e:  # a dropped connection is a failed request, not a crashed benchmark
        return {
            "t": round(t, 3),
            "status": 0,
            "client_ms": round((time.time() - t) * 1000, 2),
            "error": f"{type(e).__name__}: {e}"[:300],
        }
    rec = {"t": round(t, 3), "status": status, "client_ms": round(ms, 2)}
    if status == 200:
        rec.update(
            server_ms=p.get("latency_ms"),
            roundtrip_ms=p.get("x_roundtrip_ms"),
            tokens=(p.get("usage") or {}).get("input_tokens"),
            cpu=p.get("x_cpu"),
        )
    else:
        rec["error"] = str(p.get("detail", p))[:300] if isinstance(p, dict) else str(p)[:300]
    return rec


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q / 100 * len(xs)))] if xs else None


def stats(recs):
    ok = [r for r in recs if r["status"] == 200]
    c = [r["client_ms"] for r in ok]
    s = [r["server_ms"] for r in ok if r.get("server_ms") is not None]
    out = {
        "n": len(recs),
        "ok": len(ok),
        "refused": sum(r["status"] == 422 for r in recs),
        "errors": sum(r["status"] not in (200, 422) for r in recs),
    }
    if c:
        out.update(
            p10_ms=round(pct(c, 10), 1),
            p50_ms=round(statistics.median(c), 1),
            p90_ms=round(pct(c, 90), 1),
            p95_ms=round(pct(c, 95), 1),
            p99_ms=round(pct(c, 99), 1),
            mean_ms=round(statistics.fmean(c), 1),
        )
    if s:
        out["server_p50_ms"] = round(statistics.median(s), 1)
    toks = [r["tokens"] for r in ok if r.get("tokens")]
    if toks:
        out["tokens_mean"] = round(statistics.fmean(toks), 1)
    cpus = sorted({r["cpu"] for r in ok if r.get("cpu")})
    if cpus:
        out["cpu"] = {k: sum(r.get("cpu") == k for r in ok) for k in cpus}
    return out


# ---------------------------------------------------------------------------------------------------- metrics
class FileWorkload:
    """Real requests from a JSONL(.gz) file: SystemOneRequests ({"state", "questions"}), or rows with a single
    "question" object ({"state", "question"}), each sent as a one-question request."""

    def __init__(self, path, seed):
        import gzip

        opener = gzip.open if str(path).endswith(".gz") else open
        rows = [json.loads(line) for line in opener(path, "rt") if line.strip()]
        self.name = Path(path).name.split(".")[0]
        self.bodies = [
            json.dumps(
                r
                if "questions" in r
                else {"state": r["state"], "questions": {"q": one_question(r["question"])}}
            ).encode()
            for r in rows
        ]
        random.Random(seed).shuffle(self.bodies)
        self.i, self.lock = 0, threading.Lock()

    def next(self):
        with self.lock:
            b = self.bodies[self.i % len(self.bodies)]
            self.i += 1
            return b


def one_question(q):
    out = {"type": q["type"], "instructions": q.get("instructions")}
    if q["type"] == "noul":
        c = {
            k: v
            for k, v in (q.get("criteria") or {}).items()
            if k in ("true", "false") and v not in (None, "")
        }
        if c:
            out["criteria"] = c
    else:
        out["criteria"] = q["criteria"]
    return out


def closed_loop(url, next_body, c, n):
    """c clients send back to back. Each client's first request is a warm-up (on Lambda it may cold-start a
    sandbox); measurement covers the first n requests sent after every client has finished its warm-up."""
    lock, enough, meas, dones, st = (
        threading.Lock(),
        threading.Event(),
        [],
        [],
        {"warm": 0, "t0": None},
    )

    def client():
        timed(url, next_body())
        with lock:
            st["warm"] += 1
            if st["warm"] == c:
                st["t0"] = time.time()
        while not enough.is_set():
            r = timed(url, next_body())
            r["done"] = r["t"] + r["client_ms"] / 1000
            with lock:
                if r["status"] == 200:
                    dones.append(r["done"])
                if st["t0"] is not None and r["t"] >= st["t0"] and len(meas) < n:
                    meas.append(r)
                    if len(meas) >= n:
                        enough.set()

    threads = [threading.Thread(target=client, daemon=True) for _ in range(c)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    # throughput: every completion between the first and last measured one, so the ramp after warm-up is excluded
    if len(meas) < 2:
        return meas, None
    a, b = min(r["done"] for r in meas), max(r["done"] for r in meas)
    return meas, sum(a < d <= b for d in dones) / (b - a) if b > a else None


def open_loop(url, next_body, rate, burstiness, n, seed, max_inflight):
    """Arrivals with gamma-distributed gaps (burstiness 1 = Poisson, lower = burstier, as in vLLM's bench serve).
    Latency runs from each request's scheduled send time, so a server, or a client pool, that falls behind shows it."""
    rng = random.Random(seed)
    n_warm = max(2, n // 10)
    t, sched = 0.0, []
    for _ in range(n_warm + n):
        t += rng.gammavariate(burstiness, 1 / (rate * burstiness))
        sched.append(t)
    t0 = time.time() + 0.5
    recs = [None] * len(sched)

    def fire(i, at):
        r = timed(url, next_body())
        r["service_ms"] = r["client_ms"]
        r["done"] = r["t"] + r["client_ms"] / 1000
        r["client_ms"] = round((r["done"] - (t0 + at)) * 1000, 2)
        recs[i] = r

    with ThreadPoolExecutor(max_workers=max_inflight) as pool:
        for i, at in enumerate(sched):
            delay = t0 + at - time.time()
            if delay > 0:
                time.sleep(delay)
            pool.submit(fire, i, at)
    meas = recs[n_warm:]
    window = max(r["done"] for r in meas) - (t0 + sched[n_warm])
    return meas, (sum(r["status"] == 200 for r in meas) / window if window > 0 else None)


def ttft(m, prompts, args, sink):
    """TTFT, and the decisions per second achieved, for every input and every load."""
    inputs = []
    for n in args.lengths:
        prompts.calibrate(n)
        if prompts.chars[n] is None:
            inputs.append((f"{n} tokens", n, None))
        else:
            inputs.append((f"{n} tokens", n, lambda chars=prompts.chars[n]: prompts.body(chars)))
    for path in args.dataset:
        w = FileWorkload(path, f"{args.seed}:{m.label}")
        need = (len(args.concurrency) + len(args.rps)) * args.num_requests
        if need > len(w.bodies):
            log(
                f"  {w.name}: {need} requests from {len(w.bodies)} rows, so rows repeat and a prefix cache may hit"
            )
        inputs.append((w.name, None, w.next))
    loads = [
        {"load": f"{c} in flight", "mode": "concurrency", "concurrency": c}
        for c in args.concurrency
    ] + [
        {"load": f"{r:g} req/s", "mode": "rps", "rps": r, "burstiness": args.burstiness}
        for r in args.rps
    ]
    small = [c for c in args.concurrency if args.num_requests < 2 * c]
    if small:
        log(
            f"  --num-requests {args.num_requests} is under twice the concurrency {small}; "
            "throughput there is a rough estimate"
        )
    out = {}
    for name, n, next_body in inputs:
        if next_body is None:
            out[name] = {"not_run": "longer than the model's window"}
            continue
        if n is None:
            for _ in range(
                2
            ):  # warm up at this dataset's shapes; synthetic inputs were warmed by calibration
                timed(m.url, next_body())
        out[name] = {}
        for ld in loads:
            if ld["mode"] == "concurrency":
                meas, dps = closed_loop(m.url, next_body, ld["concurrency"], args.num_requests)
            else:
                meas, dps = open_loop(
                    m.url,
                    next_body,
                    ld["rps"],
                    args.burstiness,
                    args.num_requests,
                    f"{args.seed}:{m.label}:{name}:{ld['load']}",
                    args.max_inflight,
                )
            for i, r in enumerate(meas):
                sink({"model": m.label, "input": name, "load": ld["load"], "i": i, **r})
            st = {
                "input": name,
                **({"length": n} if n else {}),
                **ld,
                **stats(meas),
                "decisions_per_s": round(dps, 3) if dps else None,
            }
            out[name][ld["load"]] = st
            log(
                f"  {name:>14}, {ld['load']:>12}: TTFT p50 {st.get('p50_ms')} ms, p95 {st.get('p95_ms')} ms, "
                f"{st['decisions_per_s']} decisions/s ({st['ok']}/{st['n']} ok"
                + (f", {st['refused']} refused" if st["refused"] else "")
                + (f", {st['errors']} errors" if st["errors"] else "")
                + ")"
            )
    return out


def ensure_jevbench(path):
    path = Path(path).expanduser()
    if not (path / ".git").exists():
        log(f"fetching JevBench {JEVBENCH_COMMIT[:9]} into {path}")
        path.mkdir(parents=True, exist_ok=True)
        for cmd in (
            ["init", "-q"],
            ["remote", "add", "origin", JEVBENCH_REPO],
            ["fetch", "-q", "--depth", "1", "origin", JEVBENCH_COMMIT],
            ["checkout", "-q", "--detach", "FETCH_HEAD"],
        ):
            subprocess.run(["git", "-C", str(path), *cmd], check=True)
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != JEVBENCH_COMMIT or dirty:
        raise SystemExit(
            f"{path} is at {head[:9]}{' with local changes' if dirty else ''}, not the pinned {JEVBENCH_COMMIT[:9]}"
        )
    return path


def jevbench(m, args):
    jb = ensure_jevbench(args.jevbench_dir)
    d = m.dir / "jevbench"
    if d.exists() and any(d.iterdir()):
        raise SystemExit(
            f"{d} is not empty (JevBench opens its outputs exclusively); use a fresh --out"
        )
    d.mkdir(parents=True, exist_ok=True)
    tasks = d / "all.jsonl"
    tasks.write_text(
        "".join(
            (jb / "datasets/public" / f"{s}.jsonl").read_text()
            for s in ("original", "easy", "hard")
        )
    )
    n = sum(1 for line in tasks.read_text().splitlines() if line.strip())
    if n != 231:
        raise SystemExit(f"JevBench task file has {n} tasks, expected 231")
    env = {**os.environ, "PYTHONPATH": str(jb) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    jbc = [sys.executable, "-m", "jevbench.cli"]
    log(f"  jevbench: 231 tasks against {m.url}")
    t = time.time()
    with open(d / "run.log", "w") as f:
        subprocess.run(
            [
                *jbc,
                "run",
                "--tasks",
                str(tasks),
                "--adapter",
                "typesafe",
                "--endpoint",
                m.url,
                "--key-env",
                "",
                "--model",
                m.label,
                "--run-label",
                m.label,
                "--revision",
                m.spec,
                "--cost-basis",
                "no_billable_account_public_endpoint",
                "--reserve-usd",
                "0",
                "--results",
                str(d / "results.jsonl"),
                "--raw-dir",
                str(d / "raw"),
                "--ledger",
                str(d / "ledger.jsonl"),
                "--manifest",
                str(d / "manifest.json"),
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
        subprocess.run(
            [
                *jbc,
                "summarize",
                "--tasks",
                str(tasks),
                "--results",
                str(d / "results.jsonl"),
                "--ledger",
                str(d / "ledger.jsonl"),
                "--public-export",
                str(d / "summary.json"),
            ],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    s = json.loads((d / "summary.json").read_text())
    out = {k: s.get(k) for k in ("n_attempted", "n_correct", "accuracy", "schema_validity_strict")}
    out["wall_s"] = round(time.time() - t, 1)
    log(f"  jevbench: {out['n_correct']} / {out['n_attempted']} correct")
    return out


def decision_index(m, args):
    try:
        from decision_index import pipeline, runner
        from decision_index.suite.io import Suite
    except ImportError:
        raise SystemExit(
            "decision-index needs the kit: pip install 'git+https://github.com/apolinario/decision-index@9eb2dbe2a358004c8782c66e40a83ac07b953fec'"
        ) from None
    if not args.di_suite:
        raise SystemExit(
            "decision-index needs --di-suite: the suite directory built by `decision-index suite rebuild`"
        )
    suite = Suite(args.di_suite, args.di_edition)
    report = suite.verify(strict=False)
    if not report["match"]:
        if not args.di_no_verify:
            raise SystemExit(
                f"the suite in {args.di_suite} does not match the frozen {suite.edition['id']} hashes "
                f"(e.g. HLE is gated and missing); rebuild it, or pass --di-no-verify to score it anyway"
            )
        log(
            "  decision-index: the suite does not match the frozen hashes (--di-no-verify); the index is not official"
        )
    corpus = report["sha256"]
    rows = list(runner.iter_rows(suite.row_paths, suite.in_edition))
    if args.di_limit:
        rows = rows[:: max(1, len(rows) // args.di_limit)][: args.di_limit]
        log(
            f"  decision-index: {len(rows)} spread rows only (--di-limit): the index will be incomplete, not official"
        )
    d = m.dir / "decision-index"
    k = max(1, min(args.di_workers, len(rows)))
    log(f"  decision-index: {len(rows):,} requests in {k} shards against {m.url}")
    shards = []
    for i in range(k):
        sd = d / f"shard{i:03d}"
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / "rows.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[i::k])
        shards.append(sd)
    opts = {"base_url": m.url, "model": m.label}

    def one(sd):
        return runner.run(
            "http",
            opts,
            sd / "rows.jsonl",
            sd,
            compact=True,
            resume=True,
            corpus_sha256=corpus,
            keep=None,
            log=lambda line: None,
        )

    t = time.time()
    stop = threading.Event()

    def progress():
        while not stop.wait(60):
            done = sum(
                sum(1 for _ in open(sd / "results.jsonl"))
                for sd in shards
                if (sd / "results.jsonl").exists()
            )
            log(f"  decision-index: {done:,} / {len(rows):,} answered, {time.time() - t:.0f} s")

    threading.Thread(target=progress, daemon=True).start()
    with ThreadPoolExecutor(max_workers=k) as pool:
        finals = list(pool.map(one, shards))
    stop.set()
    latest = {}
    for sd in shards:
        for line in open(sd / "results.jsonl"):
            if line.strip():
                r = json.loads(line)
                if r["run_id"] not in latest or latest[r["run_id"]]["status"] == "error":
                    latest[r["run_id"]] = r
    with open(d / "results.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in latest.values())
    shutil.copy(shards[0] / "environment.json", d / "environment.json")
    scores = pipeline.score_run(suite, d / "results.jsonl", "http", d)
    counts = {}
    for fin in finals:
        for key, v in fin.get("counts", {}).items():
            counts[key] = counts.get(key, 0) + v
    out = {
        "decision_index": scores.get("decision_index"),
        "complete": scores.get("complete"),
        "edition": scores.get("edition"),
        "suite_verified": report["match"],
        "requests": len(rows),
        "counts": counts,
        "wall_s": round(time.time() - t, 1),
        "areas": {
            a["id"]: round(a["skill"] * 100, 2)
            for a in scores.get("areas", [])
            if a.get("skill") is not None
        },
    }
    log(f"  decision-index: {out['decision_index']} (complete: {out['complete']})")
    return out


# ---------------------------------------------------------------------------------------------------- environment
def this_host():
    """This host's EC2 instance type (IMDSv2), else its GPU or CPU."""
    try:
        tok = (
            urllib.request.urlopen(
                urllib.request.Request(
                    "http://169.254.169.254/latest/api/token",
                    method="PUT",
                    headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
                ),
                timeout=0.3,
            )
            .read()
            .decode()
        )
        return (
            urllib.request.urlopen(
                urllib.request.Request(
                    "http://169.254.169.254/latest/meta-data/instance-type",
                    headers={"X-aws-ec2-metadata-token": tok},
                ),
                timeout=0.3,
            )
            .read()
            .decode()
        )
    except OSError:
        env = machine()
        return (env.get("gpu") or [""])[0].split(",")[0] or env.get("cpu") or platform.node()


def machine():
    env = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "host": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
    }
    if hasattr(os, "sched_getaffinity"):
        env["cpus_usable"] = len(os.sched_getaffinity(0))
    try:
        if sys.platform == "darwin":
            env["cpu"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
        else:
            info = Path("/proc/cpuinfo").read_text()
            name = re.search(r"model name\s*:\s*(.+)", info)
            part = re.search(r"CPU part\s*:\s*(\S+)", info)
            env["cpu"] = (
                name.group(1)
                if name
                else f"arm part {part.group(1)}"
                if part
                else platform.processor()
            )
    except OSError:
        pass
    if shutil.which("nvidia-smi"):
        env["gpu"] = (
            subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
            )
            .stdout.strip()
            .splitlines()
        )
    try:
        import strands_decider

        env["strands_decider"] = getattr(strands_decider, "__version__", "installed")
    except ImportError:
        pass
    return env


# ---------------------------------------------------------------------------------------------------- plots
def ticks125(lo, hi):
    out, d = [], 10 ** int(f"{lo:e}".split("e")[1])
    while d <= hi * 10:
        out += [x for x in (d, 2 * d, 5 * d) if lo <= x <= hi]
        d *= 10
    return out


def round_down(x):
    return max([t for t in ticks125(1e-6, 1e9) if t <= x] or [x])


def round_up(x):
    return min([t for t in ticks125(1e-6, 1e9) if t >= x] or [x])


def load_runs(run_dirs):
    """[(label, result, host)] from saved runs; a label that recurs across runs is suffixed with its run's name."""
    runs, seen = [], {}
    for d in run_dirs:
        summ = json.loads((Path(d) / "summary.json").read_text())
        for label, r in summ["models"].items():
            seen.setdefault(label, set()).add(str(d))
            runs.append((label, r, (r.get("model") or {}).get("host") or "unknown", Path(d).name))
    return [
        (lab if len(seen[lab]) == 1 else f"{lab} ({run})", r, host) for lab, r, host, run in runs
    ]


def write_csv(runs, out):
    """latency.csv and accuracy.csv: one row per measurement, for analysis this script does not draw."""
    import csv

    lat_cols = [
        "model",
        "spec",
        "host",
        "gpus",
        "input",
        "length",
        "tokens_mean",
        "load",
        "mode",
        "concurrency",
        "rps",
        "burstiness",
        "n",
        "ok",
        "refused",
        "errors",
        "p10_ms",
        "p50_ms",
        "p90_ms",
        "p95_ms",
        "p99_ms",
        "mean_ms",
        "server_p50_ms",
        "decisions_per_s",
        "cpu",
    ]
    with open(out / "latency.csv", "w", newline="") as f:
        w = csv.DictWriter(f, lat_cols, extrasaction="ignore")
        w.writeheader()
        for lab, r, host in runs:
            meta = {
                "model": lab,
                "spec": (r.get("model") or {}).get("spec"),
                "host": host,
                "gpus": (r.get("model") or {}).get("gpus"),
            }
            for v in (r.get("ttft") or {}).values():
                for pt in v.values():
                    if isinstance(pt, dict):
                        w.writerow(
                            {**meta, **pt, "cpu": json.dumps(pt["cpu"]) if pt.get("cpu") else None}
                        )
    with open(out / "accuracy.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, ["model", "spec", "host", "benchmark", "score", "correct", "of", "complete"]
        )
        w.writeheader()
        for lab, r, host in runs:
            meta = {"model": lab, "spec": (r.get("model") or {}).get("spec"), "host": host}
            if r.get("jevbench"):
                j = r["jevbench"]
                w.writerow(
                    {
                        **meta,
                        "benchmark": "jevbench",
                        "score": round(100 * j["n_correct"] / j["n_attempted"], 2),
                        "correct": j["n_correct"],
                        "of": j["n_attempted"],
                        "complete": j["n_attempted"] == 231,
                    }
                )
            if r.get("decision-index"):
                d = r["decision-index"]
                w.writerow(
                    {
                        **meta,
                        "benchmark": "decision-index",
                        "score": d["decision_index"],
                        "of": d["requests"],
                        "complete": d["complete"],
                    }
                )


def plot(run_dirs, out):
    runs = load_runs(run_dirs)
    write_csv(runs, out)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("matplotlib is not installed; wrote the CSVs but no figures")
        return
    palette = plt.get_cmap("tab10").colors
    labels = list(dict.fromkeys(lab for lab, _, _ in runs))
    color = {lab: palette[i % 10] for i, lab in enumerate(labels)}
    hosts = list(dict.fromkeys(h for _, r, h in runs if r.get("ttft") or r.get("error")))

    def fmt(v):
        return f"{v:g}" if v < 1000 else f"{v / 1000:g}k" if v % 1000 == 0 else f"{v:,.0f}"

    def log_axis(ax, axis, vals):
        vals = [v for v in vals if v and v > 0]
        if not vals:
            return
        lo, hi = round_down(min(vals) / 1.2), round_up(max(vals) * 1.2)
        t = ticks125(lo, hi)
        (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
        (ax.set_xlim if axis == "x" else ax.set_ylim)(lo, hi)
        (ax.set_xticks if axis == "x" else ax.set_yticks)(t, [fmt(v) for v in t])
        (ax.xaxis if axis == "x" else ax.yaxis).set_minor_locator(matplotlib.ticker.NullLocator())

    def grid(nrows, title, ylabel, xlabel, row_names=None):
        fig, axes = plt.subplots(
            nrows,
            len(hosts),
            figsize=(4.8 * len(hosts) + 3.2, 3.6 * nrows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        for c, h in enumerate(hosts):
            axes[0][c].set_title(h, fontsize=9)
            axes[-1][c].set_xlabel(xlabel)
        for r_ in range(nrows):
            axes[r_][0].set_ylabel((f"{row_names[r_]}\n" if row_names else "") + ylabel)
            for ax in axes[r_]:
                ax.grid(True, alpha=0.25)
        fig.suptitle(title, fontsize=11)
        return fig, axes

    def finish(fig, axes, name, xs, ys):
        log_axis(axes[0][0], "x", xs)
        log_axis(axes[0][0], "y", ys)
        h = {}
        for ax in axes.flat:
            for hh, ll in zip(*ax.get_legend_handles_labels(), strict=True):
                h.setdefault(ll, hh)
        if h:
            fig.legend(
                h.values(),
                h.keys(),
                loc="center left",
                bbox_to_anchor=(1.0, 0.5),
                frameon=False,
                fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    lat = [(lab, r["ttft"], h) for lab, r, h in runs if r.get("ttft")]

    def synthetic(t):
        return sorted(
            (
                (v, p)
                for v in t.values()
                for p in v.values()
                if isinstance(p, dict) and p.get("length") and p.get("p50_ms")
            ),
            key=lambda vp: vp[1]["length"],
        )

    # 1. TTFT against prompt length: one line per model and load
    if any(synthetic(t) for _, t, _ in lat):
        fig, axes = grid(
            1,
            "Time to first token by prompt length",
            "TTFT (s), p50 with p10 to p90",
            "prompt tokens",
        )
        xs, ys = [], []
        dashes = ["-", "--", ":", "-."]
        for lab, t, h in lat:
            ax = axes[0][hosts.index(h)]
            by_load = {}
            for _, p in synthetic(t):
                by_load.setdefault(p["load"], []).append(p)
            for k, (ld, pts) in enumerate(by_load.items()):
                x = [p.get("tokens_mean") or p["length"] for p in pts]
                ax.plot(
                    x,
                    [p["p50_ms"] / 1e3 for p in pts],
                    marker="o",
                    ms=4,
                    ls=dashes[k % 4],
                    color=color[lab],
                    label=f"{lab}, {ld}",
                )
                ax.fill_between(
                    x,
                    [p["p10_ms"] / 1e3 for p in pts],
                    [p["p90_ms"] / 1e3 for p in pts],
                    color=color[lab],
                    alpha=0.10,
                    lw=0,
                )
                xs += x
                ys += [p["p10_ms"] / 1e3 for p in pts] + [p["p90_ms"] / 1e3 for p in pts]
        finish(fig, axes, "ttft_vs_length.png", xs, ys)

    # 2. p95 TTFT against decisions per second: one row per input, one line per model and kind of load
    inputs = list(
        dict.fromkeys(name for _, t, _ in lat for name, v in t.items() if "not_run" not in v)
    )
    if inputs and any(len(v) > 1 for _, t, _ in lat for v in t.values() if "not_run" not in v):
        fig, axes = grid(
            len(inputs),
            "p95 time to first token against throughput",
            "p95 TTFT (s)",
            "decisions answered per second",
            row_names=inputs,
        )
        xs, ys = [], []
        for lab, t, h in lat:
            for i, name in enumerate(inputs):
                v = t.get(name) or {}
                for mode, ls, mk, kind in (
                    ("concurrency", "-", "o", "N in flight"),
                    ("rps", "--", "s", "N req/s"),
                ):
                    pts = sorted(
                        (p["decisions_per_s"], p["p95_ms"] / 1e3)
                        for p in v.values()
                        if isinstance(p, dict)
                        and p.get("mode") == mode
                        and p.get("decisions_per_s")
                        and p.get("p95_ms")
                    )
                    if pts:
                        axes[i][hosts.index(h)].plot(
                            *zip(*pts, strict=True),
                            ls=ls,
                            marker=mk,
                            ms=4,
                            color=color[lab],
                            label=f"{lab}, {kind}",
                        )
                        xs += [x for x, _ in pts]
                        ys += [y for _, y in pts]
        for ax in axes.flat:
            ax.axhline(1, color="grey", lw=0.8, ls=":")
        finish(fig, axes, "ttft_vs_throughput.png", xs, ys)

    # 3. the highest decisions per second measured, against prompt length (per GPU where the GPU count is known)
    if any(synthetic(t) for _, t, _ in lat) or any(r.get("error") for _, r, _ in runs):
        per_gpu = all((r.get("model") or {}).get("gpus") for _, r, _ in runs if r.get("ttft"))
        fig, axes = grid(
            1,
            "Highest throughput measured, by prompt length",
            "decisions/s per GPU" if per_gpu else "decisions/s",
            "prompt tokens",
        )
        xs, ys = [], []
        for lab, r, h in runs:
            ax = axes[0][hosts.index(h)] if h in hosts else None
            if ax is None:
                continue
            if r.get("error"):
                ax.text(
                    0.5,
                    0.92 - 0.07 * len(ax.texts),
                    f"{lab}: did not start",
                    transform=ax.transAxes,
                    ha="center",
                    fontsize=8,
                    color="grey",
                )
                continue
            g = (r.get("model") or {}).get("gpus") if per_gpu else 1
            pts = []
            for v in (r.get("ttft") or {}).values():
                best = max(
                    (
                        p
                        for p in v.values()
                        if isinstance(p, dict) and p.get("length") and p.get("decisions_per_s")
                    ),
                    key=lambda p: p["decisions_per_s"],
                    default=None,
                )
                if best:
                    pts.append(
                        (best.get("tokens_mean") or best["length"], best["decisions_per_s"] / g)
                    )
            if pts:
                pts.sort()
                ax.plot(*zip(*pts, strict=True), marker="o", ms=4, color=color[lab], label=lab)
                xs += [x for x, _ in pts]
                ys += [y for _, y in pts]
        finish(fig, axes, "throughput_vs_length.png", xs, ys)

    # 4. accuracy
    panels = [
        (key, xl, f, txt)
        for key, xl, f, txt in (
            (
                "jevbench",
                "JevBench public tasks correct (%)",
                lambda v: 100 * v["n_correct"] / v["n_attempted"],
                lambda v: f" {v['n_correct']} / {v['n_attempted']}",
            ),
            (
                "decision-index",
                "Decision Index (public index)",
                lambda v: v["decision_index"],
                lambda v: (
                    f" {v['decision_index']:.2f}" + ("" if v.get("complete") else " (incomplete)")
                ),
            ),
        )
        if any(r.get(key) for _, r, _ in runs)
    ]
    if panels:
        n = max(sum(1 for _, r, _ in runs if r.get(key)) for key, *_ in panels)
        fig, axes = plt.subplots(
            1, len(panels), figsize=(5 * len(panels), 0.5 * n + 1.4), squeeze=False
        )
        for ax, (key, xl, f, txt) in zip(axes[0], panels, strict=True):
            rows = [
                (f"{lab} ({h})" if len(hosts) > 1 else lab, lab, r[key])
                for lab, r, h in runs
                if r.get(key)
            ]
            vals = [f(v) for *_, v in rows]
            ax.barh(range(len(vals)), vals, height=0.6, color=[color[lab] for _, lab, _ in rows])
            for i, ((*_, v), x) in enumerate(zip(rows, vals, strict=True)):
                ax.text(x, i, txt(v), va="center", fontsize=8)
            ax.set_yticks(range(len(vals)), [name for name, *_ in rows], fontsize=8)
            ax.set_ylim(len(vals) - 0.5, -0.5)
            ax.set_xlim(0, 100)
            ax.set_xlabel(xl)
            ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / "accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    log(f"figures and CSVs in {out}")


# ---------------------------------------------------------------------------------------------------- main
def main():
    if sys.argv[1:2] == ["plot"]:
        ap = argparse.ArgumentParser(
            prog="decider_bench.py plot", description="Draw the figures over saved runs."
        )
        ap.add_argument("runs", nargs="+", metavar="RUN_DIR")
        ap.add_argument(
            "--out",
            help="where to write (default: the first RUN_DIR when there is one, else plots/)",
        )
        a = ap.parse_args(sys.argv[2:])
        out = Path(a.out or (a.runs[0] if len(a.runs) == 1 else "plots"))
        out.mkdir(parents=True, exist_ok=True)
        return plot(a.runs, out)

    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog=__doc__.split("\n\n", 1)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = ap.add_argument_group("what to benchmark and measure")
    g.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="[NAME=]MODEL",
        help="checkpoint dir, Hub id, http(s)://URL, lambda:FUNCTION or sagemaker:ENDPOINT; repeatable",
    )
    g.add_argument(
        "--metrics",
        default="ttft",
        metavar="LIST",
        help="ttft, jevbench, decision-index, comma-separated (default ttft)",
    )
    g = ap.add_argument_group("ttft inputs (either or both; default --lengths 256,1024,4096)")
    g.add_argument(
        "--lengths",
        type=ints,
        metavar="N,N",
        help="synthetic requests of exactly these prompt-token counts",
    )
    g.add_argument(
        "--dataset",
        action="append",
        default=[],
        metavar="FILE",
        help='your own requests, JSONL (.gz ok) of {"state", "questions"}; repeatable',
    )
    g = ap.add_argument_group("ttft load (either or both; default --concurrency 1)")
    g.add_argument(
        "--concurrency", type=ints, metavar="N,N", help="keep N requests in flight, e.g. 1,8,32"
    )
    g.add_argument(
        "--rps",
        type=floats,
        default=[],
        metavar="R,R",
        help="send N requests per second at Poisson times, e.g. 2,8",
    )
    g.add_argument(
        "--num-requests",
        type=int,
        default=50,
        metavar="N",
        help="requests measured per input and load (default 50)",
    )
    g = ap.add_argument_group("local checkpoints")
    g.add_argument(
        "--device",
        metavar="DEV",
        help="cuda, mps, cpu or mlx, passed to strands-decider serve (default: its own choice)",
    )
    g.add_argument(
        "--serve-cmd",
        metavar="CMD",
        help="your own server instead of strands-decider serve, with {model} and {port}, "
        "e.g. 'python my_server.py --model {model} --port {port}'",
    )
    g = ap.add_argument_group("benchmarks")
    g.add_argument(
        "--di-suite",
        metavar="DIR",
        help="the Decision Index suite, built with the kit's `suite rebuild`",
    )
    g = ap.add_argument_group("output")
    g.add_argument("--out", metavar="DIR", help="output directory (default runs/<UTC time>)")
    g = ap.add_argument_group("advanced")
    g.add_argument(
        "--burstiness",
        type=float,
        default=1.0,
        help="--rps arrival shape: 1 Poisson, below 1 burstier",
    )
    g.add_argument(
        "--serve-args", help="extra flags for strands-decider serve, e.g. '--strict-window'"
    )
    g.add_argument(
        "--health-timeout",
        type=float,
        default=1800,
        help="seconds to wait for a local server to load",
    )
    g.add_argument("--max-inflight", type=int, default=512, help="client threads for --rps")
    g.add_argument(
        "--jevbench-dir",
        default="~/.cache/decider_bench/jevbench",
        help="where JevBench is fetched",
    )
    g.add_argument("--di-edition", help="Decision Index edition (default: read from the suite)")
    g.add_argument(
        "--di-workers", type=int, default=16, help="Decision Index requests in flight (default 16)"
    )
    g.add_argument(
        "--di-no-verify",
        action="store_true",
        help="score a suite whose hashes differ from the frozen ones",
    )
    g.add_argument(
        "--di-limit",
        type=int,
        help="run only N spread Decision Index rows: a plumbing check, not an index",
    )
    g.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.model:
        ap.error("give at least one --model")
    metrics = [x.strip() for x in args.metrics.split(",") if x.strip()]
    bad = [x for x in metrics if x not in METRICS]
    if bad:
        ap.error(f"unknown metric {', '.join(bad)}; choose from {', '.join(METRICS)}")
    if args.lengths is None:
        args.lengths = [] if args.dataset else [256, 1024, 4096]
    if args.concurrency is None:
        args.concurrency = [] if args.rps else [1]
    out = Path(args.out or f"runs/{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    if (out / "summary.json").exists():
        ap.error(f"{out} already holds a run; choose another --out")
    out.mkdir(parents=True, exist_ok=True)
    models = [Model(a, args, out) for a in args.model]
    if len({m.label for m in models}) < len(models):
        ap.error("two models have the same name; name them with NAME=")

    if "ttft" in metrics:
        inputs = [f"{n} tokens" for n in args.lengths] + [Path(d).name for d in args.dataset]
        loads = [f"{c} in flight" for c in args.concurrency] + [f"{r:g} req/s" for r in args.rps]
        log(
            f"plan: {len(models)} model(s) x {len(inputs)} input(s) [{', '.join(inputs)}] x {len(loads)} load(s) "
            f"[{', '.join(loads)}] x {args.num_requests} requests"
            + (f", then {', '.join(x for x in metrics if x != 'ttft')}" if len(metrics) > 1 else "")
        )
    sink_f, sink_lock = open(out / "requests.jsonl", "a"), threading.Lock()

    def sink(rec):
        with sink_lock:
            sink_f.write(json.dumps(rec) + "\n")

    summary = {
        "client": machine(),
        "metrics": metrics,
        "lengths": args.lengths,
        "dataset": args.dataset,
        "concurrency": args.concurrency,
        "rps": args.rps,
        "burstiness": args.burstiness,
        "num_requests": args.num_requests,
        "models": {},
    }
    for m in models:
        log(f"{m.label}: {m.spec}")
        res = summary["models"][m.label] = {}
        try:
            try:
                m.start()
            except (
                Exception,
                SystemExit,
            ) as e:  # e.g. a model that does not fit: record it and go on
                res["model"] = m.describe()
                res["error"] = str(e)[:500]
                log(f"{m.label}: did not start: {res['error']}")
                continue
            res["model"] = m.describe()
            log(f"{m.label}: on {res['model']['host']}")
            prompts = Prompts(m.url, f"{args.seed}:{m.label}", m.health.get("max_length"))
            for metric in metrics:
                try:
                    res[metric] = (
                        ttft(m, prompts, args, sink)
                        if metric == "ttft"
                        else jevbench(m, args)
                        if metric == "jevbench"
                        else decision_index(m, args)
                    )
                except (Exception, SystemExit) as e:
                    res.setdefault("errors", {})[metric] = str(e)[:500]
                    log(f"{m.label}: {metric} failed: {e}")
                sink_f.flush()
                (out / "summary.json").write_text(json.dumps(summary, indent=2))
        finally:
            m.stop()
    sink_f.close()
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    plot([out], out)


if __name__ == "__main__":
    main()
