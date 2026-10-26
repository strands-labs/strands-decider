# Figures: recipes and style

`python benchmark/decider_bench.py plot runs/*` draws the standard figures. Everything below uses
[`../scripts/figures.py`](../scripts/figures.py), which reads `latency.csv` and `accuracy.csv` into
plain rows (dicts) and needs only matplotlib.

```python
import sys; sys.path.insert(0, "skills/benchmark-decider/scripts")
import figures as F

lat = F.split_name(F.load("runs/*"), ["size", "engine", "precision"])   # names like 2b-vllm-fp8
synthetic = [r for r in lat if r["input_length"]]          # rows from --input-length; dataset rows have no input_length
```

Each row of `latency.csv` is one input at one load for one model: `model`, `host`, `gpus`, `input`,
`input_length`, `tokens_mean`, `load`, `mode` (`concurrency` or `rps`), `concurrency`, `rps`, `p10_ms` to
`p99_ms`, `server_p50_ms`, `decisions_per_s`, and `cpu` (CPU parts seen, when the server reports them).

## Recipes

**Time to first token by input length**, one panel per host, one line per model, one-at-a-time load,
with the p10 to p90 band:

```python
F.facet([r for r in synthetic if r["load"] == "1 in flight"], x="tokens_mean", y="p50_ms", col="host",
        line="model", band=("p10_ms", "p90_ms"), yscale=1e-3, hline=1,
        title="Time to first token by input length", xlabel="input tokens", ylabel="TTFT (s)",
        out="ttft_by_length.png")
```

**Tail latency against load**, rows by input, columns by host. Use achieved decisions per second on
x to put closed and open loops on one axis, or `x="rps"` over the open-loop rows only to plot against
offered traffic:

```python
F.facet(lat, x="decisions_per_s", y="p95_ms", row="input", col="host", line="model", style="mode",
        yscale=1e-3, hline=1, title="p95 time to first token against throughput",
        xlabel="decisions answered per second", ylabel="p95 TTFT (s)", out="p95_by_throughput.png")
```

**Throughput per GPU by input length**, model sizes in rows, hosts in columns, engines as lines,
with models that failed to start marked in their panel. Run a high `--concurrency` (32 or 64) so the
highest throughput measured is the ceiling:

```python
best = F.split_name(F.max_throughput(synthetic), ["size", "engine", "precision"])
for r in best:
    r["per_gpu"] = r["decisions_per_s"] / (r["gpus"] or 1)
F.facet(best, x="tokens_mean", y="per_gpu", row="size", col="host", line="engine", style="precision",
        failed=F.failed_cells("runs/*", ["size", "engine", "precision"], line=1),
        title="Decisions per second per GPU by input length", xlabel="input tokens",
        ylabel="decisions/s per GPU", out="throughput_grid.png")
```

**Cost per million decisions by input length.** Price a provisioned host at its ceiling and Lambda
at the in-function time (`server_p50_ms`). Take current prices from the provider; the
`LAMBDA_GB_S` defaults in `figures.py` are us-east-1 list prices and change.

```python
price = {"g6e.xlarge": 1.861}            # USD per hour, from the user's region and pricing page
rows = []
for r in F.max_throughput(synthetic):
    if r["host"] in price:
        rows.append({**r, "usd": F.cost_per_million(r["decisions_per_s"], price[r["host"]])})
    elif r["host"].startswith("Lambda"):
        mem = int(r["host"].split()[-2].replace(",", ""))
        rows.append({**r, "usd": F.lambda_cost_per_million(r["server_p50_ms"] / 1e3, mem)})
F.facet(rows, x="tokens_mean", y="usd", line="model", title="Cost per million decisions",
        xlabel="input tokens", ylabel="USD per million decisions", out="cost_by_input_length.png")
```

For cost against sustained traffic, a provisioned host costs its hourly price at any rate up to
its ceiling, while Lambda costs its per-decision price times the rate; the lines cross at
`F.break_even_per_s(price_per_hour, lambda_usd_per_million)`. Add any fixed monthly charge (a
snapshot cache, provisioned concurrency) to Lambda's line.

**Share of peak.** `F.pct_of_peak(tokens, seconds, peak_tflops, gflop_per_token, attn_coeff)` for each
row (see [analysis.md](analysis.md) for the inputs), plotted against input length. It shows how
much headroom an engine has and catches impossible numbers.

**Answer drift between two engines.** `F.answer_drift("runs/a/<model>/decision-index/results.jsonl",
"runs/b/<model>/decision-index/results.jsonl")` compares the per-question probabilities two runs gave
the same Decision Index requests: argmax agreement, mean and p95 total variation, and the flips with
their margins. Plot mean drift as bars, one per engine, next to the latency bars of the same engines.

**Step-by-step optimizations (an ablation).** Name one model per step (`0-baseline`, `1-int8`,
`2-int8-fused`, ...) and measure them in one run so they share a session. Draw grouped bars of p50
TTFT at three input lengths for each step, cross-hatch the baselines, and put drift from the full-precision
reference in a second panel on the same x positions.

**Cold start.** The tool measures warm requests only. Time cold starts separately (a fresh process
for a local server, a new function version for Lambda, which changes the user's resources and needs
their approval), split each into stages (start, load, first decision, network), and draw stacked
horizontal bars on a log axis that starts at a round value.

## Style rules

- Axis limits are round and labelled: log axes start and end on 1-2-5 values and tick only at those
  values; linear axes start at zero or at a round value. `figures.log_axis` does this.
- Colors come from one qualitative palette, and an engine keeps its color in every figure. Use line
  style or markers, not a second color, for a secondary condition such as the kind of load.
- Label each thing once: a legend or direct labels, not both, and no text repeated in every panel.
- Titles are short and plain and say what is plotted ("Time to first token by input length").
  Setup details (cores, threads, sample counts, where the client ran) go in the caption.
- Mark missing cells ("did not start", "does not fit in 24 GB") instead of leaving them blank.
- Nothing overflows a panel or the figure. Open every image and look at it before sharing it.
- Every figure gets a caption that says what is plotted, from how many requests, where it was
  measured, and what to notice; the text refers to it.
