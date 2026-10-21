# Analysis: first principles, cost and uncertainty

## First principles

A decider answers with one prefill: it reads the prompt once and scores the options from the final
hidden states, decoding nothing. Its work is therefore fixed by the prompt length n:

    FLOPs ≈ 2 × (matrix-multiply parameters) × n + (attention term, growing with n²)

For Strands Decider 2B (Qwen3.5 2B), the matrix multiplies hold 1.373 B parameters, so a prompt
costs about 2.746 GFLOP per token, and attention adds about 24,576 × n² FLOP. For another model,
count the parameters of its linear layers from the config and work out the attention term from its
attention layers; record how you derived both.

Divide by the device's nominal peak to get a floor:

    peak = cores × FLOPs per cycle per core × clock

Six Graviton3 vCPUs (a 10,240 MB Lambda sandbox) peak at about 0.5 TFLOP/s in fp32, 2.0 in bf16 and
4.0 TOPS in int8, so a 300-token prompt cannot take less than 1.7 s in fp32 or 0.21 s in int8.
Use the floor three ways:

- To choose a strategy: if the floor at the target length is above the latency budget for a
  precision, no engine at that precision will meet it.
- To judge an engine: `figures.pct_of_peak` gives the share of peak it sustains. Well-tuned CPU
  engines usually sustain 20 to 40%.
- To catch errors: above 100% of peak the measurement is wrong (an unsynchronized device, a cached
  answer, the wrong token count); below about 5% suggests a debug build, emulation or the wrong
  thread count.

## Throughput and latency

In a closed loop, Little's law ties the numbers: requests in flight ≈ decisions per second × mean
latency. If a run's numbers break it by more than a few percent, the client, not the server, set the
pace. Throughput stops growing at the ceiling while latency keeps rising; report the ceiling and the
load where p95 crosses the user's budget. In an open loop below the ceiling, decisions per second
equal the offered rate and latency shows the queueing.

## Cost

- A provisioned host kept busy: USD per million decisions = price per hour ÷ (3,600 × decisions per
  second) × 10⁶ (`figures.cost_per_million`). Below the ceiling, it costs the same per hour, so the
  cost per decision rises as traffic falls.
- Lambda, billed per use: memory in GB × billed seconds × price per GB-second, plus the request
  price (`figures.lambda_cost_per_million`). Billed seconds are the in-function time, which the
  shim reports as `server_ms`, not the client's round trip.
- Fixed charges (a SnapStart cache, provisioned concurrency, an always-on instance) are monthly and
  belong in a separate line.
- The break-even rate (`figures.break_even_per_s`) is the sustained decisions per second above which
  the provisioned host is cheaper.

Quote prices with their date and region, from the provider's pricing page.

## Uncertainty

- Compare systems measured in one session on one host. Absolute times move several percent between
  sessions and instances, so a difference under about 5% needs a repeat before it counts.
- A median from 50 requests is usually stable to a few percent. A p95 needs 200 or more requests, and
  a p99 from 50 requests is the slowest request, not a percentile.
- On Lambda, report latency by the CPU part the function reports (`cpu` in `latency.csv`): sandboxes
  land on different processor generations that differ by up to 2×.
- JevBench has 231 tasks. A single-run difference under about 10 tasks is unresolved; compare two
  runs task by task with `evaluation/jevbench/paired.py` (an exact McNemar test).
- The Decision Index board treats Full scores less than 0.9 points apart as tied.
- Accuracy alone is a weak check on a numerics change: answers can move without changing which ones
  are right. Compare probabilities with a full-precision reference as well (`figures.answer_drift` on
  Decision Index results, or `evaluation/device_parity.py` across local devices), and report argmax
  agreement, mean total variation, and the flips with their margins. Flips at margins under 0.02
  are near-ties.

## Writing it up

Answer the question in the first paragraph. Write each number as a value with its difference from
the baseline in parentheses, for example "38.9 decisions/s (+37.6 vs. one request at a time's 1.3)".
Give every table and figure a caption and refer to it. Say what was not measured (other hosts,
cold starts, other models) and keep failed attempts and corrections in the log.
