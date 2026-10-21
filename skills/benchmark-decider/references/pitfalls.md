# Pitfalls

Each of these has produced a wrong number before. The tool handles some itself; the rest are yours.

**Prefix caches.** A server that caches prompt prefixes answers a repeated prompt much faster.
Synthetic requests start with a random id, so they never repeat. A `--dataset` smaller than the
requests a run sends repeats rows, and the tool warns; use more rows or fewer requests.

**Warm-up and cold starts.** The first request at a new prompt shape is slow on some devices (MPS
compiles a kernel per input length), and the first request on a new Lambda sandbox is a cold start
of seconds. The tool discards each client's first request and measures only requests sent after
every client has warmed up. Open-loop runs can still trigger cold starts when traffic grows faster
than Lambda adds sandboxes; that is real behavior, but report it as such.

**Mixed hosts on Lambda.** Sandboxes of one function land on different Graviton generations, and an
older one can be twice as slow. One sandbox serves every request of a one-at-a-time run, so a whole
run can land on the slow one. Look at the `cpu` column before comparing two Lambda runs.

**Client placement.** A client on a laptop adds tens of milliseconds of network time and jitter to
every request. For latency claims about a remote endpoint, run the client in the same region.

**Thread counts.** Count cores from the process affinity and the cgroup quota, not `os.cpu_count()`,
which reports the host's cores inside some containers and sandboxes. Libraries with their own thread
pools (OpenBLAS, OpenMP) can spin idle threads on the cores the engine needs; set their threads to
1 when they are not the engine.

**Too few requests for the load.** Throughput at concurrency c from fewer than 2c requests is a
rough estimate, and the tool warns. Open loops need more requests than closed loops, because one
slow request stretches the measured window.

**Context window and truncation.** Prompts over the window are skipped (synthetic) or refused (HTTP
422). A server that truncates long prompts and one that refuses them answer different questions, so
compare like with like, and count refusals rather than dropping them.

**JevBench.** The harness opens its outputs exclusively, so each run needs a fresh `--out`. Check
that the server answering is the one you meant: the tool starts local servers on a free port for
this reason.

**Decision Index.** The suite must be built locally from its sources (some are gated, such as HLE).
A suite whose hashes differ from the frozen ones is refused unless `--di-no-verify` is given, and the
index from it is not official. A partial run (`--di-limit`) scores near zero by design, because
unanswered requests count as wrong; use it only to check the plumbing. Requests an engine refuses for
capacity count as wrong whichever status the kit records.

**Different sessions.** Two runs on different days or instances differ by several percent even for
the same system. Put the systems you compare in one run, or repeat the key point in one session.
