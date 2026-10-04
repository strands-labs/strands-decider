# Profiling validation run, October 4, 2026

This run validates the profiling pipeline; it is not a replacement for the clean AC-powered benchmarks. It ran on battery (94% initially), with 4.26 GiB of swap used at startup. FP16 ran first, then INT8. Each shape had two warmups, three ordinary timing samples, three separate phase-instrumented passes, and one Python profile.

INT8 was faster in this run: 626 vs 1190 ms (256 tokens, one question), 1779 vs 3891 ms (256, eight), 2005 vs 3981 ms (1024, one), and 3257 vs 6169 ms (1024, eight). This reverses the earlier clean benchmark ordering. The run does not reproduce the reported INT8 slowdown and cannot establish its root cause. The difference is concentrated in the decoder/hidden-state phase; the output head and other request work account for only a small part of the measured duration. These broad measurements do not identify an individual GPU kernel.

Repeat on AC power with reduced memory pressure and both precision orders before comparing performance. Kernel-level attribution then requires Metal capture replay/profiling in Xcode. No model GPU traces were collected in this run; the separate tiny-matrix Metal capture smoke test succeeded.

The initial script is preserved in commit f23f879 and its SHA256 is in manifest.json. While the FP16 worker ran, a lint cleanup changed import order and made the event-list closure binding explicit for the subsequent INT8 worker; the timing method and model computation were unchanged. The final script additionally records each worker's script hash and power/swap state. The manifest source revision predates the harness commit because the run began just before that commit.

Raw samples, phase events, Python text profiles and binary pstats files are retained alongside this note. Profiling adds synchronization and CPU bookkeeping, so only the uninstrumented samples should be treated as baseline request latency.
