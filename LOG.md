## 2026-10-14 · Fase A · g6.xlarge (Base AMI, driver 595.91.07, CUDA 13.2)
- torch 2.14.1+cu130, transformers 5.19.0, fla instalado, causal-conv1d NO (pendiente: cuda-toolkit-13-0)
- checkpoint sha256_16: 49314346cbdb88b2 · jevbench 1bcc55eb · repo 3506409e
- JevBench: 176/231, Brier 0.3227, ECE 0.0742 · p50 64 ms, p95 437 ms (prefix cache on)
- tool_selection Brier 0.0002; débiles: temporal_numeric 0.82, long_policy 0.68, probability 0.65
- bench_local: suelo 61 ms hasta 512 tok, luego 0.17 ms/tok; prefix vs batched hasta 13.5x (4096 tok, 16 q)
- Resultados en s3://avahi-silicon-bench/strands-decider/results/
- Pendiente: A5 venv sin fla; causal-conv1d; repetir bench_local con conv
