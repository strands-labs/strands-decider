# Editorial changes to the supplied warm-run report

The original report is preserved in `reports/editorial-source/m1-quant-warm-runs-report.md`.
The rewrite uses the saved `m1-bf16-clean` and `m1-fp16-clean` runs, not the earlier exploratory runs.

- **Comparable quantization base:** the source's 15.1 s int8 number is from the bfloat16-base run. The fp16-base int8 number is 10.21 s, compared with 8.39 s for fp16. Both original numbers remain in the narrative with their provenance clarified. The plots use fp16-base int8/int4.
- **Latency versus speed:** fp16 reduces latency by 13.47–20.47%, averaging 19.05% across six shapes. The original claim of about 20% at every size was narrowed. This is an average of per-shape percentage reductions, not a pooled workload-weighted result.
- **Slowdown ranges:** fp16-base int8 adds 10.71–32.11% latency and int4 adds 5.52–41.98%. Int4 is slightly faster than int8 for 256 tokens/one question, so it is not described as slower in every case.
- **Memory scope:** the plotted statistic is peak MLX allocation during inference, not total RAM or process RSS. The measured reductions are retained without calling them universally deterministic.
- **Quality scope:** 65/72 and 62/72, unchanged int8 predictions, and the three int4 changes are retained. An observed match on 72 tasks is not described as a general quality guarantee.
- **Mechanism and hardware scope:** compute/dequantization overhead is presented as a possible explanation, not a proven bottleneck. Recommendations apply to this M1 setup, not all Apple silicon or unmeasured CUDA configurations.
- **Int4 tradeoff:** its memory benefit is acknowledged rather than calling it a configuration with no upside.

The chart whiskers show observed min/max over five samples. They are not confidence intervals.
Exact chart values and SHA-256 hashes are saved alongside the PNG/SVG exports.
