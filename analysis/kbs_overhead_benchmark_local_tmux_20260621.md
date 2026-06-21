# KBS overhead benchmark (non-canonical, timing-only)

- Trace: `brightkite_50k` (`data/processed/brightkite/trace.jsonl`)
- Requests used: 5000 (of the trace's full length; not the full 50,000-request canonical scale)
- Capacities: 32, 64, 128
- Policies: lru, sieve, fifo_reinsertion, rest_v1, evict_value_v1
- Benchmark environment: local/cloud machine via tmux (not Wulver, not Slurm)
- Timestamp (UTC): 2026-06-21T18:49:00.384784+00:00
- Command: `python scripts/run_overhead_benchmark.py --trace-path data/processed/brightkite/trace.jsonl --trace-name brightkite_50k --capacities 32,64,128 --policies lru,sieve,fifo_reinsertion,rest_v1,evict_value_v1 --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl --max-requests 5000 --out-csv analysis/kbs_overhead_benchmark_local_tmux_20260621.csv --out-md analysis/kbs_overhead_benchmark_local_tmux_20260621.md`

Timing boundary: wall-clock around each `policy.on_request(req)` call only; trace loading and `policy.reset(...)` are excluded. "Eviction decision" rows are the subset of calls where `event.evicted is not None` (i.e. the cache was full and a victim was actually chosen) -- this isolates the O(k) candidate-scan cost from O(1) hit-path bookkeeping.

## Mean ms per eviction decision, by policy and capacity

| capacity | policy | n_eviction_decisions | mean_ms | median_ms | p95_ms | status |
|---|---|---|---|---|---|---|
| 32 | lru | 1611 | 0.000954 | 0.00089 | 0.001248 | ok |
| 32 | sieve | 1605 | 0.002354 | 0.002299 | 0.002703 | ok |
| 32 | fifo_reinsertion | 1625 | 0.000973 | 0.000905 | 0.001351 | ok |
| 32 | rest_v1 | 1611 | 0.038123 | 0.037597 | 0.040719 | ok |
| 32 | evict_value_v1 | 1805 | 75.014359 | 74.627909 | 76.080272 | ok |
| 64 | lru | 1395 | 0.001126 | 0.001078 | 0.001458 | ok |
| 64 | sieve | 1418 | 0.003336 | 0.003275 | 0.003797 | ok |
| 64 | fifo_reinsertion | 1394 | 0.000994 | 0.000925 | 0.001402 | ok |
| 64 | rest_v1 | 1395 | 0.074822 | 0.074305 | 0.07913 | ok |
| 64 | evict_value_v1 | 1576 | 152.122719 | 151.719789 | 152.914868 | ok |
| 128 | lru | 1177 | 0.001151 | 0.001096 | 0.001487 | ok |
| 128 | sieve | 1193 | 0.00513 | 0.005054 | 0.005691 | ok |
| 128 | fifo_reinsertion | 1188 | 0.001066 | 0.00099 | 0.001463 | ok |
| 128 | rest_v1 | 1177 | 0.18466 | 0.18294 | 0.199115 | ok |
| 128 | evict_value_v1 | 2059 | 316.010305 | 315.350158 | 317.811113 | ok |

