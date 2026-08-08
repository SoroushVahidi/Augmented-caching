# KBS heavy_r1 policy-comparison: validation-pass report

Status: validation pass complete (`VALIDATION_EXIT=0`). This is a single-trace,
single-capacity smoke pass — **not** the canonical multi-trace result. See
`reports/kbs_full_policy_comparison_execution_plan.md` for the full-scope plan.

## 1. Exact command executed

From `logs/kbs_policy_comparison_heavy_r1/validation_eval_run.log`:

```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --max-traces 1 \
  --capacities 256 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.md
```

(run under `/usr/bin/time -v`, wrapped by the tmux session `kbs_policy_comparison_heavy_r1`)

## 2. Runtime

- Wall clock: **4:02:57** (4h 3m)
- User time: 14570.75s, system time: 2.53s — **99% CPU**, confirming the process
  is single-threaded/single-core bound (no internal parallelism).
- Max resident set size: ~2.1GB.
- Exit status: **0** (clean).

## 3. Scope (trace / capacity / policies)

- **Trace:** 1 of 7 — `brightkite_50k` (family `brightkite`), 50,000 requests,
  `data/processed/brightkite/trace.jsonl`.
- **Capacity:** 256 only (the most expensive capacity in the dataset's pool of
  32/64/128/256).
- **Policies:** 7 — `lru`, `blind_oracle`, `predictive_marker`,
  `blind_oracle_lru_combiner`, `trust_and_doubt`, `rest_v1`, `evict_value_v1`.
  (`ml_gate_v1`/`ml_gate_v2` excluded automatically — their model files are not
  present on this machine.)
- **Model:** `models/evict_value_wulver_v1_best_heavy_r1.pkl`.

## 4. Output columns and values

Columns: `trace_name, trace_family, path, capacity, policy, misses, hit_rate`.

| policy | misses | hit_rate | vs LRU |
|---|---|---|---|
| lru | 14758 | 0.70484 | — |
| rest_v1 | 14758 | 0.70484 | tie |
| blind_oracle_lru_combiner | 14759 | 0.70482 | −0.01% (≈tie) |
| predictive_marker | 14923 | 0.70154 | −1.12% |
| trust_and_doubt | 15306 | 0.69388 | −3.71% |
| **evict_value_v1** | **20533** | **0.58934** | **−39.13%** |
| blind_oracle | 29554 | 0.40892 | −100.26% |

("vs LRU" here is the script's own sign convention: negative = *more* misses
than LRU, i.e. worse.)

**Notable findings, not just pipeline mechanics:**

- **`evict_value_v1` has 39% more misses than LRU/rest_v1 on this trace** — the
  model is currently *underperforming* the simplest baseline here, not just
  trailing the best one. This is a single trace, so it is not yet evidence of a
  systematic problem, but it is the opposite of the result the manuscript needs
  and should not be ignored while waiting for the full sweep.
- **`blind_oracle` is the worst performer**, well below LRU. For a policy named
  "oracle" this is counterintuitive and worth a quick semantic sanity check
  (`src/lafc/policies/blind_oracle.py`) before this number is used in any
  narrative — confirm it's deliberately a weak/naive baseline and not a
  mislabeled or miswired policy.
- `blind_oracle_lru_combiner` is statistically indistinguishable from LRU on
  this trace (14759 vs 14758 misses) — consistent with a combiner that falls
  back to LRU behavior most of the time on this family.

## 5. Does this confirm the pipeline works end to end?

**Yes, mechanically.** The script loaded the trained `heavy_r1` model, ran all
7 policies against a real 50k-request trace at the dataset's largest capacity,
and produced well-formed CSV/MD output with no exceptions. This validates:
trace loading, model loading (the earlier `joblib` fallback bugfix, commit
`b9df6f1`, did not regress), the policy-comparison harness, and the
aggregation/markdown-summary logic.

It does **not** confirm the *model's* end-to-end miss-ratio result is good —
see §4 above.

## 6. Warnings / errors

None. `validation_eval_run.log` shows a clean `/usr/bin/time -v` summary
ending in `Exit status: 0` and `VALIDATION_EXIT=0`. No stderr, no Python
tracebacks, no truncated output.

## 7. Should this output be used in the manuscript?

**No.** This is a 1-trace × 1-capacity sample (7 of the eventual 196
trace×capacity×policy rows) — it is explicitly a pipeline-correctness check,
not a statistically representative result. Per the repo's own standard
(`docs/wulver_heavy_evict_value_experiment.md`, cited in
`reports/kbs_policy_comparison_launch_plan.md` §8): smoke/validation outputs
must not substitute for the canonical 7-trace × 4-capacity sweep in any
manuscript table or figure. Keep this file as an internal validation artifact
only (filename already carries the `_validation` suffix specifically so it
cannot be confused with, or accidentally consumed by, the canonical-artifact
build script).
