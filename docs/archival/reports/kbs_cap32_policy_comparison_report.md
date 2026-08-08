# cap32 policy-comparison report (2026-06-19)

Read-only verification of the completed cap32 chunk. Nothing re-run, nothing
modified except this report.

**Status update, same day (2026-06-19), later pass: `CANONICAL CHUNK
WITHOUT SIEVE / SUPERSEDED FOR FINAL TABLE IF SIEVE-INCLUSIVE CHUNKS ARE
USED`.** This chunk (and the `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv/.md`
files it verifies) predates SIEVE's implementation and has no SIEVE column.
Per Reviewer #3's explicit SIEVE request (R3-Issue3/R3-Rec2) and
`reports/kbs_cap32_rerun_with_sieve_plan.md`, a SIEVE-inclusive rerun is
planned (output: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.csv/.md`,
status `PENDING / CANONICAL CHUNK WITH SIEVE` — **not yet launched, awaiting
explicit approval**). This file's own content below (the verified
no-SIEVE run) is **not deleted or rewritten** — Option B in the rerun plan
keeps both chunks on disk side by side until a human decides which one
backs the final manuscript table. Until that decision, this report remains
an accurate, valid description of the no-SIEVE chunk it documents; it is
"superseded" only in the conditional sense that the SIEVE-inclusive chunk,
once run and approved, is intended to be the one actually cited in the
manuscript.

## 1. Exact command (as run, from `kbs_full_policy_comparison_execution_plan.md` §6)

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap32/cap32_run.log
```

Run inside tmux session `kbs_full_policy_comparison_cap32`. Session is still
open (idle at a shell prompt); the job inside it is finished.

## 2. Runtime and exit code

- Started: Thu 2026-06-18 19:18:11 EDT
- Finished: Fri 2026-06-19 00:31:13 EDT
- **Wall-clock runtime: 5:13:02** (~1.49× the ~3.5h "optimistic/capacity-proportional"
  estimate in the execution plan, within its stated ±2x band)
- **`CAP32_EXIT=0`**
- Log file (`logs/kbs_full_policy_comparison/cap32/cap32_run.log`) is only 2
  lines — the script has no per-trace progress logging, consistent with the
  execution plan's finding that it writes the CSV/MD exactly once at the end:
  ```
  Wrote analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv and analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.md
  CAP32_EXIT=0
  ```

## 3. CSV size and line count

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv`: 5.1K, **50 lines** (1 header + 49 data rows).

## 4. Trace / capacity / policy coverage

- **7 trace families** × **1 capacity (32)** × **7 policies** = 49 rows. Matches
  expected coverage exactly for a single-capacity chunk.
- Trace families: `brightkite, citibike, cloudphysics, metacdn, metakv,
  twemcache, wiki2018`.
- Policies: `lru, blind_oracle, predictive_marker, blind_oracle_lru_combiner,
  trust_and_doubt, rest_v1, evict_value_v1`.

## 5. Per-trace results (misses, capacity 32)

| Trace family | lru | blind_oracle | predictive_marker | blind_oracle_lru_combiner | trust_and_doubt | rest_v1 | evict_value_v1 |
|---|---|---|---|---|---|---|---|
| brightkite | 18,078 | 32,262 | 18,483 | 18,081 | 18,845 | 18,078 | **19,970** |
| citibike | 19,994 | 42,268 | 20,169 | 19,995 | 20,856 | 19,994 | **21,573** |
| wiki2018 | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 | 50,000 | **50,000** |
| twemcache | 37,881 | 47,626 | 38,260 | 37,884 | 38,827 | 37,881 | **38,579** |
| metakv | 38,064 | 40,842 | 38,105 | 38,065 | 38,165 | 38,064 | **38,710** |
| metacdn | 29,380 | 38,911 | 29,506 | 29,381 | 29,811 | 29,380 | **35,951** |
| cloudphysics | 49,273 | 49,906 | 49,384 | 49,274 | 49,275 | 49,273 | **49,626** |

(wiki2018: 50,000/50,000 misses for every policy — capacity 32 ≪ unique pages
in this trace, so every policy hits 100% miss rate; not informative for
ranking policies on this trace.)

## 6. Aggregate pattern

Mean misses across all 7 families (capacity 32):

| Policy | Mean misses | vs. LRU |
|---|---|---|
| lru | 34,667.14 | 0.00% |
| rest_v1 | 34,667.14 | -0.00% |
| blind_oracle_lru_combiner | 34,668.57 | -0.00% |
| predictive_marker | 34,843.86 | -0.51% |
| trust_and_doubt | 35,111.29 | -1.28% |
| **evict_value_v1** | **36,344.14** | **-4.84%** |
| blind_oracle | 43,116.43 | -24.37% |

(Sign convention in the `.md` aggregate: negative % = more misses than LRU,
i.e. worse.)

## 7. Is `evict_value_v1` better or worse than LRU, by trace?

**Worse on 6 of 7 families, tied on 1 (wiki2018, where every policy is at the
100%-miss floor and no policy can be distinguished):**

| Trace family | Result vs. LRU |
|---|---|
| brightkite | Worse (+10.5% more misses) |
| citibike | Worse (+7.9% more misses) |
| cloudphysics | Worse (+0.7% more misses) |
| metacdn | Worse (+22.4% more misses — largest gap) |
| metakv | Worse (+1.7% more misses) |
| twemcache | Worse (+1.8% more misses) |
| wiki2018 | Tie (both at 50,000/50,000 — uninformative) |

This is the **third independent data point** showing `evict_value_v1`
underperforming LRU (after the cap256/1-trace validation pass and the
cap32/5k-request validation-sanity check), now across the full 7-trace set at
capacity 32. Direction is consistent across all three; magnitude varies
(+39% at cap256/50k brightkite-only, +11.8% at cap32/5k brightkite-only,
+10.5% at cap32/50k brightkite — same trace, same direction, decreasing
magnitude as request count/capacity context changes, but never reversing
sign).

## 8. Should `blind_oracle` remain diagnostic-only?

**Yes.** `blind_oracle` is the worst policy in every single trace family
(mean -24.37% vs. LRU, more than 5x worse than `evict_value_v1`'s gap). This
matches the sanity audit's root cause: the comparison script never attaches
real predictions to `blind_oracle`, so it degenerates to evicting the
smallest `page_id` — a known, already-diagnosed configuration gap, not a new
finding. `blind_oracle` is correctly excluded from `TABLE3_POLICIES` and this
cap32 data gives no reason to revisit that exclusion. Do not cite its number
as a meaningful baseline in any manuscript narrative.

## 9. Is cap32 safe to merge later into the final canonical CSV?

**Yes, with no reservations found in this check.** Evidence:

- Exit code 0, expected row count (49) achieved exactly.
- Column schema (`trace_name, trace_family, path, capacity, policy, misses,
  hit_rate`) matches what the merge script in
  `kbs_full_policy_comparison_execution_plan.md` §8 expects (`csv.DictReader`
  / `DictWriter` on a shared `fieldnames`).
- No duplicate `(trace_name, capacity, policy)` triples in the 49 rows
  (visually confirmed: each of the 7 traces appears exactly once per policy).
- Numbers are directionally consistent with 2 independent prior data points
  (validation pass, validation-sanity check) — no internal contradiction that
  would suggest a wiring bug introduced specifically in this run.
- The merge script itself (concatenate 4 chunk CSVs, write one header) is
  purely mechanical and chunk-order-independent — cap32's file can sit
  untouched until cap64/128/256 exist, then be merged in any order.

## 10. Recommendation for cap64

- **Do not launch automatically as a "next step" without a decision
  checkpoint.** cap32 already shows `evict_value_v1` losing on 6/7 families;
  before committing the next ~1.5–3 days of compute, confirm the team still
  wants the full canonical sweep on the current model/horizon (H=4,
  random_forest) rather than reconsidering training first.
- **If proceeding**, cap64 is the next cheapest chunk (cost scales
  ~proportionally with capacity per the execution plan's cost model) —
  recommend running it next, then spot-checking its aggregate `.md` the same
  way as this report before deciding on cap128/cap256.
- **Before cap64 specifically**, finalize the SIEVE/FIFO-Reinsertion scope
  decision (see `reports/kbs_baseline_gap_action_plan.md`) — if either is to
  be implemented and run, doing so in the same remaining chunked passes avoids
  a second multi-day sweep later.
- This report does not launch cap64 and takes no position beyond what the
  data supports — the go/no-go decision is the user's, per the task's
  read-only constraint.
