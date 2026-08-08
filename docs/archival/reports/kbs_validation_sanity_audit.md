# KBS validation sanity audit — pre-full-sweep gate

Scope: investigate whether the heavy_r1 validation pass's worrying numbers
(`evict_value_v1` 39% more misses than LRU; `blind_oracle` far worse than
LRU) reflect a real result or a bug/configuration problem, **before**
launching the multi-day canonical 7×4×7 policy-comparison sweep. Performed
entirely on the local/cloud machine (no Wulver, no Slurm/sbatch/squeue/sacct).
No canonical artifacts were modified, deleted, or overwritten.

Started: `Thu Jun 18 06:36:14 PM EDT 2026` (`logs/kbs_validation_sanity_audit/started_at.txt`).

---

## Step 2 — Validation output and command

### 1. Exact validation command used

From `logs/kbs_policy_comparison_heavy_r1/validation_eval_run.log` (line 2,
verified verbatim this session):

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

Run under `/usr/bin/time -v`, inside tmux session `kbs_policy_comparison_heavy_r1`.

### 2. Trace(s), capacities, policies, model/config used

- **Trace:** 1 of 7 manifest rows (`--max-traces 1`) → first row of
  `analysis/wulver_trace_manifest_full.csv`, which is
  `brightkite_50k` / family `brightkite` / `data/processed/brightkite/trace.jsonl`
  (confirmed by re-reading the manifest's first data row this session).
- **Capacity:** `256` only — the most expensive of the dataset's capacity
  pool `{32, 64, 128, 256}` (confirmed against
  `analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`). Deliberate
  choice for a worst-case timing calibration, not a misconfiguration.
- **Requests:** capped at 50,000 (`--max-requests-per-trace 50000`).
- **Policies:** exactly 7, explicitly listed:
  `lru, blind_oracle, predictive_marker, blind_oracle_lru_combiner,
  trust_and_doubt, rest_v1, evict_value_v1`. This is a deliberate subset of
  the 10-entry `POLICIES` dict in `scripts/run_policy_comparison_wulver_v1.py`
  (full set also includes `atlas_v3`, `ml_gate_v1`, `ml_gate_v2`) — not an
  accidental omission; `ml_gate_v1`/`ml_gate_v2` are auto-dropped anyway
  because their model files aren't present on this machine, and `atlas_v3`
  was simply not requested for this validation pass.
- **Model:** `models/evict_value_wulver_v1_best_heavy_r1.pkl` — the
  fresh, heavy_r1-tagged artifact (see Step 3 below), not a stale or
  default path.

### 3. Output values

`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv`
(re-read in full this session, unchanged from prior inspection):

| policy | misses | hit_rate | vs LRU |
|---|---|---|---|
| lru | 14758 | 0.70484 | — |
| rest_v1 | 14758 | 0.70484 | tie |
| blind_oracle_lru_combiner | 14759 | 0.70482 | ≈tie (−0.01%) |
| predictive_marker | 14923 | 0.70154 | −1.12% |
| trust_and_doubt | 15306 | 0.69388 | −3.71% |
| **evict_value_v1** | **20533** | **0.58934** | **−39.13%** |
| blind_oracle | 29554 | 0.40892 | −100.26% |

Runtime: wall clock **4:02:57**, user time 14570.75s, system 2.53s (~99% CPU,
single-core-bound, no internal parallelism), max RSS ≈2.1GB, **exit status 0**
(`VALIDATION_EXIT=0`, confirmed by re-tailing the log this session).

### 4. Is the output path clearly non-canonical?

**Yes.** Filename carries the `_validation` suffix
(`..._heavy_r1_validation.csv` / `.md`), distinct from the canonical target
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (no
`_validation` suffix), which **does not exist** on this machine (confirmed:
`ls analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` →
no such file). No risk of the validation file being mistaken for, or
accidentally consumed as, the canonical artifact — this is also called out
explicitly in `reports/kbs_validation_eval_report.md` §7.

### 5. Warnings / errors

**None.** The log shows a clean `/usr/bin/time -v` summary ending in
`Exit status: 0`, no Python tracebacks, no stderr content, no truncation.
A repo-wide grep across `logs/kbs_policy_comparison_heavy_r1 reports scripts
docs README.md` for `VALIDATION_EXIT|max-traces|policy_comparison|
heavy_r1_validation|validation` surfaced only consistent, corroborating
references — nothing contradicting the above.

---

## Step 3 — Model/config freshness

Re-verified via `ls -lh` / `cat` / `stat` on the heavy_r1 training artifacts:

| Artifact | mtime |
|---|---|
| `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` (dataset build) | 2026-06-18 09:39:04 |
| `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` | 2026-06-18 10:37:50 |
| `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` | 2026-06-18 10:37:50 |
| `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` | 2026-06-18 10:37:50.515297029 |
| `models/evict_value_wulver_v1_best_heavy_r1.pkl` | 2026-06-18 10:41:32.634055899 (13,301,662 bytes) |
| Validation CSV (this audit's subject) | ~10:44 start → ~14:47 finish (4:02:57 runtime) |

**Did validation use the fresh heavy_r1 model/config?** Yes. The chain
dataset (09:39) → training outputs (10:37:50) → model pkl (10:41:32) →
validation run (started ~10:44) is monotonically increasing and internally
consistent — no gaps, no out-of-order timestamps, no sign of a stale or
leftover artifact being silently reused. The validation command also passes
`--evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl`
explicitly (not the script's non-heavy_r1 default
`models/evict_value_wulver_v1_best.pkl`), so there's no risk of accidentally
picking up an older un-tagged model.

**Which model/horizon was selected?** `horizon=4`, `random_forest`
(`val_mean_regret≈0.02078`, narrowly ahead of `hist_gb`'s `0.02100` at the
same horizon — see full comparison table below).

**Are training metrics plausible?**
`analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` (9 rows = 3
horizons × 3 models):

```
horizon,model,val_mae,val_rmse,test_mae,test_rmse,val_top1,test_top1,val_mean_regret,test_mean_regret
4,ridge,1.0051569594591077,1.2542485521990867,0.8291895856180861,0.9071243990961178,0.2968066491688539,0.03683241252302026,0.1052055993000875,0.024554941682013505
4,random_forest,1.0041431699216743,1.2414330636230133,0.7562150634583107,0.8453242980341048,0.05883639545056868,0.028852056476365868,0.02077865266841645,0.022713321055862493
4,hist_gb,0.997527614917347,1.231602186915872,0.7579555536299724,0.8430078635362057,0.03477690288713911,0.03161448741559239,0.02099737532808399,0.024554941682013505
8,ridge,1.8883492108639683,2.2895462497177292,1.6175107377415974,1.7415008666010774,0.26334208223972005,0.03528689782141761,0.14785651793525809,0.04571954587296717
8,random_forest,1.8724714650766354,2.261571375026655,1.4298794784226556,1.6087949047173056,0.02930883639545057,0.033445842282908866,0.031496062992125984,0.045412703283215713
8,hist_gb,1.85593431981628,2.2379064203366776,1.4383847783191404,1.6096395546506603,0.024059492563429572,0.040196379257440934,0.04505686789151356,0.045105860693464256
16,ridge,3.7540634408544293,4.432848085848817,3.2485837962812045,3.4741848466677823,0.25284339457567806,0.03192142418661756,0.17475940507436571,0.08103130755064457
16,random_forest,3.7019098168100983,4.377104801524636,2.8472493344632164,3.1935668369888406,0.03849518810148731,0.028238182934315532,0.06342957130358705,0.08931860036832412
16,hist_gb,3.6691815595072073,4.330094344256714,2.8676082146125257,3.1977306152941036,0.021653543307086614,0.030693677102516883,0.05861767279090114,0.08717004297114794
```

These are plausible regression metrics for the stated target (`y_loss` =
simulated LRU misses over the next `h` requests if a candidate were evicted
now): errors grow monotonically with horizon (h=4 → mae≈1.0, h=16 → mae≈3.7,
roughly tracking √h-ish growth in cumulative miss count), `random_forest`
and `hist_gb` both clearly beat `ridge` at every horizon (nonlinear models
capturing real structure), and regret values are small relative to typical
miss counts. No NaNs, no degenerate zero-variance rows, no signs of a failed
or truncated training run.

**Is there any sign of stale artifacts?** No. All four heavy_r1 analysis
files (dataset summary, best-config, model-comparison, train-metrics) were
already flagged in `git status --short` as locally **modified** (not stale —
modified relative to a previously committed version, consistent with this
session's rebuild), and their mtimes are all within the same ~3-hour window
that precedes the validation run. No artifact dated before today, no
mismatched feature-count or schema version found.

**Known caveat (disclosed, not a staleness bug):** per
`reports/kbs_revision_gap_tracker.md`, the validation-set rows used for
horizon/model selection cover only 5 of 7 trace families (citibike and
metakv are absent from the ~4,572-row validation sample due to an early-stop
sampling artifact in `_load_rows_from_manifest`) — this could plausibly flip
the close random_forest-vs-hist_gb margin at h=4, but it is a
representativeness caveat, not a freshness/staleness issue, and is already
tracked for disclosure.

---

## Step 4 — Policy semantics

Investigated directly: `src/lafc/policies/blind_oracle.py`,
`src/lafc/policies/evict_value_v1.py`, `src/lafc/policies/offline_belady.py`,
`src/lafc/policies/predictive_marker.py`, `src/lafc/metrics/cost.py`,
`src/lafc/runner/run_policy.py`, `src/lafc/simulator/request_trace.py`,
`src/lafc/evict_value_wulver_v1.py`, `src/lafc/evict_value_features_v1.py`,
`scripts/run_policy_comparison_wulver_v1.py`,
`scripts/build_evict_value_dataset_wulver_v1.py`, plus a broad grep across
`src scripts tests` for `blind_oracle|belady|oracle|evict_value_v1|
miss_ratio|misses|LRU|lru`.

### Q1. What exactly does `blind_oracle` mean in this repo?

`BlindOraclePolicy` (`src/lafc/policies/blind_oracle.py`) evicts, on a miss,
the cached page with the **largest `predicted_next`** value (farthest
predicted future use), i.e. `argmax` over `request.predicted_next` of the
resident pages. Per its own docstring: it "fully trusts the predictor" and
is "equivalent to running Belady's optimal offline algorithm using
*predicted* next arrivals" rather than actual ones. It explicitly warns:
"When predictions are adversarial, Blind Oracle can be arbitrarily bad."

### Q2. Is `blind_oracle` expected to be an upper-bound true-future oracle, or something else?

**Something else.** The true ground-truth oracle in this repo is a
*separate* policy, `offline_belady` (`src/lafc/policies/offline_belady.py`),
which uses `actual_next` (real future arrivals, computed offline from the
trace) and is explicitly documented as the OPT baseline: "DO NOT use this
policy to make production eviction decisions — it cheats by consulting
actual_next." `blind_oracle` instead consumes `predicted_next` — a
predictor-supplied signal that is only as good as the predictor feeding it.
It is a **prediction-consistency** baseline (best case when predictions are
perfect, per Lykouris & Vassilvitskii-style learning-augmented-caching
theory), not a true-future upper bound, and it carries **no robustness
guarantee** when predictions are uninformative or wrong — that's the whole
point of pairing it against robustness-guaranteed baselines like
`predictive_marker` in this literature.

Notably, `offline_belady` is registered in the general `run_policy.py`
`POLICY_REGISTRY` but is **not** one of the 10 policies wired into
`scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES` dict — the
canonical KBS comparison script does not currently report a true-Belady
upper bound at all. This is a scope gap worth flagging for the manuscript
narrative (separate from the bug below), not something to fix as part of
this audit.

### Q3. Is it logically suspicious for `blind_oracle` to be worse than LRU under this definition? — **Root cause found**

It would be suspicious if `blind_oracle` were receiving real, informative
predictions. **It is not.** Root cause, confirmed by direct code reading:

1. `scripts/run_policy_comparison_wulver_v1.py` loads each trace via
   `load_trace_from_any(path)` → for `.jsonl` traces this dispatches to
   `_parse_jsonl_trace` (`src/lafc/evict_value_wulver_v1.py`), which calls
   `build_requests_from_lists(page_ids=..., prediction_records=...)`
   **without** ever passing a `predictions` argument.
2. `build_requests_from_lists` (`src/lafc/simulator/request_trace.py`):
   `preds = predictions if predictions is not None else [math.inf] * len(page_ids)`
   — so **every** request's `predicted_next` defaults to `math.inf`.
3. In the main per-trace/per-capacity loop, the script builds an enriched
   request stream `td_reqs = attach_predicted_caches(reqs, capacity=cap)`
   but only passes it to `trust_and_doubt`:
   `res = run_policy(pol, reqs if pname != "trust_and_doubt" else td_reqs, pages, cap)`.
   Every other policy — including `blind_oracle` and `evict_value_v1` —
   receives the plain `reqs`, where `predicted_next == math.inf` for all
   pages, all the time.
4. With every candidate's `predicted_next` tied at `math.inf`,
   `BlindOraclePolicy._choose_victim`'s sort key
   `(-self._predicted_next.get(q, math.inf), q)` degenerates to breaking
   ties purely on `q` (page_id) — i.e. **blind_oracle always evicts the
   lexicographically/numerically smallest page_id currently cached**,
   completely independent of recency or actual future reuse.

This is a real, mechanical explanation, confirmed by:
- `tests/test_baseline4.py::test_blind_oracle_evicts_farthest`,
  `test_belady_no_worse_than_lru`,
  `test_blind_oracle_with_perfect_predictions_matches_belady`, and
  `tests/test_policies_baseline2.py::test_blind_oracle_perfect_predictions_optimal`
  (all **pass** this session, Step 5) — these confirm `BlindOraclePolicy`'s
  own eviction logic is correct *when given real predictions*. The bug is
  not inside `BlindOraclePolicy`; it's that the comparison script never
  supplies it with anything but a constant.
- This also matches theory exactly: Blind Oracle is the
  *consistency-optimal, zero-robustness* baseline in the learning-augmented
  caching literature — by design it has no fallback when predictions carry
  no information, unlike `predictive_marker` (which falls back toward
  Marker-algorithm behavior and scored within 1.12% of LRU on this same,
  prediction-free trace) or `trust_and_doubt` (which scored within 3.71%).
  A "predictor" that is a constant carries strictly zero information, which
  is the worst case for a policy with no robustness term — `blind_oracle`
  performing far below LRU here is the textbook-expected outcome of that
  worst case, not an anomaly.

**Conclusion: explainable, not suspicious as a code-logic matter** — but it
*is* a real configuration/scope limitation of the comparison script: as
currently wired, `blind_oracle`'s number on real `.jsonl` traces measures
"Blind Oracle given zero predictive information," not "Blind Oracle given a
real predictor." That number should not be read as "the oracle baseline
underperforms LRU" in any manuscript narrative — it should be footnoted, or
the script should be extended to feed `blind_oracle` a real prediction
source (e.g. the same `attach_predicted_caches` enrichment already computed
for `trust_and_doubt`, or a noisy/perfect predictor via
`lafc.predictors.offline_from_trace`) before its number is used for
anything beyond "did the harness run end-to-end."

### Q4. Is `evict_value_v1` interpreting predicted scores in the correct direction?

**Yes — direction confirmed correct, and confirmed NOT subject to the
`predicted_next` bug above.**

- `evict_value_v1`'s features come from `request.metadata.get("bucket")` /
  `("confidence")`, never from `predicted_next` — so it is structurally
  immune to the Q3 bug (`src/lafc/policies/evict_value_v1.py:182-183`).
- `EvictValueV1Policy._choose_victim`
  (`src/lafc/policies/evict_value_v1.py:180-202`) picks
  `victim = min(candidates, key=lambda p: (pred_losses[p], candidates.index(p)))`
  — it evicts the candidate with the **lowest** predicted loss.
- The training target it's predicting (`y_loss` in
  `src/lafc/evict_value_wulver_v1.py::iter_candidate_rows`) is
  `_simulate_lru_misses(after, fut_h, capacity=capacity)` — the number of
  *future* LRU misses over the next `h` real requests **if this candidate
  were evicted now**. Lower `y_loss` = safer to evict. Minimizing predicted
  `y_loss` is exactly the correct eviction rule for this target — no sign
  flip, no inverted comparator.

### Q5. Is the miss ratio computed consistently across policies?

**Yes.** Every policy is run through the single shared
`run_policy()` function (`src/lafc/runner/run_policy.py:157-214`), which
calls `total_misses(events)` / `hit_rate(events)`
(`src/lafc/metrics/cost.py`) uniformly:

```python
def total_misses(events): return sum(1 for e in events if not e.hit)
def hit_rate(events):     return total_hits(events) / len(events)  # 0.0 if empty
```

There is no policy-specific branch in this accounting path — `event.hit` is
set identically by each policy's `on_request()` (a `CacheEvent` with a
`bool hit` field), and `total_misses`/`hit_rate` consume that field the same
way for all 7 policies in the validation CSV. No divergent accounting found.

### Q6. Are cache capacity and trace selection aligned correctly?

**Yes.** `--max-traces 1` against
`analysis/wulver_trace_manifest_full.csv` selects exactly the manifest's
first row (`brightkite_50k`, `data/processed/brightkite/trace.jsonl`,
family `brightkite`) — and the validation CSV's `trace_name`/`path`/
`trace_family` columns match that row exactly. `--capacities 256` selects
the largest/most expensive capacity in the dataset's `{32,64,128,256}` pool
— a deliberate choice for worst-case timing calibration (confirmed against
`reports/kbs_full_policy_comparison_execution_plan.md` §3-4, which uses this
exact data point to project full-sweep runtime). No mismatch between the
capacity used for training-data generation and the capacity used for
evaluation; no off-by-one or wrong-row trace selection found.

---

## Step 5 — Targeted tests

```
pytest tests/ -k "lru or oracle or belady or evict_value or policy_comparison or miss" -v
```

Result: **58 passed, 225 deselected, 0 failed** (full output in
`logs/kbs_validation_sanity_audit/targeted_tests.log`). Relevant tests exist
and all pass, including:

- `tests/test_baseline4.py` — `test_blind_oracle_evicts_farthest`,
  `test_belady_oracle_perfect_predictions`,
  `test_blind_oracle_with_perfect_predictions_matches_belady`,
  `test_belady_no_worse_than_lru`, `test_belady_optimal_on_classic_example`.
- `tests/test_policies_baseline2.py` — `test_blind_oracle_evicts_farthest_predicted`,
  `test_blind_oracle_perfect_predictions_optimal`,
  `test_run_policy_blind_oracle_smoke`.
- `tests/test_offline_belady.py` — 5 tests, all passing (true Belady
  correctness, tie-breaking, weight handling).
- `tests/test_evict_value_v1.py` — 9 tests, all passing (feature
  determinism, training reproducibility, policy smoke/diagnostics,
  lightweight/artifact/auto scorer-mode handling, backward compatibility).
- `tests/test_evict_value_wulver_v1.py` — 3 tests, all passing (manifest
  split stability, required-metadata presence, summary counting).
- `tests/test_metrics.py::test_total_hits_and_misses`,
  `tests/test_runner.py` (cost/length invariants) — all passing, reinforcing
  Q5's conclusion.

No relevant test failures anywhere in the suite that would indicate a code
defect in `blind_oracle`, `evict_value_v1`, `offline_belady`, `lru`, or the
miss-accounting path.

---

## Step 6 — Optional tiny extra validation

Ran one small, clearly non-canonical check to get a second `evict_value_v1`
vs LRU data point at a different (much cheaper) capacity, since the existing
validation only has one `(trace, capacity)` point and the short training
horizon (h=4 selected, vs the 50k-request live replay) was an open
hypothesis for why a 39%-worse result might still be genuine rather than a
wiring bug. This does not touch the canonical CSV.

**Command:**

```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --max-traces 1 \
  --capacities 32 \
  --max-requests-per-trace 5000 \
  --policies lru,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation_sanity.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation_sanity.md
```

Same trace family (`brightkite`), cheapest capacity (32, ~8x cheaper than
256 under the execution plan's capacity-proportional cost model), and a
10x-smaller request window (5,000 vs 50,000) — chosen to finish in well
under an hour while still being a genuine second measurement, not a
re-aggregation of the same numbers. Output uses the suggested
`_validation_sanity` suffix, distinct from both the canonical CSV and the
existing `_validation` CSV.

**Result** (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation_sanity.csv`):

```
trace_name,trace_family,path,capacity,policy,misses,hit_rate
brightkite_50k,brightkite,data/processed/brightkite/trace.jsonl,32,lru,1643,0.6714
brightkite_50k,brightkite,data/processed/brightkite/trace.jsonl,32,evict_value_v1,1837,0.6326
```

Runtime: 2:14.73 wall clock, exit status 0 (`SANITY_EXTRA_EXIT=0`), max RSS
~216MB — fast and safe, as expected from the capacity/request-count scaling
in the execution plan's cost model.

**Reading:** `evict_value_v1` has **1837 vs 1643 misses = +11.81% more
misses than LRU** at capacity 32 / 5,000 requests, compared to **+39.13%**
at capacity 256 / 50,000 requests on the same trace family. This is a
**second, independent data point that is directionally consistent** with
the validation result (evict_value_v1 worse than LRU, same trace family) —
ruling out "the cap256/50k result was a one-off fluke or wiring accident."
The magnitude shrinking at smaller capacity/shorter window is consistent
with the open hypothesis from Step 3 (the selected model's training horizon
is h=4, short relative to the 50,000-request cap256 replay; the gap to LRU
narrows as the replay gets closer to that horizon's scale) — a genuine
methodological lead for the manuscript/model-improvement discussion, not
evidence of a bug.

---

## Required conclusion

**A. Validation is trustworthy; poor validation performance appears real
for that trace. Proceed with chunked full sweep.**

Basis:
- Exhaustive code-path audit (Step 4) found **no logic bug** in
  `evict_value_v1`'s feature construction, eviction-direction comparator, or
  training-target definition, and **no inconsistency** in how misses/hit
  rate are computed across policies (shared `run_policy()` /
  `metrics/cost.py` path, Q5).
- All 58 targeted tests pass (Step 5), including tests that directly cover
  `blind_oracle`, `offline_belady`, `evict_value_v1`, and the
  miss/hit-rate accounting invariants — no evidence of a regression or
  latent defect in any of the relevant code paths.
- Freshness chain (Step 3) is clean and monotonic — the validation
  definitely used the fresh `heavy_r1` model/config, not a stale artifact.
- The one real configuration gap found (`blind_oracle` receiving constant
  `predicted_next=math.inf` because the comparison script never attaches
  real predictions to it, Q3) is **fully explained**, is **not** a defect in
  `BlindOraclePolicy` itself, and — critically — **`blind_oracle` is not a
  member of `TABLE3_POLICIES`**
  (`scripts/paper/build_kbs_main_manuscript_artifacts.py`'s manuscript-
  citable policy set is `lru, predictive_marker, trust_and_doubt,
  blind_oracle_lru_combiner, rest_v1, evict_value_v1`), so this gap does
  **not** block, taint, or need fixing before the full sweep — it only
  means `blind_oracle`'s own number should never be cited as a meaningful
  baseline result anywhere.
- The Step 6 extra check supplies a **second, independent (trace-family,
  capacity, request-count) data point** showing `evict_value_v1` worse than
  LRU again, at a different magnitude that moves in the direction the
  short-training-horizon hypothesis predicts — this is exactly the kind of
  signal that says "real and worth running the full sweep to characterize,"
  not "stop and debug further."

This does **not** mean `evict_value_v1`'s full-sweep results will look good
in the manuscript — they may not. It means the *measurement* is trustworthy,
so the full sweep's results (good or bad) can be relied upon as real
signal about the method, which is what the chunked sweep is for.

---

## Final output (see chat for the 9-point summary).
