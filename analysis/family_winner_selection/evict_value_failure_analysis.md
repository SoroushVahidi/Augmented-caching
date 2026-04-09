# evict_value_v1 focused failure analysis (family winner-selection harness)

## 1) Input inspected
- `analysis/family_winner_selection/winner_selection_results.csv` (30 slices: 5 traces × 2 regimes × 3 capacities).
- Comparator set used for direct pairwise deltas: `robust_ftp_d_marker`, `blind_oracle_lru_combiner`, `rest_v1`, and `lru`.
- Derived slice table written to `analysis/family_winner_selection/evict_value_failure_slices.csv`.

## 2) Where evict_value_v1 loses most badly
- Worst gap to *best policy in-slice*: **+4 misses**.
  - stress::predictor_good_lru_bad | clean | C=3 : evict_value=9 vs best=5 (blind_oracle_lru_combiner).

Top failure slices by `delta_vs_best`:

| trace | regime | cap | evict_value | best_policy | best_misses | delta_vs_best | delta_vs_robust | delta_vs_blind_combiner | delta_vs_rest | delta_vs_lru |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| stress::predictor_good_lru_bad | clean | 3 | 9 | blind_oracle_lru_combiner | 5 | 4 | 4 | 4 | 2 | 2 |
| stress::predictor_bad_lru_good | clean | 2 | 9 | atlas_v3 | 6 | 3 | 3 | 3 | 3 | 3 |
| stress::predictor_bad_lru_good | noisy | 2 | 9 | atlas_v3 | 6 | 3 | 3 | 3 | 3 | 3 |
| stress::predictor_good_lru_bad | clean | 2 | 10 | blind_oracle_lru_combiner | 7 | 3 | 3 | 3 | 2 | 1 |
| file::example_atlas_v1 | clean | 3 | 9 | atlas_v3 | 6 | 3 | 3 | 3 | 3 | 0 |
| file::example_atlas_v1 | noisy | 3 | 9 | atlas_v3 | 6 | 3 | 3 | 3 | 3 | 0 |
| file::example_unweighted | clean | 2 | 10 | blind_oracle_lru_combiner | 7 | 3 | 3 | 3 | 0 | 0 |
| file::example_unweighted | noisy | 2 | 10 | blind_oracle_lru_combiner | 7 | 3 | 3 | 3 | 1 | 0 |
| stress::mixed_regime | clean | 2 | 10 | blind_oracle_lru_combiner | 7 | 3 | 3 | 3 | 0 | 0 |
| stress::predictor_bad_lru_good | clean | 3 | 8 | atlas_v3 | 6 | 2 | 2 | 2 | 2 | 2 |

## 3) Direct comparison vs requested baselines

| comparator | mean miss delta (evict_value - comparator) | losses | ties | wins | worst miss delta |
|---|---:|---:|---:|---:|---:|
| robust_ftp_d_marker | 1.833 | 24 | 6 | 0 | 4 |
| blind_oracle_lru_combiner | 1.700 | 23 | 7 | 0 | 4 |
| rest_v1 | 1.267 | 21 | 9 | 0 | 3 |
| lru | 0.667 | 13 | 17 | 0 | 3 |

Interpretation: all pairwise records are non-negative; evict_value_v1 never wins a slice against any requested comparator in this harness output.

## 4) Regime/capacity/trace localization

### By regime
- clean: mean delta_vs_best=2.00, mean delta_vs_lru=0.73, lru-loss slices=7/15.
- noisy: mean delta_vs_best=1.73, mean delta_vs_lru=0.60, lru-loss slices=6/15.

### By capacity
- C=2: mean delta_vs_best=2.60, mean delta_vs_lru=0.70, lru-loss slices=3/10.
- C=3: mean delta_vs_best=2.40, mean delta_vs_lru=0.90, lru-loss slices=6/10.
- C=4: mean delta_vs_best=0.60, mean delta_vs_lru=0.40, lru-loss slices=4/10.

### By trace
- file::example_atlas_v1: mean delta_vs_best=2.33, mean delta_vs_lru=0.33, lru-loss slices=2/6.
- file::example_unweighted: mean delta_vs_best=1.67, mean delta_vs_lru=0.33, lru-loss slices=2/6.
- stress::mixed_regime: mean delta_vs_best=1.50, mean delta_vs_lru=0.00, lru-loss slices=0/6.
- stress::predictor_bad_lru_good: mean delta_vs_best=2.00, mean delta_vs_lru=2.00, lru-loss slices=6/6.
- stress::predictor_good_lru_bad: mean delta_vs_best=1.83, mean delta_vs_lru=0.67, lru-loss slices=3/6.

Most concentrated failure regime visible in this harness: `stress::predictor_bad_lru_good` (evict_value loses to LRU in 6/6 slices, avg +2 misses).

## 5) Likely failure mechanism from code + outputs (no new heavy experiments)

### A. Lightweight surrogate scoring appears directionally misaligned
- In `scripts/run_family_winner_selection.py`, evict_value is forced to `EvictValueV1Policy(scorer_mode="lightweight")`, so artifact model behavior is not evaluated in this harness.
- In `src/lafc/policies/evict_value_v1.py`, victim selection is `min(predicted_loss)`.
- In the lightweight linear weights, `candidate_predictor_score` and `candidate_lru_score` have **positive** weights. Given min-selection, this pushes eviction toward *lower* predictor/LRU scores (more keeper-like pages), which conflicts with standard eviction intuition where higher risk/older pages should usually be preferred eviction candidates.
- Negative flags (`candidate_is_predictor_victim`, `candidate_is_lru_victim`) partially counteract that, but with much smaller magnitude in aggregate than the continuous score terms across many states.

### B. Missing artifact-backed behavior is a real confounder
- The harness winner report itself is therefore a verdict on the surrogate path, not the trained checkpoint path.
- Prior first-check artifacts show evict_value_v1 can do better than LRU/rest in a different setup, so the current loser status may be mostly a lightweight-mode problem, not a full-method proof of failure.

### C. Trace/regime mismatch pattern
- Failures are worst in the explicitly adversarial `predictor_bad_lru_good` stress trace and at C=2/3.
- At C=4, deltas compress heavily (many ties), suggesting the policy is most fragile when eviction pressure is high.

### D. Bad target/supervision?
- From this harness alone, there is not enough evidence to blame the underlying supervision target directly; the dominant visible factor is surrogate scoring quality + surrogate-only evaluation path.

## 6) Conservative decision

**continue only if artifact-backed mode is tested**
