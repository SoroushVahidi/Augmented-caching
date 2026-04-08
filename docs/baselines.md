# Baselines: Learning-Augmented Caching

## Baseline 3 (Main): TRUST&DOUBT

### Paper citation

Antoniadis, A., Coester, C., Eliáš, M., Polak, A., & Simon, B. (2020). **Online Metric Algorithms with Untrusted Predictions**. ICML 2020, PMLR 119:345–355. Supplementary material contains Algorithm 3 pseudocode.

## Step-0 paper-to-code note (TRUST&DOUBT)

1. **Setting implemented**: unweighted paging (unit miss cost), cache size `k`, request sequence `r_t`.
2. **Prediction role**: TRUST&DOUBT uses **predicted cache states** `P_t` (predictor configuration at time `t`), not directly next-arrival values.
3. **State maintained**: `A` (ancient), `stale`, `U`, `M`, `C`, `T`, `D`, and for each clean page `q`: `p_q`, `trusted(q)`, threshold `t_q`, and `q_interval_change`.
4. **Eviction/update logic**: implemented from Supplementary Algorithm 3 (phase reset; steps 1–4; threshold doubling across doubted intervals).
5. **Difference vs Blind Oracle / Predictive Marker**:
   - Blind Oracle fully trusts predicted next-arrival advice.
   - Predictive Marker remains a marking algorithm.
   - TRUST&DOUBT adaptively alternates trust/doubt and may evict marked pages (through set `T`).
6. **Interpretation-required points**:
   - Paper says “arbitrary” choices in several places; we use deterministic LRU tie-breaking for reproducibility.
   - Paper describes non-lazy formulation; implementation follows Remark 10 by simulating non-lazy cache in background and serving requests lazily.
   - Caching distance for MTS-style error is interpreted as `|X \ Y|` between equal-size cache states.
7. **Mapping to paper sections**:
   - Main description: ICML paper Section 4.
   - Full operational pseudocode: Supplementary Algorithm 3.
   - Non-lazy/lazy implementation note: Remark 10.

## Implemented baselines

- `lru`
- `marker`
- `blind_oracle`
- `predictive_marker`
- `trust_and_doubt` (Baseline 3 target)

## Prediction interfaces supported

- **Next-arrival predictions** (`predictions` in trace).
- **Predicted cache states** (`predicted_caches` in trace / `metadata["predicted_cache"]`).
- Conversion utility from next-arrival predictions to `P_t` using Blind-Oracle conversion (paper Sec. 1.3).

## Error metrics exposed

- `prediction_error_eta` (legacy next-arrival eta).
- `eta_unweighted` (LV-style unweighted next-arrival eta).
- `cache_state_error_total` (MTS-style state error over `P_t` vs offline Belady states).

## Deviations / non-goals

- We did **not** implement full generic MTS algorithms; only paging-specialized machinery needed for baseline quality.
- We did **not** formally verify theorem constants/competitive-ratio proofs in code.

## Running

```bash
python -m lafc.runner.run_policy \
  --policy trust_and_doubt \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --perfect-predictions \
  --derive-predicted-caches
```

Outputs:
- `summary.json`
- `metrics.json`
- `per_step_decisions.csv`
