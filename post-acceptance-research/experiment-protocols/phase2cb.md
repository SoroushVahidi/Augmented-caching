# Phase 2C-B controlled learning and replay

The primary experiment compares fixed H=8 and fully resolved LRU-relative advantage on identical training groups, causal features, candidate ranks, estimators, weights and gate calibration. Retain the seven historical family folds, 32/64/128 slots, unit-size mandatory admission, 10,000 warm-up requests, and 40,000 scored requests per cell. Independent confirmation workloads stay sealed.

## Preregistered stage matrix

| Stage | Configurations per fold | Fits per fold | Replays per fold |
|---|---:|---:|---:|
| A. Data scaling | 16 | 16 | 96 |
| B. Horizons and features | 8 | 8 | 48 |
| C. Components and shortlists | 9 | 9 | 54 |
| D. Gates, controls, CPU | 44 | 0 | 264 plus 2 chronological reference-path checks |

For each configuration, three capacities are replayed on validation and three on the outer family. All fits are deterministic. The planned total is 231 fit components and 3,234 validation/outer replays plus 14 chronological reference-path checks. Bounded smoke runs and retrieved Phase 2B baselines are separate.

### Stage A

Run budgets 1,176, 20,000, 50,000 and 100,000 (pooled eligible decisions before family separation). The 1,176 pilot preserves its earlier decision population. Larger budgets are exactly nested. Fit Ridge and histogram gradient boosting using legacy F0 and low-frequency K=8. Archive actual train groups, target-positive and resolved-positive groups, thresholds, prediction metrics and closed-loop results. The larger budget has enough training positives for development, and sparse designated validation families can lead to abstention.

### Stage B

Choose one architecture per fold from the sum of the two 100,000-budget Stage A validation replay miss totals, then fewer validation interventions, then Ridge on exact ties. Reuse the matching 100,000 Stage A fits as controls. Compare H=32 and H=C in F0. H=4C is omitted because the verified design-weighted action disagreement with resolved is 0.012%, below the 0.05% redundancy threshold for this bounded comparison. H=8, H=32, H=C and resolved supply substantially separated label regimes. Use selected architecture and resolved target to compare F1 cleaned legacy, F2 compact history, F3 plus two decay scales, F4 plus ghost information, F5 plus selected workload context, and F6 expanded temporal history. Fit the F1 column mask on its own training rows. All feature comparisons reuse the same gated cache transition and shortlist.

### Stage C

Use the same selected architecture and 100,000 budget. Compare direct advantage with two regressors estimating prevented misses and additional-miss probability. Clip expected benefit to the nonnegative range and estimated risk to [0,1], then subtract risk. Compare low-frequency K=8 to LRU-tail, SIEVE-derived, ghost-informed, and uniform K=8 with seeds 7, 19 and 42. Select one causal K=16 using estimated captured gain on the six non-outer families only. Full-candidate expansion reuses the K=8 model and its gate as an inference-only diagnostic. Every shortlist includes LRU. Offline fixed-state coverage is distinct from replay performance.

### Stage D and E

Choose one gated finalist per architecture per fold across stages A through C using only that fold's designated validation closed-loop miss count, then fewer overrides, then the stable policy name; exclude full-candidate diagnostic. Preserve each finalist's original threshold. Compare raw argmax, oldest exact ties, oldest approximate ties, baseline-improvement gating, a causal pre-gate that abstains when the oldest object has a count below two, and one two-part risk-cap gate using a threshold and a risk-probability ceiling. Approximate ties use the offline grid 0, .01, .025, .05 and .10. The primary strict improvement grid is 0, .005, .01, .025, .05, .10, .20, .40 and always abstain. Two-part risk ceilings are .25, .50, .75 and 1. Resolve offline selection by higher weighted resolved net advantage, fewer overrides, then stronger abstention. The original single-family validation remains primary. Every stage writes its selection receipt before reading the outer stream.

For each finalist, run a rate-matched full-cache random victim, rate-matched shortlist random victim, gate-conditioned random victim on the random policy's own state, and deterministic low-frequency intervention with the validation-rate budget. Use seeds 7, 19, 42, 101 and 211 for stochastic variants. Rates come from validation-family closed-loop replay separately for each capacity. Report actual rates and paired differences. Gate-conditioned controls preserve gate activation on their own trajectories, which can yield a different outer override rate.

Measure CPU time using one instrumented cache engine for LRU and Phase 2C-B learners. Measure decoding separately. Use preallocated legacy feature buffers when trace metadata permits, incremental request windows, fused Ridge normalization, direct dot products and packed tree traversal with node-wise normalization. Compare packed and reference inference on held-out validation groups. Run one full chronological parity check at C=64 per finalist and fold. Preserve any discrepancy as a separate reference-path replay. Sample a deterministic subset of actual overrides for exact single-action diagnostics outside the main timed loop. The profiling component times nest in decision time and must not be summed across levels.

### Screening and stage gates

Keep the 0.5% relative miss reduction, positive macro-family and leave-one-family-out gain, maximum 2% worst-family regression, matched H=8 improvement where there is an exact comparator, advantage over every matched random and heuristic control, and provisional 5× CPU budget. Only a finalist with all required comparisons and decision parity can pass the combined screen. Different targets, features or shortlist arms do not substitute for an exact H=8 comparator. Keep incomplete comparisons explicit. No result opens independent confirmation.

The staged scope excludes recurrent training, LightGBM (absent in the verified runtime), a feature Cartesian product, and additional optional inner validation. The analysis will report measured rates and identify which limitations are established by a controlled comparison. Slurm uses successful dependencies from verified input smoke to A, B, C, D and the final reducer. The reducer checks stage receipts and model hashes before producing a compact aggregate archive.
