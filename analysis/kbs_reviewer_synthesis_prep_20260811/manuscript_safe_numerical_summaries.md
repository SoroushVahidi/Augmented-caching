# Reviewer Synthesis Prep - 2026-08-11

Scope: generated from existing local artifacts and status documents only. No new experiment was launched, no Wulver contact was made, and no scientific source files were edited.

## Reviewer Evidence Table

| Reviewer comment | Experiment | Status | Strongest result | Caveat | Artifact path |
|---|---|---|---|---|---|
| Reviewer #2 Major 1: fair learned-baseline comparison | Reviewer-fairness baseline pool plus corrected held-out evict_value_v1 replay | PARTIAL | Local original-protocol LRU/SIEVE/FIFO/LRB/3L-Cache/HALP/CACHEUS rows are complete; corrected evict_value_v1 replay is WULVER_ONLY_VALIDATED 42/42 per status docs. | Exact-protocol LRB/3L-Cache/CACHEUS reruns matched to corrected split remain pending on Wulver; HALP fidelity remains LOW_TO_MEDIUM. | `docs/CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md; analysis/reviewer_fairness/policy_comparison_*.csv; analysis/kbs_comparison_fairness_audit.json` |
| Reviewer #2 Major 2: supervision-objective ablation | Four-objective supervision comparison | FINAL_VALIDATED | 84/84 rows all ok; aggregate misses: pairwise 565127, reuse_distance 571456, next_arrival 573059, eviction_loss 601569; eviction_loss worst/tied-worst in 7/7 families. | Objective_pairwise changes the target construction; it is not the same condition as same-target eviction_loss_pairwise in the learning curve. | `analysis/supervision_objective_ablation_v1/policy_comparison.csv; analysis/supervision_objective_ablation_v1/model_registry.json` |
| Reviewer #2 Major 3: offline/online failure explanation | Distribution-shift ablation; same-target learning curve; exact-target oracle; pending C0/C1/C2 | PARTIAL | Sample-size H1 disfavored by 1%-50% learning curve; local distribution shift currently has 24/42 rows and DAgger worsened misses in 10/12 paired cells, improved 2. | The missing causal C0/C1/C2 continuation experiment is still not replaced by these diagnostics. | `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/; analysis/distribution_shift_ablation_v1/; analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/` |
| Reviewer #2 Major 4: practical significance / timing | Practical-significance timing/equivalence evidence | COMPLETE_WITH_CAVEATS in status docs; local timing result not present as final table | Wulver-only status docs record controlled timing complete for 420/420 rows: LRU 4.68 us/request, FIFO-Reinsertion 5.17, SIEVE 9.52, HALP-causal 870.66. | Wulver numbers are not independently verified locally in this pass; wall-clock implementation evidence, not a complexity theorem. | `docs/CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md; analysis/practical_significance_ablation_v1/ (local smoke only if present)` |
| Reviewer #3 Issues 2-3: HALP and SIEVE/FIFO differentiation | Reviewer-fairness baseline pool | ANSWERED_WITH_CAVEATS | SIEVE/FIFO/LRU and HALP comparison rows are locally complete under reviewer_fairness_v1. | HALP is an adapted reimplementation with LOW_TO_MEDIUM fidelity; keep this caveat attached. | `analysis/reviewer_fairness/policy_comparison_{lru,sieve,fifo_reinsertion,halp}.csv; docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md` |
| Reviewer #3 Issue 4 / MC1: mechanism and horizon justification | Exact-target oracle; target degeneracy; reuse-tail diagnostic pending/final not incorporated here | PARTIAL | Exact H4 target oracle single cell loses to LRU: 19079 vs 13225 misses; learned scalar agrees with exact target on 96.47% yet has 15449 misses. | Single-cell oracle result; reuse-tail outcome may change horizon/tail interpretation but will not by itself prove causal excess misses. | `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/summary.json; docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` |
| Reviewer #3 causal continuation-mismatch concern | Continuation-policy C0/C1/C2 causal ablation | MISSING / CONCEPTUAL_BUT_NOT_PRODUCTION_READY per cross-environment matrix | Available distribution-shift evidence narrows the story but does not show DAgger-style correction improves misses. | C0/C1/C2 production interface blocker remains; this campaign is still needed for causal attribution. | `docs/CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md; configs/continuation_policy_causal_ablation_v1.json` |
| Reviewer #3 Issue 6: fallback mechanism | Fallback/gating mechanism | MISSING | No current local evidence; only future diagnostic candidates are documented. | Cannot claim this concern is empirically addressed yet. | `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md; docs/reviewer/kbs_negative_results_interpretation.md` |

## Manuscript-Safe Numerical Summaries

### Four-Objective Comparison

- Artifact: `analysis/supervision_objective_ablation_v1/policy_comparison.csv`.
- Integrity: 84/84 rows are `status=ok`; families=brightkite,citibike,cloudphysics,metacdn,metakv,twemcache,wiki2018; capacities=32,64,128.
- `objective_pairwise`: aggregate misses 565127; mean miss ratio 0.672770; rows 21.
- `objective_reuse_distance`: aggregate misses 571456; mean miss ratio 0.680305; rows 21.
- `objective_next_arrival`: aggregate misses 573059; mean miss ratio 0.682213; rows 21.
- `objective_eviction_loss`: aggregate misses 601569; mean miss ratio 0.716154; rows 21.
- Cell wins/ties by miss ratio: `objective_pairwise` best in 9 cells and tied-best in 4; `objective_eviction_loss` worst in 13 cells and tied-worst in 4; eviction_loss is worst/tied-worst in 7/7 family means.
- Safe wording: this supports objective/surrogate mismatch as a plausible contributor; it does not prove a unique cause.

### 1%-50% Learning Curve

- Artifact: `analysis/supervision_objective_learning_curve_v1/`; final synthesis `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`.
- Apples-to-apples basis: the four families present at every tested fraction (`brightkite`, `citibike`, `cloudphysics`, `metacdn`), 12 family-capacity cells per fraction.
- 1%: scalar MAE 0.986665; scalar miss ratio 0.625606; pairwise miss ratio 0.829929; gap pairwise-scalar 0.204323; scalar better 12/12.
- 2%: scalar MAE 0.983932; scalar miss ratio 0.618331; pairwise miss ratio 0.829577; gap pairwise-scalar 0.211246; scalar better 12/12.
- 5%: scalar MAE 0.983804; scalar miss ratio 0.616458; pairwise miss ratio 0.829621; gap pairwise-scalar 0.213163; scalar better 12/12.
- 10%: scalar MAE 0.982503; scalar miss ratio 0.611021; pairwise miss ratio 0.829658; gap pairwise-scalar 0.218638; scalar better 12/12.
- 25%: scalar MAE 0.982593; scalar miss ratio 0.613652; pairwise miss ratio 0.829569; gap pairwise-scalar 0.215917; scalar better 12/12.
- 50%: scalar MAE 0.982667; scalar miss ratio 0.612613; pairwise miss ratio 0.829979; gap pairwise-scalar 0.217367; scalar better 12/12.
- 25%->50% apples-to-apples: scalar miss ratio change -0.001040; pairwise change +0.000410; gap change +0.001450.
- 1%->50% apples-to-apples: scalar miss ratio change -0.012994; pairwise change +0.000050; gap change +0.013044.
- Full 50% seven-family slice: 42/42 rows, all `status=ok`; scalar better on 18/21 family-capacity cells, ties 3/21, pairwise better 0/21; mean pairwise-minus-scalar miss-ratio gap approximately +0.1611.
- Safe wording: within the tested 1%-50% range, the sample-size explanation is not supported as the primary cause; do not claim more data can never help.

### Exact-Target Oracle

- Artifact: `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/summary.json`; status `COMPLETE`; cell brightkite/capacity64/H4.
- `lru`: misses 13225; miss ratio 0.330625; excess vs LRU 0; gap to Belady 1913.
- `learned_eviction_loss_scalar`: misses 15449; miss ratio 0.386225; excess vs LRU 2224; gap to Belady 4137.
- `exact_finite_horizon_eviction_loss_oracle`: misses 19079; miss ratio 0.476975; excess vs LRU 5854; gap to Belady 7767.
- `offline_belady`: misses 11312; miss ratio 0.282800; excess vs LRU -1913; gap to Belady 0.
- Learned scalar agreement with exact target: 96.4658%; mean target regret 0.035342.
- Safe wording: this disfavors pure model-fitting failure in this cell and points toward target/deployment semantics; it is one-cell diagnostic evidence.

### Distribution-Shift Evidence Available Locally

- Artifact: `analysis/distribution_shift_ablation_v1/policy_comparison.csv`; local completion 24/42 rows, families=brightkite,citibike,cloudphysics,metacdn, paired cells=12.
- DAgger-minus-off-policy mean miss-ratio delta +0.014415; improved 2, worsened 10, tied 0 of 12 paired cells.
- State-shift index decreased in 10/12 paired cells; mean state-shift delta -0.000074.
- Safe wording: trajectory/state shift exists, but the available local directional correction does not show miss-ratio improvement; causal C0/C1/C2 remains unresolved.

### Sample-Size Stopping Decision

- Decision recorded for H1: `STOP_SAMPLE_SIZE_HYPOTHESIS`.
- Fractions tested: 1%, 2%, 5%, 10%, 25%, 50%; 100% intentionally not run by stopping rule, not missing.

