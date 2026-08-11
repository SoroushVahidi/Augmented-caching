# KBS Second-Revision Repository State

Date: 2026-08-09  
Canonical checkout: this `kbs/second-revision-science` repository clone  
Canonical branch: `kbs/second-revision-science`  
Expected baseline HEAD for the current local documentation pass: `63e63acb09c39449bda0a28c8ab2d24f63b2547e`

For the consolidated mechanistic-hypothesis matrix see
[`reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md);
for the per-reviewer-concern status matrix see
[`reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md).
This file stays focused on repository/branch/run-progress status.

**Snapshot convention:** this file (and the sync-status docs it links to)
records a *current snapshot* of an actively evolving branch, not a final
scientific result. Read every number in it as one of three kinds:

- **CURRENT SNAPSHOT** -- a timestamped read of in-progress state (e.g.
  "job X had Y rows at timestamp Z"). Expected to go stale;
  trust the timestamp, not the number, once time has passed.
- **LAST FINALIZED SCIENTIFIC RESULT** -- a phase that reached a clean stop
  and passed an integrity audit (e.g. the `25%` learning-curve fraction,
  `7/7` folds, `42/42` rows). These do not change until a deliberate new
  audit supersedes them.
- **RUNNING EXPERIMENT** -- explicitly still executing; never cite its
  partial numbers as a scientific result, only as progress context.

The durable theory documents (hypothesis map, reviewer coverage map,
experiment registry) intentionally avoid embedding live fold counts for this
reason -- they reference *phase-level* status (`RUNNING`, `FINAL_VALIDATED`,
...) and point back here for the live number.

## Purpose

This note records the structural intent of the local KBS second-revision branch
before final manuscript-facing cleanup:

- `kbs/second-revision-science` is the intended source of truth for the
  reviewer-science code and frozen protocols.
- reviewer evidence under `analysis/`, `models/`, and large derived datasets is
  preserved locally but is not yet treated as fully curated, tracked release
  material.
- historical worktrees remain useful as provenance and comparison points, but
  they are not the intended long-term entrypoint for outside researchers.

## Current source-of-truth boundaries

### Tracked source/configuration

- experiment runners and gates under `scripts/experiments/`
- dataset/build/train drivers under root `scripts/`
- reviewer protocols and frozen configs under `docs/` and `configs/`
- code for external baselines and reproducibility helpers under `src/lafc/`
- fast regression tests under `tests/`

### Generated reviewer evidence kept untracked locally

- `analysis/reviewer_fairness/` policy CSVs, provenance JSONs, fairness certificates
- `analysis/reviewer_fairness_cross_family_v1/`
- `analysis/distribution_shift_ablation_v1/`
- `analysis/practical_significance_ablation_v1/`
- `analysis/supervision_objective_ablation_v1/`
- `analysis/supervision_objective_learning_curve_v1/`
- `analysis/external_learned_baselines/`
- `models/`

### Tracked small audit / provenance summaries

- contamination and temporal-order audits in `analysis/reviewer_fairness/`
- small tracked derived fixtures already committed under `data/derived/`

## Important local caveats

- `analysis/reviewer_fairness/policy_comparison_*.csv` includes both
  `primary_controlled_window` and `deployment_full_stream` rows. Only the
  primary rows are eligible for the main reviewer comparison.
- `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` is
  intentionally contaminated/ineligible and must stay labeled that way.
- `analysis/practical_significance_ablation_v1/` currently contains smoke-scale
  timing evidence plus synthetic cost analyses; the final controlled timing run
  is still separate work.
- `analysis/distribution_shift_ablation_v1/` is a valid partial checkpoint, not
  a completed campaign.
- `analysis/supervision_objective_learning_curve_v1/` is a local explanatory
  diagnostic. It is now `FINAL_50PCT_VALIDATED` for the intended H1
  stopping-rule scope: fractions `1%, 2%, 5%, 10%, 25%, 50%` have been
  tested, and `100%` is intentionally not run due
  `STOP_SAMPLE_SIZE_HYPOTHESIS`.
- the low-fraction checkpoint contains `16` validated units / `96` rows
  across `brightkite, citibike, cloudphysics, metacdn` at fractions
  `1%, 2%, 5%, 10%`.
- the `25%` local extension completed naturally on `2026-08-09`: all `7/7`
  families (`brightkite`, `citibike`, `cloudphysics`, `metacdn`, `metakv`,
  `twemcache`, `wiki2018`), `42/42` expected rows, all `status=ok`, no
  duplicate `(fraction, family, condition, capacity)` keys, no literal
  NaN/Inf strings, and all `14` model artifact SHA-256 hashes verified
  against the CSV. One data characteristic to carry forward: the
  `eviction_loss_pairwise` condition for `twemcache` at fraction `0.25`
  produced zero validation pairs at all three capacities, so
  `validation_pairwise_accuracy` is legitimately blank in the CSV, not a
  data integrity failure.
- the `50%` extension is complete after the final `wiki2018|0.5` resume:
  7/7 families, `42/42` rows, all `status=ok`, duplicate-key count 0,
  NaN/Inf count 0, 30 audit files total, 7/7 fraction-0.5 audit units, 0
  model SHA mismatches, and `campaign_state.json` contains
  `wiki2018|0.500000`. Final synthesis:
  `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`.
  Classification: `COMPLETE_7_OF_7`; scientific decision:
  `STOP_SAMPLE_SIZE_HYPOTHESIS`.
- an exact-target-oracle vs learned-online diagnostic foundation now exists
  locally in `src/lafc/oracle_diagnostics.py` with focused synthetic tests in
  `tests/test_oracle_diagnostics.py`; one local real-trace cell has now been
  run at `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/`
  for `brightkite`, capacity `64`, horizon `4`, canonical window
  `[10000,50000)`.
- that one-cell diagnostic found LRU `13225` misses, exact finite-horizon
  eviction-loss oracle `19079` misses, learned eviction-loss scalar policy
  `15449` misses, and offline Belady `11312` misses; treat this as
  diagnostic evidence for the target/learning decomposition only, not a
  family-general or horizon-sweep conclusion.
- that oracle diagnostic is intentionally distinct from the
  minimum-counterfactual or minimum-Hamming-distance suffix-attribution line:
  the former checks exact-target consistency of a decision, while the latter
  asks which earlier changed decisions are minimally sufficient to remove a
  later excess miss.
- a target-degeneracy diagnostic cell completed locally at
  `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/`. In this
  cell, all `19079` H=4 scored decisions have ordinary zero margin, `63.0%`
  have all candidates tied, and longer horizons break only a minority of H=4
  tied sets (`14.2%` at H=8, `27.6%` at H=16, `39.6%` at H=32). Treat this as
  cell-specific mechanism evidence, not a workload-general conclusion.
- `objective_pairwise` and `eviction_loss_pairwise` are not interchangeable
  labels; the former changes the supervision objective, while the latter keeps
  the eviction-loss target fixed and only changes representation.

## Unconsolidated items requiring explicit follow-up

The following locally known Wulver-dispatched files were not found in any local
worktree during the 2026-08-09 audit and must be synced back from Wulver before
the branch can be treated as fully consolidated:

- `scripts/experiments/run_distribution_shift_family.py`
- `scripts/experiments/upgrade_cross_family_manifest_metadata.py`
- `slurm/kbs_distribution_shift_wulver_smoke.sbatch`
- `slurm/kbs_distribution_shift_wulver.sbatch`
- `slurm/kbs_cross_family_heldout_smoke.sbatch`
- `slurm/kbs_cross_family_heldout_eval_wulver.sbatch`

## PASS-1 scope

Safe PASS-1 work on this branch should stay structural:

- add read-only status/validation tooling
- improve repository navigation and script-layout documentation
- clarify tracked-vs-generated boundaries
- tighten ignore rules for obvious non-scientific logs

It should not:

- rewrite manuscript conclusions
- mutate frozen result files
- delete historical evidence
- fabricate missing Wulver-only orchestration files
