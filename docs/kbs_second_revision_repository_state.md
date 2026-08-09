# KBS Second-Revision Repository State

Date: 2026-08-09  
Canonical checkout: this `kbs/second-revision-science` repository clone  
Canonical branch: `kbs/second-revision-science`  
Expected baseline HEAD for the current local documentation pass: `58f486701c304d8db5460cfd559af4051e59b938`

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
  diagnostic. Completed cells may be inspected, but incomplete aggregates must
  remain `DIAGNOSTIC_PARTIAL`.
- the last audited low-fraction learning-curve checkpoint contains `16`
  validated units / `96` rows across
  `brightkite, citibike, cloudphysics, metacdn` at fractions
  `1%, 2%, 5%, 10%`.
- a separate `25%` local extension is currently running in tmux session
  `kbs_learning_curve_highfrac_20260809` with a clean `10`-hour wall-time
  budget; no current `25%` scientific claim should be recorded until completed
  units are audited.
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
