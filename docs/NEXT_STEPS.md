# Next-work roadmap

This is a project-management note. It is not the reviewer entry point; use
[../README.md](../README.md) and
[reviewer/START_HERE.md](reviewer/START_HERE.md) for reviewer-facing
navigation.

## Current evidence state

The previous KBS second-revision evidence campaign is complete and validated.
The current primary evidence remains the curated package in
[../reports/kbs_final_evidence_20260813/](../reports/kbs_final_evidence_20260813/).

That package covers:

- matched baseline comparison against LRU, FIFO-Reinsertion, SIEVE, LRB,
  3L-Cache, CACHEUS, and HALP;
- corrected held-out `evict_value_v1` treatment integrity;
- supervision-objective ablation;
- exact-target, target-degeneracy, learned/exact, continuation, and
  distribution-shift diagnostics;
- controlled four-policy timing.

## Running controls

Two additional acceptance-risk controls are currently running:

- `kbs_common_model_objective_control_20260813_final`
- `kbs_tie_aware_exact_oracle_20260813_final`

These controls are not integrated into the manuscript, are not part of the
current primary evidence, and should not be summarized until they finish and
pass integrity review. Do not inspect or modify their output directories while
they are active.

## Safe current work

- Reviewer-facing documentation, link checks, and evidence navigation.
- Final package consistency audits that read committed summaries and
  non-running artifacts only.
- Dataset inventory or release-prep notes that do not upload data and do not
  alter generated scientific outputs.

## Do not do now

- Do not stop, restart, signal, or relaunch the two running controls.
- Do not inspect active outputs under `analysis/common_model_objective_control_v1/`
  or `analysis/tie_aware_exact_target_oracle_v1/`.
- Do not launch duplicate full campaigns for already validated evidence.
- Do not run the intentionally stopped 100% learning-curve fraction unless a
  future protocol explicitly reopens that question.
- Do not use `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
  as primary evidence; it is contaminated/historical.

## Deferred

- Query 3: citation metadata and script/config portability edits.
- Query 4: physical relocation or quarantine of contaminated historical result
  files after documentation warnings are in place.
