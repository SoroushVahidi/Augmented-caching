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

## Acceptance-risk controls

Current acceptance-risk control status:

- `analysis/common_model_objective_control_v1/` is
  `SUPERSEDED_AFTER_IMPLEMENTATION_AUDIT`. An implementation audit identified
  an orientation error in the initial common-model pairwise control; that run
  was retired before use as manuscript evidence. The corrected V2 control is
  regression-gated and pending.
- `kbs_tie_aware_exact_oracle_20260813_final` is still running/pending.

Neither control is integrated into the manuscript or current primary evidence.
Do not summarize the tie-oracle run until it finishes and passes integrity
review.

## Safe current work

- Reviewer-facing documentation, link checks, and evidence navigation.
- Final package consistency audits that read committed summaries and
  non-running artifacts only.
- Dataset inventory or release-prep notes that do not upload data and do not
  alter generated scientific outputs.

## Do not do now

- Do not stop, restart, signal, or relaunch the tie-aware oracle.
- Do not inspect or modify active outputs under
  `analysis/tie_aware_exact_target_oracle_v1/`.
- Do not use `analysis/common_model_objective_control_v1/` as scientific
  evidence; it is superseded after implementation audit.
- Do not launch duplicate full campaigns for already validated evidence.
- Do not run the intentionally stopped 100% learning-curve fraction unless a
  future protocol explicitly reopens that question.
- Do not use `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
  as primary evidence; it is contaminated/historical.

## Deferred

- Query 3: citation metadata and script/config portability edits.
- Query 4: physical relocation or quarantine of contaminated historical result
  files after documentation warnings are in place.
