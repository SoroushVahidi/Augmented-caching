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

Both authorized acceptance-risk controls are scientifically complete and
audited. They are not yet integrated into the manuscript.

- Common-model objective control V2 (`analysis/common_model_objective_control_wulver_v2/`):
  Wulver job `1176758`, 21/21 units, 84/84 rows, integrity `PASS`.
  Aggregate misses: eviction_loss 571,976 < pairwise 577,339 < reuse_distance
  615,850 < next_arrival 627,392. Eviction-loss is not materially worse;
  the matched control does **not** support blaming the eviction-loss
  *training objective* for poor policy performance. See
  [reports/common_model_v2_formal_audit_20260814/AUDIT.md](../reports/common_model_v2_formal_audit_20260814/AUDIT.md).
  V1 remains `SUPERSEDED_AFTER_IMPLEMENTATION_AUDIT`.
- Tie-aware exact-target oracle v1 (`analysis/tie_aware_exact_target_oracle_v1/`):
  21/21 units, 189/189 rows, integrity `PASS` after campaign-CSV recovery.
  `CURRENT_DETERMINISTIC` loses to LRU 18/21 (ties 3/21).
  `LRU_WITHIN_MINIMA` never loses to LRU (16 wins, 5 ties). Every tie-aware
  row has `fraction_tied_decisions = 1.0`. The old deterministic
  exact-oracle-versus-LRU result is tie-confounded. See
  [reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md).

## Safe current work

- Manuscript and response-to-reviewers rewrite using the audited controls
  above. Do not rerun either campaign.
- Reviewer-facing documentation, link checks, and evidence navigation.

## Do not do now

- Do not rerun Common V2 or any tie-oracle family/capacity unit.
- Do not use `analysis/common_model_objective_control_v1/` as scientific
  evidence; it is superseded after implementation audit.
- Do not cite the deterministic exact-oracle-versus-LRU table as proof that
  the H4 target itself is intrinsically worse than LRU.
- Do not launch duplicate full campaigns for already validated evidence.
- Do not run the intentionally stopped 100% learning-curve fraction unless a
  future protocol explicitly reopens that question.
- Do not use `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv`
  as primary evidence; it is contaminated/historical.

## Deferred

- Query 3: citation metadata and script/config portability edits.
- Query 4: physical relocation or quarantine of contaminated historical result
  files after documentation warnings are in place.
