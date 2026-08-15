# Reproduction matrix

Public repository: <https://github.com/SoroushVahidi/Augmented-caching>

Many main results can be **inspected from committed summaries** without
rerunning HPC jobs. Full campaigns require raw traces, optional dependencies,
and cluster resources. This `main` publication includes the compact
reviewer-verifiable artifacts, not every campaign tree or runner script.

Set:

```bash
export REPO_ROOT=/path/to/Augmented-caching
cd "$REPO_ROOT"
```

| Result | Committed verification path | How to check without rerunning | Full rerun |
|---|---|---|---|
| Matched primary comparison (Table 5) | [major1_full_baseline_comparison.csv](../../reports/kbs_final_evidence_20260813/major1_full_baseline_comparison.csv), [major1_reviewer_summary.md](../../reports/kbs_final_evidence_20260813/major1_reviewer_summary.md) | Recompute win/loss/tie from the seven baseline CSVs plus the held-out EV treatment CSV | HPC; not required to verify published numbers |
| LRB / 3L-Cache / CACHEUS / HALP / LRU / SIEVE / FIFO-Reinsertion | [analysis/reviewer_fairness/](../../analysis/reviewer_fairness/) `policy_comparison_*.csv` | Confirm 21 `primary_controlled_window` rows per CSV; compare misses to EV | HPC + traces |
| Workload-specific Table 4 | [evict_value_v1_final_42_20260810/policy_comparison.csv](../../analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv), [policy_comparison_lru.csv](../../analysis/reviewer_fairness/policy_comparison_lru.csv) | `(EV_misses − LRU_misses) / LRU_misses` on the 40k scored window | HPC |
| Continuation C0/C1/C2 | [c0_continuation_summary.csv](../../reports/kbs_final_evidence_20260813/c0_continuation_summary.csv) | 21 cells; C2 vs C1 counts | HPC; do not rerun casually |
| DAgger / distribution shift | [distribution_shift_summary.csv](../../reports/kbs_final_evidence_20260813/distribution_shift_summary.csv) | 21 cells; shift vs miss-ratio signs | HPC |
| Controlled timing (Table 8) | [controlled_timing_summary.csv](../../reports/kbs_final_evidence_20260813/controlled_timing_summary.csv) | Four-policy means / CIs | HPC 420-run campaign |
| Common-Model V2 (Table 7) | [AUDIT.md](../../reports/common_model_v2_formal_audit_20260814/AUDIT.md), [summary.csv](../../analysis/common_model_objective_control_wulver_v2/summary.csv) | Totals 571,976 / 577,339 / 615,850 / 627,392 | HPC; V1 is superseded |
| Tie-aware exact-target oracle | [AUDIT.md](../../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md), [summary.csv](../../analysis/tie_aware_exact_target_oracle_v1/summary.csv) | CURRENT_DETERMINISTIC vs LRU 0/3/18; LRU_WITHIN_MINIMA 16/5/0 | Production already complete; no unit rerun |

**Historical / do not reproduce as primary**

- Common-model V1 (pairwise orientation error)
- Leaky single-split `evict_value_v1` evaluation
- Older Wulver `heavy_r1` manuscript-artifact workflow docs now under `historical/`

Cost guide:

- Cheap: inspect CSVs and audits committed here; unit tests that already exist
  on `main` (`tests/test_lrb.py`, `tests/test_sieve.py`,
  `tests/test_supervision_objective_ablation.py`).
- Expensive: regenerating held-out treatment, continuation, DAgger, timing,
  Common-Model V2, or the tie-aware oracle.

See also [RESULT_VERIFICATION.md](RESULT_VERIFICATION.md).
