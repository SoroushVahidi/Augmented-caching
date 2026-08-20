# Final Submission Claim Audit — 2026-08-13

All rows below were checked against the final `main.tex`, the 44-page compiled
manuscript, the final response source, and the canonical evidence artifacts.

| Claim | Manuscript location | Response location | Evidence | Status |
|---|---|---|---|---|
| Matched comparison: `evict_value_v1` does not outperform any of seven baselines | §3.6, Table 7, pp. 23–25 | R2 Major 1; R3 Issue 2 | `major1_full_baseline_comparison.csv` | PASS |
| Eviction-loss is worst/tied-worst among four objectives | §3.7, Table 8, pp. 25–26 | R2 Major 2 | `analysis/supervision_objective_ablation_v1/policy_comparison.csv` | PASS |
| Exact horizon-4 oracle loses to LRU | §3.8, pp. 26–28 | R2 Major 3 | `analysis/exact_target_oracle_replication_v1/policy_comparison.csv` | PASS |
| Severe target degeneracy at the tested horizon | §3.8, pp. 26–28 | R2 Major 3 | `analysis/strict_preference_horizon_diagnostic_v1/cell_summary.csv` | PASS |
| Learned/exact agreement is 97.5301% and positive regret is 2.4699% | §3.8, p. 26 | R2 Major 3 | `analysis/learned_exact_target_agreement_v1/cell_summary.csv` | PASS |
| Horizon-4 reuse-tail result is an observability limitation, not a causal proof | §3.8, p. 27 | R2 Major 3 additional clarification | `analysis/reuse_tail_horizon_diagnostic_v1/report.md` | PASS |
| Continuation intervention is partially supported and regime-dependent | §3.9, pp. 28–29 | R3 Primary issue | `c0_continuation_summary.csv`, `c0_integrity_summary.md` | PASS |
| DAgger-style generic shift reduction does not improve online performance | §3.9, pp. 29–30 | R2 Major 3; R3 Primary issue | `distribution_shift_summary.csv`, `distribution_integrity_summary.md` | PASS |
| Controlled timing covers exactly LRU, FIFO-Reinsertion, SIEVE, and HALP; treatment timing is separate | §3.10, Table 9, pp. 31–33 | R2 Major 4; R3 Issue 3 | `controlled_timing_summary.csv`, `controlled_timing_interpretation.md` | PASS |

## Reviewer coverage

| Concern | Evidence complete | Manuscript changed | Response written | Exact reference | Status |
|---|---|---|---|---|---|
| R2 Major 1 | Yes | Yes | Yes | §3.6, Table 7, pp. 23–25 | ANSWERED |
| R2 Major 2 | Yes | Yes | Yes | §3.7, Table 8, pp. 25–26 | ANSWERED |
| R2 Major 3 | Yes | Yes | Yes | §§3.8–3.9, pp. 26–31 | ANSWERED |
| R2 Major 4 | Yes | Yes | Yes | §§3.10–3.11, Table 9, pp. 31–33 | ANSWERED |
| R3 primary concern | Yes | Yes | Yes | §3.9, pp. 28–30 | ANSWERED |

No new experiment was run for this audit; no major experiment remains.
