# KBS acceptance-risk follow-up — 2026-08-13

Updated 2026-08-14: both authorized controls are complete and audited. This
document does not alter the manuscript or rebuttal; that is a separate step.

| Priority | Experiment | Status |
|---|---|---|
| 1 | COMMON_MODEL_OBJECTIVE_CONTROL | `COMPLETE_AUDITED`. Wulver Slurm job `1176758` (`common-v2`); 21/21 tasks, all `ExitCode 0:0`, 84/84 rows, integrity PASS. Formal audit: [common_model_v2_formal_audit_20260814/AUDIT.md](../../reports/common_model_v2_formal_audit_20260814/AUDIT.md). Not yet integrated into the manuscript. |
| 2 | TIE_AWARE_EXACT_ORACLE | `COMPLETE_AUDITED` after campaign-CSV recovery. 21/21 units, 189/189 rows, integrity PASS. Formal audit: [tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md](../../reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md). Not yet primary manuscript evidence. |
| Optional | repeated EV timing; LRB/3L parity | deferred; not authorized in this run |

Do not rerun either campaign. Common-model V1 remains superseded after
implementation audit; use `analysis/common_model_objective_control_wulver_v2/`.
Do not cite the deterministic exact-oracle-versus-LRU table as proof that the
H4 target intrinsically loses to LRU.
