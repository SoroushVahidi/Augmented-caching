# KBS manuscript integration audit — 2026-08-14

- Status: `MANUSCRIPT_AND_RESPONSE_UPDATED`
- Branch: `kbs/second-revision-science`
- HEAD before this integration: `ea61b4ba6b3c49203932a027e76fe51cbf09af6d`
- No new experiment was run.

## Finalized evidence used

- Primary curated package: `reports/kbs_final_evidence_20260813/` (unchanged)
- Common-model V2: `reports/common_model_v2_formal_audit_20260814/AUDIT.md`,
  `analysis/common_model_objective_control_wulver_v2/`
- Tie-aware exact-target oracle:
  `reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md`,
  `reports/tie_aware_exact_oracle_recovery_20260814/`,
  `analysis/tie_aware_exact_target_oracle_v1/`
- Hypothesis map:
  `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`

## Files changed in this integration

- Canonical manuscript:
  `submission_kbs_revision_final/07_LaTeX_Source/main.tex`
- Canonical response:
  `submission_kbs_revision_final/02_Response_to_Reviewers.md`
- Workflow pointer:
  `docs/kbs_manuscript_workflow.md`
- This audit:
  `reports/kbs_final_manuscript_integration_20260814/AUDIT.md`

Scientific result files were not modified. No model binaries were added.
`manuscript_source/main.tex` was not edited (non-canonical duplicate).

## Manuscript sections modified

- Abstract
- §1.3 Contributions (items 3–4)
- §3.7 Objective comparison (pipeline Table 8 retained; new matched
  common-model V2 table)
- §3.8 Mechanistic diagnosis (tie-aware follow-up; pending language removed)
- §3.9 Continuation / DAgger (opening interpretation only)
- §3.10 Discussion and synthesis
- §3.11 Practical significance (barrier statement)
- §4.1 Summary of findings
- §4.2 Implications
- §4.3 Limitations (item 3)
- §4.4 Future research

## Newly introduced quantitative claims and sources

| Claim | Source | Check |
|---|---|---|
| V2 totals 571,976 / 577,339 / 615,850 / 627,392 | `analysis/common_model_objective_control_wulver_v2/summary.csv` | PASS |
| CURRENT_DETERMINISTIC vs LRU 0/3/18; +81,750 misses | `analysis/tie_aware_exact_target_oracle_v1/summary.csv` | PASS |
| LRU_WITHIN_MINIMA vs LRU 16/5/0; −413 misses (564,713 vs 565,126) | same | PASS |
| MRU_WITHIN_MINIMA vs LRU 0/3/18; +89,135 misses | same | PASS |
| RANDOM vs LRU 7/15/83; vs CURRENT 90/15/0 | same | PASS |
| `fraction_tied_decisions = 1.0` on 168 non-LRU rows | same | PASS |
| mean `fraction_all_tied` ≈ 0.649 | same (0.64879) | PASS |
| mean optimal-set fraction ≈ 0.991 | same (0.99130) | PASS |

Prior primary numbers (Table 7/8 pipeline, C0/C1/C2, DAgger, timing, 97.53%
agreement, unique-winner 0, $P(T>4)=0.9939$) were preserved.

## Stale claims removed or softened

- “Tie-policy sensitivity remains pending”
- Deterministic exact-oracle deficit as proof that the exact target
  intrinsically loses to LRU
- Full-pipeline Table 8 as an isolated objective-causality result
- “Eviction-loss training objective is the cause”
- “Distribution shift / DAgger explains / improves” (already negative;
  restated as not supported)

## Hypothesis-state reconciliation

- H1: DISFAVORED
- H2: DISFAVORED as a uniform model-fitting explanation
- H3: STRONGLY_SUPPORTED
- H4 target-intrinsic-failure claim: NOT_SUPPORTED (tie-confounded);
  observability limitation still noted
- Common-model objective hypothesis: NOT_SUPPORTED
- H5: PARTIALLY_SUPPORTED
- H6: NOT_SUPPORTED for the tested DAgger intervention

## Validation

- Independent Python recomputation of all newly inserted numbers: PASS
- `git diff --check`: clean
- `tectonic main.tex` in `submission_kbs_revision_final/07_LaTeX_Source/`:
  exit 0; over/underfull box warnings only; generated `main.pdf` not
  committed
- Response Markdown structure: headings intact; no LaTeX `\ref` left in
  the response file
- No new experiment run
