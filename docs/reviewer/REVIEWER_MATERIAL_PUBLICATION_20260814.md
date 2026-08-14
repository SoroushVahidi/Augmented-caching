# Reviewer-material publication (2026-08-14)

This note records how reviewer-facing materials were published to `main`
without merging the scientific working branch.

## Commits

- Source science commit: `320a06576d5711bd5cba0f9eac69e0d8c3966089`
  (`kbs/second-revision-science`)
- Target `main` baseline: `7a1aa88f414bcce4155d1eafca18b1a2ddba019b`

No scientific experiment was rerun. The manuscript PDF was recompiled from
`submission_kbs_revision_final/07_LaTeX_Source/main.tex` with `tectonic`.

## Final manuscript PDF

- Path: `submission_kbs_revision_final/01_Revised_Manuscript.pdf`
- Pages: 21
- SHA-256: `8e95dd3bf848f4859a61777cfefb281c1d3a9d9896282a1dbf702c0a82688782`
- Build timestamp: 2026-08-14 16:35:43 EDT

## Reviewer-facing files published

- `README.md` (reviewer section)
- `submission_kbs_revision_final/01_Revised_Manuscript.pdf`
- `submission_kbs_revision_final/02_Response_to_Reviewers.md`
- `docs/reviewer/START_HERE.md`
- `docs/reviewer/REPRODUCTION_MATRIX.md`
- `docs/reviewer/RESULT_VERIFICATION.md`
- `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`
- `docs/reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`
- `docs/reviewer/REVIEWER_MATERIAL_PUBLICATION_20260814.md` (this file)
- `reports/kbs_final_evidence_20260813/`
- `reports/common_model_v2_formal_audit_20260814/AUDIT.md`
- `reports/tie_aware_exact_oracle_formal_audit_20260814/AUDIT.md`
- `analysis/common_model_objective_control_wulver_v2/{summary.csv,integrity_audit.json,completion_manifest.json}`
- `analysis/tie_aware_exact_target_oracle_v1/{summary.csv,integrity_audit.json,completion_manifest.json}`
- `analysis/reviewer_fairness/policy_comparison_{lru,sieve,fifo_reinsertion,lrb,three_l_cache,cacheus,halp}.csv`
- `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/{policy_comparison.csv,family_summary.csv,primary_comparison_table.csv}`

## Response

`02_Response_to_Reviewers.md` names
https://github.com/SoroushVahidi/Augmented-caching
and `docs/reviewer/START_HERE.md`, with supporting-artifact pointers for
matched baselines, Common-Model V2, tie-aware oracle, continuation/DAgger,
timing, and the reproduction/verification maps.

## Primary vs historical

PRIMARY: corrected held-out comparison; matched Table 4 sources; Common-Model
V2; tie-aware oracle; continuation; DAgger; controlled timing.

HISTORICAL / non-primary: leaky single-split evaluation; common-model V1;
failed/truncated tie-oracle wrap-up CSVs; exploratory Wulver `heavy_r1`
workflow docs remaining elsewhere in the tree.

## Link validation

Relative Markdown links in `README.md`, the response, `START_HERE.md`,
`REPRODUCTION_MATRIX.md`, and `RESULT_VERIFICATION.md` were resolved against
this worktree before commit.
