# Corrected Held-Out `evict_value_v1` Treatment — Transfer Provenance (2026-08-13)

## Wulver source (authoritative HPC checkout, unmodified)

```
login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/
login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/reviewer_fairness_cross_family_v1/model_registry.json
login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/kbs_evict_value_heldout_synthesis_20260811/EVICT_VALUE_HELDOUT_SYNTHESIS.md
```

Source classification per Wulver-side `FINAL_HELDOUT_AUDIT.md`: `FINAL_VALIDATED`,
host `login02`, branch `kbs/second-revision-science`, HEAD
`0e24660b829b2b2b72897d8659270113fea6ac5b` (protected, unchanged throughout the
transfer). Source is read-only-derived output; it does not modify the underlying
`evict_value_v1_rerun_1168220/` or `evict_value_v1_resume_wiki128_1171090/` job
directories.

## Local destination (canonical raw synced directory, tracked outside Git per
`.gitignore` policy for raw analysis trees)

```
analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/
analysis/reviewer_fairness_cross_family_v1/model_registry.json
analysis/kbs_evict_value_heldout_synthesis_20260811/EVICT_VALUE_HELDOUT_SYNTHESIS.md
```

## Transfer method

```
rsync -avz login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/{policy_comparison.csv,FINAL_HELDOUT_AUDIT.md,final_heldout_audit.json,primary_comparison_table.csv,supplementary_full_stream_comparison.csv,baseline_eligibility.csv,paired_statistics.csv,family_summary.csv,capacity_summary.csv,finalization_audit.json,finalization_audit.md,resume_delta_2rows.csv,degeneracy_performance_link.csv,_fold_table.json} analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/

rsync -avz login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/reviewer_fairness_cross_family_v1/model_registry.json analysis/reviewer_fairness_cross_family_v1/model_registry.json

rsync -avz login02:/mmfs1/project/ikoutis/sv96/Augmented-caching/analysis/kbs_evict_value_heldout_synthesis_20260811/EVICT_VALUE_HELDOUT_SYNTHESIS.md analysis/kbs_evict_value_heldout_synthesis_20260811/EVICT_VALUE_HELDOUT_SYNTHESIS.md
```

No model binaries were transferred (none required — only CSV/JSON/Markdown evidence
artifacts).

## Hash verification (local `sha256sum -c` against the required manifest, 2026-08-13)

All 16 files matched their required SHA-256 exactly. `sha256sum -c` result: 16/16 `OK`,
0 mismatches.

| File | Required SHA-256 | Local result |
|---|---|---|
| `policy_comparison.csv` | `982bfdff...fa0a296` | OK |
| `FINAL_HELDOUT_AUDIT.md` | `646ef9d1...dd07782` | OK |
| `final_heldout_audit.json` | `07dec512...b32ea036fba` | OK |
| `primary_comparison_table.csv` | `26b35e63...da8393d4d5` | OK |
| `supplementary_full_stream_comparison.csv` | `35626e3e...c5793d4` | OK |
| `baseline_eligibility.csv` | `73874236...260fc86ac74d` | OK |
| `paired_statistics.csv` | `28434e73...67f976085c89dd2c8` | OK |
| `family_summary.csv` | `90d5742c...c43f1e125d5bb56d` | OK |
| `capacity_summary.csv` | `94795841...217854c728ac620` | OK |
| `degeneracy_performance_link.csv` | `4dfb7a3b...a90a73b76ec7aaf8ddf6` | OK |
| `finalization_audit.json` | `4b7eaeeb...93cc598943f1fbe6f` | OK |
| `finalization_audit.md` | `25196bd2...decb300b20fcbf49a` | OK |
| `resume_delta_2rows.csv` | `ceb44bc7...573a7cd854846c2b87d71e418`†| OK |
| `_fold_table.json` | `7ed38640...8151ccd69a8a54` | OK |
| `model_registry.json` | `f3ec58a2...c71e2e03b5b1e` | OK |
| `EVICT_VALUE_HELDOUT_SYNTHESIS.md` | `f396cc02...49391df35ce81e920` | OK |

† Table cells are truncated for readability; the executed `sha256sum -c` check used the
complete 64-character values from the task hash manifest and every file passed. Full
manifest retained in the transfer session log.

## Historical provenance note (preserved, not erased)

This corrected 42-row artifact supersedes the historically contaminated 40-row local
treatment result (`original_40_sha256.txt`, job `1168220`, `TIMEOUT`), which must
**not** be used for any claim. The corrected artifact is a pure copy of the validated
resume artifact (job `1171090`, `COMPLETED`) plus a 2-row recovered delta
(`resume_delta_2rows.csv`) — no new computation was performed to produce the final
42-row file. Both the original-40 and resume-42 SHA-256 values are recorded in
`FINAL_HELDOUT_AUDIT.md` for full audit trail continuity.
