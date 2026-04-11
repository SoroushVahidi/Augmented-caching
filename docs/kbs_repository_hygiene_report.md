# KBS repository hygiene pass (documentation only)

**Date:** 2026-04-12  
**Scope:** Polish navigation and warnings for the finalized **KBS** manuscript workflow. **No** experiment reruns, **no** changes to scientific numbers, **no** edits to canonical `heavy_r1` artifact **files** (only documentation and discoverability).

## Files added

| File | Purpose |
|------|---------|
| `docs/kbs_manuscript_workflow.md` | Single hub: canonical `heavy_r1` path, builder command, output directories, exploratory separation, confused-filename table. |
| `docs/kbs_repository_hygiene_report.md` | This report. |

## Files updated

| File | What changed / confusion addressed |
|------|-----------------------------------|
| `README.md` | “Start here” link to `docs/kbs_manuscript_workflow.md`; repository guide lists workflow first; link to this hygiene report. |
| `analysis/README.md` | Top navigation; stale/alternate wording for unsuffixed roots; pairwise `*_campaign/` and theorem/audit tooling called out as non-canonical. |
| `docs/repo_map.md` | New KBS workflow section; “read first” list puts workflow doc first. |
| `docs/kbs_manuscript_submission_index.md` | Opening pointer to workflow doc. |
| `docs/evict_value_v1_kbs_canonical_artifacts.md` | Navigation line to workflow doc at top. |
| `docs/reproducibility_and_artifacts.md` | Primary link to workflow doc under title. |
| `docs/lightweight_exploratory_ablations.md` | Cross-link to workflow doc for canonical boundary. |
| `docs/manuscript_evidence_map.md` | Short navigation note so this evidence table is not mistaken for the canonical Wulver `heavy_r1` line. |
| `scripts/README.md` | Canonical KBS pointer includes workflow doc. |
| `scripts/paper/README.md` | Link to workflow doc as primary manuscript navigation. |

## Remaining dependency (not fixed by docs)

Canonical **`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`** (and `.md`) must exist for `scripts/paper/build_kbs_main_manuscript_artifacts.py` to succeed. That file is produced by **`slurm/evict_value_v1_wulver_heavy_eval.sbatch`** (`EXP_TAG=heavy_r1`) after training. If a Wulver heavy eval job is still **pending** or **running**, or previously **timed out**, the manuscript bundle cannot be regenerated from canonical inputs until eval completes—watch `squeue` / `slurm/logs/evictv1-heavy-eval-*.out` and the runbook `docs/wulver_heavy_evict_value_experiment.md`.
