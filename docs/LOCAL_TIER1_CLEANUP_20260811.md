# Local Tier 1 Storage Cleanup - 2026-08-11

This cleanup removed only low-risk Tier 1 cache and duplicate artifacts from the local workstation. It did not contact Wulver, launch experiments, use Slurm, or touch current scientific evidence trees.

## Pre-Cleanup Snapshot

`df -h /home/soroush`:

| Filesystem | Size | Used | Avail | Use |
|---|---:|---:|---:|---:|
| `/dev/nvme0n1p5` | 700G | 611G | 54G | 92% |

Augmented-caching worktree sizes before cleanup:

| Path | Size |
|---|---:|
| `/home/soroush/Augmented-caching-preserve-20260808-104203` | 357M |
| `/home/soroush/Augmented-caching-kbs-parallel` | 707M |
| `/home/soroush/Augmented-caching-main` | 716M |
| `/home/soroush/Augmented-caching-kbs-second-revision` | 889M |
| `/home/soroush/Augmented-caching-3l-cache` | 1.2G |
| `/home/soroush/Augmented-caching-cacheus` | 1.2G |
| `/home/soroush/Augmented-caching-halp` | 1.2G |
| `/home/soroush/Augmented-caching-fairness` | 97G |
| `/home/soroush/Augmented-caching` | 98G |
| `/home/soroush/Augmented-caching-objective-ablation` | 122G |

## Deleted Items

| Deleted item | Previous size | Classification | Why safe |
|---|---:|---|---|
| Augmented-caching worktree `__pycache__`, `.pytest_cache`, and `*.pyc` files | 552,299,395 bytes | generated Python/test cache | Git checks found no tracked cache files; Python can regenerate them |
| `/home/soroush/.cache/pip` | 9,778,688,728 bytes | package cache | recreatable pip download/http cache, not scientific evidence |
| `/home/soroush/.cache/ms-playwright` | 1,314,335,260 bytes | package/browser cache | recreatable Playwright browser cache, not scientific evidence |
| `/home/soroush/.cache/uv` | 598,602,045 bytes | package cache | recreatable uv package cache, not scientific evidence |
| `/home/soroush/.npm` | 3,898,504,711 bytes | package cache | recreatable npm cache, not scientific evidence |
| `/home/soroush/Augmented-caching/analysis/exploratory/pairwise_chain_witness_campaign` | 726,594,736 bytes | ignored duplicate scientific artifact | byte-identical to tracked canonical copies; ignored in that worktree |
| `/home/soroush/Augmented-caching-3l-cache/.venv_3l_cache` | 335,684,610 bytes after cache removal | inactive unreferenced project virtualenv | no local references, no active process/lsof users, recreatable from `pyproject.toml` |
| `/home/soroush/Augmented-caching-cacheus/.venv_cacheus` | 336,432,847 bytes after cache removal | inactive unreferenced project virtualenv | no local references, no active process/lsof users, recreatable from `pyproject.toml` |
| `/home/soroush/Augmented-caching-halp/.venv_halp` | 335,684,510 bytes after cache removal | inactive unreferenced project virtualenv | no local references, no active process/lsof users, recreatable from `pyproject.toml` |
| `/home/soroush/Augmented-caching-objective-ablation/.venv_ablation` | 336,433,092 bytes after cache removal | inactive unreferenced project virtualenv | no local references, no active process/lsof users, recreatable from `pyproject.toml` |

Total deleted by the cleanup script: 18,213,259,934 bytes, about 16.96 GiB.

## Pairwise Duplicate Handling

The pairwise campaign tree hash was reverified before cleanup:

`cccdc8a37558381d4601e0a27b520cb79056686dbdeae619a0fe99b576462698`

All discovered copies had 120 files and 726,594,736 bytes. The ignored copy under `/home/soroush/Augmented-caching/analysis/exploratory/pairwise_chain_witness_campaign` was removed. The canonical retained copy is:

`/home/soroush/Augmented-caching-kbs-second-revision/analysis/pairwise_chain_witness_campaign`

The other seven tracked copies were left untouched to avoid intentional tracked-file deletions in separate worktrees.

## Skipped Candidates

| Candidate | Reason skipped |
|---|---|
| `/home/soroush/Augmented-caching/.venv_kbs_heavy_r1` | referenced by docs/provenance runbooks |
| `/home/soroush/Augmented-caching-fairness/.venv_fairness` | referenced by current runner/docs/tests |
| `/home/soroush/.cache/huggingface` | explicitly out of scope for this pass |
| `/home/soroush/.venvs/vllm_baseline_pilot` | explicitly out of scope for this pass |
| `/home/soroush/modal-venv` | explicitly out of scope for this pass and active process uses it |
| tracked pairwise campaign copies | retained to avoid tracked-file deletions |
| `/home/soroush/Augmented-caching/data/derived/evict_value_v1_wulver_heavy_r1` | explicitly preserved for this Tier 1 pass |
| objective-ablation and cross-family derived datasets | current scientific evidence, explicitly protected |

## Post-Cleanup Snapshot

`df -h /home/soroush`:

| Filesystem | Size | Used | Avail | Use |
|---|---:|---:|---:|---:|
| `/dev/nvme0n1p5` | 700G | 594G | 71G | 90% |

Augmented-caching worktree sizes after cleanup:

| Path | Size |
|---|---:|
| `/home/soroush/Augmented-caching-preserve-20260808-104203` | 357M |
| `/home/soroush/Augmented-caching-kbs-parallel` | 706M |
| `/home/soroush/Augmented-caching-3l-cache` | 714M |
| `/home/soroush/Augmented-caching-halp` | 714M |
| `/home/soroush/Augmented-caching-main` | 714M |
| `/home/soroush/Augmented-caching-cacheus` | 715M |
| `/home/soroush/Augmented-caching-kbs-second-revision` | 886M |
| `/home/soroush/Augmented-caching-fairness` | 97G |
| `/home/soroush/Augmented-caching` | 98G |
| `/home/soroush/Augmented-caching-objective-ablation` | 121G |

No tracked deletions appeared in any Augmented-caching worktree after cleanup.

## Protected Evidence Confirmed Untouched

The following protected paths still exist after cleanup:

| Path | Size |
|---|---:|
| `/home/soroush/Augmented-caching-objective-ablation/data/derived/supervision_objective_ablation_v1` | 121G |
| `/home/soroush/Augmented-caching-fairness/data/derived/evict_value_v1_cross_family_v1` | 93G |
| `/home/soroush/Augmented-caching/data/derived/evict_value_v1_wulver_heavy_r1` | 96G |
| `/home/soroush/Augmented-caching-fairness/models` | 355M |
| `/home/soroush/Augmented-caching-kbs-second-revision/models` | 155M |
| `/home/soroush/Augmented-caching/data/raw` | 799M |
| `/home/soroush/Augmented-caching/data/processed` | 140M |
| `/home/soroush/Augmented-caching-kbs-second-revision/analysis/reuse_tail_horizon_diagnostic_v1` | 584K |
