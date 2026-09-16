# PE Long-Horizon Preflight Residual Manifest

Created: 2026-09-16.

Archive branch: `archive/pe-long-horizon-preflight-residual-20260916`

Source branch/worktree before archiving:

- Branch: `experiment/pe-long-horizon-preflight-20260915`
- HEAD: `9b5548041e5245027a1da6a6e911307518b3ea9f`

Purpose: preserve the mixed/unattributed residual working state from the preflight root worktree before normalizing or reusing that worktree. These changes were not treated as canonical PE results.

## Classification

| Path group | Classification | Preservation action |
|---|---|---|
| `scripts/validation/revision_readiness.py`, `scripts/validation/revision_status.py`, `tests/test_revision_readiness.py`, `tests/test_revision_status.py` | UNIQUE_COHERENT | Committed on this archive branch. These appear to form a coherent revision-readiness/status improvement set. |
| Other modified tests under `tests/` | UNIQUE_UNCERTAIN | Committed on this archive branch. Blob-hash audit did not find exact duplicate objects already present in this repo. |
| Modified docs under `docs/` | UNIQUE_UNCERTAIN | Committed on this archive branch. These appear related to repository/status documentation but were not attributed to a canonical PE branch. |
| `analysis/evict_value_pairwise_first_check.md` | UNTRACKED_EVIDENCE | Committed on this archive branch. |
| `scripts/print_preflight_stats.py` | UNTRACKED_EVIDENCE | Committed on this archive branch. |
| `slurm/evict_value_v1_wulver_preflight_20260915.sbatch` | UNTRACKED_EVIDENCE | Committed on this archive branch. |
| `ystemctl --user status kerberos-renew.service` | GENERATED / SAFE_DELETE_CANDIDATE_FOR_HUMAN_APPROVAL | Not committed. Contents and hash recorded below. No deletion performed. |

## Stray 426-Byte File

- Path: `ystemctl --user status kerberos-renew.service`
- Size: `426` bytes
- SHA256: `fea4d97eb7f6fbd77448cab93e31d13d5907be9765cef39eca1e78f61229059b`
- Classification: shell redirection accident / generated terminal output, not scientific evidence, not source, not config, not provenance.
- Action: left in the worktree for human-approved deletion; not committed.

Recorded content:

```text
NEXT                           LEFT LAST                            PASSED UNIT                 ACTIVATES
Tue 2026-08-11 05:33:31 EDT 5h 59min Mon 2026-08-10 23:33:31 EDT 23s ago kerberos-renew.timer kerberos-renew.service

1 timers listed.
Pass --all to see loaded but inactive timers, too.
```

## Zero-Loss Note

All meaningful tracked and untracked residual files, except the explicitly recorded stray command-output file above, are committed on this archive branch. The original preflight branch can be restored to its committed state without losing meaningful residual work, but the stray file should only be deleted after explicit human approval.
