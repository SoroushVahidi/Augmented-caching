# Reviewer revision index

This directory (and the associated worktrees) contains experiments addressing reviewer feedback for the Augmented Caching manuscript.

## Revision status

| Concern | Experiment | Branch / Worktree | Protocol | Status |
|---------|------------|-------------------|----------|--------|
| **C1** | Cross-family Training | `feat/reviewer-fairness-protocol` | Fairness retraining | In Progress |
| **C2** | Supervision Objective | `feat/supervision-objective-ablation` | Objective ablation | In Progress |
| **C3** | Distribution Shift | `feat/reviewer-fairness-protocol` | Dataset drift | Checkpointed |
| **C4** | Practical Significance | `feat/reviewer-fairness-protocol` | Controlled timing | Deferred |

## Revision documents

- **Roadmap**: Detailed timeline and stage gates (available in `feat/reviewer-fairness-protocol` worktree).
- **[Next-Stage Runbook](../reviewer/next_stage_runbook.md)**: Operational commands for resuming/finalizing experiments.
- **[Submission Index](../reviewer/submission_index.md)**: Reviewer-facing map of revision evidence.

## Active worktrees

Scientific revision work is distributed across multiple worktrees to prevent interference:
- `/home/soroush/Augmented-caching-fairness` (C1, C3, C4)
- `/home/soroush/Augmented-caching-objective-ablation` (C2)

## Evidence location

All revision artifacts are written to `analysis/reviewer_revision/` in their respective worktrees and will be consolidated into the main repository upon completion.
