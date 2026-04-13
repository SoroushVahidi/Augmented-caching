# Learned Branch Scorer Note (v1)

## Why try a learned scorer now

After minimum-expansion safeguards and simple hand-designed ranking rules, the next question is whether a small learned scorer can better estimate branch promise for compute allocation.

## Global objective vs local target

- **Global objective:** maximize probability of correct final answer under fixed budget.
- **Local learned target (v1):** `eventually_correct_branch` where label = 1 iff `branch_id == gold_branch_id` for the episode.

This is a **proxy** target. It approximates branch promise but does not fully capture counterfactual value-of-compute for each intermediate decision state.

## Exact feature set (v1)

The v1 feature set is intentionally small and interpretable:

1. `raw_score`
2. `depth`
3. `expansions`
4. `verifications`
5. `remaining_budget`
6. `post_verify_score`
7. `score_delta` (`post_verify_score - pre_verify_score`)
8. `survived_pruning_steps`

## Model choice

- Model: `sklearn.linear_model.LogisticRegression`
- Settings: `class_weight='balanced'`, `max_iter=1000`

Rationale: transparent coefficients, lightweight training/inference, and minimal engineering overhead for first-pass research.

## Limitations of v1

- Labels are proxy labels from synthetic pilot traces, not exact continuation-value labels.
- Training and evaluation are both lightweight synthetic runs, so external validity is limited.
- The target is noisy for local branch ranking decisions; a branch can be useful without being the final gold branch.
- No temporal credit assignment or counterfactual replays yet.

So this version is meant to test feasibility and signal quality, not to claim production readiness.
