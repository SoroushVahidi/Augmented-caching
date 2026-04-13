# Branch scorer v2 continuation-value note

## Why v1 proxy target was misaligned

The v1 target (`eventually_correct_branch`) labeled each row as positive if `branch_id == gold_branch_id` at episode level. This is easy to train, but it does **not** directly answer the control question at a decision point: *is one more unit of compute on this branch useful right now?*

## New continuation-value target (practical approximation)

For each `(example_id, decision_step, branch_id)` row, v2 creates a one-step utility target that approximates the value of one additional expansion:

1. Compute a decision-point baseline pick from current candidate snapshots at that step.
2. Approximate "expand this branch once" using the next observed same-branch snapshot in the trace (if unavailable, fallback to no score change).
3. Recompute the pick under this one-branch expanded counterfactual.
4. Record:
   - `target_expand_value` (counterfactual correctness, 0/1)
   - `target_expand_delta` (counterfactual correctness minus baseline correctness)
   - `target_expand_better_than_baseline` (1 iff delta > 0)

This gives a cleaner supervision object than the episode-level branch identity proxy, while staying lightweight.

## Why episode-split evaluation is required

Rows from the same episode are highly correlated (same latent problem and branch dynamics). Splitting by row leaks episode information across train/validation and can overstate performance. v2 assigns splits by `example_id` (episode), so no episode appears in multiple splits.

## Known limitation (honest scope)

Current pilot traces do not fully support exact environment reconstruction and policy-conditional re-rollout from each decision state. Therefore v2 uses a best-feasible approximation from observed snapshots, with a marker column:

- `target_expand_approx_from_future_observation = 1` when next-snapshot evidence exists,
- `0` when fallback no-change was used.

Interpret v2 metrics as *quality-of-target-and-evaluation improvement over v1*, not as a final causal estimate of true continuation value.
