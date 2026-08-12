# Reuse-Tail Contingency Claim Changes

This file is preparatory only. It does not incorporate final reuse-tail numbers. It lists manuscript/status claims that must be revised depending on whether the diagnostic shows strong or weak outside-horizon reuse among resident candidates.

## If Reuse-Tail Shows Strong Outside-Horizon Reuse

- Mechanism claim to add: many cache-resident candidates have future reuse after the finite supervision horizon, so the H=4 target can miss potential future consequences at the decision population actually used for eviction labels.
- H4/H8 status language should move from untested/partially supported to empirically supported as a potential mechanism, with the explicit caveat that reuse after H is not itself a causal excess miss.
- The exact-target oracle interpretation should say the exact H4 target may be accurately optimized yet still myopic, consistent with learned high target agreement but poor downstream misses.
- The target-degeneracy interpretation should connect outside-horizon reuse to low H=4 label resolution only if the reuse-tail output co-moves with degeneracy metrics; otherwise keep them separate mechanisms.
- Reviewer #2 Major 3 / Reviewer #3 language should say the diagnostic narrows the failure mechanism toward horizon/tail observability, but does not replace the C0/C1/C2 continuation experiment.
- NEXT_STEPS should prioritize a tail-aware target or broader H sensitivity only after the causal continuation blocker is represented accurately; do not claim an H/C law.

## If Reuse-Tail Shows Weak/No Outside-Horizon Reuse

- Remove or soften any claim that finite-horizon truncation is a leading explanation for the poor H=4 target under the resident-candidate decision population.
- H4/H8 status should remain unsupported or be downgraded for the tested population/horizons; mechanism prose should pivot toward target degeneracy, target construction, and continuation/deployment mismatch.
- The exact-target oracle interpretation should not invoke unseen late reuse as the explanation unless another artifact supports it.
- Limitations should still mention finite horizons as a general modeling limitation, but not as a locally evidenced dominant failure mode.
- Reviewer #3 remains unresolved because C0/C1/C2 is still missing; weak reuse-tail evidence does not solve continuation mismatch.
- STOP_SAMPLE_SIZE_HYPOTHESIS remains unchanged either way; the reuse-tail result tests a different mechanism than H1.

## Claims That Must Be Audited In Manuscript/Docs

- Any statement that H=4 is adequate or inadequate without conditioning on measured resident-candidate reuse delays.
- Any statement that exact optimization of the current target proves the target is good or bad for causal misses; it only evaluates the finite-horizon target under the specified continuation semantics.
- Any statement equating future request-position delay T with classical reuse distance or stack distance; the diagnostic deliberately does not use that terminology.
- Any statement that outside-horizon reuse directly measures causal excess misses.
- Any prioritization text for H4/H8/H11, target-degeneracy, and C0/C1/C2 in `docs/NEXT_STEPS.md`, `docs/DEVELOPMENT_STATUS.md`, `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`, and reviewer coverage docs.
