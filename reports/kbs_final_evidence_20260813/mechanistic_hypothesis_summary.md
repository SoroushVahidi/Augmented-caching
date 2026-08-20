# Mechanistic Hypothesis Summary — Final Dispositions After C0/C1/C2 and Distribution-Shift

This supersedes the `RUNNING_TEST` / single-family-inconclusive status of H5 and H6 in
`docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` (that file remains the full
narrative source; this is a compact final-status summary). Full detail and reasoning
for the two newly closed hypotheses is in `c0_integrity_summary.md` and
`distribution_integrity_summary.md` in this directory.

| Hypothesis | Statement (short) | Status | Key evidence |
|---|---|---|---|
| H1 | Insufficient training data | `DISFAVORED` within tested 1%–50% scope | Same-target learning-curve campaign, 42/42 rows; scalar improvement `0.6256→0.6126` is not material; pairwise flat |
| H2 | Model-fitting / function-approximation failure | `DISFAVORED` as primary/uniform explanation | 21-cell learned/exact agreement `0.975301`; positive-regret fraction `0.024699` |
| H3 | Target degeneracy at H=4 | `STRONGLY_SUPPORTED` | `tie_event_fraction≈1.0`, `mean_optimal_set_fraction≈0.9932`, target entropy `≈0.048` bits; independently re-encountered in both new campaigns via the saturated 100%-miss Wiki2018 cells |
| H4 | Horizon truncation / observability | `SUPPORTED_AS_OBSERVABILITY_LIMITATION` (not causal) | `P(T>4\|resident)=0.9939`; even conditioned on eventual reuse, `0.9793` |
| H5 | Continuation-policy mismatch (LRU-continuation labels vs. deployed learned policy) | **`PARTIALLY_SUPPORTED`** (upgraded from single-family `RUNNING_TEST`) | Full 7-family/21-cell C0/C1/C2: C2 improves over C1 in 13/21 cells, macro mean Δ≈−0.0102, but a single large counter-example (`brightkite` cap32, +0.2433) blocks a uniform claim |
| H6 | Generic state-distribution shift reduction improves performance | **`DISFAVORED`** (upgraded from single-family `RUNNING_TEST`) | Full 7-family/21-cell distribution-shift: shift index improves in 16/21 cells while misses worsen in 16/21; 13/18 informative cells show both simultaneously (shift↓, misses↑) |
| H9 | Rare strict H=4 preferences may be myopic | `RESOLVED_DIAGNOSTIC` | 21-cell strict-preference diagnostic: no unique H4 winner found; dominated by H3 degeneracy |

## Ranked explanation, updated

Primarily a **target-design problem** (H3, extended by H4/H9) — neither new campaign
contradicts this; both independently rediscover the identical Wiki2018 degeneracy
(100%-miss regardless of policy). H5/H6 remain secondary, target/deployment-interaction
factors, now precisely characterized rather than open: H5 has real but inconsistent,
family/capacity-dependent partial support; H6, as a "reduce generic shift → reduce
misses" causal story, is disfavored at full scale even though the shift-reduction
mechanism itself functions as designed. Pure model-fitting failure (H2) and sample-size
(H1) remain the least supported explanations.

All language above is empirical and scope-limited to the tested cells. No hypothesis
here is described as "proved."

`NO_NEW_EXPERIMENT_REQUIRED` for these hypotheses under the current stopping rules;
see `docs/reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` for the individual stopping
rules that would reopen any of them.
