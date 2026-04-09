# robust_ftp_plus_X design (mechanism-harvest draft)

## Objective
Define a **minimal, conservative, interpretable** variant that keeps `robust_ftp_d_marker` as the backbone and adds exactly one mechanism harvested from sentinel.

## Chosen X
`X = disagreement-gated override`

Only allow predictor-side action when there is a contemporaneous robust/predictor disagreement; otherwise, follow robust behavior.

## Proposed policy name
`robust_ftp_plus_disagreement_gate_v1` (shorthand: `robust_ftp_plus_X`)

## Why this X (and not others)
From mechanism-harvest analysis:
- Sentinel gains are concentrated where sparse disagreement overrides are net helpful.
- Sentinel losses are associated with net harmful overrides in semi-realistic drift/scan slices.
- Importing tripwire/budget/risk controls would add coupling and calibration risk.

Therefore we keep only the disagreement-selection principle and remove all extra control layers.

## Behavior specification (conservative)
At each request:
1. Run robust and predictor shadows (as in existing robust/sentinel instrumentation).
2. Compute `disagree_now = (robust_event.evicted != predictor_event.evicted)`.
3. **If `disagree_now == 0`:** choose robust event.
4. **If `disagree_now == 1`:** allow one-step predictor override only under a strict, fixed rule:
   - baseline conservative rule: choose predictor **iff** predictor shadow cumulative misses are strictly lower than robust shadow cumulative misses; else choose robust.

Notes:
- This keeps robust as default.
- The gate prevents predictor influence when no actionable disagreement exists.
- No dynamic budgets, tripwires, or learned/logistic risk head.

## Observability / diagnostics (required)
Expose per-run diagnostics:
- `override_steps`
- `disagreement_steps`
- `override_rate_over_disagreement`
- `helpful_override_steps`
- `harmful_override_steps`
- `net_override_value = helpful - harmful`

These are mandatory to verify that the single mechanism behaves as intended.

## Non-goals for this iteration
- No early-return tripwire.
- No trust-budget throttling.
- No predictor-coverage target controller.
- No redesign of robust combiner objective.

## Evaluation plan (before further redesign)
Run exactly the two suites already used for mechanism harvest:
1. disagreement-stress
2. semi-realistic targeted

Success criteria for continuing beyond this minimal variant:
- Non-positive mean delta vs `robust_ftp_d_marker` on semi-realistic.
- Preserve at least part of disagreement-stress upside.
- Harmful overrides not exceeding helpful overrides in aggregate.

If these fail, abandon X or tighten the single gating rule further before adding any new mechanisms.
