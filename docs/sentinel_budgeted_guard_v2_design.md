# sentinel_budgeted_guard_v2 design

## Goal
`sentinel_budgeted_guard_v2` is a principled robust-first refinement of `sentinel_robust_tripwire_v1` with three explicit controls:

1. **Finite override budget** (no replenishment): predictor-following overrides are globally capped per run.
2. **Temporary guard mode with temporal memory**: suspicious behavior pushes the policy into robust-only mode and raises a short-lived memory signal.
3. **Explicit re-entry rule**: predictor overrides are blocked until post-guard stability conditions are met.

The method keeps decisions interpretable and uses the same shadows as v1:
- robust backbone: `robust_ftp_d_marker`
- consistent expert: `follow_predicted_cache`

## Decision rule (v2)
On each step, compute robust and predictor shadow outcomes and maintain online diagnostics:
- disagreement rate (windowed)
- suspicious early-return rate (windowed)
- budget pressure (`1 - remaining_budget/total_budget`)
- guard-memory pressure (`guard_memory/guard_memory_cap`)

Risk score is a fixed logistic head:

\[
r_t = \sigma\left(-2 + 2.25\cdot d_t + 3.5\cdot s_t + 1.0\cdot b_t + 1.25\cdot m_t\right)
\]

Predictor override is allowed **only if all** are true:
- not in guard mode,
- re-entry rule satisfied,
- warmup finished,
- remaining finite budget > 0,
- robust/predictor disagree on eviction,
- suspicious count under trigger threshold,
- risk score ≤ threshold.

Otherwise choose robust line.

## New ingredients and rationale

### 1) Finite override budget
- Parameter: `override_budget_total`.
- Behavior: decremented on each predictor override, never replenished.
- Rationale: strict cap on worst-case exposure to non-robust behavior.

### 2) Guard mode + temporal memory
- Guard trigger: early-return burst in sliding window or directly harmful override.
- Action: robust-only for `guard_duration` steps.
- Temporal memory:
  - incremented on suspicious events,
  - decremented by `guard_memory_decay` each non-suspicious step,
  - bounded by `guard_memory_cap`.
- Rationale: avoids immediate trust restoration after a local failure.

### 3) Explicit re-entry rule
- Re-entry requires both:
  - `steps_since_suspicious >= reentry_stable_steps`, and
  - `guard_memory <= reentry_memory_threshold`.
- Rationale: transparent hysteresis preventing oscillatory override/revert behavior.

## Knobs kept small
Compared to v1, v2 adds only the minimal controls required by the brief:
- `override_budget_total`
- `guard_memory_cap`, `guard_memory_decay`
- `reentry_stable_steps`, `reentry_memory_threshold`

Defaults are conservative and robustness-oriented.

## Diagnostics and interpretability
v2 exports:
- summary counters: robust/predictor steps, guard triggers, harmful/helpful overrides, remaining budget, re-entry blocks, memory tail.
- step log fields: chosen line, re-entry readiness, guard state, risk, suspicious count, memory, budget, and shadow hit outcomes.

This preserves the existing policy/runner diagnostic pattern and enables focused disagreement analysis.

## Why v2 is still robust-first
- Robust line is default and always available.
- Predictor can only act on disagreement steps and only under low-risk + re-entry + finite-budget constraints.
- Guard fallback is immediate and explicit.
- Trust recovery is delayed and rule-based.
