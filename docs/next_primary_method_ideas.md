# Next Primary Method Ideas (Robust-First, Learned-Second)

This design pass treats `evict_value_v1` as a secondary line and proposes **new primary directions** anchored in the current robust winners (`robust_ftp_d_marker`, `blind_oracle_lru_combiner`) and existing repo building blocks.

## 1) **sentinel_robust_tripwire_v1**

### Intuition
Use a robust policy as the default controller at all times, then add a **conservative learned tripwire** that only allows short bursts of predictor-following behavior when online evidence says risk is low. This is intentionally robust-first: unlike a direct learned eviction policy, the learned component never chooses arbitrary victims globally; it only grants (or denies) temporary permission to follow an already-implemented predictor expert.

### Exact decision rule
Let:
- `R_t` = eviction event from `robust_ftp_d_marker` shadow at step `t`.
- `P_t` = eviction event from `follow_predicted_cache` shadow (same expert already used in RobustFtP).
- `risk_t in [0,1]` = learned risk score from a lightweight model over existing online features (e.g., disagreement rate, recent early-return rate, context frequency).
- `E_t` = early-return-burst score in a sliding window (same detector family as `guard_wrapper`).
- `B_t` = trust budget (integer credits), initialized `B_0 = B_init`.

On miss with full cache:
1. Compute robust candidate `v_R` and predictor candidate `v_P` from shadows.
2. If `(risk_t <= tau_risk) AND (E_t < tau_early) AND (B_t > 0)`, choose `v_P`; else choose `v_R`.
3. If chose `v_P`, decrement budget: `B_{t+1} = max(0, B_t - 1)`.
4. Replenish budget slowly when robust is chosen and no suspicious event occurs: `B_{t+1} = min(B_max, B_t + r_plus)`.
5. Hard override: if early-return burst trigger fires (`E_t >= tau_guard`), force robust-only mode for `D` steps (no learned override).

This is a **tripwire + budgeted override** design: robust by default, learned permission second.

### Existing repo components to reuse
- `RobustFtPDeterministicMarkerCombiner` as robust backbone.
- `FollowPredictedCachePolicy` from `robust_ftp_marker_combiner.py` as predictor expert.
- Early-return detector and guard-window logic from `GuardWrapperPolicy`.
- Feature plumbing from `learned_gate/features_v2.py` and lightweight estimator path from `learned_gate/lightweight_estimator.py`.
- Decision diagnostics conventions from `rest_v1`/`ml_gate_v2`.

### New code/modules needed
- `src/lafc/policies/sentinel_robust_tripwire_v1.py` (new policy).
- Optional dataset builder for tripwire risk labels: `scripts/build_tripwire_dataset_v1.py`.
- Optional trainer for lightweight risk head: `scripts/train_tripwire_risk_v1.py`.
- Test coverage similar to `tests/test_guarded_wrapper.py` + combiner tests.

### Why it might beat current robust winners
- It preserves robust behavior in adversarial/noisy regions, but opportunistically captures predictor upside during low-risk windows where `robust_ftp_d_marker` can be conservative.
- Budgeting prevents runaway trust; burst failures are cut off quickly via tripwire.
- Because override is sparse and reversible, robustness degradation risk is lower than full learned control.

### What could go wrong
- If learned risk is miscalibrated, trusted windows may be too rare (no gain) or too frequent (robustness leak).
- Budget/threshold tuning may be sensitive to trace family.
- Running multiple shadows plus detector/model may add complexity and debugging overhead.

---

## 2) **consensus_safe_set_gate_v1**

### Intuition
Instead of picking one policy to follow, build a **safe candidate set by expert consensus**, then let learned signals act only as a tie-breaker inside that safe set. Robustness comes from constraining action space to evictions that robust experts already consider acceptable.

### Exact decision rule
On miss with full cache, compute candidate victims from existing experts:
- `v_marker` from `MarkerPolicy`
- `v_lru` from `LRUPolicy`
- `v_bo` from `BlindOraclePolicy`
- `v_ftp` = victim from active expert in `robust_ftp_d_marker` shadow (or robust expert choice only)

Build votes over cached pages:
- `vote(p) = 1[p=v_marker] + 1[p=v_lru] + 1[p=v_bo] + 1[p=v_ftp]`.

Define safe set:
- `S_t = { p in cache : vote(p) >= 2 }` (majority-backed pages).
- If `S_t` empty, fallback `S_t = {v_marker, v_lru}`.

Choose final eviction:
1. If `|S_t| = 1`, evict that page.
2. Else compute learned rank score `L_t(p)` only for `p in S_t` (existing predictor/feature stack).
3. Evict `argmax_{p in S_t} L_t(p)` **only if** margin inside `S_t` exceeds `m_min`; otherwise evict robust default `v_marker`.

So learned logic never selects pages outside consensus-safe region.

### Existing repo components to reuse
- `MarkerPolicy`, `LRUPolicy`, `BlindOraclePolicy`, `RobustFtPDeterministicMarkerCombiner` shadow mechanics.
- Existing predictor scoring utilities from `learned_gate/features.py` (or atlas bucket scoring snippets).
- Margin-trigger pattern from hybrid fallback experiment (top1-top2 threshold idea).

### New code/modules needed
- `src/lafc/policies/consensus_safe_set_gate_v1.py`.
- Shared helper for shadow-expert vote extraction (could live in `src/lafc/policies/` utility module).
- Lightweight tests for consensus construction, tie handling, and robust fallback behavior.

### Why it might beat current robust winners
- Reduces single-expert blind spots by requiring cross-expert agreement before learned intervention.
- Still allows predictor-informed gains when multiple experts expose ambiguity and safe alternatives.
- Could outperform simple two-expert follow-the-leader by using richer committee information at each decision.

### What could go wrong
- Majority set may often collapse to one robust choice, producing little headroom.
- If experts are correlated, consensus may not add much diversity.
- Additional shadow computations increase runtime.

---

## 3) **phase_guarded_robust_blender_v1**

### Intuition
Use **phase-level robustness control**: within each marker-like phase, start in robust mode, allow limited learned blending only after proving local stability, and immediately revert when regret signals rise. This differs from per-step gating by adding a structured phase state machine.

### Exact decision rule
Maintain phase state (aligned with marker clean/marked regime):
- mode `M_t in {ROBUST, BLEND}`.
- phase-local counters: disagreement `d_t`, early-return `e_t`, and trusted-good outcomes `g_t`.

Eviction score when in `BLEND`:
- `score(p) = alpha_t * robust_score(p) + (1 - alpha_t) * learned_score(p)`,
- where `alpha_t = clip(alpha_min, 1.0, 1 - u_t)`, and `u_t` is uncertainty from context/history.

Transitions:
1. Phase start: `M_t = ROBUST`, `alpha_t = 1.0`.
2. Promote to `BLEND` only if `g_t >= G_min` and `e_t = 0` for `W_promote` steps.
3. In `BLEND`, if `(e_t >= E_max) OR (d_t >= D_max)` then immediate demotion to `ROBUST` and cooldown `C` steps.
4. On phase reset, clear counters and return to `ROBUST`.

Victim choice:
- If `M_t = ROBUST`, use robust victim (marker or robust_ftp robust expert).
- If `M_t = BLEND`, use blended score with conservative tie-band; if tie-band hit, fallback robust victim.

### Existing repo components to reuse
- Phase logic intuition from `MarkerPolicy`.
- Trust-update and delayed outcome mechanics from `rest_v1` / `atlas_v3`.
- Early-return guard detector from `guard_wrapper`.
- Blending and tie-band concepts from `atlas_v3` / `atlas_cga_v2` diagnostics framework.

### New code/modules needed
- `src/lafc/policies/phase_guarded_robust_blender_v1.py`.
- Optional helper for phase state tracking (if not already exposed by marker internals).
- New diagnostics export (mode timeline, promote/demote events, cooldown occupancy).

### Why it might beat current robust winners
- Gives learned signals a chance only in stable local regimes, where they are more likely to add value.
- Phase-level hysteresis can reduce switch thrashing seen in purely per-step selectors.
- Structured demotion rules should preserve worst-case behavior better than unconstrained blending.

### What could go wrong
- Overly strict promotion conditions may keep it effectively robust-only.
- Phase definitions might not align with true regime changes in non-marker-like traces.
- More hyperparameters (promotion, demotion, cooldown) increase tuning burden.

---

## Ranked recommendation (implement one first)

1. **sentinel_robust_tripwire_v1** (**recommended first**)  
   Best balance of feasibility, robustness-first posture, and upside. It is closest to existing robust winner structure, reuses proven guard machinery, and introduces learned influence only through a tightly bounded override channel.

2. **consensus_safe_set_gate_v1**  
   Strong robustness story via constrained action set; good second step if tripwire shows either too little or too much override.

3. **phase_guarded_robust_blender_v1**  
   Promising but higher design/tuning complexity; better as a second-wave method after we establish stable tripwire baselines.
