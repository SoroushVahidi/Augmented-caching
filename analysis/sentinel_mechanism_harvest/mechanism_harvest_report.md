# Sentinel mechanism harvest report

## Scope and inputs
This report combines:
- `analysis/sentinel_disagreement_stress/disagreement_stress_results.csv`
- `analysis/sentinel_semirealistic_eval/semirealistic_results.csv`
- `analysis/sentinel_semirealistic_eval/semirealistic_report.md` (for per-slice helpful/harmful override counts)

The merged slice-level view is exported to:
- `analysis/sentinel_mechanism_harvest/mechanism_harvest_slices.csv`

## Cross-suite recap
- **Disagreement-stress:** sentinel vs robust W/T/L = **3/2/1**, mean delta = **-0.5 misses** (sentinel better), helpful/harmful overrides = **5/2**.
- **Semi-realistic:** sentinel vs robust W/T/L = **2/1/3**, mean delta = **+0.333 misses** (sentinel worse), helpful/harmful overrides = **3/5**.

Interpretation: sentinel has a **real but narrow upside mechanism** (it can convert disagreement into gains), but the full controller is not robust on average in the semi-realistic regime.

## Where sentinel helps (exact situations)
Sentinel wins are concentrated in slices with **positive net override value** (helpful overrides > harmful overrides):

1. `synthetic::predictor_disagreement_help`, cap=2  
   - disagreement steps: 4/30
   - delta vs robust: -1
   - helpful/harmful: 1/0
2. `synthetic::predictor_disagreement_mixed`, cap=2  
   - disagreement steps: 4/30
   - delta vs robust: -1
   - helpful/harmful: 1/0
3. `synthetic::predictor_disagreement_mixed`, cap=3  
   - disagreement steps: 3/30
   - delta vs robust: -2
   - helpful/harmful: 2/0
4. `semi::phase_locality_drift`, cap=4  
   - disagreement steps: 1/240
   - delta vs robust: -1
   - helpful/harmful: 1/0
5. `semi::daypart_popularity_mix`, cap=4  
   - disagreement steps: 10/240
   - delta vs robust: -1
   - helpful/harmful: 2/1

**Pattern:** wins occur when predictor overrides are **sparse and selective** and net-positive.

## Where sentinel hurts (exact situations)
Losses align with slices where predictor overrides are net harmful:

1. `synthetic::predictor_disagreement_hurt`, cap=2  
   - disagreement steps: 9/30
   - delta vs robust: +1
   - helpful/harmful: 0/1
2. `semi::phase_locality_drift`, cap=3  
   - disagreement steps: 14/240
   - delta vs robust: +1
   - helpful/harmful: 0/1
3. `semi::locality_with_short_scans`, cap=3  
   - disagreement steps: 63/240
   - delta vs robust: +2
   - helpful/harmful: 0/2
4. `semi::daypart_popularity_mix`, cap=3  
   - disagreement steps: 34/240
   - delta vs robust: +1
   - helpful/harmful: 0/1

**Pattern:** losses occur when disagreement is frequent and the predictor line is locally fragile; the full sentinel still spends predictor overrides in those periods.

## What is the single best mechanism to salvage from sentinel?
**Best single mechanism:** **disagreement-gated override**.

Concretely: only consider predictor intervention on steps where robust and predictor shadows propose different actions (`disagree_now == 1`), and otherwise default to robust behavior.

Why this is the best salvage candidate:
- It directly matches where wins come from (targeted conversion of disagreement opportunities into net-positive overrides).
- It is minimal, interpretable, and easy to audit.
- It avoids importing most of sentinel’s interacting control surfaces (risk logistic head, budget dynamics, tripwire windows, guard duration), which are likely where calibration brittleness enters.

## Why does this mechanism help in the slices where sentinel wins?
In winning slices, the predictor line appears to provide **occasional local corrections** exactly at disagreement points; these are captured as helpful overrides (predictor hit while robust would miss). The mechanism works because it acts like a sparse “delta-capture” layer rather than a full controller replacement.

## Why does the full sentinel hurt in semi-realistic evaluation?
The full sentinel bundles multiple coupled controls (risk score, trust budget depletion/recovery, early-return detection, guard mode). In semi-realistic traces with larger non-stationary segments (especially short-scan perturbations), this bundle still permits harmful predictor actions often enough (5 harmful vs 3 helpful) to lose robustness on average.

Short version: the **core disagreement-trigger idea is useful**, but the **full control stack is over-coupled and miscalibrated for mixed real-like drift regimes**.

## Is a minimal robust_ftp_plus_X variant worth implementing next?
**Yes — with strict minimalism.**

Recommendation:
- Implement exactly one additive mechanism on top of `robust_ftp_d_marker`: a **conservative disagreement-gated predictor override** layer.
- Do **not** carry over tripwire, budget, or risk-logit in v1 of `robust_ftp_plus_X`.
- Keep diagnostics first-class (override count, helpful/harmful decomposition, disagreement frequency).

This is worth testing because it preserves the only clearly transferable upside while minimizing new fragility.
