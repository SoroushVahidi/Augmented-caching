# ATLAS v2 Focused Diagnostics Report

## Executive summary
- Dynamic atlas_v2 mean match-to-LRU across baseline traces/caps: 0.639.
- Dynamic atlas_v2 mean predictor-dominated fraction: 0.306.
- Mean misses: dynamic=6.83, gamma=1 fixed=6.83, gamma=0 fixed=7.50.

## Is gamma the bottleneck?
- Parameter sweep runs better than LRU: 320/1920.
- Fixed gamma=1 vs dynamic (mean miss gap): +0.00 (negative means fixed gamma=1 better).
- Interpretation: global gamma helps in some settings but is not a dominant standalone fix.

## Is the mismatch proxy the bottleneck?
- Proxy audit events: 53; proxy positives=14; rapid-regret(H=3) positives=22; false positives=0; false negatives=8.
- Interpretation: on baseline traces the proxy is often sparse/uninformative; stress traces reveal misses when rapid regret happens with low bucket hints.

## Is score scaling the bottleneck?
- Dynamic run mean predictor contribution: 0.229.
- Dynamic run mean fallback contribution: 0.253.
- Dynamic run mean predictor-cannot-overturn fraction: 0.442.
- Interpretation: fallback term often numerically dominates ranking, limiting predictor-led overrides.

## Does atlas_v2 really collapse to LRU?
- Dynamic atlas_v2 average match-to-LRU: 0.639.
- Dynamic atlas_v2 average match-to-blind_oracle: 0.076.
- Dynamic atlas_v2 average match-to-predictive_marker: 0.167.
- On current baseline traces, atlas_v2 is frequently closer to LRU than to prediction-led references.

## What trace patterns expose the weakness best?
- Predictor-bad/LRU-good and confidence-miscalibrated stress traces sharply penalize predictor-heavy settings.
- Regime-shift traces expose lag in global trust adaptation.
- Mixed/confidence-calibrated traces show gains only when confidence and bucket quality align.

## Time-series highlights (selected dynamic runs)
- Gamma often remains high on tiny traces with few resolved predictor-dominated checks.
- Rolling mismatch can saturate quickly when threshold is small and predictor-dominated decisions are rare.

## Most likely next improvement
Move from global trust to local trust + stronger mismatch target: per-page or per-context trust updates driven by direct rapid-regret signals.