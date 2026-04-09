# sentinel_budgeted_guard_v2 sanity report

## Setup
- Trace: `synthetic::predictor_disagreement_help` (predictor_help)
- Capacity: 2
- Requests: 30

## Result
- Misses: 15
- Hits: 15
- Predictor steps: 2
- Guard triggers: 0
- Harmful overrides: 0
- Remaining override budget: 2

Sanity verdict: policy executes, logs diagnostics, and remains robust-first (predictor usage is bounded).
