# Hybrid fallback experiment report

## Key question
Can a simple confidence-aware fallback make the learned candidate-scoring policy better downstream than pointwise alone?

## Protocol
- Fallback trigger: top1-vs-top2 score margin < threshold.
- Fallback policy: LRU.
- Threshold selected on validation workloads only; held-out test used only for final comparison.

## Held-out results
- Hybrid vs pointwise wins/ties/losses: 0/36/0
- Hybrid avg trigger frequency: 0.0000

| policy | total_misses | hit_rate | avg_trigger_frequency |
|---|---:|---:|---:|
| pointwise | 223 | 0.3592 | 0.0000 |
| hybrid | 223 | 0.3592 | 0.0000 |
| lru | 223 | 0.3592 | 1.0000 |

## Fragility analysis
- `trigger_analysis.csv` reports victim-reuse-within-window rates for triggered vs non-triggered decisions.
- Higher triggered victim-reuse rate indicates low-margin decisions are indeed more fragile.
