# Cursor takeaway (exploratory only)

This lightweight history-aware ablation suggests a **small, non-uniform** downstream benefit from richer history context for the learned eviction scorer.

- Mean replay misses improved from **5.667** (base) to **5.500** (history-aware), i.e., **+0.167 base-minus-history miss delta**.
- Improvements were concentrated in **2 of 12** trace-capacity cells (`stress::predictor_bad_lru_good` at capacity 2, and `stress::mixed_regime` at capacity 2); the other **10/12** cells were ties.
- Candidate-ranking test metrics were unchanged on this run (same top-1 match and test regret), so the replay gain appears limited and scenario-specific.

Interpretation: richer history context is **promising but modest** under this lightweight setup; evidence is exploratory and should not be treated as canonical KBS `heavy_r1` support.

Contribution-story impact: **neutral-to-slightly strengthening** for the paper narrative (it supports plausibility of context enrichment), but not strong enough alone to claim a material, broadly robust downstream gain.
