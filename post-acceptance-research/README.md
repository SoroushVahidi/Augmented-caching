# Post-Acceptance Research — November 2026

This package records Soroush Vahidi's research after acceptance of [Decision-aligned eviction-value prediction for learning-augmented caching](https://doi.org/10.1016/j.knosys.2026.117353) in *Knowledge-Based Systems* on November 15, 2026. The November investigations are separate from the accepted manuscript and have not undergone its peer-review process.

- [Standalone report](REPORT.pdf) and [editable LaTeX and bibliography](report-source/)
- [Exact shared LRU labeling algorithm and proofs](EXACT-LRU.md)
- [Chronological experiment and artifact inventory](PROVENANCE.md) and [source ledger](source-provenance.csv)
- [Aggregate results](aggregate-results/), [figures](figures/), and [Phase 2B](experiment-protocols/phase2b.md), [2C-A](experiment-protocols/phase2ca.md) and [2C-B](experiment-protocols/phase2cb.md) protocols
- [Reproducibility guide](REPRODUCIBILITY.md) and [Phase 2C-B runtime configuration](phase2cb-configuration.json)
- [Manuscript boundary](MANUSCRIPT-BOUNDARY.md), [licensing review](LICENSING.md), and [artifact manifest](MANIFEST.sha256)
- [Consolidation validation](VALIDATION.md)

## Findings and stopping point

The shared LRU labeler evaluates all forced initial victims in expected O(C + H) time and O(C) core auxiliary memory for cache capacity C and an H-request suffix. Under deterministic LRU continuation, one forced alternative adds at most one miss. At least max(1, C − H) actions tie for the optimum. The median recorded speedup was 19.7× over compiled exhaustive replay, which was faster in some small cells.

On the seven historical development families and 840,000 scored requests, the optimized boosting finalist incurred **561,396 misses** versus **565,126 for LRU**. It saved **3,730 misses (0.660%)**, with **0.560%** mean family improvement and a **0.064%** regression on the worst family. It improved all three primary capacities and saved 703 misses after excluding Alibaba Block. It took **31.50 µs/request (18.35× LRU CPU)**. The preliminary abstention configuration incurred **561,424 misses (11.78× LRU CPU)**.

Longer supervision and additional data did not consistently improve closed-loop performance. Compact history and benefit/risk estimates helped in matched comparisons. Uniform shortlists covered more fixed-state opportunities than their replay gains suggested. The boosting finalist beat its 15 matched random runs. All 15 matched random runs beat the Ridge finalist. No finalist passed the full screen, and the measurements do not establish deployment readiness or independent generalization. Independent confirmation workloads remain sealed. The experimental phase concluded with Phase 2C-B.
