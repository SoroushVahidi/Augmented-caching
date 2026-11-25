# Aggregate result tables

| Files | Existing evidence used |
|---|---|
| `november-neural-replay.csv` | Terminal November campaign replay summary, with measured and retrieved origins |
| `recurrent-replay.csv` | Terminal Phase 1 reported model rows, with distinct development and historical outer roles |
| `phase2a-*` | Positive-horizon and shortlist diagnostics from the Phase 2A evidence tables |
| `phase2b-*` | Verified matched students and fallback portfolio totals, labeler benchmark and correctness counts, shortlist and sample adequacy |
| `phase2ca-*` | Verified reducer horizon, sampling, shortlist, feature information, decomposition and resource aggregates |
| `phase2cb-*` | Verified reducer aggregate policy, family, capacity, advancement, controls, resource, CPU component and parity results |

Public result columns identify groups of runs, workload families, capacities or model configurations. Both family and capacity totals for the finalist reconcile to the whole-grid policy total. Where present, relative miss reduction is a fraction unless the header or displayed value indicates percent. `cloudphysics` is the historical key for Alibaba Block.

The profile table records component times sampled in a single replay harness. The `path` column identifies the code path used for timing and has no filesystem meaning. Parent timing components include nested work. See the [reproducibility guide](../REPRODUCIBILITY.md) for interpretation and original inputs, and the [inventory](../PROVENANCE.md) for execution dates and source revisions.
