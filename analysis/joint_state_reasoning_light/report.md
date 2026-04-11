# Joint-state eviction reasoning ablation (lightweight)

## Setup
- Fully separate lightweight path; no heavy_r1 reruns and no manuscript-artifact edits.
- Same compact subset as recent lightweight ablations: 3 repository traces + 3 stress traces.
- Same capacities and horizon used in that lane (default capacities 2/3/4, horizon=8).
- Comparison targets: base `evict_value_v1` regression and history-aware pairwise objective variant.

## New decision-state dataset
- Each row is a single eviction state with incoming item, full current cache residents, compact history summary, and oracle victim label.
- Rows: 61 decisions (`joint_state_dataset.csv`).

## Candidate-decision quality (test split)
- base_regression: top1=0.5000, mean_regret=0.0000
- history_pairwise: top1=0.3750, mean_regret=0.0000
- joint_state_softmax: top1=0.3750, mean_regret=0.3750

## Downstream replay misses (mean over trace×capacity)
- base_regression: 5.556
- history_pairwise: 5.333
- joint_state_softmax: 5.833

## Explicit answers
1. Does joint cache-state reasoning show meaningful promise? **no**.
2. Does it outperform independent scoring enough for a pivot? **No** in this lightweight run.
3. Should it remain future work for now? **Yes** unless/until it consistently beats the best lightweight pairwise baseline across broader traces.
