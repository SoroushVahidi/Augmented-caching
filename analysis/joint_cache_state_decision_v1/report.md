# Joint cache-state decision dataset (experimental)

This run constructs decision-state examples only (no model training).

## Summary
- total_decisions: 29
- capacities: [2, 3, 4]
- horizon: 8
- traces: 3
- split_counts: {'train': 23, 'val': 4, 'test': 2}
- mean_candidates_per_decision: 2.448

## Sample decisions
- `data/example_unweighted.json|cap=2|t=2` incoming=C oracle_victim=B candidates=['A', 'B']
- `data/example_unweighted.json|cap=2|t=3` incoming=A oracle_victim=C candidates=['B', 'C']
- `data/example_unweighted.json|cap=2|t=4` incoming=B oracle_victim=A candidates=['C', 'A']
- `data/example_unweighted.json|cap=2|t=5` incoming=D oracle_victim=B candidates=['A', 'B']
- `data/example_unweighted.json|cap=2|t=6` incoming=A oracle_victim=B candidates=['B', 'D']
