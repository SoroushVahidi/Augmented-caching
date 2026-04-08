# atlas_v2 first check

## Questions
1. Did atlas_v2 reduce LRU collapse?
2. Did dynamic trust help on good predictions?
3. Did it improve robustness under noise?
4. What still looks weak?

## Aggregate quick stats
- LRU-collapse proxy (distance-to-LRU on trace buckets): atlas_v2 is farther from LRU in 0/6 settings.
- Good predictions (perfect buckets/conf=1.0): atlas_v2 beats atlas_v1 in 1/6 settings (ties: 5).
- Noise robustness (perfect buckets + 0.3 corruption): atlas_v2 better in 1/6, worse in 2/6 settings.

## Explicit answers
1) LRU collapse reduction: **not yet on this first check** (farther-from-LRU in 0/6 trace-bucket settings).
2) Dynamic trust on good predictions: **yes, modestly** (wins 1/6, ties 5/6).
3) Robustness under noise: **mixed-positive** (wins 1/6, losses 2/6).
4) Weak points: still sensitive on tiny traces; gains are small and not uniform across capacities.

## Command
- `PYTHONPATH=src python - <<'PY' ...` (this script)
