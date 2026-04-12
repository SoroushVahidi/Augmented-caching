# Continuation-policy lightweight ablation

- Traces: 4 matched by `data/example*.json` (capped at 300 requests/trace).
- Capacities: [2, 3]; horizon: 8.
- Protocols compared: lru, blind_oracle, fifo.

## Offline label-quality proxy (held-out decisions)
- lru: top1=0.000, mean_chosen_regret=0.333, decisions=3.
- blind_oracle: top1=0.667, mean_chosen_regret=0.000, decisions=3.
- fifo: top1=0.000, mean_chosen_regret=1.000, decisions=3.

## Downstream replay proxy (model trained per protocol)
- lru: mean delta vs LRU = -1.500 misses (negative is better).
- blind_oracle: mean delta vs LRU = -1.500 misses (negative is better).
- fifo: mean delta vs LRU = -1.500 misses (negative is better).

## Label agreement
- blind_oracle vs fifo: top1 label agreement=0.658, mean abs regret diff=0.273 over 38 decisions.
- blind_oracle vs lru: top1 label agreement=0.632, mean abs regret diff=0.216 over 38 decisions.
- fifo vs lru: top1 label agreement=0.921, mean abs regret diff=0.102 over 38 decisions.

Best mean chosen-regret protocol in this run: **blind_oracle**.

Exploratory only: this script intentionally does not touch heavy_r1 Slurm/manuscript artifacts.
