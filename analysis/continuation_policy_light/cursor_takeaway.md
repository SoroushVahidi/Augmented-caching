# Cursor takeaway: continuation-policy lightweight rollout ablation

This exploratory run compared rollout-label continuation rules (`lru`, `blind_oracle`, `fifo`) on a tiny setup (example traces, capacities 2/3, horizon 8, capped requests) and stayed fully outside canonical `heavy_r1` artifacts.

## What changed
- Added a minimal rollout continuation option `fifo` to the finite-horizon label simulator (`src/lafc/evict_value_v2_rollout.py`) so we could include a third deterministic protocol without heavy engineering.
- Extended lightweight dataset-builder CLI choices to accept `fifo` where continuation/reference policy is configured.
- Added `scripts/experiments/run_continuation_policy_light_ablation.py` to generate:
  - `summary.csv`
  - `downstream_replay.csv`
  - `label_agreement.csv`
  - `summary.json`
  - `report.md`

## Interpretation (for this tiny run only)
1. **Continuation protocol affected offline target quality metrics**: held-out top-1 eviction match and mean chosen regret differed across protocols in `summary.csv` (blind_oracle > lru > fifo in this sample).
2. **Downstream replay was inconclusive here**: mean replay delta vs LRU was identical across protocols in this small run; no material replay separation appeared.
3. **Agreement was mixed**:
   - `fifo` and `lru` labels mostly agreed (high top-1 agreement).
   - `blind_oracle` had notably lower agreement versus both, indicating meaningfully different supervision on a non-trivial subset of decisions.

## Practical takeaway
- `lru` continuation looks **competitive as a simple default proxy** for lightweight workflows: it was close to `blind_oracle` on replay in this run, and far simpler/cheaper.
- However, continuation rule can still change supervision labels noticeably, so larger follow-up checks are warranted before making strong claims.

