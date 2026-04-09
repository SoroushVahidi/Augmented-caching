# Offline Baseline 2: General Caching (LP + deterministic rounding)

## Positioning

This is the first offline baseline in the repo for **general caching / file caching** with variable sizes and retrieval costs.

- **Classification:** informed, approximation-inspired implementation.
- **Faithfulness level:** **(3) informed approximation-inspired implementation**.

It is inspired by resource-allocation / interval-packing formulations used in general caching approximation literature (including Bar-Noy et al. style reductions), but it is **not claimed as a theorem-faithful reimplementation** of a published constant-factor proof algorithm.

## Problem variant handled

Supported model:

- offline (full request sequence known)
- page/item size `s_j > 0` (arbitrary positive reals)
- retrieval cost `c_j > 0` (arbitrary positive reals)
- cache capacity `C > 0` (positive real)
- objective: minimize total retrieval cost
- **optional insertion/bypass on miss: supported**

Input requirements:

- JSON trace must include top-level `sizes: {page_id: size}`
- retrieval costs come from `weights` (existing repo convention)

## High-level method

1. Build **reuse intervals** `(t -> next_t)` per page occurrence.
2. Solve an LP relaxation of weighted interval packing:
   - maximize saved retrieval cost,
   - enforce capacity at each timeline slot.
3. Deterministically round LP fractions using capacity-aware greedy acceptance.
4. Simulate request processing under the rounded schedule with explicit bypass support.

## Outputs

Same text-only output style as offline Belady:

- `summary.json`
- `per_step_decisions.csv`
- `diagnostics.json`
- `report.md`

Diagnostics include:

- total retrieval cost
- misses/hits
- insertions and bypasses
- LP status and formulation size
- rounding metadata
- runtime

## Run command

```bash
python scripts/run_offline_general_caching_approx.py \
  --trace data/example_general_caching.json \
  --capacity 4 \
  --output-dir output/offline_general_caching_approx
```

## How this differs from offline Belady

- Belady baseline is exact for uniform paging only (unit size, unit cost).
- This baseline handles variable sizes and costs, but uses LP relaxation + rounding, so it is approximation-inspired and not exact OPT in general.

## Future extensions

- stronger rounding (e.g., dependent/pipage-style for interval packing)
- direct min-cost flow / MIP baselines for small traces
- faithful implementations of specific published approximation algorithms
- label-generation hooks from LP dual/primal signals
