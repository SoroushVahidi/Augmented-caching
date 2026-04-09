# Guarded robust wrapper for learned eviction policies

## Why add a guard layer?

A learned/prediction-heavy eviction policy can perform well when its scores are good,
but still fail sharply on regime shifts or noisy predictions. A guard wrapper adds a
lightweight safety mechanism:

- keep base policy in control when behavior is healthy,
- temporarily switch to robust fallback when risk signals spike,
- return to base mode after a bounded interval.

This repository implementation is a **framework-level wrapper**, not a new unrelated
policy family.

---

## Implemented policy

- CLI name: `evict_value_v1_guarded`
- Wrapped base policy: `evict_value_v1`
- Fallback policy: configurable (`lru` default, or `marker`)

Implementation file:

- `src/lafc/policies/guard_wrapper.py`

---

## Trigger rule (exact)

Detection signal: **base-policy early-return evictions**.

1. While base mode is active, record each page evicted by the base policy at time `t_e`.
2. If that evicted page is requested again within `W` requests (`t - t_e <= W`),
   count one suspicious event.
3. Maintain suspicious events in a sliding window of size `T` requests.
4. If suspicious count in that window reaches threshold `M`, trigger guard mode.

Config parameters:

- `W = --guard-early-return-window` (default `2`)
- `M = --guard-trigger-threshold` (default `2`)
- `T = --guard-trigger-window` (default `16`)

Interpretation: repeated very-early returns of recently evicted pages indicate the
base policy is currently making unsafe eviction choices.

---

## Fallback rule (exact)

When trigger fires, enter fallback mode for a fixed number of requests:

- duration `D = --guard-duration` (default `8`)
- fallback policy `--guard-fallback-policy` in `{lru, marker}`

After `D` requests, wrapper automatically returns to base mode.

---

## Faithfulness statement

This is a **repo-compatible guarded approximation inspired by recent guard-style
robustification literature**, not a claim of theorem-faithful implementation of a
specific paper algorithm.

What is exact in this repo:

- black-box base + robust fallback architecture
- explicit detector / trigger / finite guard interval
- low-overhead online state updates
- detailed diagnostics

What is interpreted:

- specific trigger signal (early-return burst)
- exact thresholds and guard duration choices
- deterministic tie behavior around wrapper modes

---

## Diagnostics emitted

The guarded policy exports both summary and per-step logs, including:

- number of guard triggers
- trigger timestamps
- trigger reason codes
- time spent in guarded mode
- base vs fallback step counts
- base vs fallback eviction counts
- count of early-return suspicious events
- per-step mode transitions and detector counters

Output paths via runner include:

- `metrics.json` (flattened `evict_value_v1_guarded_*` fields)
- `evict_value_v1_guarded_steps.csv`

---

## Example command

```bash
python -m lafc.runner.run_policy \
  --policy evict_value_v1_guarded \
  --trace data/example_atlas_v1.json \
  --capacity 3 \
  --evict-value-model-path models/evict_value_v1_hist_gb.pkl \
  --guard-fallback-policy lru \
  --guard-early-return-window 2 \
  --guard-trigger-threshold 2 \
  --guard-trigger-window 16 \
  --guard-duration 8
```

---

## Limitations

- Detection uses only one lightweight signal (early-return burst), not full
  confidence calibration or uncertainty estimation.
- Guard mode duration is fixed, not adaptive to environment changes.
- Base and fallback are shadow-advanced for fast switching; this is practical but
  still an implementation choice.

---

## Future directions

- Add additional principled detectors (e.g., disagreement + subsequent regret proxy).
- Learn trigger thresholds per trace family / capacity regime.
- Support wrappers for `ml_gate_v2`, `rest_v1`, and future `evict_value_v2`.
- Evaluate guard efficacy with stress traces and controlled corruption schedules.
