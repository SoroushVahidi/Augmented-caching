# Reuse-Tail Horizon Diagnostic v1 Synthesis

Status: local diagnostic closeout, completed and integrity-checked on
2026-08-11.

## Artifact

- Output directory: `analysis/reuse_tail_horizon_diagnostic_v1/`
- Final report: `analysis/reuse_tail_horizon_diagnostic_v1/report.md`
- Integrity summary:
  `analysis/reuse_tail_horizon_diagnostic_v1/integrity_summary.json`
- Log:
  `logs/kbs_second_revision/reuse_tail_horizon_diagnostic_20260811.log`

## Integrity

The diagnostic completed its declared scope:

- 21/21 family-capacity cells
- seven families: `brightkite`, `citibike`, `cloudphysics`, `metacdn`,
  `metakv`, `twemcache`, `wiki2018`
- capacities `32`, `64`, `128`
- horizons `1`, `2`, `4`, `8`, `16`
- no duplicate family-capacity cells
- no duplicate family-capacity-horizon rows
- no NaN/Inf values in the checked summary CSVs
- integrity status `COMPLETE`

## Primary H=4 Result

At `H=4`, across all family-capacity cells:

- `P(T > 4 | resident) = 0.9938544459677984`
- `P(T > 4 | resident, eventually reused) = 0.9793302186526528`
- never-reused fraction among resident candidates =
  `0.7026792916224847`

Capacity trend for `P(T > 4 | resident)`:

- `C=32`: `0.987377360773`
- `C=64`: `0.992891528854`
- `C=128`: `0.996078682684`

## Interpretation

This result establishes that the finite `H=4` window sees very little of
resident objects' eventual reuse behavior in the measured decision
population. The finding is broad across seven families and three
capacities, and its capacity trend is directionally consistent with the
Wulver-relayed broad H=4 degeneracy trend: as capacity increases, both the
target-degeneracy measurements and the resident-reuse tail become more
severe.

The result supports H4/H11 as an observability limitation: many objects
that are resident at decision time are either never reused again or are
reused only after the finite supervision window. It also weakly supports
H10 as an empirical capacity-scaling concern, because the outside-window
tail grows with capacity in the same direction as the broad degeneracy
trend.

The result does not establish that reuse after `H` causes an avoidable
miss. A reuse after `H` is a potential unseen future consequence. Whether
evicting that object creates a causal excess miss depends on the
counterfactual policy trajectory and on which alternative object would have
been evicted. That causal question remains open.

## Documentation Consequence

- H4 should move from "plausible but under-specified" to supported as a
  horizon-observability limitation, not as a complete causal mechanism.
- H10 should be treated as empirically strengthened by a capacity trend,
  but still not as a proved `H/C` law.
- H11's first, non-causal `P(T > H | resident)` pass is complete; the
  causal excess-miss attribution pass is still not implemented.
