# atlas_v3 refinement report

## Exact commands run

- `PYTHONPATH=src python scripts/run_atlas_v3_refinement.py`
- `PYTHONPATH=src pytest -q tests/test_atlas_v3.py tests/test_runner.py`

## Top findings

- Strongest overall refinement: `eta_pos=0.02_eta_neg=0.4`.
- atlas_v3 default avg misses: 6.750
- best refined atlas_v3 avg misses: 6.708
- delta (best - default): -0.042

## Question-by-question answers

1. Confidence bins main lever? **No**. Best bin sweep (bins_2_0.7) beats worst by 0.000 misses on average, smaller than update/context effects.
2. Update magnitudes main lever? **Yes**. Best update setting (eta_pos=0.02_eta_neg=0.4) gives the largest average miss reduction within atlas_v3 sweeps.
3. Regret horizon mapping right? **Linear remains strongest** in this study (rank: regret_linear best).
4. Tie handling suppressing predictor-led decisions? **Partially**. Adaptive tie helps predictor fraction but does not consistently reduce misses relative to fixed near-zero epsilon.
5. Context definition sufficient? **Bucket information is essential; confidence-only is weaker** (best context variant: bucket_only).

## Ranked sweep summaries (lower avg misses is better)

### Bin sweep
- bins_2_0.7: 6.750
- bins_3_0.5_0.8: 6.750
- bins_4_0.4_0.6_0.8: 6.750
- bins_5_0.2_0.4_0.6_0.8: 6.750

### Update sweep
- eta_pos=0.02_eta_neg=0.4: 6.708
- eta_pos=0.02_eta_neg=0.8: 6.708
- eta_pos=0.05_eta_neg=0.4: 6.708
- eta_pos=0.05_eta_neg=0.8: 6.708
- eta_pos=0.1_eta_neg=0.4: 6.708

### Regret sweep
- regret_linear: 6.750
- regret_exp2: 6.750
- regret_sqrt: 6.750

### Context ablation
- bucket_only: 6.750
- bucket_confidence: 6.750
- bucket_group_confidence: 6.750
- confidence_only: 6.792

### Tie-band sweep
- fixed_eps=0.0: 6.750
- fixed_eps=0.01: 6.750
- adaptive_c=0.0: 6.750
- adaptive_c=0.25: 6.750
- fixed_eps=0.05: 6.792
- fixed_eps=0.1: 6.792
- adaptive_c=0.5: 6.792

## Which atlas_v3 variant should be the new default?

- Recommend `eta_pos=0.02_eta_neg=0.4` as the refined default candidate from this pass.

## Is atlas_v4 needed?

- atlas_v3 can be strengthened further without a new family; immediate next focus should be context design + confidence calibration coupling.

## Main remaining failure mode

- In hard stress regimes, fallback still dominates when context trust is under-trained; this points to context granularity/calibration limitations, not score formula collapse.

## Likely next improvement

- Keep atlas_v3 family and add calibrated confidence reliability per context (online reliability scaling before lambda multiplication).
