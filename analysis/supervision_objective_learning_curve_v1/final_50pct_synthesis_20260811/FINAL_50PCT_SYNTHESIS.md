# Final 50% learning-curve synthesis

Scope: same-target scalar-vs-pairwise learning-curve diagnostic for `supervision_objective_learning_curve_v1`. This is a documentation/synthesis closeout; no new experiment was launched and the 100% fraction was intentionally not run.

## Integrity

- Fraction 0.5 rows: `42/42`
- Families: `brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018`
- Capacities: `32, 64, 128`
- Conditions: `eviction_loss_pairwise, eviction_loss_scalar`
- Status: `{'ok': 42}`
- Duplicate-key count: `0`
- NaN/Inf count: `0`
- Model SHA mismatches at 50%: `0`
- Audit files: `30` total, `7/7` fraction-0.5 units
- Campaign state marks fraction 0.5 complete: `True`
- Source `policy_comparison.csv` SHA-256: `5323eea6e3f6fb9a44b2fab2f6632f61917442ba239eababc1b2cda1fca8612a`

## Apples-to-apples curve

The apples-to-apples curve below uses the four families present at every tested fraction from 1% through 50%: `brightkite`, `citibike`, `cloudphysics`, and `metacdn`. Positive gap means pairwise has a higher downstream miss ratio than scalar.

| Fraction | Cells | Scalar MAE | Scalar miss ratio | Pairwise miss ratio | Pairwise - scalar | Scalar wins | Ties | Pairwise wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.01 | 12 | 0.986665 | 0.625606 | 0.829929 | 0.204323 | 12 | 0 | 0 |
| 0.02 | 12 | 0.983932 | 0.618331 | 0.829577 | 0.211246 | 12 | 0 | 0 |
| 0.05 | 12 | 0.983804 | 0.616458 | 0.829621 | 0.213163 | 12 | 0 | 0 |
| 0.1 | 12 | 0.982503 | 0.611021 | 0.829658 | 0.218638 | 12 | 0 | 0 |
| 0.25 | 12 | 0.982593 | 0.613652 | 0.829569 | 0.215917 | 12 | 0 | 0 |
| 0.5 | 12 | 0.982667 | 0.612613 | 0.829979 | 0.217367 | 12 | 0 | 0 |

## Changes

| Range | Scalar MAE change | Scalar miss-ratio change | Pairwise miss-ratio change | Gap change |
| --- | ---: | ---: | ---: | ---: |
| 0.25 -> 0.5 | 0.000074 | -0.001040 | 0.000410 | 0.001450 |
| 0.01 -> 0.5 | -0.003998 | -0.012994 | 0.000050 | 0.013044 |

## Full 50% seven-family comparison

Across all seven families at 50%, scalar is better on `18/21` family/capacity cells, ties on `3/21`, and pairwise is better on `0/21`. The mean pairwise-minus-scalar downstream miss-ratio gap is `0.161055`.

| Family | Cells | Scalar miss ratio | Pairwise miss ratio | Pairwise - scalar | Scalar wins | Ties | Pairwise wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| brightkite | 3 | 0.403667 | 0.632067 | 0.228400 | 3 | 0 | 0 |
| citibike | 3 | 0.410342 | 0.828825 | 0.418483 | 3 | 0 | 0 |
| cloudphysics | 3 | 0.987183 | 0.999708 | 0.012525 | 3 | 0 | 0 |
| metacdn | 3 | 0.649258 | 0.859317 | 0.210058 | 3 | 0 | 0 |
| metakv | 3 | 0.768700 | 0.820200 | 0.051500 | 3 | 0 | 0 |
| twemcache | 3 | 0.740658 | 0.947075 | 0.206417 | 3 | 0 | 0 |
| wiki2018 | 3 | 1.000000 | 1.000000 | 0.000000 | 0 | 3 | 0 |

## Interpretation

Within the tested 1%-50% range, the sample-size explanation is not supported as the primary cause of the scalar-vs-pairwise gap. Scalar downstream miss ratio changes only modestly, pairwise downstream miss ratio remains essentially flat and worse, and the 50% seven-family slice still favors scalar in every non-tied cell. This does not show that more data can never help; it records that the predefined stopping evidence supports `STOP_SAMPLE_SIZE_HYPOTHESIS`, so the 100% fraction is not active required work for this campaign.
