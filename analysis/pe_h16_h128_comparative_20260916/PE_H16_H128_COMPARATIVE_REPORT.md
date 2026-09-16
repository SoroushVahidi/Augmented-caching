# PE H16-H128 Comparative Report

Generated from existing artifacts only. H16 labels were not regenerated. H32/H64/H128 labels were not regenerated.

## Sources

- H16 source: `/home/soroush/projects/lafc-evict-dataset/worktrees/pe-results-polish/analysis/sigmod_target_discriminativeness_20260913/outputs/phase4_stratified_family_capacity_horizon.csv`
- H32/H64/H128 source: `analysis/pe_long_horizon_production_v1/summaries/table_A_family_capacity_horizon.csv`
- H16 source SHA256: `8777418be26904717568ed349c268063afb20e670a5b5eaa5a9f06098062630a`
- Long-horizon table SHA256: `401d3d5502c977129c3f2d714b71db0c62c4bd1bc854d6a396664113e39e6f68`

## Compatibility Audit

| field | compatible | note |
| --- | --- | --- |
| family | YES | Same five internal families; reader-facing mapping preserved. |
| capacity | YES | Same four capacities. |
| decision | YES | Definition matches; H128 production uses a stricter long-horizon-eligible denominator for twemcache cells. |
| candidate set | YES | Candidate row counts equal decision_count x capacity in every converted/validated cell. |
| counterfactual continuation | YES | Comparison reuses existing H16 and completed H32/H64/H128 summaries. |
| y_loss | YES | Same sign convention and target meaning. |
| optimal candidate set | YES | Both use tied argmin set. |
| all-tied criterion | YES | Equivalent criteria. |
| discriminative decision criterion | YES | Equivalent decision-level criterion. |
| random-optimal probability | YES | Per-cell capacity is fixed, so cell values match decision-weighted optimal-set fraction. |
| per-candidate/random regret | YES | Both are decision-weighted inside each cell. |
| request preprocessing | YES | Both draw from the same canonical processed family/capacity corpus identities. |
| cache initialization | YES | No source indicates a changed initialization rule. |
| admission semantics | YES | No source indicates a changed admission rule. |

The comparison is definition-compatible. One limitation is explicit: H16 is the canonical H16-eligible population, while the long-horizon production denominator is H128-eligible for each cell. The difference affects twemcache counts and is treated as a limitation, not hidden.

## Validation

- Validation status: `PASS`
- Expected cells: `80`
- H16 cells: `20`
- Long-horizon cells: `60`
- H16_REGENERATED: `NO`

## Primary Decision-Micro Summary

| horizon | decisions | all_tied_fraction | discriminative_fraction | random_optimal_probability | mean_random_regret |
| --- | --- | --- | --- | --- | --- |
| 16 | 787762 | 0.619295 | 0.380705 | 0.986942 | 0.013216 |
| 32 | 706888 | 0.636698 | 0.363302 | 0.985497 | 0.015178 |
| 64 | 706888 | 0.608204 | 0.391796 | 0.979246 | 0.023238 |
| 128 | 706888 | 0.596114 | 0.403886 | 0.971944 | 0.036698 |

Important scope note: the primary H16 row is the canonical H16-eligible population. H32/H64/H128 use the completed long-horizon production denominator. The paired cell-level deltas are therefore the preferred descriptive evidence for monotone within-cell horizon behavior; decision-micro rows are still reported because they match manuscript-style pooled summaries, but H16-to-H32 changes should be read with this denominator caveat.

## Cap32 Sanity / Scope Audit

| capacity | horizon | candidate_rows | decisions | all_tied_fraction | discriminative_fraction | random_optimal_probability | mean_random_regret |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | 16 | 6542016 | 204438 | 0.661800 | 0.338200 | 0.976588 | 0.023808 |
| 32 | 32 | 6072320 | 189760 | 0.674315 | 0.325685 | 0.975870 | 0.024541 |
| 32 | 64 | 6072320 | 189760 | 0.667011 | 0.332989 | 0.971094 | 0.029658 |
| 32 | 128 | 6072320 | 189760 | 0.666816 | 0.333184 | 0.970834 | 0.029939 |

The H32/H64/H128 cap32 rows match the earlier cap32 reference denominator (`6,072,320` candidate rows; `189,760` decisions). The canonical H16 source contains `6,542,016` candidate rows and `204,438` decisions at cap32, so the earlier H16 cap32 sanity values are not forced onto this adapter output. No existing H16-on-H128-eligible cap32 summary was found during the audit.

## Cell-Macro Summary

| horizon | cell_count | all_tied_fraction_mean | all_tied_fraction_median | discriminative_fraction_mean | random_optimal_probability_mean | mean_random_regret_mean |
| --- | --- | --- | --- | --- | --- | --- |
| 16 | 20 | 0.544824 | 0.714143 | 0.455176 | 0.984626 | 0.015546 |
| 32 | 20 | 0.502088 | 0.599844 | 0.497912 | 0.975614 | 0.025420 |
| 64 | 20 | 0.477530 | 0.505490 | 0.522470 | 0.961717 | 0.042143 |
| 128 | 20 | 0.468077 | 0.473593 | 0.531923 | 0.940269 | 0.074179 |

## Excluding Wiki2018 Sensitivity

| horizon | decisions | all_tied_fraction | discriminative_fraction | random_optimal_probability | mean_random_regret |
| --- | --- | --- | --- | --- | --- |
| 16 | 588242 | 0.490167 | 0.509833 | 0.982513 | 0.017698 |
| 32 | 507368 | 0.493831 | 0.506169 | 0.979794 | 0.021147 |
| 64 | 507368 | 0.454132 | 0.545868 | 0.971085 | 0.032376 |
| 128 | 507368 | 0.437288 | 0.562712 | 0.960911 | 0.051130 |

## Paired Horizon Deltas

| metric | delta | mean_delta | median_delta | min_delta | max_delta | n_positive | n_negative | n_effectively_zero |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_tied_fraction | H32-H16 | -0.042736 | -0.011758 | -0.169461 | 0.000000 | 0 | 16 | 4 |
| all_tied_fraction | H64-H32 | -0.024558 | -0.000657 | -0.154879 | 0.000000 | 0 | 15 | 5 |
| all_tied_fraction | H128-H64 | -0.009454 | 0.000000 | -0.098927 | 0.000000 | 0 | 8 | 12 |
| all_tied_fraction | H128-H16 | -0.076748 | -0.025702 | -0.423267 | 0.000000 | 0 | 16 | 4 |
| discriminative_fraction | H32-H16 | 0.042736 | 0.011758 | 0.000000 | 0.169461 | 16 | 0 | 4 |
| discriminative_fraction | H64-H32 | 0.024558 | 0.000657 | 0.000000 | 0.154879 | 15 | 0 | 5 |
| discriminative_fraction | H128-H64 | 0.009454 | 0.000000 | 0.000000 | 0.098927 | 8 | 0 | 12 |
| discriminative_fraction | H128-H16 | 0.076748 | 0.025702 | 0.000000 | 0.423267 | 16 | 0 | 4 |
| random_optimal_probability | H32-H16 | -0.009011 | -0.003627 | -0.075187 | 0.004515 | 1 | 15 | 4 |
| random_optimal_probability | H64-H32 | -0.013897 | -0.005719 | -0.092488 | 0.000000 | 0 | 16 | 4 |
| random_optimal_probability | H128-H64 | -0.021448 | -0.001033 | -0.175102 | 0.000000 | 0 | 14 | 6 |
| random_optimal_probability | H128-H16 | -0.044356 | -0.017249 | -0.342777 | 0.000000 | 0 | 16 | 4 |
| mean_random_regret | H32-H16 | 0.009874 | 0.003661 | -0.005273 | 0.075269 | 15 | 1 | 4 |
| mean_random_regret | H64-H32 | 0.016723 | 0.005858 | 0.000000 | 0.096345 | 16 | 0 | 4 |
| mean_random_regret | H128-H64 | 0.032035 | 0.001345 | 0.000000 | 0.312730 | 14 | 0 | 6 |
| mean_random_regret | H128-H16 | 0.058632 | 0.018645 | 0.000000 | 0.437205 | 16 | 0 | 4 |

## Family-Block Sensitivity

| family | horizon | decisions | all_tied_fraction | discriminative_fraction | random_optimal_probability | mean_random_regret |
| --- | --- | --- | --- | --- | --- | --- |
| alibaba-block | 16 | 192073 | 0.719227 | 0.280773 | 0.994306 | 0.005765 |
| alibaba-block | 32 | 192073 | 0.612293 | 0.387707 | 0.988960 | 0.011411 |
| alibaba-block | 64 | 192073 | 0.527591 | 0.472409 | 0.983170 | 0.018296 |
| alibaba-block | 128 | 192073 | 0.487721 | 0.512279 | 0.978735 | 0.023230 |
| metacdn | 16 | 112320 | 0.009455 | 0.990545 | 0.969113 | 0.031056 |
| metacdn | 32 | 112320 | 0.000650 | 0.999350 | 0.959894 | 0.040596 |
| metacdn | 64 | 112320 | 0.000009 | 0.999991 | 0.950028 | 0.050927 |
| metacdn | 128 | 112320 | 0.000009 | 0.999991 | 0.942009 | 0.059342 |
| metakv | 16 | 151351 | 0.854570 | 0.145430 | 0.997692 | 0.002308 |
| metakv | 32 | 151351 | 0.844580 | 0.155420 | 0.997409 | 0.002593 |
| metakv | 64 | 151351 | 0.834636 | 0.165364 | 0.997006 | 0.003001 |
| metakv | 128 | 151351 | 0.829654 | 0.170346 | 0.996554 | 0.003453 |
| twemcache | 16 | 132498 | 0.149368 | 0.850632 | 0.959436 | 0.041252 |
| twemcache | 32 | 51624 | 0.097784 | 0.902216 | 0.937340 | 0.069446 |
| twemcache | 64 | 51624 | 0.053309 | 0.946691 | 0.895941 | 0.130523 |
| twemcache | 128 | 51624 | 0.050713 | 0.949287 | 0.831225 | 0.276844 |
| wiki2018 | 16 | 199520 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| wiki2018 | 32 | 199520 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| wiki2018 | 64 | 199520 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |
| wiki2018 | 128 | 199520 | 1.000000 | 0.000000 | 1.000000 | 0.000000 |

## Capacity Sensitivity

| capacity | horizon | decisions | all_tied_fraction | discriminative_fraction | random_optimal_probability | mean_random_regret |
| --- | --- | --- | --- | --- | --- | --- |
| 32 | 16 | 204438 | 0.661800 | 0.338200 | 0.976588 | 0.023808 |
| 32 | 32 | 189760 | 0.674315 | 0.325685 | 0.975870 | 0.024541 |
| 32 | 64 | 189760 | 0.667011 | 0.332989 | 0.971094 | 0.029658 |
| 32 | 128 | 189760 | 0.666816 | 0.333184 | 0.970834 | 0.029939 |
| 64 | 16 | 199724 | 0.626770 | 0.373230 | 0.985488 | 0.014670 |
| 64 | 32 | 185366 | 0.626355 | 0.373645 | 0.983234 | 0.018874 |
| 64 | 64 | 185366 | 0.604760 | 0.395240 | 0.972914 | 0.035605 |
| 64 | 128 | 185366 | 0.603989 | 0.396011 | 0.959073 | 0.072468 |
| 128 | 16 | 194226 | 0.603390 | 0.396610 | 0.991349 | 0.008697 |
| 128 | 32 | 167563 | 0.637945 | 0.362055 | 0.990174 | 0.009851 |
| 128 | 64 | 167563 | 0.596635 | 0.403365 | 0.984271 | 0.015832 |
| 128 | 128 | 167563 | 0.576297 | 0.423703 | 0.975065 | 0.025212 |
| 256 | 16 | 189374 | 0.581838 | 0.418162 | 0.995131 | 0.004881 |
| 256 | 32 | 164199 | 0.603627 | 0.396373 | 0.994404 | 0.005621 |
| 256 | 64 | 164199 | 0.555935 | 0.444065 | 0.990688 | 0.009415 |
| 256 | 128 | 164199 | 0.525740 | 0.474260 | 0.984572 | 0.015850 |

## Figure

- Figure status: `PASS`
- PNG: `analysis/pe_h16_h128_comparative_20260916/figures/h16_h128_metric_trajectories.png`
- PDF: `analysis/pe_h16_h128_comparative_20260916/figures/h16_h128_metric_trajectories.pdf`

## Scientific Questions

### Q1. Does increasing H beyond 16 materially reduce tie density?

FACT: Decision-micro all-tied fraction changes from 0.619295 at H16 to 0.596114 at H128.

INTERPRETATION: The reduction is real, but the target remains highly tied.

### Q2. Does it increase discriminative supervision?

FACT: Decision-micro discriminative fraction changes from 0.380705 at H16 to 0.403886 at H128.

INTERPRETATION: Paired cell-level evidence shows longer horizons usually increase discriminative supervision, but the decision-micro aggregate is non-monotone from H16 to H32 because H16 and H32 use different eligibility denominators.

### Q3. Does random-optimal probability meaningfully decline?

FACT: Decision-micro random-optimal probability changes from 0.986942 at H16 to 0.971944 at H128.

INTERPRETATION: The probability declines, meaning a random candidate is less often optimal at longer horizons, but it remains high.

### Q4. Does random regret increase?

FACT: Decision-micro mean random regret changes from 0.013216 at H16 to 0.036698 at H128.

INTERPRETATION: Candidate choice becomes more consequential, especially in the nondegenerate family/capacity cells.

### Q5. Are effects consistent across families and capacities?

FACT: Family and capacity tables above show heterogeneity; wiki2018 remains fully degenerate, while metacdn and twemcache carry much of the discriminative signal.

INTERPRETATION: The paired cell direction is broadly toward more discrimination, but not uniformly across all cells and not monotonically in the pooled decision-micro view.

### Q6. Does H64 to H128 add substantial information relative to earlier steps?

FACT: Decision-micro discriminative fraction changes by 0.012090 from H64 to H128, compared with -0.017403 from H16 to H32 and 0.028494 from H32 to H64. In paired cell-macro deltas, the mean discriminative-fraction changes are 0.042736, 0.024558, and 0.009454.

INTERPRETATION: Changes diminish between H64 and H128. Saying the target "saturates by H64" would be too strong because family/capacity heterogeneity remains.

### Q7. Does strong tie degeneracy remain true at H128?

FACT: At H128 the decision-micro all-tied fraction is 0.596114, and the cell-macro all-tied mean is 0.468077.

INTERPRETATION: Yes. Even at H128, tie degeneracy remains a central property of the supervision target.

## Manuscript-Ready Factual Claims

- H16/H32/H64/H128 comparison uses existing H16 and completed H32/H64/H128 artifacts; no labels were regenerated.
- Paired cell-level summaries show longer horizons generally reduce tie density and increase discriminative decisions, but the pooled decision-micro H16-to-H32 comparison is affected by denominator scope.
- The H64 to H128 increment is smaller than earlier increments in the decision-micro aggregate.
- Wiki2018 remains fully tied across all tested horizons and is included in the primary corpus, with a separate excluding-wiki2018 sensitivity.
- Strong tie degeneracy persists at H128.

## Limitations

- The H16 source is the canonical H16-eligible population. The H32/H64/H128 campaign uses the long-horizon-eligible production population, which changes twemcache denominators.
- These are descriptive family/capacity summaries, not IID inferential tests over 20 independent workloads.
- The comparison does not change public release scope and does not include learned closed-loop or LFU results.
