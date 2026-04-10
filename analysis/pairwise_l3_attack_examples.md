# L3 interference/additivity attack examples

This note is exploratory and not a proof.

## Search setup
- Perturbation type: top-swap at chosen eviction decisions.
- Configs (k, alphabet): [(2,3), (2,4), (3,4)]
- Max sequence length: 8
- Pair-event cases evaluated: 133272
- Strong-L3 failure cases found: 39888

Strong L3 attack condition tested: single events individually bounded (+0/+1), but joint run is super-additive.

## Restricted-form super-additivity rates
- Bounded-overlap subset (same perturbed victim page): 18530/66198
- Separated-return-times subset (event time gap >= 3): 21432/64176
- One-shot local subset (both singles <= +1): 39888/133272

## Targeted example families
|family|seq|k|alpha|events(i,j)|times|phases|victims|extra_i|extra_j|extra_ij|sum_singles|joint_same_phase_extra|super_additive?|
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|
|two events add independently|ABCAD|2|4|(0,1)|(2,4)|(1,2)|(A,A)|1|0|1|1|1|N|
|overlap but bounded|AABCD|2|4|(0,1)|(3,4)|(1,1)|(A,A)|0|0|0|0|0|N|
|merge into cascade|DDDDCBDC|2|4|(0,1)|(5,7)|(1,2)|(D,B)|0|0|1|0|1|Y|
|separated returns prevent interference|ABCAAD|2|4|(0,1)|(2,5)|(1,2)|(A,A)|1|0|1|1|1|N|

## Interpretation
1. Strong L3 fails on this grid; max super-additivity gap observed is 1.
2. Interaction is sensitive to event footprint overlap and timing, but this grid does not show a clean monotone protection from simple time separation alone.
3. Phase boundary proximity remains a risk variable: near-boundary events should be audited separately in theorem wording.
4. Among tested restrictions, bounded-overlap with an explicit residual currently looks more plausible than strict separation-only additivity.
