# Step-4 targeted attack examples

This note is exploratory only and does not claim a proof.

## Search setup
- Configs (k, alphabet): [(2, 3), (2, 4), (3, 4)]
- Max sequence length: 9
- Evaluated perturbation cases: 1402650
- Strongest-claim violations found: 0

Strongest tested Step-4 form: `extra_misses <= local_inversions` for a single perturbed decision.

## Restricted-form violation rates
- Reuse-gap-separated subset (gap >= 3): 0/514980
- Phase-local subset (all extra misses stay in inversion phase): 0/874326
- One-inversion-per-decision subset: 0/1141098

## Family table
|family|seq|k|alphabet|perturb|event_t|phase|inversions|gap|belady|test|extra|extra_same_phase|
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
|same inversion, near-zero damage|AABC|2|3|top_swap|3|1|1|0|3|3|0|0|
|same inversion, high damage|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|
|large reuse-gap separation|DDDDDDCBD|2|4|top_swap|7|1|1|999999999|3|4|1|1|
|shortlist/full ranking difference|AABCD|3|4|lower_swap|4|1|1|0|4|4|0|0|
|one inversion stays local|ABCA|2|3|top_swap|2|1|1|999999999|3|4|1|1|
|one inversion triggers cascade|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|

## Family interpretation
1. Same inversion count can produce different damage in this grid (observed range for top-swap one-inversion: 0 to 1 extra misses).
2. No strongest-form violation was found on this grid; this is evidence of plausibility, not a proof.
3. Lower-rank inversions can be visible to full-ranking metrics while being invisible to shortlist eviction behavior.
4. No one-inversion non-local cascade was found at this search depth.
5. Restricted forms remain candidate theorem targets; broadened search may still overturn them.
