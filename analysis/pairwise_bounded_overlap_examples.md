# Bounded-overlap residual examples

Exploratory only; not a theorem proof.

## Search setup
- Configs (k, alphabet): [(2,3), (2,4), (3,4)]
- Max sequence length: 8
- Pair-event cases evaluated: 133272

## Candidate residual quantities and required constants
- shared-victim indicator: no finite c (residual with q=0 exists)
- same-phase indicator: no finite c (residual with q=0 exists)
- reinsertion-collision indicator: no finite c (residual with q=0 exists)
- interaction-depth score (1 + shared-victim + same-phase + near-time): c >= 1.00

## Grouped tiny examples
|group|seq|k|alpha|events|times|phases|victims|extra_i|extra_j|extra_ij|residual|interaction_depth|
|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|
|low-overlap low-damage|ABCBAD|2|4|(0,2)|(2,5)|(1,2)|(B,A)|0|0|0|0|1|
|high-overlap bounded-damage|AABCD|2|4|(0,1)|(3,4)|(1,1)|(A,A)|0|0|0|0|4|
|high-overlap super-additive damage|DDDDCBDC|2|4|(0,1)|(5,7)|(1,2)|(D,B)|0|0|1|1|2|
|reuse-gap separation helps|ABCDA|2|4|(0,1)|(2,3)|(1,1)|(A,A)|1|1|1|0|4|

## Interpretation
1. Super-additive interference appears in 39888/133272 pair-event cases on this grid.
2. Pure binary overlap indicators are too weak alone: residual cases exist even when those indicators are zero.
3. The interaction-depth score is currently the most practical predictor among tested simple quantities because it avoids zero-denominator failures and yields a finite small constant.
4. Reuse-gap-separated behavior still appears helpful in selected examples, so it remains the best backup theorem path if overlap-residual formalization breaks.
