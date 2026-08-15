# Reuse-Tail Horizon Diagnostic

Protocol: `reuse_tail_horizon_diagnostic_v1`

## Definition

`T` is the number of future request positions until a currently resident
candidate object is next requested. If the object is never requested again
in the remaining trace, `T = infinity`. This is not classical reuse
distance or stack distance; those count distinct intervening objects,
while this diagnostic counts request positions.

Primary quantity: `P(T > H | object is resident at the decision point)`.
The conditioning population is every resident candidate at each full-cache
miss decision under the same LRU reference state used by the eviction-loss
candidate-row construction, restricted to the canonical score window.

## Causal Guardrail

This measures potential unseen future reuse. A reuse after `H` does not
prove that evicting the object caused an avoidable miss; causal excess
misses require a policy counterfactual. This diagnostic addresses only
whether potentially relevant reuse lies outside the finite supervision
horizon.

## Scope

- Families: `brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018`
- Capacities: `32, 64, 128`
- Horizons reported from the same T samples: `1, 2, 4, 8, 16`

## Overall H=4 Result

- Resident-candidate observations: `41615776`
- Decision points: `565126`
- `P(T > 4 | resident)`, including never-reused: `0.9938544459677984`
- `P(T > 4 | resident, eventually reused)`: `0.9793302186526528`
- `P(T <= 4 | resident)`: `0.006145554032201635`

## Capacity Trend at H=4

Descriptive only. Do not infer an H/C law from this diagnostic.

| Family | C=32 | C=64 | C=128 |
| --- | ---: | ---: | ---: |
| brightkite | 0.971990676556218 | 0.9846112948960303 | 0.9918541280148423 |
| citibike | 0.9699282509185086 | 0.9833057482971544 | 0.9903570549342336 |
| cloudphysics | 0.9982672341666244 | 0.9984077610596708 | 0.9988073969584433 |
| metacdn | 0.968082098073857 | 0.9830827171699282 | 0.9910408220207726 |
| metakv | 0.9962942441689293 | 0.9981194970869337 | 0.9990511620387046 |
| twemcache | 0.9791319698887483 | 0.9864935578661844 | 0.9917572527395335 |
| wiki2018 | 1.0 | 1.0 | 1.0 |

Compare this trend qualitatively with the already observed target-
degeneracy trend: zero-margin pair fraction and mean optimal-set
fraction increased with capacity. Any co-movement here is exploratory
correlation only.
