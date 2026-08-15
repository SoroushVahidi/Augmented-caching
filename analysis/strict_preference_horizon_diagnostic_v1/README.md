# Strict-Preference Horizon Diagnostic v1

This diagnostic measures finite-horizon eviction-loss target resolution and
preference stability. It does not train a model and does not modify either
active policy campaign.

Protocol: seven canonical trace families, capacities 32/64/128, history
`[0,10000)`, scored window `[10000,50000)`, and target horizons H=4, 8, 16,
and 32. The trajectory follows the exact H=4 eviction-loss oracle with the
existing lexicographic tie break. Longer horizons are diagnostic target
comparisons evaluated on the same decision states.

Definitions:

- **Unique winner:** the H target minimum is attained by exactly one candidate.
- **Multiple optimum:** the minimum is attained by two or more candidates.
- **All tied:** every candidate has the same target value.
- **Strict preference:** a unique minimum or a positive ordinary margin; no
  total ordering is imposed on tied candidates.
- **Pairwise reversal:** not computed. Comparisons use optimal sets and unique
  winners, avoiding arbitrary ordering of ties.

Partial units are not citable. Aggregate files are generated only after all
21 units pass validation, including trace hashes, protocol checks, finite
metrics, and the Brightkite/capacity-64 regression against the validated
degeneracy diagnostic.
