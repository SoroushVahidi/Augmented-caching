# Reviewer-closure audit after compressed PDF review — 2026-08-14

- Branch: `kbs/second-revision-science`
- HEAD before this pass: `d7259df93e69699876caf42a7785ed69b6c83653`
- No experiment was run.

## Table 4 replacement

Leaky historical family-gap percentages were replaced with matched-protocol
relative miss gaps vs LRU, computed from:

- `analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv`
  (`primary_controlled_window`, 21 rows)
- `analysis/reviewer_fairness/policy_comparison_lru.csv`
  (`primary_controlled_window`, 21 rows)

Formula: `(EV_misses - LRU_misses) / LRU_misses`, 40{,}000 scored requests.

## Wording fixes

- 97.53% set-aware agreement: no longer implied as strong discrimination.
- LRU-within-minima: not claimed as H=4 improving LRU.
- Degeneracy 0.649/0.991 vs 76.4%/0.9949: two diagnostics, not interchangeable.
- Decision alignment framed as structural, not empirical superiority.
- Removed H1/H3 jargon from manuscript prose.

## Validation

See parent task compile after this commit.
