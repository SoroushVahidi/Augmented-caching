# Final baseline set decision before cap64 (2026-06-19)

**Update 2026-06-19 (approved stop-and-relaunch pass).** The obsolete
`cap32_with_sieve` job was stopped because it omitted
`fifo_reinsertion`, and the corrected canonical cap32 chunk is now running
in tmux session `kbs_full_policy_comparison_cap32_with_sieve_fifo` with
policy list
`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`
writing to
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md`.
cap64/cap128/cap256 remain not launched.

Original memo written while `cap32_with_sieve` was still running
read-only. This update records the later approved stop-and-relaunch action.

## 1. Should the final canonical sweep include `fifo_reinsertion`?

Yes.

Current justification:

- Reviewer #3 named FIFO-Reinsertion explicitly.
- The repo implementation now passes the stricter readiness bar:
  implemented, source-verified, runner-wired, tested, tiny-smoke checked.
- Excluding it now and adding it later would likely force re-running any
  already-finished canonical chunks.

## 2. Should the old `cap32_with_sieve` run be considered incomplete if FIFO-Reinsertion is included?

Yes, with respect to the final manuscript-citable baseline set.

It would have been a valid SIEVE-inclusive cap32 chunk, but it was not sufficient
for the final baseline set if that set includes `fifo_reinsertion`, because
its current policy list omits FIFO-Reinsertion entirely.

Recommended labeling after it finishes, if FIFO-Reinsertion is adopted:

- `cap32_with_sieve`: aborted / superseded intermediate chunk
- `cap32_with_sieve_fifo`: running corrected final cap32 chunk

## 3. Should we rerun cap32 again later with both SIEVE and FIFO-Reinsertion?

Yes.

If the final canonical policy set is:

`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

then cap32 should eventually be rerun with exactly that set so cap32 is
schema-consistent with the later capacity chunks.

## 4. Should cap64 wait until this is decided?

Yes. This pass resolves the decision in favor of including
`fifo_reinsertion`, so the practical recommendation is:

- do not launch cap64 until the final canonical policy list is fixed
- launch cap64 only once that list includes both `sieve` and
  `fifo_reinsertion`

That avoids the avoidable rerun scenario.

## 5. Recommended final policy list for manuscript-citable table

Recommended final canonical list:

`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

Reasoning:

- `blind_oracle` stays excluded from the main manuscript-citable table
  because it is already outside `TABLE3_POLICIES` and remains
  diagnostic-only.
- `sieve` and `fifo_reinsertion` are the lightweight modern additions
  directly responsive to Reviewer #3.
- the existing predictive/combiner/reference baselines stay in place.

## 6. Practical implication for scheduling

Because the earlier `cap32_with_sieve` run omitted FIFO-Reinsertion:

1. that old chunk should not be treated as the final cap32 chunk
2. the corrected `cap32_with_sieve_fifo` chunk should finish first
3. cap64/cap128/cap256 should still wait for the corrected cap32 result

## 7. Bottom line

- final canonical sweep should include `fifo_reinsertion`
- the old `cap32_with_sieve` run was not enough on its own once
  FIFO-Reinsertion was confirmed in scope
- the corrected `cap32_with_sieve_fifo` chunk is now the cap32 job that
  matters for the final baseline set
- cap64 should wait until the final list is fixed to include both modern
  lightweight baselines
