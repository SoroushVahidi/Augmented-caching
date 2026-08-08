# cap32_with_sieve aborted / superseded report (2026-06-19)

This report documents the approved stop of the obsolete
`cap32_with_sieve` run and its replacement by the corrected
`cap32_with_sieve_fifo` canonical cap32 chunk on the current local/cloud
machine. No Slurm or cluster commands were used.

## 1. Why `cap32_with_sieve` was stopped

The in-flight `cap32_with_sieve` run omitted `fifo_reinsertion`, while the
final reviewer-responsive baseline set had already been fixed to:

`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

Because Reviewer #3 explicitly requested both SIEVE and FIFO-Reinsertion,
and the FIFO baseline had already been confirmed implemented, wired,
tested, source-verified, and smoke-validated, continuing the old run would
have produced a cap32 chunk that was not manuscript-citable for the final
baseline set.

## 2. What was stopped

At the time of the stop request, the obsolete tmux session
`kbs_full_policy_comparison_cap32_with_sieve` was still present and the
runner process was:

- `python scripts/run_policy_comparison_wulver_v1.py ... --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.csv --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.md`

The stop sequence was:

1. record status to
   `logs/kbs_full_policy_comparison/cap32_with_sieve_aborted_status.txt`
2. send `C-c` to the obsolete tmux session
3. verify the policy-comparison runner was gone
4. kill the now-idle obsolete tmux session shell to avoid confusion

No unrelated Python processes were terminated.

## 3. Did any partial output exist?

No manuscript-output files existed for the obsolete run at stop time:

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.md`

Both were absent when checked. The associated log
`logs/kbs_full_policy_comparison/cap32_with_sieve.log` existed but was
still empty.

No partial output was deleted.

## 4. Was the older cap32 chunk preserved?

Yes.

The earlier completed cap32 chunk without SIEVE remains untouched and
preserved:

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.md`

Those files remain the historical no-SIEVE / no-FIFO cap32 result and were
not renamed, deleted, or overwritten in this pass.

## 5. Corrected final baseline set

The corrected manuscript-citable cap32 policy list is:

`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

`blind_oracle` remains excluded from the main canonical table because it is
diagnostic-only and already outside the manuscript's main policy table
logic.

## 6. cap64 / cap128 / cap256 status

They remain not launched.

This pass only stopped the obsolete `cap32_with_sieve` run and launched the
corrected `cap32_with_sieve_fifo` cap32 chunk. No cap64/cap128/cap256 work
was started.
