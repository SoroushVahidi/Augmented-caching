# cap32_with_sieve_fifo launch report (2026-06-19)

This report documents the launch of the corrected canonical cap32 chunk
that includes both SIEVE and FIFO-Reinsertion. It was launched in `tmux` on
the current local/cloud machine after stopping the obsolete
`cap32_with_sieve` run.

## 1. Exact corrected command

```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log
```

The tmux session also appends:

```bash
echo "CAP32_WITH_SIEVE_FIFO_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log
date | tee logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_finished_at.txt
```

## 2. tmux session name

- `kbs_full_policy_comparison_cap32_with_sieve_fifo`

## 3. Start time

- `Fri Jun 19 02:42:27 PM EDT 2026`

Recorded in:

- `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_started_at.txt`

## 4. Expected runtime

Expected wall-clock runtime is approximately the same as the earlier cap32
chunk:

- prior no-SIEVE cap32 runtime: `5:13:02`
- practical expectation for this corrected cap32 chunk: roughly `4.5-6 hours`

Reason: same trace set, same capacity, same request cap, same number of
policies, with `fifo_reinsertion` replacing the earlier diagnostic-only
`blind_oracle`.

## 5. Output filenames

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.md`
- `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log`

These filenames were confirmed absent before launch, so nothing was
overwritten.

## 6. Policy list

`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

## 7. Safety checks passed before launch

- corrected output paths confirmed absent
- trace manifest present
- heavy_r1 model present
- heavy_r1 dataset manifest present
- `src/lafc/policies/sieve.py` present
- `src/lafc/policies/fifo_reinsertion.py` present
- `tests/test_sieve.py` and `tests/test_fifo_reinsertion.py` present
- targeted policy tests passed: `16 passed in 0.11s`
- available disk/memory checked
- obsolete `cap32_with_sieve` tmux session stopped first
- cap64/cap128/cap256 not launched

## 8. Status at launch verification

Immediate post-launch verification showed:

- tmux session exists: `kbs_full_policy_comparison_cap32_with_sieve_fifo`
- runner PID: `198488`
- tee PID: `198489`
- launch log exists:
  `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log`
- start marker exists:
  `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_started_at.txt`

At the first verification point, the log was still size 0 and the output
CSV/MD had not yet been written, which is normal early in the run.
