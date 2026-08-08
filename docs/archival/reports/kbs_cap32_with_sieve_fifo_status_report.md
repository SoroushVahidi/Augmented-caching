# cap32_with_sieve_fifo status report (2026-06-19, audit at 15:58 EDT)

This is a status-only audit. No process was started, stopped, or interrupted.
Environment: current local/cloud machine (not Wulver, not Slurm).

## 1. Case determined

**Case A — running.** The corrected `cap32_with_sieve_fifo` job is actively
executing. No action other than reporting was taken.

## 2. tmux session

- `kbs_full_policy_comparison_cap32_with_sieve_fifo` (created Fri Jun 19
  14:42:26 2026), confirmed via `tmux ls`.
- The string `kbs_full_policy_comparison_cap32_with_sieve` (without `_fifo`)
  does **not** exist as a separate session in `tmux ls`. `tmux has-session`
  and `tmux capture-pane` against that shorter name resolve via tmux's
  prefix-matching to this same `_fifo` session — there is no second,
  obsolete session running. This is consistent with
  `reports/kbs_cap32_with_sieve_fifo_launch_report.md`, which records that
  the obsolete `cap32_with_sieve` session was stopped *before* this `_fifo`
  job was launched.

## 3. PIDs

- `198488` — `python scripts/run_policy_comparison_wulver_v1.py ...
  cap32_with_sieve_fifo` runner. State `Rl+`, **100% CPU**, elapsed
  `01:16:xx` (and climbing) at audit time. Actively computing, not hung.
- `198489` — `tee logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log`,
  idle/sleeping waiting on pipe input (expected for `tee`).

## 4. Start time

- `Fri Jun 19 02:42:27 PM EDT 2026`
  (from `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_started_at.txt`
  and confirmed by `ps -o lstart`).
- Audit time: `Fri Jun 19 03:58:16 PM EDT 2026` → elapsed ≈ **1h16m**.
- Prior (no-SIEVE) cap32 chunk took `5:13:02` wall-clock for a similar
  (7-policy) workload; this corrected chunk has 8 policies
  (`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,
  trust_and_doubt,rest_v1,evict_value_v1`) so an expected total runtime of
  roughly **4.5–6 hours** (per the launch report) is still consistent with
  1h16m elapsed — the job is on track, not overdue.

## 5. Log size

- `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log`: **0 bytes**.
- This is expected, not a hang indicator: the runner script
  (`scripts/run_policy_comparison_wulver_v1.py`) only calls `print()` once,
  at the very end (`Wrote {out_csv} and {out_md}`), so stdout/the log stays
  empty for the entire run until completion. The prior successful cap32 run
  showed the identical pattern (single "Wrote ..." line appearing only at
  the end). The runner process's open file descriptors confirm `stdout`/
  `stderr` are still connected to the live `tee` pipe (`pipe:[735271]`).

## 6. Output CSV/MD

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv`
  — does not exist yet.
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.md`
  — does not exist yet.
- Both are written together only at the very end of the run (`Wrote ...`
  log line happens at the same point the files are written).

## 7. Exit marker

- `CAP32_WITH_SIEVE_FIFO_EXIT=` not present anywhere under
  `logs/kbs_full_policy_comparison/`. The run has not finished.

## 8. Recommendation

**Wait.** The job is healthy: 100% CPU, elapsed time well within the
expected 4.5–6 hour envelope, no error indicators, no separate/obsolete
session competing for resources. Do not interrupt, do not relaunch, and do
not start cap64/cap128/cap256 while this is in flight. Re-audit later (e.g.
once elapsed time approaches 5 hours, or via `tmux capture-pane`/`tail` on
the log) to check for the `CAP32_WITH_SIEVE_FIFO_EXIT=` marker and the
output CSV/MD before producing the completion report
(`reports/kbs_cap32_with_sieve_fifo_policy_comparison_report.md`).
