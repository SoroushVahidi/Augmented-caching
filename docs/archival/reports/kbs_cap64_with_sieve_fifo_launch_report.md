# cap64_with_sieve_fifo launch report (2026-06-19)

This report documents the launch of the cap64 chunk using the same final
8-policy baseline set as the completed `cap32_with_sieve_fifo` run. This is
Option C: launch cap64 only, on the local/cloud machine (not Wulver, not
Slurm), in `tmux` so SSH/network disconnects cannot interrupt the run.

## 1. Exact command

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
date | tee logs/kbs_full_policy_comparison/cap64_with_sieve_fifo_started_at.txt
set -o pipefail
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 64 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log
echo "CAP64_WITH_SIEVE_FIFO_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log
date | tee logs/kbs_full_policy_comparison/cap64_with_sieve_fifo_finished_at.txt
```

This is the same command as the `cap32_with_sieve_fifo` run with only the
capacity (`32` -> `64`) and output filenames changed.

## 2. tmux session name

- `kbs_full_policy_comparison_cap64_with_sieve_fifo`

## 3. Start timestamp

- `Fri Jun 19 10:29:23 PM EDT 2026`

Recorded in:

- `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo_started_at.txt`

## 4. Expected runtime

- `cap32_with_sieve_fifo` actual runtime: start `02:42:27 PM EDT`, finish
  `07:58:48 PM EDT` -> approximately `5h16m`.
- Practical expectation for `cap64_with_sieve_fifo`: roughly `5-7 hours`.

Reason: same trace set (7 traces), same request cap
(`--max-requests-per-trace 50000`), same 8 policies, only the cache capacity
changes from 32 to 64. Slightly larger capacity can modestly increase
per-request bookkeeping cost for some policies (e.g. ranking/scoring over a
larger resident set), so the estimate is biased slightly upward from the
cap32 baseline rather than assumed identical.

## 5. Policy list

`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`

This is the identical final 8-policy baseline set used in the completed
`cap32_with_sieve_fifo` run (includes reviewer-requested SIEVE and
FIFO-Reinsertion).

## 6. Output filenames

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.md`
- `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log`
- `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo_started_at.txt`
- `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo_finished_at.txt` (written on completion)

These filenames were confirmed absent before launch (Step 2 of the launch
procedure), so nothing was overwritten.

## 7. Safety checks passed before launch

- current branch confirmed: `main`
- `git status --short` reviewed; no destructive state found
- `analysis/wulver_trace_manifest_full.csv` present
- `models/evict_value_wulver_v1_best_heavy_r1.pkl` present
- `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` present
- `src/lafc/policies/sieve.py` present
- `src/lafc/policies/fifo_reinsertion.py` present
- `pytest tests/test_sieve.py tests/test_fifo_reinsertion.py -v` -> `16 passed`
- disk space checked: `499G` available on `/`
- memory checked: `59Gi` available
- load average checked: `0.18, 0.07, 0.02` (idle)
- corrected cap64 output paths (`.csv`, `.md`) confirmed absent before launch
- launched in `tmux` (session persists across SSH/network disconnects, no
  Slurm/Wulver involved)
- cap128 and cap256 not launched (and not planned until cap64 results are
  reviewed)

## 8. Status at launch verification

Immediate post-launch verification showed:

- tmux session exists: `kbs_full_policy_comparison_cap64_with_sieve_fifo`
- runner PID: `261643`
- tee PID: `261644`
- process confirmed alive ~30s after launch via `ps -p 261643`
- launch log exists: `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log`
  (size 0 at verification time, consistent with the same early-stage
  behavior observed in the `cap32_with_sieve_fifo` run)
- start marker exists:
  `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo_started_at.txt`

## 9. Rationale for cap64 only (not cap128/cap256)

Reviewer #2 and Reviewer #3 requested evaluation across multiple cache
capacities. Results for `evict_value_v1` were scientifically negative at
cap32. Rather than committing several more multi-hour runs (cap128, cap256)
before knowing whether the negative result is capacity-dependent, the
decision is to run cap64 first as the next data point and only decide on
cap128/cap256 after reviewing cap64 results. This limits compute spend to
what is needed to answer the immediate question ("does the result change
between cap32 and cap64?") before committing to the larger and slower
cap128/cap256 chunks.
