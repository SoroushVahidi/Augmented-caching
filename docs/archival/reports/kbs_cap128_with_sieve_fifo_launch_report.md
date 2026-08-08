# cap128_with_sieve_fifo launch report (2026-06-20)

**Approved action.** Following `reports/kbs_after_cap64_decision_memo.md`
(Option B), the cap128 canonical chunk was launched. cap256 was **not**
launched, per the same memo's decision to reassess before committing to
the final capacity point.

---

## 1. Exact command

Run inside the `kbs_full_policy_comparison_cap128_with_sieve_fifo` tmux
session, with `.venv_kbs_heavy_r1` activated:

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
date | tee logs/kbs_full_policy_comparison/cap128_with_sieve_fifo_started_at.txt
set -o pipefail
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 128 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap128_with_sieve_fifo.log
echo "CAP128_WITH_SIEVE_FIFO_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_full_policy_comparison/cap128_with_sieve_fifo.log
date | tee logs/kbs_full_policy_comparison/cap128_with_sieve_fifo_finished_at.txt
```

Run on the local/cloud machine via `tmux`. No `sbatch`/`squeue`/`sacct` or
any Slurm/Wulver tooling was used.

## 2. tmux session name

`kbs_full_policy_comparison_cap128_with_sieve_fifo` (created Sat Jun 20
2026 09:31:02, confirmed live via `tmux ls`).

## 3. Start timestamp

`Sat Jun 20 09:31:03 AM EDT 2026` (written to
`logs/kbs_full_policy_comparison/cap128_with_sieve_fifo_started_at.txt`).
Driver process confirmed running as PID 276106
(`python scripts/run_policy_comparison_wulver_v1.py ...`), piped through a
`tee` process (PID 276107) into the log file.

## 4. Expected runtime

Observed history for this same script/manifest at smaller capacities:

| Chunk | Wall-clock |
|-------|------------|
| cap32_with_sieve_fifo | ~5h 16m |
| cap64_with_sieve_fifo | ~10h 04m |
| **cap128_with_sieve_fifo (this run)** | **~15–20h estimated**, if the ~2x-per-doubling pattern observed from cap32→cap64 continues |

No checkpointing exists in this script — it writes the CSV/MD exactly once
at the end of the full run, so there is no partial-progress signal beyond
the live log/process state until it exits.

## 5. Policy list (8, unchanged from cap32/cap64)

`lru, sieve, fifo_reinsertion, predictive_marker, blind_oracle_lru_combiner, trust_and_doubt, rest_v1, evict_value_v1`

## 6. Output filenames

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.md`

Confirmed absent before launch (`test ! -e` on both paths returned true).

## 7. Safety checks passed

| Check | Result |
|-------|--------|
| On `main` branch | ✓ (`git branch --show-current` → `main`) |
| `analysis/wulver_trace_manifest_full.csv` present | ✓ (624 bytes) |
| `models/evict_value_wulver_v1_best_heavy_r1.pkl` present | ✓ (13M) |
| `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` present | ✓ (104K) |
| `tests/test_sieve.py` + `tests/test_fifo_reinsertion.py` | ✓ **16/16 passed**, exit code 0 |
| cap128 CSV does not already exist | ✓ confirmed via `test !-e` |
| cap128 MD does not already exist | ✓ confirmed via `test !-e` |
| Disk space | ✓ 499G available of 700G (25% used) |
| Memory | ✓ 59Gi available of 62Gi total |
| Load average | ✓ 0.02 / 0.15 / 0.13 (idle) |
| Launched via tmux, not sbatch/Slurm | ✓ confirmed — this machine is local/cloud, not Wulver |

Pre-existing uncommitted changes/untracked files from prior sessions were
observed in `git status --short` but were **not** touched, modified, or
staged as part of this launch.

## 8. Rationale for running cap128 but not cap256

Per `reports/kbs_after_cap64_decision_memo.md` (Option B, the approved
decision):

- cap32→cap64 showed an improving, family-dependent gap-narrowing trend
  for evict_value_v1 (vs LRU, SIEVE, FIFO-Reinsertion) that two data
  points cannot yet distinguish from a 1-2-trace-driven effect. cap128
  is the lowest-risk way to get a third point and tell whether the trend
  is a stable multi-family pattern or a near-term artifact.
- A previously observed single-trace cap256 validation point already
  reversed direction once — a concrete reason to avoid committing to
  cap256 before seeing what cap128 shows.
- `run_policy_comparison_wulver_v1.py` has no mid-run checkpointing; a
  combined cap128+cap256 run risks losing both to a single failure, with
  no partial output to recover, right as the 2026-07-08 deadline window
  narrows. Running cap128 alone isolates that risk to one chunk and
  preserves a deliberate decision point before cap256.
- This continues the same staged escalation pattern already used for
  cap32 → cap64 (one chunk, evaluate, decide the next), rather than
  committing to the full remaining sweep on optimism about the trend
  continuing.

cap256 remains **not launched**. The next decision point is after cap128
completes and is analyzed, mirroring `reports/kbs_after_cap64_decision_memo.md`.
