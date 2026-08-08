# cap256_with_sieve_fifo launch template

**DO NOT RUN UNTIL USER APPROVES AFTER CAP128 ANALYSIS**

This is a staged template only, mirroring the convention used for
`reports/kbs_cap128_with_sieve_fifo_launch_report.md`. It is not an approved
action. cap256 must not be launched until:

1. `kbs_full_policy_comparison_cap128_with_sieve_fifo` has completed
   (`CAP128_WITH_SIEVE_FIFO_EXIT=0` present in
   `logs/kbs_full_policy_comparison/cap128_with_sieve_fifo.log`),
2. cap128 output has passed `scripts/paper/verify_kbs_policy_chunks.py`
   alongside cap32/cap64, and
3. `reports/kbs_post_cap128_decision_template.md` has been filled in and the
   "is cap256 worth running" section recommends GO.

## 1. Exact command (DO NOT RUN YET)

Run inside a `kbs_full_policy_comparison_cap256_with_sieve_fifo` tmux
session, with `.venv_kbs_heavy_r1` activated, on this local/cloud machine
(no `sbatch`/`squeue`/`sacct`/Slurm/Wulver tooling):

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
date | tee logs/kbs_full_policy_comparison/cap256_with_sieve_fifo_started_at.txt
set -o pipefail
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 256 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap256_with_sieve_fifo.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap256_with_sieve_fifo.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap256_with_sieve_fifo.log
echo "CAP256_WITH_SIEVE_FIFO_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_full_policy_comparison/cap256_with_sieve_fifo.log
date | tee logs/kbs_full_policy_comparison/cap256_with_sieve_fifo_finished_at.txt
```

## 2. tmux session name (to be created at launch time, not yet)

`kbs_full_policy_comparison_cap256_with_sieve_fifo`

## 3. Policy list (8, unchanged from cap32/cap64/cap128)

`lru, sieve, fifo_reinsertion, predictive_marker, blind_oracle_lru_combiner, trust_and_doubt, rest_v1, evict_value_v1`

## 4. Output filenames (must not exist before launch — verify with `test ! -e` first)

- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap256_with_sieve_fifo.csv`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap256_with_sieve_fifo.md`

## 5. Expected runtime (estimate only)

| Chunk | Wall-clock |
|-------|------------|
| cap32_with_sieve_fifo | ~5h 16m |
| cap64_with_sieve_fifo | ~10h 04m |
| cap128_with_sieve_fifo | ~15-20h estimated (in progress as of this template's creation) |
| **cap256_with_sieve_fifo (not yet launched)** | **~25-40h estimated**, if the roughly-2x-per-doubling pattern continues; treat as a rough bound, not a commitment, until cap128's actual runtime is known |

No mid-run checkpointing exists in `run_policy_comparison_wulver_v1.py` — the
CSV/MD is written exactly once at the end. A failed or interrupted cap256 run
loses all progress with no partial output to recover.

## 6. Pre-launch safety checklist (re-verify at actual launch time, do not assume these still hold)

- [ ] On `main` branch
- [ ] `analysis/wulver_trace_manifest_full.csv` present
- [ ] `models/evict_value_wulver_v1_best_heavy_r1.pkl` present
- [ ] `tests/test_sieve.py` + `tests/test_fifo_reinsertion.py` pass
- [ ] cap256 CSV/MD do not already exist
- [ ] Disk space sufficient (check `df -h .`)
- [ ] Memory/load healthy (check `free -h`, `uptime`)
- [ ] cap128 fully complete and verified (see section above)
- [ ] No other heavy job (cap128, heavy_r1 retrain, etc.) still running that would contend for CPU
- [ ] Explicit user approval obtained for this specific launch, after reviewing the filled-in `reports/kbs_post_cap128_decision_template.md`

## 7. Rationale this template depends on

Per `reports/kbs_after_cap64_decision_memo.md` and
`reports/kbs_cap128_with_sieve_fifo_launch_report.md`, the staged
cap32 -> cap64 -> cap128 -> (cap256) escalation exists specifically because:

- a single previously observed cap256 validation point already reversed
  direction once, and
- the script has no checkpointing, so committing two large capacities at
  once risks losing both to one failure.

cap256 should only be launched as a deliberate next step after cap128's
trend is analyzed, not bundled automatically.

---
_Template only. cap256 has NOT been launched. No tmux session, process, or
output file for cap256 exists as of this template's creation._
