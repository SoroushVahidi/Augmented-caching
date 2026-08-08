# cap32 rerun-with-SIEVE plan (2026-06-19)

**Update 2026-06-19 (approved launch supersession).** This plan is now
historical. The originally drafted `cap32_with_sieve` rerun was launched,
then approved to stop because it omitted `fifo_reinsertion`. The corrected
canonical cap32 run is now `cap32_with_sieve_fifo`, launched in tmux
session `kbs_full_policy_comparison_cap32_with_sieve_fifo` with outputs
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md`.
See `reports/kbs_cap32_with_sieve_aborted_superseded_report.md` and
`reports/kbs_cap32_with_sieve_fifo_launch_report.md`.

Zero-compute planning report. No rerun was launched in producing this
report. We are on the current local/cloud machine (not Wulver, no Slurm) —
any future launch uses `tmux`, not `sbatch`/`squeue`/`sacct`. cap64/cap128/
cap256 are **not launched** and remain out of scope for this report.

## 1. Output naming strategy: Option B chosen (write-new, don't rename old)

**Recommendation: Option B.** Keep the existing no-SIEVE cap32 files
(`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv/.md`)
completely untouched, and write the new SIEVE-inclusive run to new,
distinctly-named files:

```
analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.csv
analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.md
```

**Why Option B over Option A:**

- Option A requires **renaming** the current canonical-named cap32 files
  (to `..._no_sieve_legacy.*`) before the new run can claim the canonical
  name. That is an extra filesystem operation on existing artifacts, and
  the instruction in force for this task is explicit: *"Do not push, merge,
  delete, or overwrite important artifacts."* A rename is not a delete, but
  it is also not nothing — it changes the path that every other file in
  this repo currently points to.
- This is not hypothetical: `reports/kbs_cap32_policy_comparison_report.md`,
  `reports/kbs_revision_gap_tracker.md`, `reports/kbs_revision_evidence_ledger.md`,
  `reports/kbs_revision_evidence/evidence_manifest.json`, and
  `reports/kbs_full_policy_comparison_execution_plan.md` all currently
  reference `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv/.md`
  by their exact current name. Renaming those two files under Option A would
  require auditing and fixing every one of those cross-references in the
  same pass to avoid leaving stale/broken pointers — extra work and extra
  risk for no benefit over Option B.
- Option B requires **zero changes to any existing file's name or
  content**. The only new filesystem actions are two brand-new file writes.
  This is the minimum-blast-radius choice, consistent with the same
  reasoning already applied to the SIEVE smoke test in
  `reports/kbs_sieve_implementation_report.md` §5 ("no canonical output
  file was overwritten").
- The "merge only `*_with_sieve` chunks later" step Option B defers is a
  deliberate, later, explicit decision point — exactly the kind of
  judgment call that should wait for direct approval rather than being
  pre-committed to by a naming choice made now.

**Consequence**: until a human explicitly decides to retire the no-SIEVE
cap32 files, **two cap32 datasets will coexist on disk**:
`..._cap32.csv/.md` (no SIEVE, 7 traces × 7 policies incl. `blind_oracle`,
already complete) and, once approved and run, `..._cap32_with_sieve.csv/.md`
(SIEVE included, `blind_oracle` excluded — see §4 of the final summary).
This is intentional and clearly labeled in the planning files updated in
Task 5 below, not an oversight.

## 2. Pre-run inventory (read-only, see commands/output in this session)

- `git status --short` — only previously-known untracked/modified files;
  nothing surprising, no uncommitted destructive state.
- Existing cap32 files confirmed present and untouched:
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv` (5.1K),
  `...cap32.md` (1.8K), both dated `Jun 19 00:31` (from the original cap32
  run, predating SIEVE).
- SIEVE code confirmed present: `src/lafc/policies/sieve.py` (6.4K),
  `tests/test_sieve.py` (5.0K).
- `grep -Rni "sieve" src/lafc scripts tests reports` → 264 matches across 19
  files (the policy implementation, both registries, the test file, the
  smoke-test outputs' filenames where referenced, and the SIEVE-related
  reports from the prior phase). Confirms SIEVE is consistently wired and
  documented, no stray/duplicate implementation under a different name.
- All four required inputs confirmed present:
  `analysis/wulver_trace_manifest_full.csv` (624B),
  `models/evict_value_wulver_v1_best_heavy_r1.pkl` (13M),
  `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` (104K),
  `data/derived/evict_value_v1_wulver_heavy_r1/split_summary.csv` (8.1K).

## 3. SIEVE integration re-verification

Ran exactly the two requested lightweight commands (no full suite):

```
pytest tests/test_sieve.py -v                          # 8 passed
pytest tests/ -k "sieve or policy_comparison" -v        # 8 passed, 283 deselected
```

Both green. Note: the second command's `policy_comparison` half of the `-k`
filter matched 0 additional tests — there is no test function or file in
this repo whose *name* contains `policy_comparison` (the substring only
appears inside `tests/test_sieve.py`'s own source text, e.g. in a comment
referencing `scripts/run_policy_comparison_wulver_v1.py`, which `-k` does
not search). This is not a gap in coverage; the canonical pipeline script
is exercised indirectly via `tests/test_sieve.py::test_runner_policies_dict_accepts_sieve`
and via the Step 6 smoke test already completed in
`reports/kbs_sieve_implementation_report.md` §5, not via a dedicated
`test_policy_comparison*.py` file (none exists in this repo).

## 4. `blind_oracle` inclusion decision

**Exclude `blind_oracle` from the SIEVE-inclusive cap32 rerun's
manuscript-canonical set** — already the right call independent of SIEVE:

- `scripts/paper/build_kbs_main_manuscript_artifacts.py`'s own
  `TABLE3_POLICIES` tuple (the set the manuscript's Table 3 is actually
  built from) is `(lru, predictive_marker, trust_and_doubt,
  blind_oracle_lru_combiner, rest_v1, evict_value_v1)` — **`blind_oracle`
  was never in the manuscript-canonical set**, even though it was included
  as an extra diagnostic column in the original (no-SIEVE) cap32 run.
- `reports/kbs_validation_sanity_audit.md` (Q3, prior phase) already
  root-caused why: as currently wired, `blind_oracle` consumes
  `request.predicted_next`, which is never populated by the canonical
  pipeline (only `attach_predicted_caches`-style metadata is, which feeds
  `trust_and_doubt`/`evict_value_v1` instead) — so `blind_oracle` always
  evicts on a page-id tiebreak with zero real predictive information. Its
  number (−100% vs. LRU in the original cap32 run) is explainable, not a
  bug, but explicitly **"should not be read as... in any manuscript
  narrative... beyond 'did the harness run end-to-end.'"**
- The user's proposed policy list for the SIEVE-inclusive rerun
  (`lru,sieve,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`)
  already excludes `blind_oracle` and already matches `TABLE3_POLICIES` plus
  `sieve` — this plan endorses that list as-is, no change recommended.
- This does not delete `blind_oracle`'s existing diagnostic number — that
  remains intact in the untouched, no-SIEVE `..._cap32.csv/.md` (§1) for
  anyone who wants to reference it as a "ran end-to-end" diagnostic check.

## 5. Prepared (not launched) tmux command

Mirrors the exact wrapper pattern already established in
`reports/kbs_full_policy_comparison_execution_plan.md` §7 (session name,
venv activation, log redirection, exit-code capture), changed only in
session name, policy list, and output paths:

```bash
tmux new -s kbs_full_policy_comparison_cap32_with_sieve
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
set -o pipefail
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap32_with_sieve.log
echo "CAP32_WITH_SIEVE_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_full_policy_comparison/cap32_with_sieve.log
```

**Not launched.** Requires explicit approval before running, per
instruction.

## 6. Estimated runtime

**Approximately 5 hours, in the same range as the original cap32 run.**

- Original no-SIEVE cap32 (7 traces × 7 policies × capacity 32, identical
  `--max-requests-per-trace 50000`): measured wall-clock **5:13:02**
  (`reports/kbs_cap32_policy_comparison_report.md`).
- The new policy list is also 7 policies (swaps out `blind_oracle` for
  `sieve`, same count) — both are O(1)-per-request policies (`blind_oracle`
  per `BlindOraclePolicy`'s simple farthest-comparator logic;
  `sieve` per `kbs_sieve_implementation_report.md` §6's amortized-O(1)
  argument), so no asymmetric cost is expected from this swap.
- Same trace set, same capacity, same request cap → expect a similar total,
  most plausibly in the **4.5–6 hour** band. This is an extrapolation from
  one prior data point, not a re-measurement; the only way to get an exact
  number is to actually run it.

## 7. Should cap64 wait?

**Yes, unchanged from `reports/kbs_before_cap64_baseline_decision_memo.md`.**
This plan does not alter that recommendation: cap64/cap128/cap256 remain
**not launched**, and this report does not authorize launching them. The
SIEVE-inclusive cap32 rerun (once approved) is a prerequisite check, not a
gate that's already been cleared — it should complete and be sanity-checked
before cap64 is considered, exactly as the original (no-SIEVE) cap32 was
used as a gate before this point.
