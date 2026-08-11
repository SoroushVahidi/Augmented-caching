# KBS Second-Revision Repository State

Date: 2026-08-09  
Canonical checkout: this `kbs/second-revision-science` repository clone  
Canonical branch: `kbs/second-revision-science`  
Expected baseline HEAD for the current local documentation pass: `63e63acb09c39449bda0a28c8ab2d24f63b2547e`

For the consolidated mechanistic-hypothesis matrix see
[`reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](reviewer/KBS_SECOND_REVISION_HYPOTHESIS_MAP.md);
for the per-reviewer-concern status matrix see
[`reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](reviewer/KBS_SECOND_REVISION_REVIEWER_COVERAGE.md).
This file stays focused on repository/branch/run-progress status.

**Snapshot convention:** this file (and the sync-status docs it links to)
records a *current snapshot* of an actively evolving branch, not a final
scientific result. Read every number in it as one of three kinds:

- **CURRENT SNAPSHOT** -- a timestamped read of in-progress state (e.g.
  "`4/7` folds complete as of `2026-08-10 21:59`"). Expected to go stale;
  trust the timestamp, not the number, once time has passed.
- **LAST FINALIZED SCIENTIFIC RESULT** -- a phase that reached a clean stop
  and passed an integrity audit (e.g. the `25%` learning-curve fraction,
  `7/7` folds, `42/42` rows). These do not change until a deliberate new
  audit supersedes them.
- **RUNNING EXPERIMENT** -- explicitly still executing; never cite its
  partial numbers as a scientific result, only as progress context.

The durable theory documents (hypothesis map, reviewer coverage map,
experiment registry) intentionally avoid embedding live fold counts for this
reason -- they reference *phase-level* status (`RUNNING`, `FINAL_VALIDATED`,
...) and point back here for the live number.

## Purpose

This note records the structural intent of the local KBS second-revision branch
before final manuscript-facing cleanup:

- `kbs/second-revision-science` is the intended source of truth for the
  reviewer-science code and frozen protocols.
- reviewer evidence under `analysis/`, `models/`, and large derived datasets is
  preserved locally but is not yet treated as fully curated, tracked release
  material.
- historical worktrees remain useful as provenance and comparison points, but
  they are not the intended long-term entrypoint for outside researchers.

## Current source-of-truth boundaries

### Tracked source/configuration

- experiment runners and gates under `scripts/experiments/`
- dataset/build/train drivers under root `scripts/`
- reviewer protocols and frozen configs under `docs/` and `configs/`
- code for external baselines and reproducibility helpers under `src/lafc/`
- fast regression tests under `tests/`

### Generated reviewer evidence kept untracked locally

- `analysis/reviewer_fairness/` policy CSVs, provenance JSONs, fairness certificates
- `analysis/reviewer_fairness_cross_family_v1/`
- `analysis/distribution_shift_ablation_v1/`
- `analysis/practical_significance_ablation_v1/`
- `analysis/supervision_objective_ablation_v1/`
- `analysis/supervision_objective_learning_curve_v1/`
- `analysis/external_learned_baselines/`
- `models/`

### Tracked small audit / provenance summaries

- contamination and temporal-order audits in `analysis/reviewer_fairness/`
- small tracked derived fixtures already committed under `data/derived/`

## Important local caveats

- `analysis/reviewer_fairness/policy_comparison_*.csv` includes both
  `primary_controlled_window` and `deployment_full_stream` rows. Only the
  primary rows are eligible for the main reviewer comparison.
- `analysis/reviewer_fairness/policy_comparison_evict_value_v1.csv` is
  intentionally contaminated/ineligible and must stay labeled that way.
- `analysis/practical_significance_ablation_v1/` currently contains smoke-scale
  timing evidence plus synthetic cost analyses; the final controlled timing run
  is still separate work.
- `analysis/distribution_shift_ablation_v1/` is a valid partial checkpoint, not
  a completed campaign.
- `analysis/supervision_objective_learning_curve_v1/` is a local explanatory
  diagnostic. Completed cells may be inspected, but incomplete aggregates must
  remain `DIAGNOSTIC_PARTIAL`.
- the last audited low-fraction learning-curve checkpoint contains `16`
  validated units / `96` rows across
  `brightkite, citibike, cloudphysics, metacdn` at fractions
  `1%, 2%, 5%, 10%`.
- as of the `2026-08-10` read-only local audit, the `25%` local extension
  has **completed naturally**: the tmux session `kbs_learning_curve_highfrac_20260809`
  is no longer running (no tmux server process is present on this host), and
  no experiment process remains in the process table. `campaign_state.json`,
  `provenance.json`, and `policy_comparison.csv` all agree that all `7/7`
  families (`brightkite`, `citibike`, `cloudphysics`, `metacdn`, `metakv`,
  `twemcache`, `wiki2018`) finished for fraction `0.25`, producing `42/42`
  expected rows, all with `status=ok`, no duplicate
  `(fraction, family, condition, capacity)` keys, no literal NaN/Inf strings,
  and all `14` model artifact SHA-256 hashes verified against the CSV. Total
  logged unit runtime (`27753.65s` / `7.71h`) matches observed wall time
  (`12:34:09`-`20:17:16` on `2026-08-09`, `7h43m`) with no gaps, so this was a
  natural completion, not the `10`-hour clean-budget stop (`~2.3h` of budget
  headroom remained). One data characteristic to carry forward: the
  `eviction_loss_pairwise` condition for `twemcache` at fraction `0.25`
  produced zero validation pairs at all three capacities, so
  `validation_pairwise_accuracy` is legitimately `NaN` (encoded as a blank
  CSV cell by `run_supervision_objective_learning_curve.py`), not a data
  integrity failure. Classification: `25_PERCENT_FINAL = VALID`,
  `READY_FOR_50_PERCENT = YES`. A first `50%` launch attempt on
  `2026-08-10` was **not** started in tmux as planned; it ran in a
  foreground SSH shell, which was terminated after ~80 minutes when that
  SSH session closed, with `0/42` rows committed
  (`50% = INTERRUPTED_BEFORE_FIRST_COMPLETED_UNIT`, cause: foreground SSH
  session terminated, not a clean 10-hour budget stop). One orphan model
  artifact
  (`models/supervision_objective_learning_curve_v1/brightkite/fraction_0.5/eviction_loss_scalar.pkl`)
  was left on disk, not referenced by any completed unit audit, CSV row,
  or provenance record. The phase was relaunched correctly the same day
  in tmux session `kbs_learning_curve_50pct_20260810`
  (`--resume --fractions 0.5 --max-wall-hours 10`) and is
  `RUNNING_LOCAL_RESUME` as of a `2026-08-10 22:10` read-only checkpoint
  (confirmed healthy: worker PID `3376086` alive `~8h11m` into the
  `10`-hour budget, CPU pegged at `~164%`, RSS dropped to `~18.6GiB` right
  after `metakv` finished, `51` threads, no errors, `<=25%` evidence
  unaffected). `5/7` folds complete for `0.5`
  (`brightkite`, `citibike`, `cloudphysics`, `metacdn`, `metakv`; `metakv`
  fold confirmed via its `unit_audits/metakv/fraction_0.5.json` and both
  model artifacts on disk), `twemcache` now in progress as fold `6/7`,
  `wiki2018` remaining. Per-family `0.5`
  runtimes observed so far range `~87-141` minutes; at that pace this
  invocation is likely to clean-stop at its `10`-hour budget with `twemcache`
  complete but `wiki2018` only partially through, or not
  started — a further tmux resume of the remaining fold should be
  expected. No `1.0` fraction units have been started. (Note: the tmux
  pane/log shows no stdout — this is a known Python stdout-buffering
  artifact, not a hang; `stdbuf -oL` does not intercept Python's internal
  `TextIOWrapper` buffering. Progress is confirmed via file-timestamp and
  hash evidence instead.)
- an exact-target-oracle vs learned-online diagnostic foundation now exists
  locally in `src/lafc/oracle_diagnostics.py` with focused synthetic tests in
  `tests/test_oracle_diagnostics.py`; one local real-trace cell has now been
  run at `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/`
  for `brightkite`, capacity `64`, horizon `4`, canonical window
  `[10000,50000)`.
- that one-cell diagnostic found LRU `13225` misses, exact finite-horizon
  eviction-loss oracle `19079` misses, learned eviction-loss scalar policy
  `15449` misses, and offline Belady `11312` misses; treat this as
  diagnostic evidence for the target/learning decomposition only, not a
  family-general or horizon-sweep conclusion.
- that oracle diagnostic is intentionally distinct from the
  minimum-counterfactual or minimum-Hamming-distance suffix-attribution line:
  the former checks exact-target consistency of a decision, while the latter
  asks which earlier changed decisions are minimally sufficient to remove a
  later excess miss.
- a target-degeneracy diagnostic cell completed locally at
  `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/`. In this
  cell, all `19079` H=4 scored decisions have ordinary zero margin, `63.0%`
  have all candidates tied, and longer horizons break only a minority of H=4
  tied sets (`14.2%` at H=8, `27.6%` at H=16, `39.6%` at H=32). Treat this as
  cell-specific mechanism evidence, not a workload-general conclusion.
- `objective_pairwise` and `eviction_loss_pairwise` are not interchangeable
  labels; the former changes the supervision objective, while the latter keeps
  the eviction-loss target fixed and only changes representation.

## Unconsolidated items requiring explicit follow-up

The following locally known Wulver-dispatched files were not found in any local
worktree during the 2026-08-09 audit and must be synced back from Wulver before
the branch can be treated as fully consolidated:

- `scripts/experiments/run_distribution_shift_family.py`
- `scripts/experiments/upgrade_cross_family_manifest_metadata.py`
- `slurm/kbs_distribution_shift_wulver_smoke.sbatch`
- `slurm/kbs_distribution_shift_wulver.sbatch`
- `slurm/kbs_cross_family_heldout_smoke.sbatch`
- `slurm/kbs_cross_family_heldout_eval_wulver.sbatch`

## PASS-1 scope

Safe PASS-1 work on this branch should stay structural:

- add read-only status/validation tooling
- improve repository navigation and script-layout documentation
- clarify tracked-vs-generated boundaries
- tighten ignore rules for obvious non-scientific logs

It should not:

- rewrite manuscript conclusions
- mutate frozen result files
- delete historical evidence
- fabricate missing Wulver-only orchestration files
