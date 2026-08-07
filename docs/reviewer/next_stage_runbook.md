# Reviewer Revision — Next-Stage Runbook

Companion to `docs/reviewer_revision_roadmap.md` / `configs/reviewer_revision_roadmap.json`
(both live in the `feat/reviewer-fairness-protocol` worktree, see "Worktree
ownership" below — they are NOT present in this `main` checkout). This
runbook records the exact command sequence for each concern's remaining
stages, prepared in advance so nothing needs to be designed or improvised
under time pressure once the currently-running jobs finish.

All tools referenced here are read-only status/gate checks or thin,
fail-closed wrappers around already-existing runners — none of them was
run to completion against real data as part of preparing this runbook
(beyond safe dry-runs / read-only audits explicitly noted).

## Worktree ownership

| Worktree | Branch | Owns |
|---|---|---|
| `/home/soroush/Augmented-caching` | `main` | This runbook, `scripts/revision_status.py`, `scripts/revision_readiness.py` (cross-worktree, read-only) |
| `/home/soroush/Augmented-caching-fairness` | `feat/reviewer-fairness-protocol` | Concern 1 (cross-family retraining), Concern 3 (distribution shift), Concern 4 (practical significance), and the roadmap doc itself |
| `/home/soroush/Augmented-caching-objective-ablation` | `feat/supervision-objective-ablation` | Concern 2 (supervision-objective ablation) |

Find current paths yourself rather than trusting this table blindly:
`git -C /home/soroush/Augmented-caching worktree list`.

## Quick status check (run this first, always)

```bash
cd /home/soroush/Augmented-caching
python3 scripts/revision_status.py       # per-concern artifact counts, active tmux sessions
python3 scripts/revision_readiness.py    # + registry/eval/resume/timing readiness, one NEXT_ACTION line
```

Both are 100% read-only (no writes, no subprocess launches of any
scientific job) and safe to run at any time, including while other jobs
are active.

---

## Concern 1 — Cross-family retraining (`Augmented-caching-fairness`)

```
wait for 7/7 final models (models/evict_value_v1_cross_family_v1_<family>.pkl)
  ->
python scripts/experiments/finalize_cross_family_model_registry.py
  ->
(inspect analysis/reviewer_fairness_cross_family_v1/model_registry.json: MODEL_SELECTION_FROZEN=true)
  ->
python scripts/experiments/run_cross_family_heldout_eval.py --dry-run   # sanity-check the plan first
python scripts/experiments/run_cross_family_heldout_eval.py             # 7 x 3 x 2 = 42 rows (21 primary_controlled_window)
  ->
python scripts/experiments/generate_fairness_certificate_v5.py         # already exists, already gates on row completeness
  ->
(inspect analysis/reviewer_fairness_v5/fairness_certificate.json: evict_value_v1_cross_family_v1 overall=PASS)
  ->
build primary_comparison.csv / oracle_comparison.csv (script TBD if not already covered by v5 certificate outputs)
  ->
run the frozen paired statistics (configs/reviewer_fairness_statistics.json) -- ONLY after the certificate PASSes
```

tmux: `evict_cross_family_resume` (currently running). Log:
`/tmp/evict_cross_family_resume.log`. Output:
`data/derived/evict_value_v1_cross_family_v1/`,
`analysis/reviewer_fairness_cross_family_v1/`.

`finalize_cross_family_model_registry.py` fails closed (exit 1, writes
nothing) below 7/7 folds — verified against real state (2/7 as of this
writing). `run_cross_family_heldout_eval.py` additionally refuses to run
unless the registry is frozen AND every model hash on disk still matches
the registry's recorded hash.

## Concern 2 — Supervision-objective ablation (`Augmented-caching-objective-ablation`)

**Important:** unlike Concern 1, this one is already fully automated. The
currently-running tmux session (`objective_ablation_pipeline`, driven by
`/tmp/run_supervision_objective_ablation_pipeline.sh`) chains, with **no
manual step in between**:

```
7 folds (dataset build -> train 4 objectives)
  -> Stage 3: scripts/build_supervision_objective_ablation_registry.py   (already exists, fails closed <28/28)
  -> Stage 4: scripts/experiments/run_supervision_objective_ablation.py  (already exists, fails closed on unfrozen registry)
```

The same-example and fairness audits prepared in this session are **not**
wired into that auto-chain:

```
python scripts/experiments/audit_supervision_objective_examples.py --partial-audit   # can run on completed folds today
python scripts/experiments/audit_supervision_objective_fairness.py --partial-audit   # can run on completed folds today
```

If these audits are meant to gate the 84-row evaluation, the running
pipeline must be **deliberately interrupted after Stage 3** (registry
freeze) and before Stage 4 (eval) — that is a conscious decision to make
when the time comes, not something either audit script or this runbook
does automatically. `scripts/revision_readiness.py` prints an explicit
reminder of this while Concern 2 training is still running.

Once 84/84 rows exist:

```
run the frozen paired statistics (configs/supervision_objective_ablation_statistics.json)
```

Log: `/tmp/objective_ablation_pipeline.log`. Output:
`data/derived/supervision_objective_ablation_v1/`,
`models/supervision_objective_ablation_v1/`,
`analysis/supervision_objective_ablation_v1/`.

## Concern 3 — Distribution shift (`Augmented-caching-fairness`)

```
python scripts/experiments/resume_distribution_shift.py --dry-run    # confirm plan: completed folds, next fold, artifact integrity
  ->
python scripts/experiments/resume_distribution_shift.py --max-wall-hours <N>    # launch in a NEW tmux session when authorized
  ->
(repeat resume as needed -- each pass stops cleanly at its own wall-clock budget)
  ->
python scripts/experiments/audit_distribution_shift_completion.py    # only meaningful once 42/42
  ->
(only if classification == COMPLETE_VALID) run the frozen statistics (configs/distribution_shift_statistics.json)
```

Suggested tmux naming for the next resume: `distribution_shift_resume_<date>`
(the wrapper does not pick a tmux session name for you — launch it inside
one explicitly, e.g. `tmux new-session -d -s distribution_shift_resume_<date>
'python scripts/experiments/resume_distribution_shift.py --max-wall-hours 4 2>&1 | tee /tmp/distribution_shift_resume_<date>.log'`).

Current real state (read-only, as of this writing): 18/42 primary rows,
3/7 folds complete (brightkite, citibike, cloudphysics), stopped cleanly
on its own wall-clock budget, all three output CSVs clean (no duplicates/
failures/NaN). `resume_distribution_shift.py --dry-run` reproduces this
exactly from live artifacts, not from a hardcoded snapshot.

Output: `analysis/distribution_shift_ablation_v1/`.

## Concern 4 — Practical significance (`Augmented-caching-fairness`)

```
python scripts/experiments/run_practical_significance_controlled.py            # prints TIMING_GATE=READY/BLOCKED + exact plan, never launches
  ->
(only once TIMING_GATE=READY)
python scripts/experiments/run_practical_significance_controlled.py --launch   # reuses run_practical_significance_ablation.py --all --controlled-final
  ->
inspect analysis/practical_significance_ablation_v1/controlled_final/ (or wherever --controlled-final routes output)
  ->
run the frozen statistics (configs/practical_significance_statistics.json)
  ->
re-evaluate the API-case-study decision gate (see roadmap's Concern 4 "API decision" field) once the controlled numbers exist
```

The gate blocks on: any Concern 1/2/3 process, the archival legacy LRB
process, load average above ~0.5/core, <20GB free RAM, <20GB free disk, or
a missing/unhashable protocol config. All four blocking-process categories
and both resource thresholds are checked from live `ps`/`/proc` state, not
cached — re-run the gate check immediately before actually launching.

Smoke-scale artifacts (`analysis/practical_significance_ablation_v1/*.csv`,
9 files) already exist and are complete; only the controlled/idle-machine
run is pending.

---

## What NOT to do (carried over from the roadmap's "Do not forget")

- Do not write the final reviewer response before each concern's
  experiments are complete.
- Do not choose hyperparameters/thresholds/optimization variants based on
  held-out/test outcomes.
- Do not compute any frozen statistics until the corresponding concern's
  completion/fairness gates all pass.
- Do not run Concern 4's controlled campaign while any other concern's job
  (or the legacy LRB) is active.
- Re-run `scripts/revision_status.py` / `scripts/revision_readiness.py`
  immediately before citing any progress number in the manuscript or
  reviewer response — this campaign's state changes within minutes.
