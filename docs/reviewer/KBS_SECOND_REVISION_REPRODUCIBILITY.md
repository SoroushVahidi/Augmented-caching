# KBS Second-Revision Reproducibility Guide

Status: authoritative reproducibility reference for the current
`kbs/second-revision-science` reviewer-science line (the diagnostics indexed
in
[`KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md`](KBS_SECOND_REVISION_EXPERIMENT_REGISTRY.md)).

This document is scoped to second-revision reviewer-science work. It does
**not** cover the older `heavy_r1` Wulver manuscript pipeline -- see
`docs/reproducibility_and_artifacts.md` and `docs/kbs_manuscript_workflow.md`
for that separate, historical line.

## A. Environment assumptions

- Python 3.12 (matches the interpreter recorded in every protocol/provenance
  JSON on this branch).
- Install with `pip install -e ".[dev]"` from the repository root (see the
  top-level `README.md`).
- `lightgbm` is required for some learned-baseline tests and policies.
- Commands below assume the current working directory is the repository
  root. They intentionally avoid embedding this machine's absolute path.

## B. `PYTHONPATH`

Every entry point under `scripts/` imports from the `lafc` package in `src/`.
Set:

```bash
export PYTHONPATH=src
```

before running any script or `pytest` invocation in this repository. Scripts
do not add `src/` to `sys.path` themselves.

## C. Canonical windows

Every second-revision reviewer diagnostic (fairness baselines, exact-target
oracle, target degeneracy, learning-curve, continuation C1/C2) shares one
canonical score window, imported from a single constants module
(`lafc.experiments.reviewer_fairness_common`) rather than being redefined
per script:

- `HISTORY_START = 0`
- `SCORE_START = 10000`
- `SCORE_END = 50000`
- i.e. history window `[0, 10000)`, scored window `[10000, 50000)`, `40000`
  scored requests per trace.

This was verified by direct import-source inspection (Pass #2 of this
branch's polish work), not by re-reading documentation -- if you change any
of these constants, every diagnostic listed in the experiment registry moves
together.

## D. Family / capacity conventions

- Full campaign scope: 7 held-out families --
  `brightkite, citibike, cloudphysics, metacdn, metakv, twemcache, wiki2018`.
- Full campaign capacities: `[32, 64, 128]` (object-slot semantics).
- Single-cell diagnostics (exact-target oracle, target degeneracy) default
  to `--family brightkite --capacity 64` but accept any family/capacity via
  CLI flags -- they are not hardcoded to that one cell, just piloted there.
- Folds are leave-one-family-out, defined in `configs/fair_cross_family_v1/folds/`.

## E. Config-driven execution

Frozen protocols live as JSON under `configs/`:

- `configs/supervision_objective_learning_curve_v1.json`
- `configs/continuation_policy_causal_ablation_v1.json`
- `configs/reviewer_fairness_protocol.json`
- `configs/fair_cross_family_v1/folds/*.json`

Do not hand-edit a frozen config to "fix" a run; if a config value is wrong,
that is a protocol-change decision, not a bug fix (see the top-level task
instructions across this branch's polish passes: never change scientific
protocol merely for convenience).

## F. Deterministic seeds

- `seed = 0` throughout (learning-curve config, continuation config,
  objective-ablation training).
- Nested training-fraction subsets are built once via
  `lafc.reviewer_diagnostics.build_nested_fraction_subsets`: every decision
  id gets a stable per-seed hash rank, and each fraction takes a prefix of
  that single global order. This guarantees strict subset nesting
  (`subset(1%) ⊂ subset(2%) ⊂ ... ⊂ subset(100%)`) and full determinism for
  a fixed seed -- verified directly in source, Pass #2.

## G. Generated-output policy

- Tracked in git: `src/`, `scripts/`, `tests/`, `configs/`, `docs/`.
- Not tracked (see `.gitignore`): most `analysis/<experiment>/` result
  trees, `models/<experiment>/`, large `data/derived/` datasets, `logs/`.
  `analysis/kbs_local_current_evidence_synthesis_*/` is similarly ignored.
- A few small canonical audits/fixtures under `analysis/` are intentionally
  tracked (e.g. `analysis/continuation_policy_light/`, a historical result,
  see the experiment registry).
- `docs/reviewer/kbs_second_revision_artifact_map.md` and the experiment
  registry are the source of truth for which generated outputs are complete,
  partial, smoke-only, contaminated, historical, or currently citable.

## H. How checkpoint/resume works (learning-curve campaign)

- `scripts/experiments/run_supervision_objective_learning_curve.py --resume`
  reads `campaign_state.json`'s `completed_units` set and skips any
  `(family, fraction)` unit already recorded there.
- A unit is marked complete (via an atomic tmp-file-then-rename JSON write)
  **only after** its scalar model, pairwise model, all CSV rows, and its
  `unit_audits/<family>/fraction_<f>.json` file have all been written
  successfully -- so a mid-unit failure never marks that unit complete, and
  `--resume` will safely redo the whole unit rather than silently accepting
  a partial one.
- CSV row writes are separately idempotent (`IncrementalCsvWriter.already_done`
  keyed on condition/fraction/family/capacity), so redoing an interrupted
  unit does not duplicate rows.
- `--max-wall-hours` is a **clean stop before the next unit**, not a mid-unit
  kill: the runner checks `remaining_time > average_completed_unit_seconds`
  before starting each new unit and stops if that fails. It can therefore
  overrun the budget somewhat while finishing the unit already in progress.
- A model `.pkl` file being present on disk is **not** by itself proof that
  its unit completed -- always check `campaign_state.json` /
  `unit_audits/` / the CSV row, not just file existence. (A previously
  interrupted foreground run left exactly this kind of orphan model file;
  see `docs/kbs_second_revision_repository_state.md`.)

## I. Where provenance/hashes are stored

- `analysis/<experiment>/provenance.json`: repository commit/branch,
  platform, python version, protocol id, fraction/family/capacity scope.
- `analysis/<experiment>/protocol_snapshot.json`: the exact frozen config
  used, so a later run can detect protocol drift before resuming.
- `analysis/<experiment>/unit_audits/<family>/fraction_<f>.json` (learning
  curve) or `.../<family>_cap<capacity>_h<horizon>/summary.json`
  (oracle/degeneracy diagnostics): per-unit model hashes, decision-subset
  hashes, and same-example guarantees.
- Model artifact SHA-256 hashes are recorded in both the per-unit audit JSON
  and the corresponding CSV row (`model_sha256` column); these should match
  the actual file hash -- spot-checking this was part of this branch's
  integrity audits.

## J. Local vs. Wulver execution

- This repository clone is local-only; Wulver is a separate cluster
  environment not contacted by any of this branch's polish passes.
- Some experiments are fully local (exact-target oracle, target degeneracy,
  learning curve, objective ablation, fairness baselines). Others are
  local-implementation-only with the full campaign requiring Wulver scale
  (continuation C1/C2, seven-family distribution-shift continuation,
  historical-tail diagnostic).
- See
  [`KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md`](KBS_LOCAL_TO_WULVER_MASTER_MANIFEST.md)
  and
  [`KBS_LOCAL_WULVER_CONFLICT_MATRIX.md`](KBS_LOCAL_WULVER_CONFLICT_MATRIX.md)
  for what would need to transfer and what carries merge risk.
- Do not assert a Wulver-side numerical result as available locally; the
  experiment registry marks such items `KNOWN_WULVER_RESULT / NOT_LOCAL`
  rather than fabricating a path.

## K. What must NOT be recomputed blindly

- Do not rerun a `FINAL_VALIDATED` or `COMPLETE_DIAGNOSTIC` experiment "to
  double check" without a specific reason -- these already have recorded
  hashes/provenance; recomputation risk (different machine, different
  library versions) is a real threat to reproducibility claims.
- Do not resume or relaunch the learning-curve campaign while its tmux
  worker is already running (check `tmux ls` first).
- Do not treat a currently-`RUNNING` fraction's partial rows as evidence;
  wait for a clean stop and an integrity audit.
- Do not hand-edit a frozen `configs/*.json` protocol file to make a
  specific run "work" -- that is a protocol change, not a bug fix, and
  invalidates comparability with already-completed units under the old
  config.
- Do not treat a `SMOKE_ONLY` result (continuation smoke, practical-
  significance smoke) as a scientific finding -- both are explicitly
  self-disclaiming in their own output artifacts.

## Stable reproduction commands

These are the exact, currently-supported invocations (not aspirational).
Each reads its scope from the frozen config/CLI defaults described above.

```bash
# Read-only campaign status (safe to run any time, does not launch anything)
PYTHONPATH=src python3 scripts/validation/revision_status.py
PYTHONPATH=src python3 scripts/validation/revision_readiness.py

# Lightweight test suite for the local-only diagnostic groups (fast, synthetic data)
PYTHONPATH=src python3 -m pytest \
  tests/test_oracle_diagnostics.py \
  tests/test_target_degeneracy.py \
  tests/test_continuation_policy_ablation.py \
  tests/test_supervision_objective_learning_curve.py \
  tests/test_reviewer_diagnostics.py \
  tests/test_revision_status.py \
  tests/test_revision_readiness.py \
  tests/test_supervision_objective_ablation.py -q

# Exact-target oracle diagnostic, default cell (brightkite, capacity 64, H=4)
PYTHONPATH=src python3 scripts/experiments/run_exact_target_oracle_diagnostic.py

# Target-degeneracy diagnostic, default cell, with the H=8/16/32 tie-resolution extension
PYTHONPATH=src python3 scripts/experiments/analyze_eviction_loss_target_degeneracy.py

# Learning-curve campaign (example: resume an existing campaign at a given fraction)
PYTHONPATH=src python3 scripts/experiments/run_supervision_objective_learning_curve.py \
  --resume --fractions 0.5 --max-wall-hours 10 \
  --config configs/supervision_objective_learning_curve_v1.json \
  --out-dir analysis/supervision_objective_learning_curve_v1 \
  --models-dir models/supervision_objective_learning_curve_v1
```

Do not copy the last command verbatim while a campaign is already running in
tmux -- check `tmux ls` first (see Section K).
