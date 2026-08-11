# Reproducibility

This page describes how to reproduce what is actually available on `main`
today, plus how to locate the deeper diagnostic work that currently lives on
the `kbs/second-revision-science` development branch (see
[`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) for exactly which
experiments that applies to). Commands below are only included when they are
stable and currently supported -- nothing aspirational, nothing tied to a
specific machine.

## A. Environment assumptions

- Python 3.12.
- No absolute paths or machine-specific assumptions are required for
  anything documented here; run everything from the repository root.

## B. Install

```bash
pip install -e ".[dev]"
```

This installs the `lafc` package in editable mode from `src/`
(`[tool.setuptools.packages.find] where = ["src"]` in `pyproject.toml`), so
`python -m lafc...` and `pytest` work directly afterward without needing to
set `PYTHONPATH` manually. If you ever run a script without an editable
install, set `PYTHONPATH=src` explicitly first.

Optional: `lightgbm` is required for some learned-baseline tests and
policies (e.g. `lrb`, `three_l_cache` where present).

## C. Canonical protocol constants

Different experiments define their own frozen protocol (folds, capacities,
horizon, seed) in their own config/doc -- there is intentionally no single
global constant set on `main`. Where a diagnostic reuses a shared window or
constant, that is documented in its own protocol file; do not assume one
experiment's window applies to another without checking.

## D. Deterministic seeds

Training and evaluation scripts on this branch take an explicit `--seed` or
read a seed from their config; results are deterministic for a fixed seed
and fixed input data. If you are comparing results across runs, confirm the
seed and config match.

## E. Config-driven execution

Frozen protocols and configs live under `configs/`. Do not hand-edit a
frozen config to make a specific run "work" -- a config value being wrong is
a protocol-design decision to revisit deliberately, not a bug to patch
silently, since it changes what earlier results are comparable to.

## F. Generated-output behavior

- Tracked in git: `src/`, `scripts/`, `tests/`, `configs/`, `docs/`.
- Not tracked (see `.gitignore`): most experiment result trees under
  `analysis/<experiment>/`, trained model artifacts under `models/`, and
  large derived datasets under `data/derived/`.
- A handful of small, stable, canonical audits/fixtures are intentionally
  tracked under `analysis/` -- see [`../analysis/README.md`](../analysis/README.md)
  for the current list and rationale.

## G. Provenance

Experiment output directories generally include a `provenance.json` (commit,
branch, platform, python version, protocol scope) and, where relevant, a
`protocol_snapshot.json` recording the exact config used, so a later run can
detect protocol drift before resuming or comparing. Model artifacts are
hashed (SHA-256); treat a hash mismatch between a recorded provenance file
and the actual file on disk as an integrity problem to investigate, not to
paper over.

## H. Stable reproduction commands

```bash
# Install
pip install -e ".[dev]"

# Full test suite
pytest tests/ -v

# A literature baseline
python -m lafc.runner.run_policy \
  --policy predictive_marker \
  --trace data/example_unweighted.json \
  --capacity 3

# A robust combiner baseline
python -m lafc.runner.run_policy \
  --policy robust_ftp_d_marker \
  --trace data/example_unweighted.json \
  --capacity 3 \
  --derive-predicted-caches

# Local evict_value_v1 first check (small; not the full evaluation line)
python scripts/build_evict_value_dataset_v1.py --max-rows 200000
python scripts/train_evict_value_v1.py --horizon 8
python scripts/run_evict_value_v1_first_check.py

# Dataset preparation
python scripts/datasets/prepare_all.py \
  --dataset <brightkite|citibike|spec_cpu2006|wiki2018|twemcache|metakv|metacdn|cloudphysics|all>
```

See the root [`README.md`](../README.md) for the full policy roster and the
canonical Knowledge-Based Systems manuscript reproduction path
([`CANONICAL_KBS_SUBMISSION.md`](../CANONICAL_KBS_SUBMISSION.md)), which has
its own more specific (Slurm-based, larger-scale) reproduction steps.

## I. Resume / checkpoint semantics (general note)

Longer-running experiments on the development branch (the diagnostic work
described in [`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)) use an
atomic, per-unit checkpoint pattern: a unit of work is marked complete only
after all of its outputs are written successfully, using an atomic
write-then-rename, so a checkpoint file's existence never reflects a
half-finished unit, and resuming safely redoes any unit that was interrupted
mid-way rather than accepting a partial one. If you build new long-running
experiments on this codebase, this pattern is worth following.

## J. Local vs. larger-scale execution

Some experiments in this line of work (documented in
[`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md)) are single-machine
diagnostics that run in minutes to hours; others (a full cross-family
comparison, or a full causal continuation-policy campaign) are designed to
run at a larger, cluster scale and are not expected to complete on a laptop.
Where a registry row says an experiment needs larger-scale execution, do not
assume the numbers exist locally just because the code does.

## K. What not to recompute blindly

- Do not treat a currently-running or partial experiment's numbers as final;
  wait for its own completion/audit before citing them.
- Do not rerun a `FINAL_VALIDATED` experiment "just to check" without a
  specific reason -- recomputation on a different machine/library-version
  combination is itself a threat to reproducibility claims, not a
  neutral action.
- Do not hand-edit a frozen config to make a specific run succeed (see
  Section E).
- Do not cite smoke-scale timing or implementation-only results as final
  scientific findings -- see [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md).
