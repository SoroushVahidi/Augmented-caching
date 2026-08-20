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

A few more specific provenance mechanisms used across this line of work,
in case you need to trace where a specific number came from:

- **Trace hashes.** Fairness-protocol comparisons record a `trace_sha256`
  column alongside each row, hashing the exact input trace file used for
  that row. This lets you confirm two rows that claim to use "the same
  trace" actually used byte-identical input, not just a file with the same
  name.
- **Split / fold provenance.** Cross-family comparisons use a `fold_id`
  naming convention (`cross_family_v1_<held_out_family>`) that records
  which single family was held out for that fold, alongside explicit
  `history_start`/`history_end`/`score_start`/`score_end` request-index
  columns defining the exact evaluation window used.
- **Leakage gates.** Where a split is meant to be held-out (no family used
  for both training and evaluation), the eligibility rule is checked
  explicitly rather than assumed from the config alone -- a split that
  fails this check (e.g. re-using the same trace streams for train and
  eval) must be labeled contaminated and excluded from primary comparisons,
  never silently included; see
  [`RESULTS_AND_LIMITATIONS.md`](RESULTS_AND_LIMITATIONS.md) section F for
  a concrete example of a split that failed this check.
- **Seeds.** See section D above; a fixed seed is recorded per run and
  should match across anything being compared.
- **Generated-result status.** A generated result is only as trustworthy as
  its accompanying `provenance.json`/hash pair -- a CSV or model file
  without a matching provenance record next to it should not be treated as
  reproducible evidence, only as an unverified artifact.
- **Why bulk data stays outside git.** Large derived datasets, trained
  model weights, and most per-experiment result trees are reproducible from
  tracked source + tracked config + a recorded seed, so they are
  deliberately not committed (see section F above) -- committing them would
  bloat the repository without adding anything the tracked inputs plus
  provenance records don't already determine.

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

# Target-degeneracy diagnostic (trace-only; skips the trained-model leg)
python scripts/experiments/analyze_eviction_loss_target_degeneracy.py \
  --family brightkite --capacity 64 --horizon 4 --no-learned

# Exact-target-oracle diagnostic (trace-only; skips the trained-model leg)
python scripts/experiments/run_exact_target_oracle_diagnostic.py \
  --family brightkite --capacity 64 --horizon 4 --no-learned
```

Both diagnostics above default to comparing against a trained
`evict_value_v1` model loaded from a model registry
(`analysis/supervision_objective_ablation_v1/model_registry.json`) produced
by the objective-ablation campaign, which is not yet on `main` -- pass
`--no-learned` to run only the trace-derived legs (LRU / exact-target-oracle
/ Belady for the oracle diagnostic; tie-fraction / entropy metrics for the
degeneracy diagnostic) without that dependency. See
[`EXPERIMENT_REGISTRY.md`](EXPERIMENT_REGISTRY.md) experiments 4 and 5 for
what each diagnostic measures and its current evidence status.

See the root [`README.md`](../README.md) for the full policy roster and the
canonical Knowledge-Based Systems manuscript reproduction path
([`CANONICAL_KBS_SUBMISSION.md`](../historical/CANONICAL_KBS_SUBMISSION.md)), which has
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
