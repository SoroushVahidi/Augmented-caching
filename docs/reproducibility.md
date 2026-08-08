# Reproducibility guide

This document provides a comprehensive guide to reproducing the scientific results in this repository.

## 1. Environment setup

### Requirements
- **Python**: 3.9 or higher (tested on 3.10 and 3.12).
- **Dependencies**: `numpy`, `scikit-learn`, `pulp`, `matplotlib`, `pytest`.
- **Optional**: `lightgbm` (required for LRB baseline).

### Installation
We recommend using a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

If you intend to run the LRB baseline, install the `lrb` extra:
```bash
pip install -e ".[lrb]"
```

**Note on `pulp`**: Ensure that a linear programming solver (like CBC, which usually comes with `pulp`) is available in your environment. If you encounter solver errors, you may need to install `coinor-cbc` via your system package manager.

## 2. Dataset preparation

Follow the [Dataset Setup Guide](data_setup.md) to obtain and standardize the traces. At a minimum, for canonical reproduction, you need the `brightkite`, `citibike`, and `wiki2018` traces.

## 3. Simulator usage

The core simulator is accessed via `python -m lafc.runner.run_policy`:

```bash
python -m lafc.runner.run_policy \
  --policy <name> \
  --trace data/processed/<dataset>/trace.jsonl \
  --capacity <k>
```

Available policies include `lru`, `offline_belady`, `predictive_marker`, and `evict_value_v1`.

## 4. Reproducing canonical results

The canonical results for the KBS manuscript (tagged `heavy_r1`) involve a multi-phase pipeline: **Build Dataset** → **Train Model** → **Evaluate**.

Refer to the [Canonical Experiment Guide](canonical_experiments.md) for the exact command sequences and Slurm batch scripts.

## 5. Seed and scoring policy

- **Random seeds**: All experiments use a fixed seed (default `0`) for reproducibility.
- **Scoring windows**: Learned policies are evaluated on a held-out scoring window (typically the last 20% of the trace).
- **Deterministic simulation**: Given the same trace, capacity, and model, simulators produce bit-identical results.

## 6. Hardware and runtime expectations

| Phase | Dataset Size | CPU | Memory | Runtime |
|-------|--------------|-----|--------|---------|
| **Prepare** | ~100MB | 1 core | < 2GB | Seconds |
| **Train** | ~1GB | 1 core | 4-16GB | Minutes |
| **Evaluate** | All | 1-8 cores | 2-8GB | Minutes |
| **LRB (External)** | All | 1 core | 32GB+ | Hours |

## 7. Artifacts and provenance

All canonical runs generate a `provenance.json` or `summary.json` containing:
- Commit hash.
- Command-line arguments.
- Environment metadata.
- Result checksums.

See the [Artifact Policy](artifact_policy.md) for details on where these files are stored and how they are tracked.
