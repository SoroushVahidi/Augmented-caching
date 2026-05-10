# AGENTS.md

## Cursor Cloud specific instructions

### Overview

`lafc` is a pure-Python research codebase for learning-augmented caching. It has no external services (no database, no web server, no Docker). Everything runs as Python scripts or `pytest`.

### Dependencies

```
pip install -e ".[dev]"
```

This installs `numpy`, `scikit-learn`, `pulp`, `pytest`, and `matplotlib`. See `pyproject.toml` for details.

### Running tests

```
pytest tests/ -v
```

All 283 tests should pass. There are no linters configured in this repo.

### Running policies (CLI)

```
python3 -m lafc.runner.run_policy --policy <name> --trace <trace.json> --capacity <k>
```

Use `python3` (not `python`) — the environment does not alias `python` to `python3`. Example traces are in `data/example_unweighted.json` and `data/example_general_caching.json`.

### Quick-start ML pipeline

The full evict_value_v1 quick start (dataset build → train → eval) is documented in `README.md` under "Quick start". It requires running three scripts in sequence.

### Gotchas

- `~/.local/bin` must be on `PATH` for `pytest` and other pip-installed scripts to work. The update script handles this via `export PATH`.
- The `scripts/train_ml_gate_v1.py` script has a pre-existing bug (`AttributeError: 'LinearProbabilityEstimator' object has no attribute 'named_steps'`). This does not affect the core evict_value_v1 pipeline or the test suite.
- The `RuntimeWarning` about `lafc.runner.run_policy` found in `sys.modules` when using `python3 -m` is harmless and can be ignored.
