# Augmented Caching

Research repository for **learning-augmented caching**, focusing on decision-aligned eviction-value prediction. This project provides a robust framework for training and evaluating learned cache eviction policies against literature-faithful baselines across diverse production workloads.

## Key Ideas

Classical learned caching often focuses on predicting the "time-to-next-access" (reuse distance). This repository explores **eviction-value prediction**, where a supervised model predicts the expected loss (in terms of future misses) if an item is evicted now.

- **Decision Alignment**: We optimize for prediction quality specifically where it affects the eviction decision, improving closed-loop cache performance.
- **Robustness**: Integrated "guarded" wrappers provide fallback mechanisms to classical policies (e.g., LRU) when learned models are uncertain.
- **Trace-Driven**: Evaluated on real-world traces from Twitter, Meta, CloudPhysics, and public sources.

## Repository Layout

- `src/lafc/`: Core library implementation (policies, simulator, models).
- `scripts/`: Structured entry points for setup, training, and evaluation.
- `configs/`: Experimental protocols and frozen configurations.
- `docs/`: Detailed methodology, reproducibility, and result guides.
- `analysis/`: Generated experiment artifacts (CSV, JSON, Markdown).
- `data/`: Small example traces and dataset preparation drivers.
- `models/`: Trained model artifacts and staging areas.

For a detailed map, see [Repository Map](docs/repo_map.md).

## Quick Start

### 1. Installation

Requires Python 3.9+.

```bash
# Clone the repository
git clone https://github.com/soroush/Augmented-caching.git
cd Augmented-caching

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install base dependencies and dev tools
pip install -e ".[dev]"
```

### 2. Run a smoke test

Verify the installation by running a simple LRU simulation on a small example trace:

```bash
python -m lafc.runner.run_policy \
  --policy lru \
  --trace data/example_unweighted.json \
  --capacity 3
```

### 3. Run unit tests

```bash
pytest tests/ -v
```

## Reproducing Results

Detailed instructions for reproducing published results are available in the following guides:

- **[Reproducibility Guide](docs/reproducibility.md)**: Environment setup and general workflow.
- **[Dataset Setup](docs/data_setup.md)**: How to obtain and prepare trace data.
- **[Canonical Experiments](docs/canonical_experiments.md)**: Command sequences for the primary manuscript results.

## Baselines and Policies

This repository integrates a wide range of cache eviction policies:

- **Classical**: LRU, FIFO, Marker, Belady (Optimal).
- **Learned (Internal)**: `evict_value_v1` (our primary model), `atlas`, `ml_gate`.
- **Learned (External)**: LRB (songjiayang/LRB), 3L-Cache, HALP, CACHEUS.

For details on baseline provenance and implementation status, see [External Baselines](docs/external_baselines.md) and [Baseline Roster](docs/baselines.md).

## Reviewer Revision

Ongoing experiments addressing reviewer feedback (Fairness, Objective Ablation, Distribution Shift) are tracked in the [Reviewer Revision Index](docs/reviewer_revision/README.md).

## Artifacts and Provenance

Scientific outputs are written to `analysis/`. We use a strict [Artifact Policy](docs/artifact_policy.md) to distinguish between frozen canonical results and exploratory diagnostics.

## License

This project is licensed under the terms included in the [LICENSE](LICENSE) file. External baselines may have separate licensing terms; see [Third-Party Notices](docs/external_baselines.md).

## Citation

If you use this code in your research, please cite our manuscript:

```bibtex
@article{lafc2026,
  title={Decision-aligned eviction-value prediction for robust learning-augmented caching},
  author={...},
  journal={Knowledge-Based Systems},
  year={2026}
}
```
