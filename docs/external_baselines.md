# External baselines

This document describes the external learned baselines integrated into this repository for comparison.

## Baseline status definitions

- **Official Source**: Implementation provided by the original authors.
- **Independent Reimplementation**: Re-written from scratch based on the published paper.
- **Adapted Implementation**: Original code modified to fit this repository's simulator and dataset formats.

## Integrated baselines

### 1. LRB (Learning Relaxed Belady)
- **Paper**: *Learning Relaxed Belady for Content Delivery Network Caching* (NSDI '20).
- **Status**: Adapted implementation.
- **Provenance**: Based on the official [songjiayang/LRB](https://github.com/songjiayang/LRB) logic, integrated via `scripts/experiments/run_lrb_external_baseline.py`.
- **Note**: This is a heavyweight baseline requiring significant training time and memory.
- **Method Spec**: See `docs/lrb_method_spec.md`.

### 2. 3L-Cache
- **Paper**: *3L-Cache: A Three-Layer Cache Management for Cloud Storage* (FAST '21).
- **Status**: Independent reimplementation.
- **Provenance**: Reimplemented based on the paper's description of the admission and eviction policies.

### 3. HALP
- **Paper**: *HALP: Heuristic-Aided Learned Caching* (OSDI '23).
- **Status**: Independent reimplementation / Adaptation.
- **Note**: Reimplemented to follow the heuristic-augmented learning structure described in the paper.

### 4. CACHEUS
- **Paper**: *CACHEUS: Machine Learning-based Caching for Block I/O* (FAST '21).
- **Status**: Author-released source (no license file found at pinned commit).
- **Provenance**: Integrated via `lafc.policies.external.cacheus`.

## Setup and usage

External baselines often require additional dependencies (e.g., specific versions of `scikit-learn` or `LightGBM`) or external binaries.

For LRB, you must use the designated runner:
```bash
python scripts/experiments/run_lrb_external_baseline.py --capacities 32,64,128 --out-dir analysis/external_learned_baselines/lrb
```

Refer to `docs/baselines.md` for the full roster of internal classical baselines (LRU, Belady, etc.).
