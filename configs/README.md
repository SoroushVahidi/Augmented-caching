# Configuration guide

This directory contains experimental configurations and protocol definitions.

## Directory structure

- `configs/canonical/` — frozen configurations for the main manuscript results.
- `configs/reviewer/` — configurations for reviewer-revision experiments (fairness, ablation, sensitivity).
- `configs/sensitivity/` — configurations for parameter sensitivity sweeps.
- `configs/diagnostics/` — configurations for diagnostic and auditing runs.
- `configs/historical/` — archival and superseded configurations.

## Usage

Most active configurations for ongoing reviewer-revision experiments currently live in their respective feature branches and worktrees:
- `feat/reviewer-fairness-protocol`
- `feat/supervision-objective-ablation`
- `feat/distribution-shift-ablation`

Once these experiments are finalized, their authoritative configurations will be merged into the `configs/reviewer/` directory of the primary repository.
