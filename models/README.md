# Models guide

This directory contains trained model artifacts and their associated metadata.

## Directory structure

- `models/*.pkl` — serialized model files (typically excluded from version control if large).
- `models/archival/` — models from superseded or historical experiments.
- `models/staging/` — models currently being trained or evaluated in active worktrees.

## Version Control Policy

- **Tracked**: small model-selection summaries, best-config JSONs, and small frozen models if essential for reproducibility.
- **Not Tracked**: large ensemble models, checkpoint files, and temporary training artifacts.

## Naming Convention

Models should ideally follow the pattern:
`<experiment>_<fold>_<objective>_<version>_<hash>.pkl`

Example: `evict_value_v1_brightkite_h8_r1_a60e582.pkl`
