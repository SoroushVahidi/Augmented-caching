# Data guide

This directory contains trace data and derived datasets used for experiments.

## Directory structure

- `data/raw/` — raw trace data (typically excluded from version control).
- `data/processed/` — preprocessed traces in standard repository format.
- `data/derived/` — feature tables and supervision targets generated for model training.

## Version Control Policy

- **Tracked**: small example traces, dataset specifications, and READMEs.
- **Not Tracked**: large raw traces, huge derived feature tables, and temporary simulator outputs.

Individual experiment subdirectories in `data/derived/` should include a `manifest.json` or `provenance.json` describing how they were generated.
