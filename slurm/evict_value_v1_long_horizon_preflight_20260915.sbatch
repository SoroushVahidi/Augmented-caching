#!/bin/bash
#
# Long-horizon (H=16/32/64/128) candidate-row dataset PREFLIGHT — NOT production.
# Single workload (metacdn), single capacity (32), full trace length.
# Purpose: measure runtime/memory/output scaling with horizon before designing
# the full PE long-horizon production campaign. See docs/canonical_experiments.md
# for the canonical (short-horizon, H<=32 historically 8/16/32) pipeline this
# extends, and scripts/build_evict_value_dataset_wulver_v1.py for the generator
# this preflight drives directly (same script/library used by the known-good
# slurm/evict_value_v1_wulver_dataset.sbatch template).
#
# Provenance (filled in at submit time, recorded again in stdout):
#   Source repo:   ~/lafc-work/Augmented-caching (Wulver checkout, branch main)
#   Wulver SHA:    3a04a25094de1e8c57a603e2662b321bcf1d9e88
#   Local SHA:     ceb36705b59d0d16db55d0fc0bfb63a5c7a2a6a1 (branch chore/repository-polish)
#   Scientific-code equivalence: verified byte-identical for
#     src/lafc/evict_value_wulver_v1.py and build_evict_value_dataset_wulver_v1.py
#     (the latter lives at scripts/build_evict_value_dataset_wulver_v1.py on this
#     Wulver checkout (pre-reorg main); the local chore/repository-polish branch
#     relocated it, unmodified, to scripts/experiments/canonical/ — path differs,
#     content is byte-identical).
#
#SBATCH --job-name=lafc-long-horizon-preflight
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

mkdir -p slurm/logs

echo "[$(date -Is)] Starting long-horizon preflight on $(hostname)"
echo "[provenance] wulver_checkout_sha=$(git rev-parse HEAD)"
echo "[provenance] wulver_checkout_branch=$(git rev-parse --abbrev-ref HEAD)"

cd ~/lafc-work/Augmented-caching

source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT_DIR="data/derived/evict_value_v1_long_horizon_preflight_metacdn_20260915"

# NEW output directory only — never touches canonical H={4,8,16} data
# (analysis/*_heavy_r1*, data/derived/evict_value_v1_wulver/) or the earlier
# failed 5-family preflight at
# data/derived/evict_value_v1_wulver_preflight_20260915.

python scripts/build_evict_value_dataset_wulver_v1.py \
  --trace-glob "data/processed/metacdn/trace.jsonl" \
  --capacities "32" \
  --horizons "16,32,64,128" \
  --split-mode trace_chunk \
  --chunk-size 4096 \
  --max-rows-per-shard 500000 \
  --split-seed 7 \
  --min-requests-per-trace 1000 \
  --min-unique-pages-per-trace 100 \
  --out-dir "${OUT_DIR}" \
  --overwrite

echo "[$(date -Is)] Dataset build finished successfully."
echo "[manifest]"
cat "${OUT_DIR}/manifest.json"
echo
echo "[split_summary]"
cat "${OUT_DIR}/split_summary.csv"
echo "[$(date -Is)] Preflight job complete."
