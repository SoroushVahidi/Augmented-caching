#!/bin/bash
#
# Dataset-generation helper only — not a full heavy_r1 train/eval/manuscript pipeline.
# KBS canonical path: docs/wulver_heavy_evict_value_experiment.md
#
#SBATCH --job-name=evictv1-dsgen
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

mkdir -p slurm/logs

# ---------------------------
# Environment setup (EDIT ME)
# ---------------------------
# Option A:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <YOUR_ENV_NAME>
#
# Option B:
# module load python/3.10

# Ensure local package imports work even without pip install -e.
export PYTHONPATH="${PYTHONPATH:-src}"

# ---------------------------
# Configurable arguments
# ---------------------------
OUT_DIR="${OUT_DIR:-data/derived/evict_value_v1_wulver}"
TRACE_MANIFEST="${TRACE_MANIFEST:-}"
TRACE_GLOB="${TRACE_GLOB:-data/processed/*/trace.jsonl}"
CAPACITIES="${CAPACITIES:-64,128,256}"
HORIZONS="${HORIZONS:-8,16,32}"
SPLIT_MODE="${SPLIT_MODE:-trace_chunk}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-500000}"
HISTORY_WINDOW="${HISTORY_WINDOW:-64}"
SPLIT_SEED="${SPLIT_SEED:-0}"
SPLIT_TRAIN_PCT="${SPLIT_TRAIN_PCT:-70}"
SPLIT_VAL_PCT="${SPLIT_VAL_PCT:-15}"
MIN_REQUESTS_PER_TRACE="${MIN_REQUESTS_PER_TRACE:-1000}"
MIN_UNIQUE_PAGES_PER_TRACE="${MIN_UNIQUE_PAGES_PER_TRACE:-100}"

CMD=(
  python scripts/build_evict_value_dataset_wulver_v1.py
  --trace-glob "${TRACE_GLOB}"
  --capacities "${CAPACITIES}"
  --horizons "${HORIZONS}"
  --split-mode "${SPLIT_MODE}"
  --chunk-size "${CHUNK_SIZE}"
  --max-rows-per-shard "${ROWS_PER_SHARD}"
  --history-window "${HISTORY_WINDOW}"
  --split-seed "${SPLIT_SEED}"
  --split-train-pct "${SPLIT_TRAIN_PCT}"
  --split-val-pct "${SPLIT_VAL_PCT}"
  --min-requests-per-trace "${MIN_REQUESTS_PER_TRACE}"
  --min-unique-pages-per-trace "${MIN_UNIQUE_PAGES_PER_TRACE}"
  --out-dir "${OUT_DIR}"
)

if [[ -n "${TRACE_MANIFEST}" ]]; then
  CMD+=(--trace-manifest "${TRACE_MANIFEST}")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
