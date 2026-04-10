#!/bin/bash
#
# Sharded dataset build — not the KBS canonical heavy_r1 bundle by itself.
# See docs/kbs_manuscript_submission_index.md
#
#SBATCH --job-name=evictv1-dsarr
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

mkdir -p slurm/logs

# ---------------------------
# Environment setup (EDIT ME)
# ---------------------------
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate <YOUR_ENV_NAME>
export PYTHONPATH="${PYTHONPATH:-src}"

# ---------------------------
# Array sharding strategy
# ---------------------------
# Provide a CSV manifest with columns:
# path,trace_name,dataset_source,trace_family
# and split it externally into N slices:
# trace_manifest.part0.csv ... trace_manifest.partN.csv
MANIFEST_PREFIX="${MANIFEST_PREFIX:-trace_manifest.part}"
PART_FILE="${MANIFEST_PREFIX}${SLURM_ARRAY_TASK_ID}.csv"

if [[ ! -f "${PART_FILE}" ]]; then
  echo "Missing manifest part: ${PART_FILE}"
  exit 2
fi

OUT_ROOT="${OUT_ROOT:-data/derived/evict_value_v1_wulver_array}"
OUT_DIR="${OUT_ROOT}/task_${SLURM_ARRAY_TASK_ID}"
CAPACITIES="${CAPACITIES:-64,128,256}"
HORIZONS="${HORIZONS:-8,16,32}"
SPLIT_MODE="${SPLIT_MODE:-trace_chunk}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-400000}"
MIN_REQUESTS_PER_TRACE="${MIN_REQUESTS_PER_TRACE:-1000}"
MIN_UNIQUE_PAGES_PER_TRACE="${MIN_UNIQUE_PAGES_PER_TRACE:-100}"

python scripts/build_evict_value_dataset_wulver_v1.py \
  --trace-manifest "${PART_FILE}" \
  --capacities "${CAPACITIES}" \
  --horizons "${HORIZONS}" \
  --split-mode "${SPLIT_MODE}" \
  --chunk-size "${CHUNK_SIZE}" \
  --max-rows-per-shard "${ROWS_PER_SHARD}" \
  --min-requests-per-trace "${MIN_REQUESTS_PER_TRACE}" \
  --min-unique-pages-per-trace "${MIN_UNIQUE_PAGES_PER_TRACE}" \
  --out-dir "${OUT_DIR}"
