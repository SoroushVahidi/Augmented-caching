#!/bin/bash
#
# KBS canonical Wulver train stage (EXP_TAG=heavy_r1). Pair with heavy_eval, then
# scripts/paper/build_kbs_main_manuscript_artifacts.py — docs/kbs_manuscript_submission_index.md
#
#SBATCH --job-name=evictv1-heavy-train
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Heavy Wulver training stage for artifact-backed evict_value_v1.
# Phase order: dataset build -> dataset summary -> model training.
#
# Optional environment overrides:
#   EXP_TAG=heavy_r1
#   TRACE_MANIFEST=analysis/wulver_trace_manifest_full.csv
#   CAPACITIES=32,64,128,256
#   HORIZONS=4,8,16
#   MAX_REQUESTS_PER_TRACE=50000
#   CHUNK_SIZE=4096
#   MAX_ROWS_PER_SHARD=500000
#   MAX_TRAIN_ROWS=800000
#   MAX_VAL_ROWS=250000
#   MAX_TEST_ROWS=250000
#   BUILD_OVERWRITE=1
#   SPLIT_SEED=7
#
# Environment setup (if needed):
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate <YOUR_ENV_NAME>

set -euo pipefail

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis models data/derived
export PYTHONPATH="${PYTHONPATH:-src}"

EXP_TAG="${EXP_TAG:-heavy_r1}"
TRACE_MANIFEST="${TRACE_MANIFEST:-analysis/wulver_trace_manifest_full.csv}"
CAPACITIES="${CAPACITIES:-32,64,128,256}"
HORIZONS="${HORIZONS:-4,8,16}"
MAX_REQUESTS_PER_TRACE="${MAX_REQUESTS_PER_TRACE:-50000}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
MAX_ROWS_PER_SHARD="${MAX_ROWS_PER_SHARD:-500000}"
SPLIT_SEED="${SPLIT_SEED:-7}"

OUT_DS="data/derived/evict_value_v1_wulver_${EXP_TAG}"
TRAIN_METRICS="analysis/evict_value_wulver_v1_train_metrics_${EXP_TAG}.json"
MODEL_CMP="analysis/evict_value_wulver_v1_model_comparison_${EXP_TAG}.csv"
BEST_CFG="analysis/evict_value_wulver_v1_best_config_${EXP_TAG}.json"
SUMMARY_MD="analysis/evict_value_v1_wulver_dataset_summary_${EXP_TAG}.md"
SUMMARY_JSON="${OUT_DS}/dataset_summary_extended_${EXP_TAG}.json"
BEST_MODEL="models/evict_value_wulver_v1_best_${EXP_TAG}.pkl"

if [[ ! -f "${TRACE_MANIFEST}" ]]; then
  log "ERROR: missing trace manifest: ${TRACE_MANIFEST}"
  exit 2
fi

log "=== Wulver heavy training start ==="
log "job_id=${SLURM_JOB_ID:-?} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
log "exp_tag=${EXP_TAG} trace_manifest=${TRACE_MANIFEST} capacities=${CAPACITIES} horizons=${HORIZONS}"

BUILD_ARGS=(
  python scripts/build_evict_value_dataset_wulver_v1.py
  --trace-manifest "${TRACE_MANIFEST}"
  --capacities "${CAPACITIES}"
  --horizons "${HORIZONS}"
  --split-mode trace_chunk
  --chunk-size "${CHUNK_SIZE}"
  --split-seed "${SPLIT_SEED}"
  --max-rows-per-shard "${MAX_ROWS_PER_SHARD}"
  --max-requests-per-trace "${MAX_REQUESTS_PER_TRACE}"
  --out-dir "${OUT_DS}"
)
if [[ "${BUILD_OVERWRITE:-0}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
fi
log "Running dataset build: ${BUILD_ARGS[*]}"
"${BUILD_ARGS[@]}"
test -f "${OUT_DS}/manifest.json" || { log "ERROR: missing ${OUT_DS}/manifest.json"; exit 3; }

log "Running dataset summary"
python scripts/summarize_wulver_evict_value_dataset.py \
  --manifest "${OUT_DS}/manifest.json" \
  --out-json "${SUMMARY_JSON}" \
  --out-md "${SUMMARY_MD}"

log "Running training"
python scripts/train_evict_value_wulver_v1.py \
  --manifest "${OUT_DS}/manifest.json" \
  --horizons "${HORIZONS}" \
  --max-train-rows "${MAX_TRAIN_ROWS:-800000}" \
  --max-val-rows "${MAX_VAL_ROWS:-250000}" \
  --max-test-rows "${MAX_TEST_ROWS:-250000}" \
  --models-dir models \
  --metrics-json "${TRAIN_METRICS}" \
  --comparison-csv "${MODEL_CMP}" \
  --best-config-json "${BEST_CFG}"

test -f "${TRAIN_METRICS}" || { log "ERROR: missing ${TRAIN_METRICS}"; exit 4; }
test -f "${BEST_CFG}" || { log "ERROR: missing ${BEST_CFG}"; exit 4; }
test -f "models/evict_value_wulver_v1_best.pkl" || { log "ERROR: missing models/evict_value_wulver_v1_best.pkl"; exit 4; }
cp "models/evict_value_wulver_v1_best.pkl" "${BEST_MODEL}"

log "=== Wulver heavy training complete ==="
log "Artifacts:"
log "- dataset manifest: ${OUT_DS}/manifest.json"
log "- split summary: ${OUT_DS}/split_summary.csv"
log "- dataset summary: ${SUMMARY_JSON}"
log "- training metrics: ${TRAIN_METRICS}"
log "- model comparison: ${MODEL_CMP}"
log "- best config: ${BEST_CFG}"
log "- best model copy: ${BEST_MODEL}"
