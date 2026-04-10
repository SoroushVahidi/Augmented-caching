#!/bin/bash
#
# NOT the KBS canonical heavy_r1 pipeline. For the submission-critical Wulver line
# (heavy_r1 filenames + build_kbs inputs), use evict_value_v1_wulver_heavy_train.sbatch
# and evict_value_v1_wulver_heavy_eval.sbatch — see docs/kbs_manuscript_submission_index.md
#
#SBATCH --job-name=evictv1-multi
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Full multi-family Wulver validation for evict_value_v1 (CPU-only).
# No GPU: dataset generation + sklearn Ridge/RF/HistGB + policy simulators are CPU-bound.
#
# Environment overrides (optional):
#   TRACE_MANIFEST, OUT_DS, CAPACITIES, HORIZONS, CHUNK_SIZE, MAX_ROWS_PER_SHARD
#   MAX_REQUESTS_PER_TRACE — optional cap per trace (omit for full 50k rows)
#   BUILD_OVERWRITE=1 — pass --overwrite to dataset builder (e.g. after a partial/cancelled build)
#   MAX_TRAIN_ROWS, MAX_VAL_ROWS, MAX_TEST_ROWS — training subsample caps
#   POLICY_CAPS, POLICY_MAX_REQ — policy grid (default: 32,64,128 and 50000)
#
# Wiki2018 trace is a pageview-derived proxy, not a CDN object trace — interpret accordingly.

set -euo pipefail

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs models analysis

# EDIT: conda/module if your Python deps are not on the default PATH
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>

export PYTHONPATH="${PYTHONPATH:-src}"

log "=== evict_value_v1 Wulver multi-phase job start ==="
log "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem_gb=${SLURM_MEM_PER_NODE:-?} job_id=${SLURM_JOB_ID:-?}"
log "PWD=$(pwd)"

OUT_DS="${OUT_DS:-data/derived/evict_value_v1_wulver_multi_full}"
TRACE_MANIFEST="${TRACE_MANIFEST:-analysis/wulver_trace_manifest_full.csv}"
CAPACITIES="${CAPACITIES:-32,64,128}"
HORIZONS="${HORIZONS:-4,8,16}"

if [[ ! -f "${TRACE_MANIFEST}" ]]; then
  log "ERROR: trace manifest missing: ${TRACE_MANIFEST}"
  exit 2
fi
log "Using TRACE_MANIFEST=${TRACE_MANIFEST} (first 3 data lines):"
head -n 4 "${TRACE_MANIFEST}" | sed 's/^/[manifest] /' || true

# --- Optional: ml_gate artifacts for policy comparison (tabular; same Python env) ---
if [[ "${SKIP_ML_GATE_TRAIN:-0}" != "1" ]]; then
  if [[ ! -f models/ml_gate_v1.pkl ]] || [[ ! -f models/ml_gate_v2_random_forest.pkl ]]; then
    log "=== Phase ml_gate: training missing gate models (quick) ==="
    [[ -f data/derived/ml_gate_train.csv ]] || { log "WARN: data/derived/ml_gate_train.csv missing; ml_gate policies may be skipped by run_policy_comparison."; }
    [[ -f models/ml_gate_v1.pkl ]] || python scripts/train_ml_gate_v1.py
    [[ -f models/ml_gate_v2_random_forest.pkl ]] || python scripts/train_ml_gate_v2.py
    log "=== Phase ml_gate: done ==="
  else
    log "=== Phase ml_gate: skipped (models already present) ==="
  fi
else
  log "=== Phase ml_gate: skipped (SKIP_ML_GATE_TRAIN=1) ==="
fi

# --- Phase 1: candidate dataset (full 7-family manifest) ---
log "=== Phase 1: dataset generation START ==="
log "OUT_DS=${OUT_DS} capacities=${CAPACITIES} horizons=${HORIZONS}"

BUILD_ARGS=(
  python scripts/build_evict_value_dataset_wulver_v1.py
  --trace-manifest "${TRACE_MANIFEST}"
  --capacities "${CAPACITIES}"
  --horizons "${HORIZONS}"
  --split-mode trace_chunk
  --chunk-size "${CHUNK_SIZE:-4096}"
  --max-rows-per-shard "${MAX_ROWS_PER_SHARD:-500000}"
  --out-dir "${OUT_DS}"
)
if [[ -n "${MAX_REQUESTS_PER_TRACE:-}" ]]; then
  BUILD_ARGS+=(--max-requests-per-trace "${MAX_REQUESTS_PER_TRACE}")
fi
if [[ "${BUILD_OVERWRITE:-0}" == "1" ]]; then
  BUILD_ARGS+=(--overwrite)
  log "BUILD_OVERWRITE=1: using --overwrite on dataset builder"
fi

log "Running: ${BUILD_ARGS[*]}"
"${BUILD_ARGS[@]}"
log "=== Phase 1: dataset generation COMPLETE ==="
test -f "${OUT_DS}/manifest.json" || { log "ERROR: missing ${OUT_DS}/manifest.json"; exit 3; }
test -f "${OUT_DS}/split_summary.csv" || { log "ERROR: missing ${OUT_DS}/split_summary.csv"; exit 3; }
log "Checkpoint: $(wc -l < "${OUT_DS}/split_summary.csv") lines in split_summary.csv; shard_count=$(python -c "import json;print(json.load(open('${OUT_DS}/manifest.json'))['shard_count'])")"

# --- Phase 1b: extended summary (rows by split/family/capacity/horizon) ---
log "=== Phase 1b: summarize_wulver_evict_value_dataset START ==="
python scripts/summarize_wulver_evict_value_dataset.py \
  --manifest "${OUT_DS}/manifest.json" \
  --out-json "${OUT_DS}/dataset_summary_extended.json" \
  --out-md analysis/evict_value_v1_wulver_dataset_summary.md
log "=== Phase 1b: summarize COMPLETE ==="
log "Checkpoint: wrote ${OUT_DS}/dataset_summary_extended.json and analysis/evict_value_v1_wulver_dataset_summary.md"

# --- Phase 2: train Ridge / RF / HistGB ---
log "=== Phase 2: training START ==="
python scripts/train_evict_value_wulver_v1.py \
  --manifest "${OUT_DS}/manifest.json" \
  --horizons 4,8,16 \
  --max-train-rows "${MAX_TRAIN_ROWS:-400000}" \
  --max-val-rows "${MAX_VAL_ROWS:-100000}" \
  --max-test-rows "${MAX_TEST_ROWS:-100000}" \
  --models-dir models \
  --metrics-json analysis/evict_value_wulver_v1_train_metrics.json \
  --comparison-csv analysis/evict_value_wulver_v1_model_comparison.csv \
  --best-config-json analysis/evict_value_wulver_v1_best_config.json
log "=== Phase 2: training COMPLETE ==="
test -f analysis/evict_value_wulver_v1_train_metrics.json || exit 4
test -f models/evict_value_wulver_v1_best.pkl || exit 4
log "Checkpoint: best model models/evict_value_wulver_v1_best.pkl"

# --- Phase 3: policy comparison (all policies; ml_gate omitted if pkl missing) ---
log "=== Phase 3: policy comparison START ==="
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest "${TRACE_MANIFEST}" \
  --capacities "${POLICY_CAPS:-32,64,128}" \
  --max-requests-per-trace "${POLICY_MAX_REQ:-50000}" \
  --evict-value-model models/evict_value_wulver_v1_best.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison.md
log "=== Phase 3: policy comparison COMPLETE ==="
test -f analysis/evict_value_wulver_v1_policy_comparison.csv || exit 5

log "=== ALL PHASES DONE ==="
log "Artifacts: ${OUT_DS}/manifest.json, ${OUT_DS}/dataset_summary_extended.json"
log "Artifacts: analysis/evict_value_wulver_v1_train_metrics.json, analysis/evict_value_wulver_v1_model_comparison.csv"
log "Artifacts: analysis/evict_value_wulver_v1_policy_comparison.csv, analysis/evict_value_wulver_v1_policy_comparison.md"
log "Slurm stdout/stderr: slurm/logs/evictv1-multi-${SLURM_JOB_ID}.out / .err"
