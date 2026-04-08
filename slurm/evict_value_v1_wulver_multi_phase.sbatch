#!/bin/bash
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

set -euo pipefail
mkdir -p slurm/logs

# EDIT: conda/module
# source ~/miniconda3/etc/profile.d/conda.sh && conda activate <env>

export PYTHONPATH="${PYTHONPATH:-src}"

# --- Phase 1: candidate dataset (full 7-family manifest) ---
# Capacities: 32,64,128 — small enough to stay tractable; wide enough for multi-scale signal.
# Horizons: 4,8,16 — aligned with evict_value_v1 labeling.
OUT_DS="${OUT_DS:-data/derived/evict_value_v1_wulver_multi_full}"
TRACE_MANIFEST="${TRACE_MANIFEST:-analysis/wulver_trace_manifest_full.csv}"

BUILD_ARGS=(
  python scripts/build_evict_value_dataset_wulver_v1.py
  --trace-manifest "${TRACE_MANIFEST}"
  --capacities "${CAPACITIES:-32,64,128}"
  --horizons "${HORIZONS:-4,8,16}"
  --split-mode trace_chunk
  --chunk-size "${CHUNK_SIZE:-4096}"
  --max-rows-per-shard "${MAX_ROWS_PER_SHARD:-500000}"
  --out-dir "${OUT_DS}"
)
if [[ -n "${MAX_REQUESTS_PER_TRACE:-}" ]]; then
  BUILD_ARGS+=(--max-requests-per-trace "${MAX_REQUESTS_PER_TRACE}")
fi
"${BUILD_ARGS[@]}"

python scripts/summarize_wulver_evict_value_dataset.py \
  --manifest "${OUT_DS}/manifest.json" \
  --out-json "${OUT_DS}/dataset_summary_extended.json" \
  --out-md analysis/evict_value_v1_wulver_dataset_summary.md

# --- Phase 2: train (adjust row caps if needed for RAM) ---
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

# --- Phase 3: policy comparison (requires ml_gate *.pkl if you want those policies) ---
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest "${TRACE_MANIFEST}" \
  --capacities "${POLICY_CAPS:-64}" \
  --max-requests-per-trace "${POLICY_MAX_REQ:-50000}" \
  --evict-value-model models/evict_value_wulver_v1_best.pkl

echo "Done multi-phase Wulver evict_value_v1 pipeline."
