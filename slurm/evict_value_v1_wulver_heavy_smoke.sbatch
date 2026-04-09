#!/bin/bash
#SBATCH --job-name=evictv1-heavy-smoke
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Quick smoke check for the heavy Wulver package.
# Runs a tiny version of dataset build -> train -> eval to validate wiring.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis models data/derived
export PYTHONPATH="${PYTHONPATH:-src}"

EXP_TAG="${EXP_TAG:-heavy_smoke}"
TRACE_MANIFEST="${TRACE_MANIFEST:-analysis/wulver_trace_manifest_full.csv}"
OUT_DS="data/derived/evict_value_v1_wulver_${EXP_TAG}"
BEST_MODEL="models/evict_value_wulver_v1_best_${EXP_TAG}.pkl"

python scripts/build_evict_value_dataset_wulver_v1.py \
  --trace-manifest "${TRACE_MANIFEST}" \
  --max-traces 2 \
  --capacities "32,64" \
  --horizons "4,8" \
  --max-requests-per-trace 5000 \
  --max-rows-per-shard 100000 \
  --split-mode trace_chunk \
  --chunk-size 2048 \
  --split-seed 7 \
  --out-dir "${OUT_DS}"

python scripts/train_evict_value_wulver_v1.py \
  --manifest "${OUT_DS}/manifest.json" \
  --horizons "4,8" \
  --max-train-rows 100000 \
  --max-val-rows 30000 \
  --max-test-rows 30000 \
  --models-dir models \
  --metrics-json "analysis/evict_value_wulver_v1_train_metrics_${EXP_TAG}.json" \
  --comparison-csv "analysis/evict_value_wulver_v1_model_comparison_${EXP_TAG}.csv" \
  --best-config-json "analysis/evict_value_wulver_v1_best_config_${EXP_TAG}.json"

cp "models/evict_value_wulver_v1_best.pkl" "${BEST_MODEL}"

python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest "${TRACE_MANIFEST}" \
  --max-traces 2 \
  --capacities "32,64" \
  --max-requests-per-trace 5000 \
  --policies "lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1" \
  --evict-value-model "${BEST_MODEL}" \
  --out-csv "analysis/evict_value_wulver_v1_policy_comparison_${EXP_TAG}.csv" \
  --out-md "analysis/evict_value_wulver_v1_policy_comparison_${EXP_TAG}.md"
