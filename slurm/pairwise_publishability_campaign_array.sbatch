#!/bin/bash
#SBATCH --job-name=pairwise-publish
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=20:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#
# Pairwise publishability campaign array:
# trains/evaluates multiple pairwise supervision variants and model families.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis/pairwise_publishability_campaign/jobs
export PYTHONPATH="${PYTHONPATH:-src}"

EXP_TAG="${EXP_TAG:-pairwise_pub_r1}"
BASE_SEED="${BASE_SEED:-3000}"
TRACE_CHUNK="${TRACE_CHUNK:-3}"
MAX_REQ="${MAX_REQ:-40000}"
CAPACITIES="${CAPACITIES:-32,64,128}"
HORIZONS="${HORIZONS:-4,8,16}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

# Grid mapping:
# task%3 -> model_family
# (task/3)%2 -> label_variant
# (task/6)%3 -> label_noise in {0.0,0.1,0.2}
# (task/18)%3 -> trace_start chunk index over full manifest
MODEL_IX=$(( TASK % 3 ))
VAR_IX=$(( (TASK / 3) % 2 ))
NOISE_IX=$(( (TASK / 6) % 3 ))
CHUNK_IX=$(( (TASK / 18) % 3 ))

if (( MODEL_IX == 0 )); then MODEL="logistic"; elif (( MODEL_IX == 1 )); then MODEL="rf"; else MODEL="histgb"; fi
if (( VAR_IX == 0 )); then LABEL="head_pair"; else LABEL="regret_diff"; fi
if (( NOISE_IX == 0 )); then NOISE="0.0"; elif (( NOISE_IX == 1 )); then NOISE="0.1"; else NOISE="0.2"; fi
TRACE_START=$(( CHUNK_IX * TRACE_CHUNK ))

SEED=$(( BASE_SEED + TASK * 101 ))
JOB_LABEL="${EXP_TAG}_t${TASK}_${MODEL}_${LABEL}_n${NOISE}_s${TRACE_START}"

python scripts/run_pairwise_publishability_campaign_task.py \
  --manifest analysis/wulver_trace_manifest_full.csv \
  --trace-start "${TRACE_START}" \
  --trace-count "${TRACE_CHUNK}" \
  --capacities "${CAPACITIES}" \
  --horizons "${HORIZONS}" \
  --max-requests-per-trace "${MAX_REQ}" \
  --seed "${SEED}" \
  --model-family "${MODEL}" \
  --label-variant "${LABEL}" \
  --label-noise "${NOISE}" \
  --job-label "${JOB_LABEL}" \
  --out-root analysis/pairwise_publishability_campaign/jobs
