#!/bin/bash
#SBATCH --job-name=pairwise-chain-camp
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
#
# High-computation theorem validation campaign worker array.
# Each array task runs one shard of exhaustive/random/structured/real checks.
#
# Optional env overrides:
#   EXP_TAG=campaign_r1
#   BASE_SEED=1000
#   MAX_INSTANCES=25000
#   MAX_LEN=11
#   REAL_PREFIX=30000
#   REAL_MAX_TRACES=7

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis/pairwise_chain_witness_campaign/jobs
export PYTHONPATH="${PYTHONPATH:-src}"

EXP_TAG="${EXP_TAG:-campaign_r1}"
BASE_SEED="${BASE_SEED:-1000}"
MAX_INSTANCES="${MAX_INSTANCES:-25000}"
MAX_LEN="${MAX_LEN:-11}"
REAL_PREFIX="${REAL_PREFIX:-30000}"
REAL_MAX_TRACES="${REAL_MAX_TRACES:-7}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

# 0..3 exhaustive, 4..15 random, 16..21 structured, 22..23 real
if (( TASK <= 3 )); then
  MODE="exhaustive"
  INSTANCES="${MAX_INSTANCES}"
elif (( TASK <= 15 )); then
  MODE="random"
  INSTANCES="${MAX_INSTANCES}"
elif (( TASK <= 21 )); then
  MODE="structured"
  INSTANCES="${MAX_INSTANCES}"
else
  MODE="real"
  INSTANCES="200"
fi

SEED=$(( BASE_SEED + TASK * 7919 ))
JOB_LABEL="${EXP_TAG}_task${TASK}_${MODE}"
OUT_DIR="analysis/pairwise_chain_witness_campaign/jobs"

echo "[pairwise-chain-campaign] task=${TASK} mode=${MODE} seed=${SEED} label=${JOB_LABEL}"

python scripts/run_pairwise_chain_witness_theorem_checks.py \
  --mode "${MODE}" \
  --seed "${SEED}" \
  --job-label "${JOB_LABEL}" \
  --out-dir "${OUT_DIR}" \
  --max-instances "${INSTANCES}" \
  --max-len "${MAX_LEN}" \
  --real-prefix "${REAL_PREFIX}" \
  --real-max-traces "${REAL_MAX_TRACES}"
