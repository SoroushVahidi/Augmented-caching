#!/bin/bash
#SBATCH --job-name=common-v2
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-20
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs

export PYTHONPATH="${PYTHONPATH:-src}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

FAMILIES=(brightkite citibike cloudphysics metacdn metakv twemcache wiki2018)
CAPACITIES=(32 64 128)
TASK="${SLURM_ARRAY_TASK_ID:-0}"

if (( TASK < 0 || TASK > 20 )); then
  echo "Invalid SLURM_ARRAY_TASK_ID=${TASK}; expected 0..20" >&2
  exit 2
fi

FAMILY_INDEX=$(( TASK / 3 ))
CAPACITY_INDEX=$(( TASK % 3 ))
FAMILY="${FAMILIES[$FAMILY_INDEX]}"
CAPACITY="${CAPACITIES[$CAPACITY_INDEX]}"
OUT_ROOT="${OUT_ROOT:-analysis/common_model_objective_control_wulver_v2}"
UNIT_OUT="${OUT_ROOT}/units/${FAMILY}_cap${CAPACITY}"

echo "task=${TASK} family=${FAMILY} capacity=${CAPACITY} out=${UNIT_OUT}"

python scripts/experiments/run_common_model_objective_control_v2.py \
  --config configs/common_model_objective_control_v2.json \
  --family "${FAMILY}" \
  --capacity "${CAPACITY}" \
  --out "${UNIT_OUT}"
