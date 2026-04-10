#!/bin/bash
#SBATCH --job-name=pairwise-pub-smoke
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Smoke run for pairwise publishability campaign.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis/pairwise_publishability_campaign/jobs
export PYTHONPATH="${PYTHONPATH:-src}"

python scripts/run_pairwise_publishability_campaign_task.py \
  --manifest analysis/wulver_trace_manifest_full.csv \
  --trace-start 0 \
  --trace-count 2 \
  --capacities "32,64" \
  --horizons "4,8" \
  --max-requests-per-trace 8000 \
  --seed 4242 \
  --model-family logistic \
  --label-variant head_pair \
  --label-noise 0.1 \
  --job-label pairwise_pub_smoke_task0 \
  --out-root analysis/pairwise_publishability_campaign/jobs
