#!/bin/bash
#SBATCH --job-name=pairwise-pub-agg
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Aggregate pairwise publishability campaign artifacts.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis/pairwise_publishability_campaign
export PYTHONPATH="${PYTHONPATH:-src}"

python scripts/aggregate_pairwise_publishability_campaign.py \
  --jobs-root analysis/pairwise_publishability_campaign/jobs \
  --out-dir analysis/pairwise_publishability_campaign
