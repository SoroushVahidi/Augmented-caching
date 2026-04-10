#!/bin/bash
#SBATCH --job-name=pairwise-chain-smoke
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Smoke run for theorem validation pipeline wiring.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis/pairwise_chain_witness_campaign/jobs
export PYTHONPATH="${PYTHONPATH:-src}"

EXP_TAG="${EXP_TAG:-campaign_smoke}"
JOB_LABEL="${EXP_TAG}_task0_smoke"

python scripts/run_pairwise_chain_witness_theorem_checks.py \
  --mode random \
  --seed 4242 \
  --job-label "${JOB_LABEL}" \
  --out-dir analysis/pairwise_chain_witness_campaign/jobs \
  --max-instances 300 \
  --max-len 9
