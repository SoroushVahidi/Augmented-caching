#!/bin/bash
#
# Aggregation for corrected PE long-horizon canonical-regeneration output.
#
#SBATCH --job-name=pe-lh-canregen-agg
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/pe-lh-canregen-agg-%j.out
#SBATCH --error=slurm/logs/pe-lh-canregen-agg-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs
if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi
export PYTHONPATH=src

python scripts/pe_long_horizon/aggregate.py \
  --manifest configs/pe_long_horizon_canonical_regen_20260916/manifest.json
