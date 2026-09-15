#!/bin/bash
#SBATCH --job-name=lafc-long-horizon-validate
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

mkdir -p slurm/logs
cd ~/lafc-work/Augmented-caching

source .venv/bin/activate

OUT_DIR="data/derived/evict_value_v1_long_horizon_preflight_metacdn_20260915"

python scripts/validate_long_horizon_preflight.py "${OUT_DIR}" --horizons 16,32,64,128
