#!/bin/bash
#SBATCH --job-name=pe-learned-cl-validate
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
# Use the same isolated sklearn-1.8.0 venv as the production array (see
# analysis/pe_publication_learned_closed_loop_20260916/WULVER_RUNTIME_ENVIRONMENT.json)
PE_LEARNED_CL_VENV="${PE_LEARNED_CL_VENV:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916/env/sklearn180}"
source "${PE_LEARNED_CL_VENV}/bin/activate"
export PYTHONPATH="${PWD}/src"
python scripts/pe_learned_closed_loop.py validate --manifest analysis/pe_publication_learned_closed_loop_20260916/campaign_manifest.json --out-root "${OUT_ROOT:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916}" --report "${OUT_ROOT:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916}/validation_report.json"
