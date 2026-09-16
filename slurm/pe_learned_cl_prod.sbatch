#!/bin/bash
# Frozen publication-grade held-out closed-loop production array.
# Cluster settings follow the successful PE Wulver campaign.
#SBATCH --job-name=pe-learned-cl-prod
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --array=0-249%20
#SBATCH --output=slurm/logs/%x-%A_%a.out
#SBATCH --error=slurm/logs/%x-%A_%a.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate 2>/dev/null || true
export PYTHONPATH="${PWD}/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MODEL="${MODEL_PATH:-/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/model/pe_evict_value_h16_hist_gb_attempt2_20260915.pkl}"
OUT="${OUT_ROOT:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916}"
mkdir -p slurm/logs "${OUT}"
sha256sum "${MODEL}" | grep -F '8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6'
python scripts/pe_learned_closed_loop.py run-task --manifest analysis/pe_publication_learned_closed_loop_20260916/campaign_manifest.json --task-id "${SLURM_ARRAY_TASK_ID}" --model "${MODEL}" --out-root "${OUT}"
