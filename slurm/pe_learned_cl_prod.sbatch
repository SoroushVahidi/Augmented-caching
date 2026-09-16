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
# The frozen model was pickled under scikit-learn 1.8.0; the shared
# /mmfs1/home/sv96/lafc-work/Augmented-caching/.venv runs 1.9.0 and cannot
# unpickle it (sklearn 1.9 removed _hist_gradient_boosting._loss). Use the
# campaign-scoped, isolated venv pinned to 1.8.0 instead. See
# analysis/pe_publication_learned_closed_loop_20260916/WULVER_RUNTIME_ENVIRONMENT.json
PE_LEARNED_CL_VENV="${PE_LEARNED_CL_VENV:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916/env/sklearn180}"
source "${PE_LEARNED_CL_VENV}/bin/activate"
python3 -c "import sklearn; assert sklearn.__version__ == '1.8.0', sklearn.__version__"
export PYTHONPATH="${PWD}/src"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MODEL="${MODEL_PATH:-/mmfs1/project/ikoutis/sv96/lafc-evict/pe_publication_learned_model_attempt2_20260916/model/pe_evict_value_h16_hist_gb_attempt2_20260915.pkl}"
OUT="${OUT_ROOT:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_publication_learned_closed_loop_20260916}"
mkdir -p slurm/logs "${OUT}"
sha256sum "${MODEL}" | grep -F '8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6'
python scripts/pe_learned_closed_loop.py run-task --manifest analysis/pe_publication_learned_closed_loop_20260916/campaign_manifest.json --task-id "${SLURM_ARRAY_TASK_ID}" --model "${MODEL}" --out-root "${OUT}"
