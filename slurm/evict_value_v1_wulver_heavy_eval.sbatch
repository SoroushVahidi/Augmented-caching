#!/bin/bash
#
# KBS canonical Wulver eval stage — writes analysis/evict_value_wulver_v1_policy_comparison_${EXP_TAG}.csv
# (use EXP_TAG=heavy_r1). docs/kbs_manuscript_submission_index.md
#
#SBATCH --job-name=evictv1-heavy-eval
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#
# Heavy Wulver evaluation stage for artifact-backed evict_value_v1.
# Compares the trained model against strong baselines on the full trace-family manifest.
#
# Optional environment overrides:
#   EXP_TAG=heavy_r1
#   TRACE_MANIFEST=analysis/wulver_trace_manifest_full.csv
#   EVAL_CAPACITIES=32,64,128,256
#   EVAL_MAX_REQUESTS=50000
#   EVICT_VALUE_MODEL=models/evict_value_wulver_v1_best_heavy_r1.pkl
#   BASELINE_POLICIES=lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1
#
# Environment setup (if needed):
#   source ~/miniconda3/etc/profile.d/conda.sh
#   conda activate <YOUR_ENV_NAME>

set -euo pipefail

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*"; }

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs analysis
export PYTHONPATH="${PYTHONPATH:-src}"

EXP_TAG="${EXP_TAG:-heavy_r1}"
TRACE_MANIFEST="${TRACE_MANIFEST:-analysis/wulver_trace_manifest_full.csv}"
EVAL_CAPACITIES="${EVAL_CAPACITIES:-32,64,128,256}"
EVAL_MAX_REQUESTS="${EVAL_MAX_REQUESTS:-50000}"
EVICT_VALUE_MODEL="${EVICT_VALUE_MODEL:-models/evict_value_wulver_v1_best_${EXP_TAG}.pkl}"
BASELINE_POLICIES="${BASELINE_POLICIES:-lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1}"
OUT_CSV="analysis/evict_value_wulver_v1_policy_comparison_${EXP_TAG}.csv"
OUT_MD="analysis/evict_value_wulver_v1_policy_comparison_${EXP_TAG}.md"

if [[ ! -f "${TRACE_MANIFEST}" ]]; then
  log "ERROR: missing trace manifest: ${TRACE_MANIFEST}"
  exit 2
fi
if [[ ! -f "${EVICT_VALUE_MODEL}" ]]; then
  log "ERROR: missing evict_value model: ${EVICT_VALUE_MODEL}"
  exit 2
fi

log "=== Wulver heavy evaluation start ==="
log "job_id=${SLURM_JOB_ID:-?} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} mem=${SLURM_MEM_PER_NODE:-?}"
log "exp_tag=${EXP_TAG} capacities=${EVAL_CAPACITIES} max_requests=${EVAL_MAX_REQUESTS}"
log "policies=${BASELINE_POLICIES}"

python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest "${TRACE_MANIFEST}" \
  --capacities "${EVAL_CAPACITIES}" \
  --max-requests-per-trace "${EVAL_MAX_REQUESTS}" \
  --policies "${BASELINE_POLICIES}" \
  --evict-value-model "${EVICT_VALUE_MODEL}" \
  --out-csv "${OUT_CSV}" \
  --out-md "${OUT_MD}"

test -f "${OUT_CSV}" || { log "ERROR: missing ${OUT_CSV}"; exit 3; }
test -f "${OUT_MD}" || { log "ERROR: missing ${OUT_MD}"; exit 3; }

log "=== Wulver heavy evaluation complete ==="
log "- policy comparison csv: ${OUT_CSV}"
log "- policy comparison md: ${OUT_MD}"
