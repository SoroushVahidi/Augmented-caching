#!/bin/bash
#
# CAP256 long-horizon timing PROBE — single family (cloudphysics/alibaba-block),
# NOT production. Purpose: bound the capacity-scaling term (the one dimension
# job 1287838's cap32 preflight did not measure) before designing the full
# production campaign.
#
# Family selection: cloudphysics tied wiki2018 for most decisions at cap32
# (49,241 vs 49,968, measured from job 1287838's split_summary.csv) but
# wiki2018 is a documented degenerate/negative-control workload (every
# request is a miss); cloudphysics is a realistic, non-degenerate trace at
# essentially the same decision volume, so it is the conservative,
# representative choice per the tie-breaker rule (default to cloudphysics
# when evidence does not clearly separate the candidates).
#
# Uses --trace-manifest (not --trace-glob) deliberately: the first preflight's
# --trace-glob was silently appended to the argparse default
# ["data/processed/*/trace.jsonl"] (action="append" + non-empty default),
# causing it to process all 5 families instead of one. --trace-manifest
# bypasses that default entirely (parse_trace_manifest ignores fallback_globs
# whenever a manifest path is given), so this is the safe single-family path.
#
# Provenance:
#   Wulver checkout: ~/lafc-work/Augmented-caching, branch main, SHA 3a04a25094de1e8c57a603e2662b321bcf1d9e88
#   Local branch:    experiment/pe-long-horizon-preflight-20260915 (chore/repository-polish base ceb3670)
#   Generator: scripts/build_evict_value_dataset_wulver_v1.py (flat path, pre-reorg on this checkout;
#     byte-identical to local src/lafc/evict_value_wulver_v1.py)
#
#SBATCH --job-name=lafc-cap256-probe
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err

set -euo pipefail

mkdir -p slurm/logs

echo "[$(date -Is)] Starting cap256 long-horizon probe on $(hostname)"
echo "[provenance] wulver_checkout_sha=$(git rev-parse HEAD)"
echo "[provenance] wulver_checkout_branch=$(git rev-parse --abbrev-ref HEAD)"

cd ~/lafc-work/Augmented-caching

source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

OUT_DIR="data/derived/evict_value_v1_long_horizon_probe_cap256_cloudphysics_20260915"

# NEW output directory only -- never touches canonical H={4,8,16} data,
# job 1287838's cap32 preflight output, or any other existing artifact.

python scripts/build_evict_value_dataset_wulver_v1.py \
  --trace-manifest "slurm/trace_manifest_cap256_probe_cloudphysics.csv" \
  --capacities "256" \
  --horizons "16,32,64,128" \
  --split-mode trace_chunk \
  --chunk-size 4096 \
  --max-rows-per-shard 500000 \
  --split-seed 7 \
  --min-requests-per-trace 1000 \
  --min-unique-pages-per-trace 100 \
  --out-dir "${OUT_DIR}" \
  --overwrite

echo "[$(date -Is)] Dataset build finished successfully."
echo "[manifest]"
cat "${OUT_DIR}/manifest.json"
echo
echo "[split_summary]"
cat "${OUT_DIR}/split_summary.csv"
echo "[$(date -Is)] Cap256 probe job complete."
