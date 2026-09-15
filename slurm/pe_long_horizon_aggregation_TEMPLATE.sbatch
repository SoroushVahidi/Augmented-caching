#!/bin/bash
#
# PE long-horizon AGGREGATION job template -- DO NOT SUBMIT AS-IS.
#
# Single (non-array) job, gated on the validation array completing.
#
# Intended full DAG (NOT executed here):
#   PROD_JOBID=$(sbatch --parsable slurm/pe_long_horizon_production_TEMPLATE.sbatch)
#   VALIDATE_JOBID=$(sbatch --parsable --dependency=afterok:${PROD_JOBID} \
#       slurm/pe_long_horizon_validation_TEMPLATE.sbatch)
#   AGGREGATE_JOBID=$(sbatch --parsable --dependency=afterok:${VALIDATE_JOBID} \
#       slurm/pe_long_horizon_aggregation_TEMPLATE.sbatch)
#
# aggregate.py itself refuses to run to completion unless all 60 logical
# cells (20 tasks x 3 horizons) are present and PASS -- this job's failure
# mode IS that refusal, not a separate check.
#
#SBATCH --job-name=pe-long-horizon-aggregate
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=slurm/logs/pe-long-horizon-aggregate-%j.out
#SBATCH --error=slurm/logs/pe-long-horizon-aggregate-%j.err

set -euo pipefail

cd ~/lafc-work/Augmented-caching
source .venv/bin/activate

python scripts/pe_long_horizon/aggregate.py \
  --manifest configs/pe_long_horizon_production/manifest.json \
  --h16-summary configs/pe_long_horizon_production/h16_canonical_summary.json
