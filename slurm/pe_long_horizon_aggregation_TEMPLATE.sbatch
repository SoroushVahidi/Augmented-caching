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

# NOTE: --h16-summary is intentionally omitted. No converter from the
# existing canonical H16 artifact (tab_dataset_characterization_heavy_r1_pre_eval.csv
# / raw candidate-row shards) into the {cells: {...}} JSON format aggregate.py
# expects has been built yet -- this is a known, flagged gap (see final
# launch-gate report), not a silent omission. aggregate.py's
# load_h16_canonical() gracefully no-ops (returns {}) without this flag, so
# Table A/B (the core H=32/64/128 scientific deliverable) are unaffected;
# only Table C's H16-delta columns will be None until that converter exists.
python scripts/pe_long_horizon/aggregate.py \
  --manifest configs/pe_long_horizon_production/manifest.json
