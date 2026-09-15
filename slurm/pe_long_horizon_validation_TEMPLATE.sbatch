#!/bin/bash
#
# PE long-horizon VALIDATION array template -- DO NOT SUBMIT AS-IS.
#
# One-to-one with the 20 production tasks (array indices match the
# production array exactly, since Plan A's decomposition permits a direct
# 1:1 mapping -- no adaptation needed for a 60-task decomposition since
# production is NOT using one).
#
# Intended invocation (NOT executed here):
#   PROD_JOBID=$(sbatch --parsable slurm/pe_long_horizon_production_TEMPLATE.sbatch)
#   VALIDATE_JOBID=$(sbatch --parsable --dependency=afterok:${PROD_JOBID} \
#       slurm/pe_long_horizon_validation_TEMPLATE.sbatch)
#
#SBATCH --job-name=pe-long-horizon-validate
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=__MEM__
#SBATCH --time=00:30:00
#SBATCH --array=__ARRAY_SPEC__
#SBATCH --output=slurm/logs/pe-long-horizon-validate-%A_%a.out
#SBATCH --error=slurm/logs/pe-long-horizon-validate-%A_%a.err

set -euo pipefail

cd ~/lafc-work/Augmented-caching
source .venv/bin/activate

MANIFEST="configs/pe_long_horizon_production/manifest.json"
EXPECTED_SHA="3a04a25094de1e8c57a603e2662b321bcf1d9e88"

TASK_JSON="$(python3 -c "
import json
m = json.load(open('${MANIFEST}'))
idx = ${SLURM_ARRAY_TASK_ID}
print(json.dumps([t for t in m['tasks'] if t['array_index'] == idx][0]))
")"
FAMILY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['family'])" "${TASK_JSON}")"
CAPACITY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['capacity'])" "${TASK_JSON}")"
OUT_DIR="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['out_dir'])" "${TASK_JSON}")"

python scripts/pe_long_horizon/validate_cell.py "${OUT_DIR}" \
  --family "${FAMILY}" --capacity "${CAPACITY}" --horizons 32,64,128 \
  --expected-sha "${EXPECTED_SHA}"
