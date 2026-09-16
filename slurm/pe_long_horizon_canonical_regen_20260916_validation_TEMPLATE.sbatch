#!/bin/bash
#
# Validation for corrected PE long-horizon canonical-regeneration output.
#
#SBATCH --job-name=pe-lh-canregen-val
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --array=0-19%10
#SBATCH --output=slurm/logs/pe-lh-canregen-val-%A_%a.out
#SBATCH --error=slurm/logs/pe-lh-canregen-val-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs
source .venv/bin/activate
export PYTHONPATH=src

MANIFEST="configs/pe_long_horizon_canonical_regen_20260916/manifest.json"
TASK_JSON="$(python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
idx = int('${SLURM_ARRAY_TASK_ID}')
task = [t for t in m['tasks'] if t['array_index'] == idx]
if len(task) != 1:
    print(f'FATAL: expected one task for array index {idx}, found {len(task)}', file=sys.stderr)
    sys.exit(4)
print(json.dumps(task[0]))
")"
TASK_ID="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['task_id'])" "${TASK_JSON}")"
FAMILY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['family'])" "${TASK_JSON}")"
CAPACITY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['capacity'])" "${TASK_JSON}")"
OUT_DIR="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['out_dir'])" "${TASK_JSON}")"

python scripts/pe_long_horizon/validate_cell.py "${OUT_DIR}" \
  --family "${FAMILY}" --capacity "${CAPACITY}" --horizons 32,64,128
python scripts/pe_long_horizon_canonical_regen_20260916/validate_key_identity.py "${OUT_DIR}" \
  --manifest "${MANIFEST}" --task-id "${TASK_ID}"
