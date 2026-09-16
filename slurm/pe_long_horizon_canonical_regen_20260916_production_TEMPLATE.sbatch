#!/bin/bash
#
# Corrected PE long-horizon canonical-regeneration production array.
# One array task = one canonical H16 family x capacity cell; H32/H64/H128
# are generated together. H16 is never regenerated.
#
# Submit only after configs/pe_long_horizon_canonical_regen_20260916/
# provenance/preflight_report.json reports PASS.
#
#SBATCH --job-name=pe-lh-canregen-prod
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-19%10
#SBATCH --output=slurm/logs/pe-lh-canregen-prod-%A_%a.out
#SBATCH --error=slurm/logs/pe-lh-canregen-prod-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p slurm/logs
source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

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
TRACE_PATH="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['trace_path'])" "${TASK_JSON}")"
CANONICAL_MANIFEST_TRACE_PATH="$(python3 -c "import json,sys; print(json.loads(sys.argv[1]).get('canonical_manifest_trace_path', json.loads(sys.argv[1])['trace_path']))" "${TASK_JSON}")"
TRACE_NAME="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['trace_name'])" "${TASK_JSON}")"
DATASET_SOURCE="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['dataset_source'])" "${TASK_JSON}")"
TRACE_FAMILY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['trace_family'])" "${TASK_JSON}")"
OUT_DIR="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['out_dir'])" "${TASK_JSON}")"
TRACE_SOURCE_ROOT="${TRACE_SOURCE_ROOT:-/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_canonical_regen_20260916/source_traces}"
if [[ "${TRACE_PATH}" != /* ]]; then
  TRACE_PATH="${TRACE_SOURCE_ROOT%/}/${TRACE_PATH}"
fi

COMPLETION_MARKER="${OUT_DIR}/COMPLETE.json"
INCOMPLETE_MARKER="${OUT_DIR}/INCOMPLETE.json"
if [[ -f "${COMPLETION_MARKER}" ]]; then
  echo "[skip] ${TASK_ID}: COMPLETE.json already present; refusing duplicate production."
  exit 0
fi
if [[ -d "${OUT_DIR}" && ! -f "${COMPLETION_MARKER}" ]]; then
  mkdir -p "${OUT_DIR}"
  python3 -c "import json,datetime; json.dump({'detected_at': datetime.datetime.now().isoformat(), 'task_id': '${TASK_ID}'}, open('${INCOMPLETE_MARKER}', 'w'), indent=2)"
  echo "FATAL: ${TASK_ID} output exists without COMPLETE.json; not overwriting partial output." >&2
  exit 5
fi

if [[ ! -f "${TRACE_PATH}" ]]; then
  echo "FATAL: missing canonical trace source ${TRACE_PATH}" >&2
  exit 6
fi

mkdir -p "${OUT_DIR}"
TASK_TRACE_MANIFEST="${OUT_DIR}/.trace_manifest.csv"
python3 -c "
import csv
with open('${TASK_TRACE_MANIFEST}', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['path', 'trace_name', 'dataset_source', 'trace_family'])
    w.writerow(['${TRACE_PATH}', '${TRACE_NAME}', '${DATASET_SOURCE}', '${TRACE_FAMILY}'])
"

ACTUAL_SHA="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current || true)"
START_TS="$(date -Is)"
CMD=(python scripts/experiments/canonical/build_evict_value_dataset_wulver_v1.py
  --trace-manifest "${TASK_TRACE_MANIFEST}"
  --capacities "${CAPACITY}"
  --horizons "32,64,128"
  --split-mode trace_chunk
  --chunk-size 4096
  --split-seed 7
  --split-train-pct 70
  --split-val-pct 15
  --history-window 64
  --max-rows-per-shard 500000
  --max-requests-per-trace 50000
  --min-requests-per-trace 1000
  --min-unique-pages-per-trace 100
  --out-dir "${OUT_DIR}"
  --overwrite)

echo "[${START_TS}] Starting ${TASK_ID} on $(hostname), source=${ACTUAL_BRANCH}@${ACTUAL_SHA}"
echo "[cmd] ${CMD[*]}"
"${CMD[@]}"
END_TS="$(date -Is)"

TMP_MARKER="$(mktemp "${OUT_DIR}/.complete.XXXXXX.json")"
python3 -c "
import json, sys
json.dump({
  'task_id': '${TASK_ID}',
  'family': '${FAMILY}',
  'capacity': int('${CAPACITY}'),
  'horizons': [32, 64, 128],
  'trace_path': '${TRACE_PATH}',
  'canonical_manifest_trace_path': '${CANONICAL_MANIFEST_TRACE_PATH}',
  'trace_source_root': '${TRACE_SOURCE_ROOT}',
  'trace_name': '${TRACE_NAME}',
  'dataset_source': '${DATASET_SOURCE}',
  'trace_family': '${TRACE_FAMILY}',
  'start_ts': '${START_TS}',
  'end_ts': '${END_TS}',
  'slurm_array_job_id': '${SLURM_ARRAY_JOB_ID:-}',
  'slurm_array_task_id': '${SLURM_ARRAY_TASK_ID:-}',
  'hostname': '$(hostname)',
  'git_sha': '${ACTUAL_SHA}',
  'git_branch': '${ACTUAL_BRANCH}',
  'command': sys.argv[1:],
}, open('${TMP_MARKER}', 'w'), indent=2)
" "${CMD[@]}"
mv "${TMP_MARKER}" "${COMPLETION_MARKER}"
echo "[${END_TS}] ${TASK_ID} complete"
