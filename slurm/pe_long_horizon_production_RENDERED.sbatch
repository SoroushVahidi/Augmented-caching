#!/bin/bash
#
# PE long-horizon PRODUCTION array -- RENDERED, resource-resolved. Still requires explicit human authorization before sbatch.
#
# One array task = one (family, capacity) cell; computes horizons
# {32,64,128} together in a single generator invocation (shares feature
# computation across horizons; avoids the 3x outer-loop duplication that a
# per-horizon task split would cause -- see docs/pe_long_horizon_design_notes.md
# and the manifest at configs/pe_long_horizon_production/manifest.json).
#
# RESOURCE FIELDS ARE PARAMETRIC. Job 1288047 (cap256 timing probe) hit its
# 90-minute walltime and TIMED OUT before finishing -- it never COMPLETED --
# but its partial output (30 complete 500k-row shards, stable throughput)
# was sufficient to calibrate an evidence-based runtime model together with
# job 1287838 (cap32, COMPLETED). See
# configs/pe_long_horizon_production/resource_params.env for the resolved
# values and their derivation. Submitting this file with the placeholders
# below still in place will fail Slurm's parser (or, if a wrapper resolves
# them first, that wrapper must refuse to run while any __TBD__ token
# remains -- see scripts/pe_long_horizon/launch_guard.py).
#
#SBATCH --job-name=pe-long-horizon-prod
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=10:00:00
#SBATCH --array=0-19%10
#SBATCH --output=slurm/logs/pe-long-horizon-prod-%A_%a.out
#SBATCH --error=slurm/logs/pe-long-horizon-prod-%A_%a.err

set -euo pipefail

cd ~/lafc-work/Augmented-caching
mkdir -p slurm/logs

MANIFEST="configs/pe_long_horizon_production/manifest.json"
EXPECTED_SHA="3a04a25094de1e8c57a603e2662b321bcf1d9e88"

# --- explicit source-SHA check before any computation ---
ACTUAL_SHA="$(git rev-parse HEAD)"
if [[ "${ACTUAL_SHA}" != "${EXPECTED_SHA}" ]]; then
  echo "FATAL: checked-out source SHA ${ACTUAL_SHA} != expected ${EXPECTED_SHA}" >&2
  exit 3
fi

# --- deterministic array-index -> manifest-record mapping ---
TASK_JSON="$(python3 -c "
import json, sys
m = json.load(open('${MANIFEST}'))
idx = ${SLURM_ARRAY_TASK_ID}
task = [t for t in m['tasks'] if t['array_index'] == idx]
if not task:
    print('FATAL: no manifest task for array_index=' + str(idx), file=sys.stderr)
    sys.exit(4)
print(json.dumps(task[0]))
")"

FAMILY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['family'])" "${TASK_JSON}")"
CAPACITY="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['capacity'])" "${TASK_JSON}")"
TRACE_PATH="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['trace_path'])" "${TASK_JSON}")"
OUT_DIR="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['out_dir'])" "${TASK_JSON}")"
TASK_ID="$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['task_id'])" "${TASK_JSON}")"
HORIZONS_CSV="32,64,128"

COMPLETION_MARKER="${OUT_DIR}/COMPLETE.json"
INCOMPLETE_MARKER="${OUT_DIR}/INCOMPLETE.json"

# --- detect already-completed task: exit cleanly, do not recompute ---
if [[ -f "${COMPLETION_MARKER}" ]]; then
  echo "[skip] ${TASK_ID}: COMPLETE.json already present, not recomputing"
  exit 0
fi

# --- detect a prior incomplete/partial output: flag it, do not silently reuse ---
if [[ -d "${OUT_DIR}" && ! -f "${COMPLETION_MARKER}" ]]; then
  echo "[warn] ${TASK_ID}: output dir exists without a completion marker -- prior run was likely incomplete."
  echo "[warn] Writing INCOMPLETE marker and refusing to silently reuse partial shards."
  mkdir -p "${OUT_DIR}"
  python3 -c "
import json, datetime
json.dump({'detected_at': datetime.datetime.now().isoformat(), 'task_id': '${TASK_ID}'}, open('${INCOMPLETE_MARKER}', 'w'), indent=2)
"
  # A human/resume-planner decision is required before this task can be
  # safely rerun with --overwrite; this template does not auto-decide that.
  exit 5
fi

mkdir -p "${OUT_DIR}"

source .venv/bin/activate
export PYTHONPATH=src
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

START_TS="$(date -Is)"

# --- exactly one trace per task, via --trace-manifest, NOT --trace-glob ---
# build_evict_value_dataset_wulver_v1.py's --trace-glob uses
# argparse action="append" with a non-empty default
# (default=["data/processed/*/trace.jsonl"]); passing --trace-glob on the
# command line APPENDS to that default rather than replacing it, which is
# exactly what caused preflight job 1287838 to process all 5 families
# instead of the intended 1. --trace-manifest bypasses the default globs
# entirely (parse_trace_manifest ignores fallback_globs whenever a manifest
# path is given) -- this is the same, already-validated mechanism used by
# the cap256 probe (job 1288047), which correctly processed only
# cloudphysics. Write a one-row manifest for this task only.
TASK_TRACE_MANIFEST="${OUT_DIR}/.trace_manifest.csv"
python3 -c "
import csv
path = '${TRACE_PATH}'
family = '${FAMILY}'
with open('${TASK_TRACE_MANIFEST}', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['path', 'trace_name', 'dataset_source', 'trace_family'])
    w.writerow([path, path, family, family])
"

CMD=(python scripts/build_evict_value_dataset_wulver_v1.py
  --trace-manifest "${TASK_TRACE_MANIFEST}"
  --capacities "${CAPACITY}"
  --horizons "${HORIZONS_CSV}"
  --split-mode trace_chunk
  --chunk-size 4096
  --max-rows-per-shard 500000
  --split-seed 7
  --min-requests-per-trace 1000
  --min-unique-pages-per-trace 100
  --out-dir "${OUT_DIR}"
  --overwrite)

echo "[${START_TS}] Starting ${TASK_ID} on $(hostname), array_index=${SLURM_ARRAY_TASK_ID}"
echo "[cmd] ${CMD[*]}"
"${CMD[@]}"
END_TS="$(date -Is)"

# --- atomic completion marker + per-task provenance, written only after success ---
TMP_MARKER="$(mktemp "${OUT_DIR}/.complete.XXXXXX.json")"
python3 -c "
import json, sys
cmd = sys.argv[1:]
json.dump({
    'task_id': '${TASK_ID}',
    'family': '${FAMILY}',
    'capacity': ${CAPACITY},
    'horizons': [32, 64, 128],
    'start_ts': '${START_TS}',
    'end_ts': '${END_TS}',
    'slurm_array_job_id': '${SLURM_ARRAY_JOB_ID:-}',
    'slurm_array_task_id': '${SLURM_ARRAY_TASK_ID:-}',
    'hostname': '$(hostname)',
    'git_sha': '${ACTUAL_SHA}',
    'command': cmd,
}, open('${TMP_MARKER}', 'w'), indent=2)
" "${CMD[@]}"
mv "${TMP_MARKER}" "${COMPLETION_MARKER}"
echo "[${END_TS}] ${TASK_ID} complete, wrote ${COMPLETION_MARKER}"
