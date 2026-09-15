"""Render the production sbatch TEMPLATE into a concrete, submittable file.

Slurm parses #SBATCH directives literally at submission time -- they cannot
contain runtime placeholders. This script substitutes __CPUS__, __MEM__,
__WALLTIME__, and __ARRAY_SPEC__ in the template using the resolved values
in configs/pe_long_horizon_production/resource_params.env, refusing to
render (exit != 0) unless launch_guard.py's checks already pass.

Writes the rendered file but does NOT call sbatch. Printing/inspecting the
result is the caller's job (see Task 12 of the final launch gate).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = REPO_ROOT / "slurm" / "pe_long_horizon_production_TEMPLATE.sbatch"
RENDERED = REPO_ROOT / "slurm" / "pe_long_horizon_production_RENDERED.sbatch"
PARAMS_PATH = REPO_ROOT / "configs" / "pe_long_horizon_production" / "resource_params.env"


def load_params() -> dict[str, str]:
    params: dict[str, str] = {}
    for line in PARAMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        params[key.strip()] = value.strip()
    return params


def main() -> int:
    guard = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "pe_long_horizon" / "launch_guard.py")],
        capture_output=True, text=True,
    )
    if guard.returncode != 0:
        print("REFUSING TO RENDER: launch_guard.py did not pass.", file=sys.stderr)
        print(guard.stdout, guard.stderr, file=sys.stderr)
        return 1

    params = load_params()
    text = TEMPLATE.read_text(encoding="utf-8")
    text = text.replace("__CPUS__", params["CPUS_PER_TASK"])
    text = text.replace("__MEM__", params["MEMORY_PER_TASK"])
    text = text.replace("__WALLTIME__", params["WALLTIME_PER_TASK"])
    array_spec = f"{params['ARRAY_SPEC']}%{params['ARRAY_CONCURRENCY']}"
    text = text.replace("__ARRAY_SPEC__", array_spec)
    text = text.replace(
        "PRODUCTION array template -- DO NOT SUBMIT AS-IS.",
        "PRODUCTION array -- RENDERED, resource-resolved. Still requires "
        "explicit human authorization before sbatch.",
    )

    RENDERED.write_text(text, encoding="utf-8")
    print(f"Wrote {RENDERED}")
    print(f"  cpus={params['CPUS_PER_TASK']} mem={params['MEMORY_PER_TASK']} "
          f"walltime={params['WALLTIME_PER_TASK']} array={array_spec}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
