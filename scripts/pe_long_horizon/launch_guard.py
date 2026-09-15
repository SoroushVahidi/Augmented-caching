"""Refuse to launch PE long-horizon production while resource params are TBD.

Reads configs/pe_long_horizon_production/resource_params.env and exits
non-zero (with a clear message, no Slurm interaction) if any value still
contains a placeholder token (__TBD__ or __WAIT_FOR_<job>__). This is the
single gate that must pass before the production sbatch template is
rendered with concrete values and submitted.

Does not submit anything itself -- it is a precondition check only.
"""
from __future__ import annotations

import sys
from pathlib import Path

PARAMS_PATH = Path(__file__).resolve().parents[2] / "configs" / "pe_long_horizon_production" / "resource_params.env"


def load_params(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.partition("=")
        params[key.strip()] = value.strip()
    return params


def find_placeholders(params: dict[str, str]) -> list[str]:
    return [k for k, v in params.items() if "__TBD__" in v or v.startswith("__WAIT_FOR_")]


def main() -> int:
    if not PARAMS_PATH.exists():
        print(f"FATAL: resource params file not found: {PARAMS_PATH}", file=sys.stderr)
        return 2
    params = load_params(PARAMS_PATH)
    placeholders = find_placeholders(params)
    if placeholders:
        print("PRODUCTION LAUNCH BLOCKED: unresolved resource placeholders:", file=sys.stderr)
        for k in placeholders:
            print(f"  {k}={params[k]}", file=sys.stderr)
        print(
            "Substitute measured values (see cap256 probe job) into "
            f"{PARAMS_PATH} before rendering/submitting the production sbatch.",
            file=sys.stderr,
        )
        return 1
    print("All resource parameters resolved. Production launch gate: OPEN.")
    for k, v in params.items():
        print(f"  {k}={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
