"""Gated launcher for the FINAL controlled practical-significance timing
campaign (Reviewer 1, Concern 4).

Never launches by itself -- prints TIMING_GATE = READY or BLOCKED (with
reasons) and, only when explicitly asked via --launch AND the gate is
READY, invokes the already-implemented
scripts/experiments/run_practical_significance_ablation.py with
--controlled-final --all (reusing that runner's own modes rather than
duplicating its profiling/optimization/selective/pareto logic -- see that
script's docstring and argparse).

Gate checks:
  - no revision-critical scientific process is actively consuming CPU
    (Concern 1's cross-family training, Concern 2's objective-ablation
    pipeline, Concern 3's distribution-shift campaign, AND the archival
    legacy LRB baseline -- the roadmap explicitly treats LRB as a blocker
    too, since no CPU-affinity isolation is set up to exempt it);
  - load average is not elevated relative to core count;
  - sufficient free/available RAM;
  - sufficient free disk;
  - the frozen protocol config still exists and hashes are recorded for
    provenance.

Usage:
    python scripts/experiments/run_practical_significance_controlled.py
    python scripts/experiments/run_practical_significance_controlled.py --launch
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

from lafc.experiments.external_baseline_common import sha256_of_file

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_READ_ROOT = REPO_ROOT.parent / "Augmented-caching"
DEFAULT_HEAVY_MODEL = DEFAULT_DATA_READ_ROOT / "models/evict_value_wulver_v1_best_heavy_r1.pkl"
CONFIG_PATH = Path("configs/practical_significance_ablation_v1.json")
UNDERLYING_RUNNER = Path("scripts/experiments/run_practical_significance_ablation.py")

# Process command-line substrings for every revision-critical job that must
# be absent (or, for the legacy LRB, explicitly acknowledged) before a
# controlled timing run is scientifically valid.
BLOCKING_PATTERNS = {
    "concern_1_cross_family_training": ["run_evict_cross_family_pipeline.py", "train_evict_value_wulver_v1.py",
                                          "build_evict_value_dataset_wulver_v1.py"],
    "concern_2_objective_ablation": ["build_supervision_objective_ablation_dataset.py",
                                       "train_supervision_objective_ablation.py",
                                       "run_supervision_objective_ablation.py"],
    "concern_3_distribution_shift": ["run_distribution_shift_ablation.py"],
    "legacy_lrb_archival": ["run_lrb_external_baseline.py"],
}

MIN_FREE_RAM_GB = 20.0
MIN_FREE_DISK_GB = 20.0
MAX_LOAD_PER_CORE = 0.5  # load average per core above this is treated as non-idle


def _ps_lines() -> List[str]:
    out = subprocess.run(["ps", "-eo", "pid,%cpu,cmd"], capture_output=True, text=True, check=True).stdout
    return out.splitlines()


def _find_blocking_processes() -> Dict[str, List[str]]:
    lines = _ps_lines()
    found: Dict[str, List[str]] = {}
    for label, patterns in BLOCKING_PATTERNS.items():
        matches = [
            ln.strip() for ln in lines
            if any(p in ln for p in patterns) and "grep" not in ln
            and "run_practical_significance_controlled.py" not in ln
        ]
        if matches:
            found[label] = matches
    return found


def _load_average() -> tuple:
    with open("/proc/loadavg") as fh:
        parts = fh.read().split()
    return float(parts[0]), float(parts[1]), float(parts[2])


def _free_ram_gb() -> float:
    fields = {}
    with open("/proc/meminfo") as fh:
        for line in fh:
            k, v = line.split(":", 1)
            fields[k] = int(v.strip().split()[0])  # kB
    return fields.get("MemAvailable", 0) / (1024 * 1024)


def _free_disk_gb(path: str = ".") -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def check_gate() -> Dict[str, object]:
    reasons: List[str] = []

    blocking = _find_blocking_processes()
    for label, matches in blocking.items():
        reasons.append(f"{label}: {len(matches)} active process(es) still running")

    nproc = subprocess.run(["nproc"], capture_output=True, text=True, check=True).stdout.strip()
    n_cores = int(nproc) if nproc.isdigit() else 1
    load1, load5, load15 = _load_average()
    load_per_core = load1 / n_cores
    if load_per_core > MAX_LOAD_PER_CORE:
        reasons.append(f"load average {load1:.2f} on {n_cores} cores ({load_per_core:.2f}/core) exceeds "
                        f"{MAX_LOAD_PER_CORE}/core idle threshold")

    free_ram = _free_ram_gb()
    if free_ram < MIN_FREE_RAM_GB:
        reasons.append(f"only {free_ram:.1f}GB RAM available, need >= {MIN_FREE_RAM_GB}GB")

    free_disk = _free_disk_gb()
    if free_disk < MIN_FREE_DISK_GB:
        reasons.append(f"only {free_disk:.1f}GB disk free, need >= {MIN_FREE_DISK_GB}GB")

    protocol_hash = sha256_of_file(CONFIG_PATH) if CONFIG_PATH.exists() else None
    if protocol_hash is None:
        reasons.append(f"frozen protocol config not found: {CONFIG_PATH}")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "blocking_processes": blocking,
        "n_cores": n_cores,
        "load_average": {"1min": load1, "5min": load5, "15min": load15},
        "free_ram_gb": round(free_ram, 1),
        "free_disk_gb": round(free_disk, 1),
        "protocol_config_sha256": protocol_hash,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--max-requests", type=int, default=50000, help="Full request budget for the controlled run "
                     "(vs. the smoke-scale default of 1500 in the underlying runner).")
    ap.add_argument("--data-read-root", default=str(DEFAULT_DATA_READ_ROOT))
    ap.add_argument("--evict-value-model", default=str(DEFAULT_HEAVY_MODEL))
    ap.add_argument("--launch", action="store_true",
                     help="Actually invoke the underlying controlled-final campaign if the gate is READY. "
                     "Without this flag the script only reports the gate result and the exact command "
                     "that would run.")
    args = ap.parse_args()

    gate = check_gate()
    print(f"[gate] n_cores={gate['n_cores']} load_average={gate['load_average']} "
          f"free_ram_gb={gate['free_ram_gb']} free_disk_gb={gate['free_disk_gb']} "
          f"protocol_config_sha256={gate['protocol_config_sha256']}")
    if gate["blocking_processes"]:
        print("[gate] blocking processes:")
        for label, matches in gate["blocking_processes"].items():
            for m in matches:
                print(f"  - [{label}] {m[:160]}")

    cmd = [
        sys.executable, str(UNDERLYING_RUNNER),
        "--all", "--controlled-final", "--resume",
        "--capacities", args.capacities,
        "--max-requests", str(args.max_requests),
        "--data-read-root", args.data_read_root,
        "--evict-value-model", args.evict_value_model,
    ]

    if gate["ready"]:
        print("\nTIMING_GATE = READY")
        print(f"[plan] would run: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 {' '.join(cmd)}")
    else:
        print("\nTIMING_GATE = BLOCKED")
        for r in gate["reasons"]:
            print(f"  - {r}")

    if args.launch:
        if not gate["ready"]:
            print("\n[refuse] --launch was passed but the gate is BLOCKED -- not launching.", file=sys.stderr)
            raise SystemExit(1)
        print(f"\n[launch] {' '.join(cmd)}")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
