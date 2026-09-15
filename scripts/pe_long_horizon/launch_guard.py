"""Final pre-launch gate for PE long-horizon production.

Refuses to proceed (exit != 0) unless ALL of the following hold:
  - no resource-parameter placeholder remains in resource_params.env
  - the manifest has exactly 20 physical tasks / 60 logical cells
  - every task resolves, via the REAL generator library's
    parse_trace_manifest, to exactly one family (proves the
    --trace-glob action="append" default-list bug -- the one that made
    job 1287838 process 5 families instead of 1 -- cannot recur)
  - no brightkite/citibike family present
  - no H16 (or any horizon other than 32/64/128) present as a production
    horizon
  - families are exactly the 5 PE families, capacities exactly {32,64,128,256},
    each family x capacity appears exactly once
  - output directories are all distinct (no duplicate)
  - the local generator library/driver files' SHA256 match the hashes
    recorded in resource_params.env (EXPECTED_GENERATOR_*_SHA256) -- this
    is a local pre-check; the production sbatch template independently
    re-verifies both the git SHA and can be extended to re-hash on the
    Wulver side at job-start time
  - the production run_root does not already exist with ambiguous partial
    content (if it exists, every task subdirectory must either be absent
    or carry a COMPLETE.json -- ambiguous partial dirs block launch)
  - the run_root is NOT under the user's HOME (added after the first
    production launch, job 1288536, failed campaign-wide within ~1-3
    minutes on OSError: [Errno 122] Disk quota exceeded writing to a HOME
    fileset; run_root must resolve under an approved SCRATCH or PROJECT
    prefix instead)
  - a quota-audit artifact (configs/pe_long_horizon_production/provenance/
    quota_audit.json) exists, is reasonably fresh, and shows
    scratch_available_bytes >= the required minimum -- cluster-wide `df`
    free space is explicitly NOT accepted as evidence (see the same
    incident: /mmfs1 showed 809TB free while the actual per-fileset HOME
    quota was already exhausted)

IMPORTANT REMAINING GAP (documented, not silently hidden): the ambiguous-
partial-output check above is LOCAL-FILESYSTEM-ONLY -- it inspects this
repo's local working copy, which never mirrors Wulver's actual remote
run_root (this was true even when run_root was HOME-based, since the local
checkout and the Wulver checkout are different machines). A true
pre-launch ambiguity check requires an SSH-based inspection of the Wulver
run_root immediately before submission; this script does not perform that
and should not be treated as sufficient on its own for that specific
check.

Does not submit anything, does not touch Wulver, does not modify canonical
scientific code. Read-only / local-analysis only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = REPO_ROOT / "configs" / "pe_long_horizon_production" / "resource_params.env"
MANIFEST_PATH = REPO_ROOT / "configs" / "pe_long_horizon_production" / "manifest.json"
QUOTA_AUDIT_PATH = REPO_ROOT / "configs" / "pe_long_horizon_production" / "provenance" / "quota_audit.json"

HOME_PREFIXES = ("/home/", "/mmfs1/home/")
APPROVED_PREFIXES = ("/mmfs1/scratch/", "/mmfs1/project/", "/scratch/", "/project/")
MIN_SCRATCH_AVAILABLE_BYTES = 200 * (1024 ** 3)  # 200GB, per Task 6's conservative gate
MAX_QUOTA_AUDIT_AGE_SECONDS = 24 * 3600  # require a same-day quota snapshot

EXPECTED_FAMILIES = {"cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"}
FORBIDDEN_FAMILIES = {"brightkite", "citibike"}
EXPECTED_CAPACITIES = {32, 64, 128, 256}
EXPECTED_HORIZONS = {32, 64, 128}
FORBIDDEN_HORIZONS = {4, 8, 16}

sys.path.insert(0, str(REPO_ROOT / "src"))


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


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def check_generator_hashes(params: dict[str, str]) -> list[str]:
    errors = []
    lib_path = REPO_ROOT / "src" / "lafc" / "evict_value_wulver_v1.py"
    driver_candidates = list(REPO_ROOT.glob("scripts/**/build_evict_value_dataset_wulver_v1.py"))
    expected_lib = params.get("EXPECTED_GENERATOR_LIB_SHA256")
    expected_driver = params.get("EXPECTED_GENERATOR_DRIVER_SHA256")

    if not lib_path.exists():
        errors.append(f"generator library not found: {lib_path}")
    elif expected_lib and sha256_of(lib_path) != expected_lib:
        errors.append(
            f"generator library SHA256 mismatch: {lib_path} has "
            f"{sha256_of(lib_path)}, expected {expected_lib}"
        )

    if not driver_candidates:
        errors.append("generator driver script (build_evict_value_dataset_wulver_v1.py) not found locally")
    elif expected_driver:
        matches = [p for p in driver_candidates if sha256_of(p) == expected_driver]
        if not matches:
            errors.append(
                f"generator driver SHA256 mismatch: found at {driver_candidates}, "
                f"none match expected {expected_driver}"
            )
    return errors


def check_manifest_structure(manifest: dict) -> list[str]:
    errors = []
    tasks = manifest.get("tasks", [])
    cells = manifest.get("cells", [])
    if len(tasks) != 20:
        errors.append(f"expected 20 physical tasks, found {len(tasks)}")
    if len(cells) != 60:
        errors.append(f"expected 60 logical cells, found {len(cells)}")

    families = {t["family"] for t in tasks}
    if families != EXPECTED_FAMILIES:
        errors.append(f"unexpected family set: {families} (expected {EXPECTED_FAMILIES})")
    forbidden_found = families & FORBIDDEN_FAMILIES
    if forbidden_found:
        errors.append(f"forbidden family present: {forbidden_found}")

    capacities = {t["capacity"] for t in tasks}
    if capacities != EXPECTED_CAPACITIES:
        errors.append(f"unexpected capacity set: {capacities} (expected {EXPECTED_CAPACITIES})")

    seen_fc = set()
    for t in tasks:
        key = (t["family"], t["capacity"])
        if key in seen_fc:
            errors.append(f"duplicate family+capacity task: {key}")
        seen_fc.add(key)

    for t in tasks:
        horizons = set(t.get("horizons", []))
        if horizons != EXPECTED_HORIZONS:
            errors.append(f"{t['task_id']}: unexpected horizon set {horizons} (expected {EXPECTED_HORIZONS})")
        forbidden_h = horizons & FORBIDDEN_HORIZONS
        if forbidden_h:
            errors.append(f"{t['task_id']}: forbidden production horizon present: {forbidden_h}")

    out_dirs = [t["out_dir"] for t in tasks]
    if len(out_dirs) != len(set(out_dirs)):
        errors.append("duplicate output directory across tasks")

    return errors


def check_trace_selection(manifest: dict) -> list[str]:
    """Re-run the real end-to-end trace-selection test inline (mirrors
    tests/test_pe_long_horizon_trace_selection.py) so the launch guard
    itself -- not just the test suite -- proves this before any submit."""
    import csv
    import tempfile

    from lafc.evict_value_wulver_v1 import parse_trace_manifest

    errors = []
    wildcard_default = ["data/processed/*/trace.jsonl"]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        for task in manifest["tasks"]:
            manifest_csv = tdp / f"{task['task_id']}.csv"
            with manifest_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["path", "trace_name", "dataset_source", "trace_family"])
                w.writerow([task["trace_path"], task["trace_path"], task["family"], task["family"]])
            specs = parse_trace_manifest(str(manifest_csv), wildcard_default)
            resolved = {s.dataset_source for s in specs}
            if len(specs) != 1 or resolved != {task["family"]}:
                errors.append(
                    f"{task['task_id']}: trace-manifest resolution ambiguous -- "
                    f"got {len(specs)} spec(s), families={resolved}"
                )
    return errors


def check_run_root_location(manifest: dict) -> list[str]:
    """Refuses a HOME-based run_root; requires an approved SCRATCH/PROJECT
    prefix. Pure string check -- works without any Wulver/SSH access."""
    errors = []
    run_root = manifest.get("run_root", "")
    if any(run_root.startswith(p) for p in HOME_PREFIXES):
        errors.append(
            f"run_root is under HOME ({run_root!r}) -- this is exactly what caused "
            "job 1288536's campaign-wide disk-quota failure. Must be under an "
            f"approved prefix: {APPROVED_PREFIXES}"
        )
        return errors
    if not any(run_root.startswith(p) for p in APPROVED_PREFIXES):
        errors.append(
            f"run_root {run_root!r} is not under any approved prefix {APPROVED_PREFIXES} "
            "(and is not recognized as HOME either -- treat as unapproved by default)"
        )
    return errors


def check_quota_audit(params: dict[str, str]) -> list[str]:
    """Requires a frozen, reasonably fresh quota-audit artifact showing
    sufficient available space -- never accepts cluster-wide `df` free
    space (809TB was free cluster-wide while the actual per-user HOME
    fileset quota was already exhausted)."""
    errors = []
    if not QUOTA_AUDIT_PATH.exists():
        errors.append(
            f"no quota-audit artifact at {QUOTA_AUDIT_PATH} -- generate one "
            "(see scripts/pe_long_horizon/freeze_quota_audit.py) immediately "
            "before launch; cluster-wide df free space is not accepted as evidence"
        )
        return errors

    audit = json.loads(QUOTA_AUDIT_PATH.read_text(encoding="utf-8"))
    age = time.time() - audit.get("captured_at_epoch", 0)
    if age > MAX_QUOTA_AUDIT_AGE_SECONDS:
        errors.append(
            f"quota-audit artifact is {age/3600:.1f}h old (max allowed "
            f"{MAX_QUOTA_AUDIT_AGE_SECONDS/3600:.0f}h) -- regenerate immediately before launch"
        )

    available = audit.get("scratch_available_bytes")
    if available is None or available < MIN_SCRATCH_AVAILABLE_BYTES:
        errors.append(
            f"quota-audit shows scratch_available_bytes={available}, below the "
            f"required minimum {MIN_SCRATCH_AVAILABLE_BYTES} (200GB)"
        )
    if audit.get("quota_domain") == "HOME":
        errors.append("quota-audit itself reports quota_domain=HOME -- this is the forbidden domain")
    return errors


def check_run_root_ambiguity(manifest: dict) -> list[str]:
    errors = []
    run_root = REPO_ROOT / manifest["run_root"]
    if not run_root.exists():
        return errors
    for task in manifest["tasks"]:
        out_dir = REPO_ROOT / task["out_dir"]
        if not out_dir.exists():
            continue
        complete_marker = out_dir / "COMPLETE.json"
        if not complete_marker.exists():
            errors.append(
                f"{task['task_id']}: output dir {out_dir} exists without COMPLETE.json "
                "-- ambiguous partial output, must be resolved (see resume_planner.py) before launch"
            )
    return errors


def check_probe_output_not_reused(manifest: dict) -> list[str]:
    """job 1288047's partial cap256 probe output must never be mistaken for
    a production cell."""
    errors = []
    probe_dir_name = "evict_value_v1_long_horizon_probe_cap256_cloudphysics_20260915"
    for task in manifest["tasks"]:
        if probe_dir_name in task["out_dir"]:
            errors.append(f"{task['task_id']}: out_dir points at the timeout probe's directory -- forbidden")
    return errors


def main() -> int:
    if not PARAMS_PATH.exists():
        print(f"FATAL: resource params file not found: {PARAMS_PATH}", file=sys.stderr)
        return 2
    if not MANIFEST_PATH.exists():
        print(f"FATAL: manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        return 2

    params = load_params(PARAMS_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    all_errors: list[str] = []

    placeholders = find_placeholders(params)
    if placeholders:
        all_errors.append("Unresolved resource placeholders: " + ", ".join(f"{k}={params[k]}" for k in placeholders))

    all_errors.extend(check_manifest_structure(manifest))
    all_errors.extend(check_trace_selection(manifest))
    all_errors.extend(check_generator_hashes(params))
    all_errors.extend(check_run_root_location(manifest))
    all_errors.extend(check_quota_audit(params))
    all_errors.extend(check_run_root_ambiguity(manifest))
    all_errors.extend(check_probe_output_not_reused(manifest))

    if all_errors:
        print("PRODUCTION LAUNCH BLOCKED:", file=sys.stderr)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("All launch-gate checks passed. Production launch gate: OPEN.")
    print(f"  physical_tasks={len(manifest['tasks'])} logical_cells={len(manifest['cells'])}")
    print(f"  CPUS_PER_TASK={params.get('CPUS_PER_TASK')} MEMORY_PER_TASK={params.get('MEMORY_PER_TASK')}")
    print(f"  ARRAY_CONCURRENCY={params.get('ARRAY_CONCURRENCY')} WALLTIME_PER_TASK={params.get('WALLTIME_PER_TASK')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
