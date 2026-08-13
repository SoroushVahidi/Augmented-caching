"""Resumable 21-cell exact finite-horizon target-oracle replication.

This runner delegates replay semantics to the validated single-cell diagnostic
and only adds unit orchestration, atomic completion, and final aggregation.
It intentionally disables the optional learned model by default: LRU and the
future-aware exact target oracle are the required scientific comparison.

Unit metadata is finalized after the temporary unit directory is renamed, so
output references always identify the canonical unit directory rather than a
staging path. This is a diagnostic runner, not a deployable cache policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import HISTORY_START, SCORE_END, SCORE_START

try:
    from .run_exact_target_oracle_diagnostic import (
        REPO_ROOT,
        _run_diagnostic,
        _sha256_of_file,
        _load_fold,
        _resolve_trace_path,
    )
except ImportError:  # Direct execution from scripts/experiments/.
    from run_exact_target_oracle_diagnostic import (  # type: ignore[no-redef]
        REPO_ROOT,
        _run_diagnostic,
        _sha256_of_file,
        _load_fold,
        _resolve_trace_path,
    )


CONFIG_PATH = REPO_ROOT / "configs/exact_target_oracle_replication_v1.json"
DEFAULT_OUT = REPO_ROOT / "analysis/exact_target_oracle_replication_v1"
REQUIRED_POLICIES = ("lru", "exact_finite_horizon_eviction_loss_oracle")


def _read_config(path: Path) -> Mapping[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg["history"] != [HISTORY_START, SCORE_START] or cfg["score"] != [SCORE_START, SCORE_END]:
        raise ValueError("replication config does not match canonical history/score windows")
    if cfg["horizon"] != 4:
        raise ValueError("replication requires H=4")
    return cfg


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        tmp = Path(fh.name)
    os.replace(tmp, path)


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _key(family: str, capacity: int) -> str:
    return f"{family}_cap{capacity}_h4"


def _finalize_output_paths(summary: Dict[str, Any], final_dir: Path) -> None:
    summary["outputs"] = {key: str(final_dir / Path(value).name) for key, value in summary.get("outputs", {}).items()}


def _finite(value: Any) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def _validate_summary(summary: Mapping[str, Any], family: str, capacity: int, trace_sha: str) -> None:
    if summary.get("status") != "COMPLETE":
        raise ValueError("unit status is not COMPLETE")
    protocol = summary.get("protocol", {})
    if protocol.get("horizon") != 4 or protocol.get("history_start") != 0:
        raise ValueError("unit protocol does not use H=4 and history start 0")
    if protocol.get("score_start") != 10000 or protocol.get("score_end") != 50000:
        raise ValueError("unit protocol does not use [10000,50000) scoring")
    if protocol.get("capacity") != capacity:
        raise ValueError("unit capacity mismatch")
    trace = summary.get("trace", {})
    if trace.get("family") != family or trace.get("sha256") != trace_sha:
        raise ValueError("unit trace identity/hash mismatch")
    policies = summary.get("policies", {})
    for policy in REQUIRED_POLICIES:
        if policy not in policies:
            raise ValueError(f"missing required policy: {policy}")
        if not all(_finite(policies[policy].get(name)) for name in ("misses", "miss_ratio")):
            raise ValueError(f"non-finite metrics for {policy}")


def _load_manifest(path: Path, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "status": "RUNNING",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": {"head": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"])},
        "protocol": dict(cfg),
        "expected_units": len(cfg["families"]) * len(cfg["capacities"]),
        "completed_units": 0,
        "units": {},
    }


def _run_unit(*, family: str, capacity: int, cfg: Mapping[str, Any], data_root: Path, out_root: Path, determinism_check: bool) -> Dict[str, Any]:
    fold = _load_fold(family)
    trace_path = _resolve_trace_path(fold, data_root)
    trace_sha = _sha256_of_file(trace_path)
    expected_sha = str(fold.get("test_trace_sha256", ""))
    if expected_sha and trace_sha != expected_sha:
        raise ValueError(f"trace hash mismatch for {family}: fold={expected_sha} disk={trace_sha}")
    requests, pages, _ = load_trace_from_any(str(trace_path))
    final_dir = out_root / "units" / _key(family, capacity)
    if final_dir.exists() and (final_dir / "summary.json").exists():
        summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
        _validate_summary(summary, family, capacity, trace_sha)
        return summary
    tmp_dir = out_root / "units" / f".{_key(family, capacity)}.tmp-{os.getpid()}"
    if tmp_dir.exists():
        raise RuntimeError(f"partial temporary unit exists; refusing to reuse: {tmp_dir}")
    summary = _run_diagnostic(
        requests=requests,
        pages=pages,
        trace_name=str(fold["test_trace_name"]),
        trace_family=family,
        trace_sha256=trace_sha,
        trace_path=trace_path,
        fold=fold,
        learned_model_path=None,
        learned_model_record=None,
        learned_provenance_error="disabled by replication protocol; optional learned comparison not required",
        capacity=capacity,
        horizon=4,
        score_start=SCORE_START,
        score_end=SCORE_END,
        out_dir=tmp_dir,
        overwrite=False,
        determinism_check=determinism_check,
    )
    _finalize_output_paths(summary, final_dir)
    (tmp_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_summary(summary, family, capacity, trace_sha)
    if family == "brightkite" and capacity == 64:
        policies = summary["policies"]
        if policies["lru"]["misses"] != 13225 or policies["exact_finite_horizon_eviction_loss_oracle"]["misses"] != 19079:
            raise ValueError("historical Brightkite/cap64 regression mismatch")
    os.replace(tmp_dir, final_dir)
    return summary


def _aggregate(out_root: Path, cfg: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    rows = []
    for family in cfg["families"]:
        for capacity in cfg["capacities"]:
            summary = json.loads((out_root / "units" / _key(family, capacity) / "summary.json").read_text(encoding="utf-8"))
            for policy in REQUIRED_POLICIES:
                result = summary["policies"][policy]
                rows.append({"family": family, "capacity": capacity, "horizon": 4, "score_start": 10000, "score_end": 50000, "policy": policy, "status": "ok", "misses": result["misses"], "miss_ratio": result["miss_ratio"], "trace_sha256": summary["trace"]["sha256"]})
    if len(rows) != 42 or len({(r["family"], r["capacity"], r["policy"]) for r in rows}) != 42:
        raise ValueError("aggregate row/key count mismatch")
    path = out_root / "policy_comparison.csv"
    with tempfile.NamedTemporaryFile("w", dir=out_root, prefix=".policy_comparison.", delete=False, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        tmp = Path(fh.name)
    os.replace(tmp, path)
    _atomic_json(out_root / "integrity_summary.json", {"status": "COMPLETE", "units": 21, "rows": 42, "unique_keys": 42, "required_policies": list(REQUIRED_POLICIES), "manifest_completed_units": manifest["completed_units"]})
    _atomic_json(out_root / "provenance.json", {"source": manifest["source"], "protocol": manifest["protocol"], "hostname": socket.gethostname(), "trace_hashes": {f: json.loads((out_root / "units" / _key(f, c) / "summary.json").read_text())["trace"]["sha256"] for f in cfg["families"] for c in cfg["capacities"]}})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--data-read-root", type=Path, default=REPO_ROOT.parent / "Augmented-caching")
    ap.add_argument("--determinism-check", action="store_true")
    args = ap.parse_args()
    cfg = _read_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "config_snapshot.json", cfg)
    manifest_path = args.out_dir / "unit_completion_manifest.json"
    manifest = _load_manifest(manifest_path, cfg)
    for family in cfg["families"]:
        for capacity in cfg["capacities"]:
            key = _key(family, capacity)
            summary = _run_unit(family=family, capacity=capacity, cfg=cfg, data_root=args.data_read_root, out_root=args.out_dir, determinism_check=args.determinism_check)
            manifest["units"][key] = {"status": "COMPLETE", "family": family, "capacity": capacity, "summary": str(args.out_dir / "units" / key / "summary.json"), "trace_sha256": summary["trace"]["sha256"]}
            manifest["completed_units"] = len(manifest["units"])
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "unit_complete", "unit": key, "completed_units": manifest["completed_units"]}, sort_keys=True), flush=True)
    if manifest["completed_units"] != 21:
        raise RuntimeError("campaign ended without all 21 units")
    _aggregate(args.out_dir, cfg, manifest)
    manifest["status"] = "COMPLETE"
    _atomic_json(manifest_path, manifest)
    print(json.dumps({"event": "campaign_complete", "units": 21, "rows": 42}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
