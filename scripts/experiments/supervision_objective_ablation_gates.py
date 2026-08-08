"""Shared fail-closed gates for the supervision-objective ablation campaign.

This module is the single source of truth for the future/reproduction
pipeline ordering enforced by Reviewer Concern 2:

    28/28 models complete
        ->
    same-example audit FINAL=true + PASS (7/7 folds, protocol hash agrees)
        ->
    fairness audit FINAL=true + PASS (7/7 folds, protocol hash agrees)
        ->
    protocol/hash consistency verified
        ->
    registry freeze
        ->
    held-out evaluation

There is intentionally NO normal CLI bypass flag. Every check here is
fail-closed: a missing/incomplete/failing gate returns a non-empty list of
reasons, and the caller prints ``<GATE>_BLOCKED: <reason>`` and exits
nonzero WITHOUT writing any rows or freezing any registry.

These helpers are shared by:
  - scripts/build_supervision_objective_ablation_registry.py  (registry gate)
  - scripts/experiments/run_supervision_objective_ablation.py (evaluator gate)
  - scripts/experiments/run_supervision_objective_ablation_campaign.py
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import List, Optional

from lafc.experiments.external_baseline_common import sha256_of_file

PROTOCOL_ID = "supervision_objective_ablation_v1"
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
EXPECTED_MODELS = 28
EXPECTED_EVAL_ROWS = 84

DEFAULT_PROTOCOL_CONFIG = Path("configs/supervision_objective_ablation_v1.json")
DEFAULT_SAME_EXAMPLE_AUDIT = Path("analysis/supervision_objective_ablation_v1/same_example_audit.json")
DEFAULT_FAIRNESS_AUDIT = Path("analysis/supervision_objective_ablation_v1/fairness_audit.json")
DEFAULT_REGISTRY = Path("analysis/supervision_objective_ablation_v1/model_registry.json")


def protocol_config_sha256(config_path: Optional[Path] = None) -> str:
    """SHA-256 of the frozen protocol config (the machine-readable protocol)."""
    path = config_path or DEFAULT_PROTOCOL_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Protocol config not found: {path}")
    return sha256_of_file(path)


def _load_audit(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"__corrupt__": True}


def _audit_protocol_ok(payload: Optional[dict], expected_config_hash: str, label: str,
                       failures: List[str]) -> None:
    """Append protocol-id / config-hash mismatches for one audit payload."""
    if payload is None or payload.get("__corrupt__"):
        return  # presence / FINAL / overall handled elsewhere
    if payload.get("protocol_id") != PROTOCOL_ID:
        failures.append(
            f"{label}: protocol_id={payload.get('protocol_id')!r} != expected {PROTOCOL_ID!r}"
        )
    audit_hash = payload.get("protocol_config_sha256")
    if audit_hash is None:
        failures.append(f"{label}: missing protocol_config_sha256 (stale/pre-gate audit output)")
    elif audit_hash != expected_config_hash:
        failures.append(
            f"{label}: protocol_config_sha256={audit_hash[:12]}... != config "
            f"{expected_config_hash[:12]}... (protocol changed after audit ran?)"
        )


def _registry_protocol_ok(registry: dict, expected_config_hash: str, failures: List[str]) -> None:
    if registry.get("protocol_id") != PROTOCOL_ID:
        failures.append(
            f"registry protocol_id={registry.get('protocol_id')!r} != expected {PROTOCOL_ID!r}"
        )
    registry_hash = registry.get("protocol_config_sha256")
    if registry_hash is None:
        failures.append("registry: missing protocol_config_sha256 (stale/pre-gate registry output)")
    elif registry_hash != expected_config_hash:
        failures.append(
            f"registry: protocol_config_sha256={registry_hash[:12]}... != config "
            f"{expected_config_hash[:12]}... (protocol changed after registry freeze?)"
        )


def audit_gate_failures(
    same_example_path: Optional[Path] = None,
    fairness_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    *,
    require_folds: int = len(FAMILIES),
) -> List[str]:
    """Fail-closed checks for the two final audits.

    Returns an empty list iff BOTH audits exist, are valid JSON, have
    FINAL=true, overall=PASS, cover all ``require_folds`` folds, and their
    protocol_id / protocol_config_sha256 agree with the frozen config.
    """
    failures: List[str] = []
    same_path = same_example_path or DEFAULT_SAME_EXAMPLE_AUDIT
    fair_path = fairness_path or DEFAULT_FAIRNESS_AUDIT

    expected_hash = protocol_config_sha256(config_path)

    for path, label in ((same_path, "same-example audit"), (fair_path, "fairness audit")):
        if not path.exists():
            failures.append(f"{label}: audit file not found ({path})")
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            failures.append(f"{label}: audit file is not valid JSON ({path})")
            continue
        if not payload.get("FINAL"):
            failures.append(f"{label}: FINAL={payload.get('FINAL')} (need true) -- re-run without --partial-audit")
        if payload.get("overall") != "PASS":
            failures.append(f"{label}: overall={payload.get('overall')!r} (need PASS)")
        built = payload.get("folds_built") or []
        not_built = payload.get("folds_not_built") or []
        if len(built) != require_folds or not_built:
            failures.append(
                f"{label}: folds_built={len(built)}/{require_folds} not_built={not_built}"
            )
        _audit_protocol_ok(payload, expected_hash, label, failures)
    return failures


def registry_gate_failures(
    registry_path: Optional[Path] = None,
    same_example_path: Optional[Path] = None,
    fairness_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    *,
    expected_models: int = EXPECTED_MODELS,
) -> List[str]:
    """Fail-closed gate run BEFORE writing a frozen registry.

    All of: exactly ``expected_models`` records, no missing model, every
    record's on-disk artifact hash matches its recorded hash, both final
    audits FINAL PASS, and protocol ids/hashes agree across config/audits/
    registry inputs.
    """
    failures = audit_gate_failures(same_example_path, fairness_path, config_path)
    if failures:
        return failures  # audits gate everything downstream

    reg_path = registry_path or DEFAULT_REGISTRY
    if not reg_path.exists():
        return [f"registry file not found ({reg_path}) -- nothing to gate"]
    try:
        registry = json.loads(reg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"registry file is not valid JSON ({reg_path})"]

    expected_hash = protocol_config_sha256(config_path)
    records = registry.get("records") or []
    if len(records) != expected_models:
        failures.append(
            f"registry has {len(records)} model records, expected {expected_models}"
        )
    missing = registry.get("missing_models") or []
    if missing:
        failures.append(f"registry lists missing models: {missing}")
    if registry.get("MODEL_SELECTION_FROZEN") is not True:
        failures.append(f"MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')} (need true)")
    _registry_protocol_ok(registry, expected_hash, failures)
    for rec in records:
        rec_hash = rec.get("model_artifact_sha256")
        art_path = rec.get("model_artifact_path")
        if not art_path or not rec_hash:
            failures.append(f"registry record for {rec.get('objective')}/{rec.get('held_out_family')} "
                            f"missing model_artifact_path/sha256")
            continue
        on_disk = Path(art_path)
        if not on_disk.exists():
            failures.append(f"model artifact not found on disk: {art_path}")
            continue
        if sha256_of_file(on_disk) != rec_hash:
            failures.append(f"model artifact hash mismatch: {art_path}")
    return failures


def _other_evaluator_pids() -> List[int]:
    """PIDs of other alive ``run_supervision_objective_ablation.py`` processes."""
    try:
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True, text=True, check=True).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    mine = os.getpid()
    others = []
    for line in out.splitlines():
        if "run_supervision_objective_ablation.py" not in line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        if pid != mine:
            others.append(pid)
    return others


def evaluator_startup_failures(
    registry_path: Optional[Path] = None,
    same_example_path: Optional[Path] = None,
    fairness_path: Optional[Path] = None,
    config_path: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    *,
    expected_models: int = EXPECTED_MODELS,
    check_second_evaluator: bool = True,
    check_resume_safe: bool = True,
) -> List[str]:
    """Fail-closed gate run AT EVALUATOR STARTUP before any row is written.

    Requires:
      1. frozen registry exists with exactly ``expected_models`` records;
      2. every on-disk model hash matches the registry's recorded hash;
      3. both final audits FINAL PASS (7/7 folds, protocol hash agrees);
      4. no second evaluator is already writing the same output;
      5. any existing output is resume-safe (no duplicate keys / non-ok rows).
    """
    reg_path = registry_path or DEFAULT_REGISTRY
    if not reg_path.exists():
        return [f"frozen registry not found ({reg_path})"]
    try:
        registry = json.loads(reg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"registry file is not valid JSON ({reg_path})"]

    expected_hash = protocol_config_sha256(config_path)
    records = registry.get("records") or []
    failures: List[str] = []
    if len(records) != expected_models:
        failures.append(
            f"registry has {len(records)} model records, expected {expected_models}"
        )
    if registry.get("MODEL_SELECTION_FROZEN") is not True:
        failures.append(f"MODEL_SELECTION_FROZEN={registry.get('MODEL_SELECTION_FROZEN')} (need true)")
    _registry_protocol_ok(registry, expected_hash, failures)
    for rec in records:
        rec_hash = rec.get("model_artifact_sha256")
        art_path = rec.get("model_artifact_path")
        if not art_path or not rec_hash:
            failures.append(f"record for {rec.get('objective')}/{rec.get('held_out_family')} "
                            f"missing model_artifact_path/sha256")
            continue
        on_disk = Path(art_path)
        if not on_disk.exists():
            failures.append(f"model artifact not found on disk: {art_path}")
            continue
        if sha256_of_file(on_disk) != rec_hash:
            failures.append(f"model artifact hash mismatch: {art_path}")

    failures.extend(audit_gate_failures(same_example_path, fairness_path, config_path))

    if check_second_evaluator:
        others = _other_evaluator_pids()
        if others:
            failures.append(f"another evaluator is already running (pid(s) {others})")

    if check_resume_safe and out_dir is not None:
        failures.extend(_resume_safety_failures(out_dir / "policy_comparison.csv"))

    return failures


def _resume_safety_failures(csv_path: Path) -> List[str]:
    """Verify an existing eval CSV (if any) is resume-safe and consistent."""
    if not csv_path.exists():
        return []
    failures: List[str] = []
    seen = set()
    import csv
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("objective"), row.get("held_out_family"), row.get("capacity"))
            if key in seen:
                failures.append(f"existing output has duplicate key {key}")
                continue
            seen.add(key)
            if str(row.get("status", "")).lower() in ("fail", "failed", "error", "blocked"):
                failures.append(f"existing output has non-ok row for {key}: status={row.get('status')}")
            for value in row.values():
                if not value or value in ("n/a", "NA"):
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue
                import math
                if math.isnan(float(value)) or math.isinf(float(value)):
                    failures.append(f"existing output has NaN/Inf in row {key}")
    return failures


def assert_gate_clear(failures: List[str], gate_label: str) -> None:
    """Raise SystemExit(1) printing ``<GATE>_BLOCKED: <reason>`` per failure."""
    if failures:
        print(f"{gate_label}_BLOCKED: {'; '.join(failures)}", flush=True)
        raise SystemExit(1)
