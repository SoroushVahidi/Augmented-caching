"""Production runner for the C0/C1/C2 continuation-policy causal ablation.

The atomic scientific unit is (held_out_family, capacity). Each unit writes
into a temporary unit directory, validates its own outputs and model hashes,
then atomically promotes the directory to a completed unit. Global CSVs are
rebuilt from completed units, so resume never appends duplicate rows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import sklearn
from sklearn.exceptions import InconsistentVersionWarning

from lafc.continuation_policy_ablation import (
    CONDITION_C0_BASELINE_LRU,
    CONDITION_C1_LRU_CONTINUATION_LEARNED_PI1,
    CONDITION_C2_PI1_CONTINUATION_LEARNED_PI2,
    CONTINUATION_FROZEN_PI1,
    CONTINUATION_LRU,
    ContinuationAblationConfig,
    FrozenPi1Provenance,
    build_decision_aligned_continuation_rows,
    label_agreement_metrics,
    load_frozen_pi1_from_registry,
    sha256_of_file,
    train_pi2_from_c2_labels,
)
from lafc.distribution_shift_ablation import DistributionShiftEvalPolicy
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import score_window
from lafc.policies.base import BasePolicy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy


DEFAULT_CONFIG = Path("configs/continuation_policy_causal_ablation_production_v1.json")
DEFAULT_DATA_READ_ROOT = Path(".")

POLICY_FIELDNAMES = [
    "protocol_id",
    "unit_id",
    "held_out_family",
    "fold_id",
    "capacity",
    "condition",
    "policy",
    "trace_name",
    "trace_path",
    "trace_sha256",
    "history_start",
    "history_end",
    "score_start",
    "score_end",
    "history_requests",
    "scored_requests",
    "hits",
    "misses",
    "miss_ratio",
    "seed",
    "horizon",
    "model_path",
    "model_sha256",
    "source_sha",
    "runtime_seconds",
    "status",
    "failure_reason",
]

LABEL_FIELDNAMES = [
    "protocol_id",
    "unit_id",
    "held_out_family",
    "fold_id",
    "capacity",
    "horizon",
    "train_rows",
    "val_rows",
    "train_decisions",
    "val_decisions",
    "c1_c2_label_agreement",
    "mean_abs_label_delta",
    "median_abs_label_delta",
    "fraction_candidate_rankings_changed",
    "fraction_top1_eviction_changed",
    "pi1_hash",
    "source_sha",
    "status",
]

TRAINING_FIELDNAMES = [
    "protocol_id",
    "unit_id",
    "held_out_family",
    "fold_id",
    "capacity",
    "horizon",
    "seed",
    "train_rows",
    "val_rows",
    "train_decisions",
    "val_decisions",
    "pi1_model_path",
    "pi1_model_sha256",
    "pi1_registry_path",
    "pi1_registry_sha256",
    "pi2_model_path",
    "pi2_model_sha256",
    "pi2_model_name",
    "best_model_name",
    "sklearn_current_version",
    "sklearn_pickle_warning_count",
    "sklearn_pickle_compatibility_status",
    "train_runtime_seconds",
    "status",
    "failure_reason",
]

REQUIRED_UNIT_FILES = [
    "policy_comparison.csv",
    "label_agreement.csv",
    "training_summary.csv",
    "unit_summary.json",
]
HASHED_UNIT_FILES = [
    "policy_comparison.csv",
    "label_agreement.csv",
    "training_summary.csv",
]


class ProtocolError(RuntimeError):
    pass


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    config_path: Path
    output_root: Path
    model_root: Path
    data_read_root: Path
    fold_dir: Path
    registry_path: Path


@dataclass(frozen=True)
class RuntimeLimits:
    max_train_rows: int
    max_val_rows: int
    max_train_decisions: Optional[int]
    max_val_decisions: Optional[int]
    max_requests_per_train_trace: Optional[int]
    score_start: int
    score_end: int


@dataclass(frozen=True)
class Unit:
    held_out_family: str
    capacity: int

    @property
    def unit_id(self) -> str:
        return f"{self.held_out_family}_cap{self.capacity}"


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_path(path: str | Path, *, base: Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def _git_sha(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"


def _git_dirty(repo_root: Path) -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except Exception:
        return True


def _scientific_snapshot(config: Mapping[str, object]) -> Dict[str, object]:
    return {
        "protocol_id": config["protocol_id"],
        "version": config["version"],
        "conditions": config["conditions"],
        "families": list(config["folds"]["held_out_families"]),
        "capacities": list(config["capacities"]),
        "horizon": config["horizon"],
        "feature_schema": config["feature_schema"],
        "seed": config["seed"],
        "training_budget": config["training_budget"],
        "evaluation_window": config["evaluation_window"],
        "frozen_pi1_provenance": config["frozen_pi1_provenance"],
    }


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    tmp.replace(path)


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fold_path(paths: Paths, family: str) -> Path:
    return paths.fold_dir / f"{family}.json"


def _split_map_path(paths: Paths, family: str) -> Path:
    return paths.fold_dir / f"{family}_family_split_map.json"


def _load_fold(paths: Paths, family: str) -> Dict[str, object]:
    return _load_json(_fold_path(paths, family))


def _load_split_map(paths: Paths, family: str) -> Dict[str, str]:
    return {str(k): str(v) for k, v in _load_json(_split_map_path(paths, family)).items()}


def _load_train_manifest(paths: Paths, family: str) -> List[Dict[str, str]]:
    fold = _load_fold(paths, family)
    manifest_path = _resolve_path(str(fold["train_manifest"]), base=paths.repo_root)
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _budget_to_decisions(row_budget: int, capacity: int) -> int:
    return int(math.ceil(float(row_budget) / float(max(capacity, 1))))


def _collect_rows_for_split(
    *,
    paths: Paths,
    held_out_family: str,
    split_name: str,
    capacity: int,
    row_budget: int,
    max_decisions_override: Optional[int],
    max_requests_per_trace: Optional[int],
    cfg: ContinuationAblationConfig,
    pi1_model,
    pi1_provenance: FrozenPi1Provenance,
) -> List[Dict[str, object]]:
    split_map = _load_split_map(paths, held_out_family)
    rows: List[Dict[str, object]] = []
    max_decisions_total = max_decisions_override or _budget_to_decisions(row_budget, capacity)
    seen_decisions: set[str] = set()

    for rec in _load_train_manifest(paths, held_out_family):
        trace_family = rec["trace_family"]
        if trace_family == held_out_family:
            raise ProtocolError(f"held-out family {held_out_family} appears in train manifest")
        if split_map.get(trace_family) != split_name:
            continue
        remaining_decisions = max_decisions_total - len(seen_decisions)
        if remaining_decisions <= 0 or len(rows) >= row_budget:
            break
        trace_path = paths.data_read_root / rec["path"]
        reqs, _pages, _src = load_trace_from_any(str(trace_path))
        if max_requests_per_trace:
            reqs = reqs[:max_requests_per_trace]
        new_rows = build_decision_aligned_continuation_rows(
            requests=reqs,
            capacity=capacity,
            trace_name=rec["trace_name"],
            trace_family=trace_family,
            cfg=cfg,
            pi1_model=pi1_model,
            pi1_provenance=pi1_provenance,
            max_decisions=remaining_decisions,
        )
        for row in new_rows:
            row["split"] = split_name
        rows.extend(new_rows)
        seen_decisions.update(str(row["decision_id"]) for row in new_rows)
        if len(rows) >= row_budget:
            rows = rows[:row_budget]
            break

    return rows


def _validate_same_example_gate(rows: Sequence[Mapping[str, object]], *, unit: Unit) -> Dict[str, object]:
    if not rows:
        raise ProtocolError(f"{unit.unit_id}: no aligned C1/C2 rows")
    keys: List[Tuple[str, str]] = []
    for row in rows:
        if int(row["capacity"]) != unit.capacity:
            raise ProtocolError(f"{unit.unit_id}: capacity drift in aligned rows")
        if str(row["trace_family"]) == unit.held_out_family:
            raise ProtocolError(f"{unit.unit_id}: held-out row entered label data")
        if row.get("continuation_mode_c1") != CONTINUATION_LRU:
            raise ProtocolError(f"{unit.unit_id}: bad C1 continuation mode")
        if row.get("continuation_mode_c2") != CONTINUATION_FROZEN_PI1:
            raise ProtocolError(f"{unit.unit_id}: bad C2 continuation mode")
        for required in ("c1_label", "c2_label", "label_delta", "pi1_hash"):
            if required not in row:
                raise ProtocolError(f"{unit.unit_id}: missing {required} in aligned row")
        keys.append((str(row["decision_id"]), str(row["candidate_id"])))
    if len(set(keys)) != len(keys):
        raise ProtocolError(f"{unit.unit_id}: duplicate decision/candidate keys")
    return {
        "aligned_row_count": len(rows),
        "aligned_decision_count": len({k[0] for k in keys}),
        "alignment_status": "PASS",
    }


def _validate_leakage_gate(
    *,
    paths: Paths,
    unit: Unit,
    train_rows: Sequence[Mapping[str, object]],
    val_rows: Sequence[Mapping[str, object]],
    pi1_provenance: FrozenPi1Provenance,
    cfg: ContinuationAblationConfig,
    score_start: int,
    score_end: int,
) -> Dict[str, object]:
    fold = _load_fold(paths, unit.held_out_family)
    manifest = _load_train_manifest(paths, unit.held_out_family)
    manifest_families = {rec["trace_family"] for rec in manifest}
    if unit.held_out_family in manifest_families:
        raise ProtocolError(f"{unit.unit_id}: held-out family appears in training manifest")
    if unit.held_out_family in pi1_provenance.training_families:
        raise ProtocolError(f"{unit.unit_id}: pi1 provenance leaks held-out family")
    if pi1_provenance.validation_family == unit.held_out_family:
        raise ProtocolError(f"{unit.unit_id}: pi1 validation family is held-out")
    if tuple(fold["training_families"]) != pi1_provenance.training_families:
        raise ProtocolError(f"{unit.unit_id}: pi1 training families do not match fold")
    if str(fold["validation_family"]) != pi1_provenance.validation_family:
        raise ProtocolError(f"{unit.unit_id}: pi1 validation family does not match fold")

    train_ids = {str(row["decision_id"]) for row in train_rows}
    val_ids = {str(row["decision_id"]) for row in val_rows}
    overlap = train_ids & val_ids
    if overlap:
        raise ProtocolError(f"{unit.unit_id}: train/validation decision overlap: {sorted(overlap)[:3]}")
    for row in list(train_rows) + list(val_rows):
        if int(row["horizon"]) != int(cfg.horizon):
            raise ProtocolError(f"{unit.unit_id}: horizon drift in label row")
        if row.get("pi1_hash") != pi1_provenance.model_sha256:
            raise ProtocolError(f"{unit.unit_id}: C2 row used wrong pi1 hash")
    if score_start < 0 or score_end <= score_start:
        raise ProtocolError(f"{unit.unit_id}: invalid evaluation window")
    return {
        "leakage_status": "PASS",
        "train_decision_count": len(train_ids),
        "val_decision_count": len(val_ids),
        "held_out_family_absent_from_training_manifest": True,
        "pi1_hash": pi1_provenance.model_sha256,
    }


def _audit_model_load(
    *,
    paths: Paths,
    held_out_family: str,
    objective: str,
) -> Tuple[object, FrozenPi1Provenance, Dict[str, object]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model, prov = load_frozen_pi1_from_registry(
            registry_path=paths.registry_path,
            held_out_family=held_out_family,
            folds_dir=paths.fold_dir,
            objective=objective,
        )
    warning_rows = []
    training_versions = []
    for item in caught:
        msg = str(item.message)
        warning_rows.append({"category": item.category.__name__, "message": msg})
        if " from version " in msg:
            training_versions.append(msg.split(" from version ", 1)[1].split(" ", 1)[0])

    zero_row = {col: 0.0 for col in EVICT_VALUE_V1_FEATURE_COLUMNS}
    try:
        p1 = model.predict_loss_batch([zero_row, zero_row])
        p2 = model.predict_loss_batch([zero_row, zero_row])
    except Exception as exc:
        raise ProtocolError(f"pi1 compatibility prediction check failed for {held_out_family}: {exc}") from exc
    if p1 != p2:
        raise ProtocolError(f"pi1 deterministic prediction check failed for {held_out_family}")

    incompatible_warnings = [w for w in warning_rows if w["category"] == InconsistentVersionWarning.__name__]
    status = "OK"
    if incompatible_warnings:
        status = "OK_WITH_SKLEARN_VERSION_WARNINGS_DETERMINISTIC_PREDICTION_CHECK_PASSED"
    audit = {
        "held_out_family": held_out_family,
        "model_path": prov.model_path,
        "model_sha256": prov.model_sha256,
        "current_sklearn_version": sklearn.__version__,
        "training_sklearn_versions_from_warnings": sorted(set(training_versions)),
        "warning_count": len(warning_rows),
        "warnings": warning_rows,
        "deterministic_prediction_check": "PASS",
        "compatibility_status": status,
    }
    return model, prov, audit


def _misses_for_policy(
    policy: BasePolicy,
    reqs,
    pages,
    capacity: int,
    *,
    score_start: int,
    score_end: int,
) -> Dict[str, object]:
    t0 = time.time()
    result = run_policy(policy, reqs, pages, capacity)
    runtime = time.time() - t0
    window = score_window(result.events, score_start, score_end)
    return {
        "history_requests": window.history_requests,
        "scored_requests": window.scored_requests,
        "hits": window.hits,
        "misses": window.misses,
        "miss_ratio": window.miss_ratio,
        "runtime_seconds": runtime,
    }


def _policy_rows_for_unit(
    *,
    protocol_id: str,
    unit: Unit,
    fold: Mapping[str, object],
    trace_sha: str,
    score_start: int,
    score_end: int,
    seed: int,
    horizon: int,
    source_sha: str,
    pi1_model_path: str,
    pi1_hash: str,
    pi2_model_path: str,
    pi2_hash: str,
    c0_result: Mapping[str, object],
    c1_result: Mapping[str, object],
    c2_result: Mapping[str, object],
) -> List[Dict[str, object]]:
    common = {
        "protocol_id": protocol_id,
        "unit_id": unit.unit_id,
        "held_out_family": unit.held_out_family,
        "fold_id": fold["fold_id"],
        "capacity": unit.capacity,
        "trace_name": fold["test_trace_name"],
        "trace_path": fold["test_trace_path"],
        "trace_sha256": trace_sha,
        "history_start": 0,
        "history_end": score_start,
        "score_start": score_start,
        "score_end": score_end,
        "seed": seed,
        "horizon": horizon,
        "source_sha": source_sha,
        "status": "ok",
        "failure_reason": "",
    }
    specs = [
        (CONDITION_C0_BASELINE_LRU, "lru", "", "", c0_result),
        (CONDITION_C1_LRU_CONTINUATION_LEARNED_PI1, "pi1_recursive", pi1_model_path, pi1_hash, c1_result),
        (CONDITION_C2_PI1_CONTINUATION_LEARNED_PI2, "pi2_recursive", pi2_model_path, pi2_hash, c2_result),
    ]
    rows = []
    for condition, policy, model_path, model_hash, result in specs:
        row = dict(common)
        row.update(
            {
                "condition": condition,
                "policy": policy,
                "model_path": model_path,
                "model_sha256": model_hash,
                "history_requests": result["history_requests"],
                "scored_requests": result["scored_requests"],
                "hits": result["hits"],
                "misses": result["misses"],
                "miss_ratio": result["miss_ratio"],
                "runtime_seconds": result["runtime_seconds"],
            }
        )
        rows.append(row)
    return rows


def _pi2_best_model_name(model_name: str) -> str:
    prefix = "continuation_policy_causal_ablation_pi2_"
    if model_name.startswith(prefix):
        return model_name[len(prefix) :]
    return model_name


def _hash_unit_files(unit_dir: Path) -> Dict[str, str]:
    hashes = {}
    for rel in HASHED_UNIT_FILES:
        path = unit_dir / rel
        if not path.exists():
            raise ProtocolError(f"unit missing required file {path}")
        hashes[rel] = sha256_of_file(path)
    return hashes


def _unit_dir(paths: Paths, unit: Unit) -> Path:
    return paths.output_root / "units" / unit.unit_id


def _final_pi2_path(paths: Paths, unit: Unit) -> Path:
    return paths.model_root / unit.held_out_family / f"cap{unit.capacity}" / "pi2.pkl"


def validate_completed_unit(paths: Paths, unit: Unit, config: Mapping[str, object]) -> Dict[str, object]:
    unit_dir = _unit_dir(paths, unit)
    summary_path = unit_dir / "unit_summary.json"
    if not summary_path.exists():
        raise ProtocolError(f"{unit.unit_id}: missing unit_summary.json")
    summary = _load_json(summary_path)
    if summary.get("status") != "complete":
        raise ProtocolError(f"{unit.unit_id}: status is not complete")
    if summary.get("unit_id") != unit.unit_id:
        raise ProtocolError(f"{unit.unit_id}: unit_id mismatch")
    if summary.get("protocol_id") != config["protocol_id"]:
        raise ProtocolError(f"{unit.unit_id}: protocol_id mismatch")
    hashes = _hash_unit_files(unit_dir)
    recorded = summary.get("unit_file_sha256", {})
    for rel, digest in hashes.items():
        if recorded.get(rel) != digest:
            raise ProtocolError(f"{unit.unit_id}: file hash mismatch for {rel}")
    pi2_path = Path(str(summary["pi2_model_path"]))
    if not pi2_path.exists():
        raise ProtocolError(f"{unit.unit_id}: missing pi2 model {pi2_path}")
    if sha256_of_file(pi2_path) != summary.get("pi2_model_sha256"):
        raise ProtocolError(f"{unit.unit_id}: pi2 model hash mismatch")
    for csv_name, expected_rows in {
        "policy_comparison.csv": 3,
        "label_agreement.csv": 1,
        "training_summary.csv": 1,
    }.items():
        rows = _read_csv(unit_dir / csv_name)
        if len(rows) != expected_rows:
            raise ProtocolError(f"{unit.unit_id}: {csv_name} row count {len(rows)} != {expected_rows}")
    return summary


def _run_unit(
    *,
    config: Mapping[str, object],
    paths: Paths,
    limits: RuntimeLimits,
    unit: Unit,
    source_sha: str,
) -> Dict[str, object]:
    final_unit_dir = _unit_dir(paths, unit)
    tmp_unit_dir = paths.output_root / "units" / f".{unit.unit_id}.tmp.{os.getpid()}"
    if tmp_unit_dir.exists():
        shutil.rmtree(tmp_unit_dir)
    tmp_unit_dir.mkdir(parents=True)

    protocol_id = str(config["protocol_id"])
    objective = str(config["frozen_pi1_provenance"]["required_objective"])
    cfg = ContinuationAblationConfig(horizon=int(config["horizon"]))
    started = time.time()
    train_started = time.time()

    pi1_model, pi1_prov, model_audit = _audit_model_load(
        paths=paths,
        held_out_family=unit.held_out_family,
        objective=objective,
    )
    train_rows = _collect_rows_for_split(
        paths=paths,
        held_out_family=unit.held_out_family,
        split_name="train",
        capacity=unit.capacity,
        row_budget=limits.max_train_rows,
        max_decisions_override=limits.max_train_decisions,
        max_requests_per_trace=limits.max_requests_per_train_trace,
        cfg=cfg,
        pi1_model=pi1_model,
        pi1_provenance=pi1_prov,
    )
    val_rows = _collect_rows_for_split(
        paths=paths,
        held_out_family=unit.held_out_family,
        split_name="val",
        capacity=unit.capacity,
        row_budget=limits.max_val_rows,
        max_decisions_override=limits.max_val_decisions,
        max_requests_per_trace=limits.max_requests_per_train_trace,
        cfg=cfg,
        pi1_model=pi1_model,
        pi1_provenance=pi1_prov,
    )
    if not train_rows or not val_rows:
        raise ProtocolError(f"{unit.unit_id}: insufficient train/val rows")
    alignment_train = _validate_same_example_gate(train_rows, unit=unit)
    alignment_val = _validate_same_example_gate(val_rows, unit=unit)
    leakage = _validate_leakage_gate(
        paths=paths,
        unit=unit,
        train_rows=train_rows,
        val_rows=val_rows,
        pi1_provenance=pi1_prov,
        cfg=cfg,
        score_start=limits.score_start,
        score_end=limits.score_end,
    )

    pi2 = train_pi2_from_c2_labels(
        train_rows=train_rows,
        val_rows=val_rows,
        seed=int(config["seed"]),
        pi1_provenance=pi1_prov,
    )
    train_runtime = time.time() - train_started
    tmp_pi2 = tmp_unit_dir / "pi2.pkl"
    pi2.save(tmp_pi2)
    pi2_hash = sha256_of_file(tmp_pi2)
    final_pi2 = _final_pi2_path(paths, unit)
    final_pi2.parent.mkdir(parents=True, exist_ok=True)
    tmp_final_pi2 = final_pi2.with_name(final_pi2.name + f".tmp.{os.getpid()}")
    shutil.copy2(tmp_pi2, tmp_final_pi2)
    tmp_final_pi2.replace(final_pi2)

    fold = _load_fold(paths, unit.held_out_family)
    test_trace_path = paths.data_read_root / str(fold["test_trace_path"])
    test_reqs, test_pages, _src = load_trace_from_any(str(test_trace_path))
    if limits.score_end > len(test_reqs):
        raise ProtocolError(
            f"{unit.unit_id}: score_end {limits.score_end} exceeds test trace length {len(test_reqs)}"
        )
    eval_reqs = test_reqs[: limits.score_end]
    trace_sha = sha256_of_file(test_trace_path)
    c0_result = _misses_for_policy(
        LRUPolicy(),
        eval_reqs,
        test_pages,
        unit.capacity,
        score_start=limits.score_start,
        score_end=limits.score_end,
    )
    c1_result = _misses_for_policy(
        DistributionShiftEvalPolicy(model_path=pi1_prov.model_path),
        eval_reqs,
        test_pages,
        unit.capacity,
        score_start=limits.score_start,
        score_end=limits.score_end,
    )
    c2_result = _misses_for_policy(
        DistributionShiftEvalPolicy(model_path=str(final_pi2)),
        eval_reqs,
        test_pages,
        unit.capacity,
        score_start=limits.score_start,
        score_end=limits.score_end,
    )

    policy_rows = _policy_rows_for_unit(
        protocol_id=protocol_id,
        unit=unit,
        fold=fold,
        trace_sha=trace_sha,
        score_start=limits.score_start,
        score_end=limits.score_end,
        seed=int(config["seed"]),
        horizon=int(config["horizon"]),
        source_sha=source_sha,
        pi1_model_path=pi1_prov.model_path,
        pi1_hash=pi1_prov.model_sha256,
        pi2_model_path=str(final_pi2),
        pi2_hash=pi2_hash,
        c0_result=c0_result,
        c1_result=c1_result,
        c2_result=c2_result,
    )
    label_metrics = label_agreement_metrics(train_rows + val_rows)
    label_row = {
        "protocol_id": protocol_id,
        "unit_id": unit.unit_id,
        "held_out_family": unit.held_out_family,
        "fold_id": fold["fold_id"],
        "capacity": unit.capacity,
        "horizon": int(config["horizon"]),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_decisions": len({str(r["decision_id"]) for r in train_rows}),
        "val_decisions": len({str(r["decision_id"]) for r in val_rows}),
        "c1_c2_label_agreement": label_metrics["c1_c2_label_agreement"],
        "mean_abs_label_delta": label_metrics["mean_abs_label_delta"],
        "median_abs_label_delta": label_metrics["median_abs_label_delta"],
        "fraction_candidate_rankings_changed": label_metrics["fraction_candidate_rankings_changed"],
        "fraction_top1_eviction_changed": label_metrics["fraction_top1_eviction_changed"],
        "pi1_hash": pi1_prov.model_sha256,
        "source_sha": source_sha,
        "status": "ok",
    }
    training_row = {
        "protocol_id": protocol_id,
        "unit_id": unit.unit_id,
        "held_out_family": unit.held_out_family,
        "fold_id": fold["fold_id"],
        "capacity": unit.capacity,
        "horizon": int(config["horizon"]),
        "seed": int(config["seed"]),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_decisions": len({str(r["decision_id"]) for r in train_rows}),
        "val_decisions": len({str(r["decision_id"]) for r in val_rows}),
        "pi1_model_path": pi1_prov.model_path,
        "pi1_model_sha256": pi1_prov.model_sha256,
        "pi1_registry_path": pi1_prov.registry_path,
        "pi1_registry_sha256": pi1_prov.registry_sha256,
        "pi2_model_path": str(final_pi2),
        "pi2_model_sha256": pi2_hash,
        "pi2_model_name": pi2.model_name,
        "best_model_name": _pi2_best_model_name(pi2.model_name),
        "sklearn_current_version": sklearn.__version__,
        "sklearn_pickle_warning_count": model_audit["warning_count"],
        "sklearn_pickle_compatibility_status": model_audit["compatibility_status"],
        "train_runtime_seconds": train_runtime,
        "status": "ok",
        "failure_reason": "",
    }

    _write_csv(tmp_unit_dir / "policy_comparison.csv", POLICY_FIELDNAMES, policy_rows)
    _write_csv(tmp_unit_dir / "label_agreement.csv", LABEL_FIELDNAMES, [label_row])
    _write_csv(tmp_unit_dir / "training_summary.csv", TRAINING_FIELDNAMES, [training_row])

    unit_summary: Dict[str, object] = {
        "protocol_id": protocol_id,
        "unit_id": unit.unit_id,
        "held_out_family": unit.held_out_family,
        "capacity": unit.capacity,
        "status": "complete",
        "source_sha": source_sha,
        "source_tree_dirty": _git_dirty(paths.repo_root),
        "runtime_seconds": time.time() - started,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_decisions": len({str(r["decision_id"]) for r in train_rows}),
        "val_decisions": len({str(r["decision_id"]) for r in val_rows}),
        "same_example_gate": {"train": alignment_train, "val": alignment_val},
        "leakage_gate": leakage,
        "model_compatibility_audit": model_audit,
        "pi1_model_path": pi1_prov.model_path,
        "pi1_model_sha256": pi1_prov.model_sha256,
        "pi2_model_path": str(final_pi2),
        "pi2_model_sha256": pi2_hash,
    }
    _atomic_write_json(tmp_unit_dir / "unit_summary.json", unit_summary)
    unit_summary["unit_file_sha256"] = _hash_unit_files(tmp_unit_dir)
    _atomic_write_json(tmp_unit_dir / "unit_summary.json", unit_summary)

    if final_unit_dir.exists():
        raise ProtocolError(f"{unit.unit_id}: final unit dir already exists; use --resume to skip valid units")
    tmp_unit_dir.replace(final_unit_dir)
    return validate_completed_unit(paths, unit, config)


def _completed_units(paths: Paths, config: Mapping[str, object], units: Sequence[Unit]) -> Dict[str, Dict[str, object]]:
    completed = {}
    for unit in units:
        unit_dir = _unit_dir(paths, unit)
        if not unit_dir.exists():
            continue
        completed[unit.unit_id] = validate_completed_unit(paths, unit, config)
    return completed


def _dedupe_rows(rows: Iterable[Dict[str, str]], key_fields: Sequence[str], *, source: str) -> List[Dict[str, str]]:
    out = []
    seen = set()
    for row in rows:
        key = tuple(row.get(k, "") for k in key_fields)
        if key in seen:
            raise ProtocolError(f"duplicate {source} row key: {key}")
        seen.add(key)
        out.append(row)
    return out


def rebuild_global_outputs(
    *,
    paths: Paths,
    config: Mapping[str, object],
    units: Sequence[Unit],
    source_sha: str,
    run_status: str,
) -> Dict[str, object]:
    policy_rows: List[Dict[str, str]] = []
    label_rows: List[Dict[str, str]] = []
    training_rows: List[Dict[str, str]] = []
    summaries = []
    for unit in units:
        unit_dir = _unit_dir(paths, unit)
        if not unit_dir.exists():
            continue
        summaries.append(validate_completed_unit(paths, unit, config))
        policy_rows.extend(_read_csv(unit_dir / "policy_comparison.csv"))
        label_rows.extend(_read_csv(unit_dir / "label_agreement.csv"))
        training_rows.extend(_read_csv(unit_dir / "training_summary.csv"))

    policy_rows = _dedupe_rows(policy_rows, ["held_out_family", "capacity", "condition"], source="policy")
    label_rows = _dedupe_rows(label_rows, ["held_out_family", "capacity"], source="label")
    training_rows = _dedupe_rows(training_rows, ["held_out_family", "capacity"], source="training")

    _write_csv(paths.output_root / "policy_comparison.csv", POLICY_FIELDNAMES, policy_rows)
    _write_csv(paths.output_root / "label_agreement.csv", LABEL_FIELDNAMES, label_rows)
    _write_csv(paths.output_root / "training_summary.csv", TRAINING_FIELDNAMES, training_rows)

    manifest = {
        "protocol_id": config["protocol_id"],
        "status": run_status,
        "unit_granularity": "held_out_family_x_capacity",
        "expected_units": len(units),
        "completed_units": len(summaries),
        "units": sorted(summaries, key=lambda x: str(x["unit_id"])),
    }
    integrity = {
        "protocol_id": config["protocol_id"],
        "status": "PASS" if len(summaries) == len(units) else "PARTIAL",
        "expected_policy_rows": len(summaries) * 3,
        "actual_policy_rows": len(policy_rows),
        "expected_label_rows": len(summaries),
        "actual_label_rows": len(label_rows),
        "expected_training_rows": len(summaries),
        "actual_training_rows": len(training_rows),
        "duplicate_check": "PASS",
    }
    provenance = {
        "protocol_id": config["protocol_id"],
        "source_sha": source_sha,
        "source_tree_dirty": _git_dirty(paths.repo_root),
        "python_version": sys.version,
        "sklearn_version": sklearn.__version__,
        "config_path": str(paths.config_path),
        "output_root": str(paths.output_root),
        "model_root": str(paths.model_root),
        "data_read_root": str(paths.data_read_root),
        "created_at_unix": time.time(),
    }
    _atomic_write_json(paths.output_root / "unit_completion_manifest.json", manifest)
    _atomic_write_json(paths.output_root / "integrity_summary.json", integrity)
    _atomic_write_json(paths.output_root / "provenance.json", provenance)
    _atomic_write_text(
        paths.output_root / "README.md",
        "# Continuation Policy Causal Ablation Production Output\n\n"
        f"Status: `{run_status}`\n\n"
        "This directory is produced by `scripts/experiments/run_continuation_policy_causal_ablation.py`.\n"
        "Rows are aggregated from atomically completed `(held_out_family, capacity)` units.\n",
    )
    return integrity


def _resolve_runtime_limits(config: Mapping[str, object], args: argparse.Namespace) -> RuntimeLimits:
    budget = config["training_budget"]
    window = config["evaluation_window"]
    return RuntimeLimits(
        max_train_rows=int(args.max_train_rows or budget["max_train_rows"]),
        max_val_rows=int(args.max_val_rows or budget["max_val_rows"]),
        max_train_decisions=args.max_train_decisions,
        max_val_decisions=args.max_val_decisions,
        max_requests_per_train_trace=args.max_requests_per_train_trace,
        score_start=int(args.score_start if args.score_start is not None else window["score_start"]),
        score_end=int(args.score_end if args.score_end is not None else window["score_end"]),
    )


def _select_units(config: Mapping[str, object], args: argparse.Namespace) -> List[Unit]:
    all_families = [str(x) for x in config["folds"]["held_out_families"]]
    all_caps = [int(x) for x in config["capacities"]]
    families = args.families or all_families
    capacities = args.capacities or all_caps
    unknown_families = sorted(set(families) - set(all_families))
    unknown_caps = sorted(set(int(c) for c in capacities) - set(all_caps))
    if unknown_families:
        raise ProtocolError(f"unknown families requested: {unknown_families}")
    if unknown_caps:
        raise ProtocolError(f"unknown capacities requested: {unknown_caps}")
    return [Unit(family, int(cap)) for family in families for cap in capacities]


def _make_paths(config: Mapping[str, object], args: argparse.Namespace, repo_root: Path) -> Paths:
    output_cfg = config["output"]
    frozen = config["frozen_pi1_provenance"]
    return Paths(
        repo_root=repo_root,
        config_path=_resolve_path(args.config, base=repo_root),
        output_root=_resolve_path(args.output_root or output_cfg["analysis_root"], base=repo_root),
        model_root=_resolve_path(args.model_root or output_cfg["model_root"], base=repo_root),
        data_read_root=_resolve_path(args.data_read_root or DEFAULT_DATA_READ_ROOT, base=repo_root),
        fold_dir=_resolve_path(config["folds"]["fold_dir"], base=repo_root),
        registry_path=_resolve_path(args.registry or frozen["registry"], base=repo_root),
    )


def _check_existing_output(paths: Paths, config: Mapping[str, object], *, resume: bool, preflight_only: bool) -> None:
    snapshot_path = paths.output_root / "config_snapshot.json"
    if snapshot_path.exists():
        snapshot = _load_json(snapshot_path)
        if _scientific_snapshot(snapshot) != _scientific_snapshot(config):
            raise ProtocolError("existing output config_snapshot.json is incompatible with current config")
    elif paths.output_root.exists() and any(paths.output_root.iterdir()) and not preflight_only:
        raise ProtocolError("output directory exists without config_snapshot.json; refusing to mix outputs")
    if paths.output_root.exists() and any(paths.output_root.iterdir()) and not resume and not preflight_only:
        raise ProtocolError("output directory already contains files; rerun with --resume after validation")


def preflight(
    *,
    config: Mapping[str, object],
    paths: Paths,
    units: Sequence[Unit],
    limits: RuntimeLimits,
    resume: bool,
) -> Dict[str, object]:
    _check_existing_output(paths, config, resume=resume, preflight_only=True)
    failures = []
    warnings_out = []
    if str(config.get("feature_schema")) != "EVICT_VALUE_V1_FEATURE_COLUMNS":
        failures.append("feature_schema must be EVICT_VALUE_V1_FEATURE_COLUMNS")
    if int(config["horizon"]) <= 0:
        failures.append("horizon must be positive")
    if int(config["seed"]) != 0:
        failures.append("production seed must remain 0")
    if limits.score_end <= limits.score_start:
        failures.append("score_end must be greater than score_start")
    for path, label in [
        (paths.fold_dir, "fold_dir"),
        (paths.registry_path, "registry"),
        (paths.data_read_root, "data_read_root"),
    ]:
        if not path.exists():
            failures.append(f"{label} missing: {path}")
    try:
        paths.output_root.mkdir(parents=True, exist_ok=True)
        tmp = paths.output_root / f".preflight_write_test.{os.getpid()}"
        tmp.write_text("ok\n", encoding="utf-8")
        tmp.unlink()
        paths.model_root.mkdir(parents=True, exist_ok=True)
        tmp_model = paths.model_root / f".preflight_write_test.{os.getpid()}"
        tmp_model.write_text("ok\n", encoding="utf-8")
        tmp_model.unlink()
    except Exception as exc:
        failures.append(f"output/model paths not writable: {exc}")

    objective = str(config["frozen_pi1_provenance"]["required_objective"])
    model_audits_by_family: Dict[str, Dict[str, object]] = {}
    for unit in units:
        try:
            fold = _load_fold(paths, unit.held_out_family)
            split = _load_split_map(paths, unit.held_out_family)
            manifest = _load_train_manifest(paths, unit.held_out_family)
            if unit.held_out_family in {rec["trace_family"] for rec in manifest}:
                failures.append(f"{unit.unit_id}: held-out family appears in train manifest")
            if str(fold["validation_family"]) == unit.held_out_family:
                failures.append(f"{unit.unit_id}: validation family equals held-out")
            test_trace = paths.data_read_root / str(fold["test_trace_path"])
            if not test_trace.exists():
                failures.append(f"{unit.unit_id}: missing test trace {test_trace}")
            else:
                try:
                    reqs, _pages, _src = load_trace_from_any(str(test_trace))
                    if len(reqs) < limits.score_end:
                        failures.append(f"{unit.unit_id}: test trace length {len(reqs)} < score_end {limits.score_end}")
                except Exception as exc:
                    failures.append(f"{unit.unit_id}: cannot load test trace: {exc}")
            for rec in manifest:
                if rec["trace_family"] not in split:
                    failures.append(f"{unit.unit_id}: manifest family {rec['trace_family']} missing from split map")
                trace_path = paths.data_read_root / rec["path"]
                if not trace_path.exists():
                    failures.append(f"{unit.unit_id}: missing training trace {trace_path}")
            if unit.held_out_family not in model_audits_by_family:
                _model, _prov, audit = _audit_model_load(
                    paths=paths,
                    held_out_family=unit.held_out_family,
                    objective=objective,
                )
                model_audits_by_family[unit.held_out_family] = audit
                if audit["warning_count"]:
                    warnings_out.append(audit)
        except Exception as exc:
            failures.append(f"{unit.unit_id}: {type(exc).__name__}: {exc}")

    report = {
        "protocol_id": config["protocol_id"],
        "status": "PASS" if not failures else "FAIL",
        "unit_count": len(units),
        "failures": failures,
        "model_warning_audits": warnings_out,
        "current_sklearn_version": sklearn.__version__,
        "output_root": str(paths.output_root),
        "model_root": str(paths.model_root),
    }
    if failures:
        raise ProtocolError(json.dumps(report, indent=2))
    return report


def _initialize_output(paths: Paths, config: Mapping[str, object], source_sha: str) -> None:
    paths.output_root.mkdir(parents=True, exist_ok=True)
    paths.model_root.mkdir(parents=True, exist_ok=True)
    snapshot = dict(config)
    snapshot["source_sha_at_runner_start"] = source_sha
    _atomic_write_json(paths.output_root / "config_snapshot.json", snapshot)


def run(config: Mapping[str, object], paths: Paths, units: Sequence[Unit], limits: RuntimeLimits, args) -> Dict[str, object]:
    source_sha = _git_sha(paths.repo_root)
    _check_existing_output(paths, config, resume=args.resume, preflight_only=False)
    _initialize_output(paths, config, source_sha)
    completed = _completed_units(paths, config, units) if args.resume else {}

    started = time.time()
    max_seconds = None if args.max_wall_hours is None else float(args.max_wall_hours) * 3600.0
    ran = 0
    skipped = 0
    for unit in units:
        if unit.unit_id in completed:
            print(f"[skip] {unit.unit_id} already complete")
            skipped += 1
            continue
        if max_seconds is not None and time.time() - started >= max_seconds:
            print("[stop] wall-time budget reached before starting next unit")
            break
        print(f"[run] {unit.unit_id}")
        _run_unit(config=config, paths=paths, limits=limits, unit=unit, source_sha=source_sha)
        ran += 1
        rebuild_global_outputs(
            paths=paths,
            config=config,
            units=units,
            source_sha=source_sha,
            run_status="PARTIAL",
        )
    integrity = rebuild_global_outputs(
        paths=paths,
        config=config,
        units=units,
        source_sha=source_sha,
        run_status="COMPLETE" if len(_completed_units(paths, config, units)) == len(units) else "PARTIAL",
    )
    return {"ran_units": ran, "skipped_units": skipped, "integrity": integrity}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--families", nargs="+", default=None)
    ap.add_argument("--capacities", nargs="+", type=int, default=None)
    ap.add_argument("--data-read-root", type=Path, default=None)
    ap.add_argument("--output-root", type=Path, default=None)
    ap.add_argument("--model-root", type=Path, default=None)
    ap.add_argument("--registry", type=Path, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--preflight-only", action="store_true")
    ap.add_argument("--max-wall-hours", type=float, default=None)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--max-train-decisions", type=int, default=None)
    ap.add_argument("--max-val-decisions", type=int, default=None)
    ap.add_argument("--max-requests-per-train-trace", type=int, default=None)
    ap.add_argument("--score-start", type=int, default=None)
    ap.add_argument("--score-end", type=int, default=None)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    repo_root = _repo_root()
    config_path = _resolve_path(args.config, base=repo_root)
    config = _load_json(config_path)
    paths = _make_paths(config, args, repo_root)
    units = _select_units(config, args)
    limits = _resolve_runtime_limits(config, args)
    try:
        report = preflight(config=config, paths=paths, units=units, limits=limits, resume=args.resume)
        if args.preflight_only:
            print(json.dumps(report, indent=2, sort_keys=True))
            return 0
        result = run(config, paths, units, limits, args)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except ProtocolError as exc:
        print(f"[blocked] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
