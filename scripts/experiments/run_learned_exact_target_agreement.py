"""Resumable compact agreement diagnostic for frozen held-out models.

This is a diagnostic of target agreement and regret, not a deployable-policy
training runner. Unit metadata is finalized with canonical paths before the
temporary unit directory is atomically renamed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, Mapping, Sequence

from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_END, SCORE_START
from lafc.oracle_diagnostics import replay_score_driven_policy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.supervision_objective_ablation import ObjectiveAblationConfig

try:
    from .run_exact_target_oracle_diagnostic import (
        REPO_ROOT,
        _find_registry_record,
        _load_registry,
        _learned_scorer,
        _verify_eligible_model,
    )
except ImportError:  # Direct execution from scripts/experiments/.
    from run_exact_target_oracle_diagnostic import (  # type: ignore[no-redef]
        REPO_ROOT,
        _find_registry_record,
        _load_registry,
        _learned_scorer,
        _verify_eligible_model,
    )


CONFIG_PATH = REPO_ROOT / "configs/learned_exact_target_agreement_v1.json"
DEFAULT_OUT = REPO_ROOT / "analysis/learned_exact_target_agreement_v1"
REGISTRY_PATH = REPO_ROOT / "analysis/supervision_objective_ablation_v1/model_registry.json"
FOLDS_DIR = REPO_ROOT / "configs/fair_cross_family_v1/folds"
OBJECTIVE = "eviction_loss"
REGISTRY_OBJECTIVE = "objective_eviction_loss"


def _finalized_output_paths(outputs: Mapping[str, Any], final_dir: Path) -> Dict[str, str]:
    return {key: str(final_dir / Path(value).name) for key, value in outputs.items()}


def _git(args: Sequence[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", delete=False, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temporary = Path(fh.name)
    os.replace(temporary, path)


def _fold(family: str) -> Mapping[str, Any]:
    return json.loads((FOLDS_DIR / f"{family}.json").read_text(encoding="utf-8"))


def _trace_path(fold: Mapping[str, Any], data_root: Path) -> Path:
    relative = Path(str(fold["test_trace_path"]))
    for candidate in (data_root / relative, REPO_ROOT / relative, REPO_ROOT.parent / "Augmented-caching" / relative):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"trace not found: {relative}")


def _finite(value: Any) -> bool:
    return value is None or not isinstance(value, float) or math.isfinite(value)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summary(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None, "p95": None, "p99": None, "min": None, "max": None}
    xs = sorted(float(value) for value in values)
    def quantile(q: float) -> float:
        index = min(len(xs) - 1, max(0, math.ceil(q * len(xs)) - 1))
        return xs[index]
    return {"mean": sum(xs) / len(xs), "median": median(xs), "p90": quantile(0.90), "p95": quantile(0.95), "p99": quantile(0.99), "min": xs[0], "max": xs[-1]}


def _margin(values: Mapping[str, float]) -> tuple[float, float | None]:
    ordered = sorted(float(value) for value in values.values())
    distinct = sorted(set(ordered))
    ordinary = ordered[1] - ordered[0] if len(ordered) > 1 else 0.0
    strict = distinct[1] - distinct[0] if len(distinct) > 1 else None
    return ordinary, strict


def _verify_registry(registry_path: Path, families: Sequence[str]) -> Dict[str, Any]:
    registry = _load_registry(registry_path)
    records = {}
    for family in families:
        model_path, record = _verify_eligible_model(registry_path, family)
        if record.get("objective") != REGISTRY_OBJECTIVE:
            raise ValueError(f"wrong registry objective for {family}")
        actual_hash = _sha256(model_path)
        if actual_hash != record.get("model_artifact_sha256"):
            raise ValueError(f"model hash mismatch for {family}")
        training = list(record.get("training_families", []))
        if family in training or record.get("held_out_family") != family:
            raise ValueError(f"held-out-family leakage for {family}")
        records[family] = {
            "model_path": str(model_path),
            "model_sha256": actual_hash,
            "objective": record.get("objective"),
            "held_out_family": record.get("held_out_family"),
            "training_families": training,
            "validation_family": record.get("validation_family"),
            "selected_hyperparameters": record.get("selected_hyperparameters"),
        }
    if len(records) != 7:
        raise ValueError(f"expected seven held-out models, found {len(records)}")
    return {"path": str(registry_path), "MODEL_SELECTION_FROZEN": registry.get("MODEL_SELECTION_FROZEN"), "records": records}


def _analyze_unit(*, requests, pages, family: str, capacity: int, trace_path: Path, trace_sha: str, fold: Mapping[str, Any], model_record: Mapping[str, Any], model_path: Path, out_dir: Path) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry_record = dict(model_record)
    scorer = _learned_scorer(model_path)
    result = replay_score_driven_policy(
        requests=requests,
        capacity=capacity,
        trace_name=str(fold["test_trace_name"]),
        trace_family=family,
        cfg=ObjectiveAblationConfig(horizon=4),
        objective=OBJECTIVE,
        scorer=scorer,
        policy_name="learned_eviction_loss_scalar",
    )
    scored = [decision for decision in result.decisions if SCORE_START <= decision.request_t < SCORE_END]
    if not scored:
        raise ValueError(f"no scored decisions for {family}/cap{capacity}")
    set_agree = sum(decision.chosen_candidate in decision.optimal_candidates for decision in scored)
    lex_agree = sum(decision.chosen_candidate == decision.exact_candidate for decision in scored)
    unique = [d for d in scored if len(d.optimal_candidates) == 1]
    multiple = [d for d in scored if len(d.optimal_candidates) > 1]
    all_tied = [d for d in scored if len(set(d.candidate_values.values())) == 1]
    regrets = [float(d.target_regret) for d in scored]
    ordinary_margins = []
    strict_margins = []
    positive_ordinary = []
    for decision in scored:
        ordinary, strict = _margin(decision.candidate_values)
        ordinary_margins.append(ordinary)
        if ordinary > 0:
            positive_ordinary.append(ordinary)
        if strict is not None:
            strict_margins.append(strict)
    bins = {"ordinary_zero": [], "ordinary_positive_le_0_5": [], "ordinary_positive_gt_0_5": []}
    for decision in scored:
        ordinary, _strict = _margin(decision.candidate_values)
        if ordinary == 0:
            bins["ordinary_zero"].append(decision)
        elif ordinary <= 0.5:
            bins["ordinary_positive_le_0_5"].append(decision)
        else:
            bins["ordinary_positive_gt_0_5"].append(decision)
    conditioned = {}
    for name, subset in (("all", scored), ("unique_exact_winner", unique), ("multiple_exact_optima", multiple), ("all_candidates_tied", all_tied)):
        subset_regrets = [float(d.target_regret) for d in subset]
        conditioned[name] = {"decision_count": len(subset), "set_aware_agreement_fraction": _fraction(sum(d.chosen_candidate in d.optimal_candidates for d in subset), len(subset)), "lexicographic_agreement_fraction": _fraction(sum(d.chosen_candidate == d.exact_candidate for d in subset), len(subset)), "zero_regret_fraction": _fraction(sum(d.target_regret == 0.0 for d in subset), len(subset)), "regret_summary": _summary(subset_regrets)}
    margin_conditioned = {}
    for name, subset in bins.items():
        margin_conditioned[name] = {"decision_count": len(subset), "set_aware_agreement_fraction": _fraction(sum(d.chosen_candidate in d.optimal_candidates for d in subset), len(subset)), "lexicographic_agreement_fraction": _fraction(sum(d.chosen_candidate == d.exact_candidate for d in subset), len(subset)), "regret_summary": _summary([float(d.target_regret) for d in subset])}
    lru = run_policy(LRUPolicy(), requests, pages, capacity)
    learned_window_misses = sum(not hit for hit in result.hit_sequence[SCORE_START:SCORE_END])
    lru_window_misses = sum(not hit for hit in [event.hit for event in lru.events][SCORE_START:SCORE_END])
    summary = {
        "status": "COMPLETE",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"head": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"]), "dirty_status": _git(["status", "--short"])},
        "protocol": {"family": family, "capacity": capacity, "horizon": 4, "history": [0, 10000], "score": [10000, 50000], "trajectory": "learned evict_value_v1 scalar deployment trajectory; exact target evaluated on each learned decision state", "target_semantics": "Finite-horizon H4 eviction-loss target with LRU continuation", "regret_definition": "chosen exact target value minus minimum exact target value"},
        "trace": {"family": family, "name": fold["test_trace_name"], "path": str(trace_path), "sha256": trace_sha, "fold": dict(fold)},
        "model": registry_record,
        "decision_metrics": {"decision_count": len(scored), "set_aware_agreement_fraction": _fraction(set_agree, len(scored)), "lexicographic_agreement_fraction": _fraction(lex_agree, len(scored)), "set_aware_disagreement_count": len(scored) - set_agree, "lexicographic_disagreement_count": len(scored) - lex_agree, "unique_exact_winner_count": len(unique), "multiple_exact_optima_count": len(multiple), "all_candidates_tied_count": len(all_tied)},
        "regret_metrics": {"zero_regret_fraction": _fraction(sum(value == 0.0 for value in regrets), len(regrets)), "positive_regret_fraction": _fraction(sum(value > 0.0 for value in regrets), len(regrets)), "summary": _summary(regrets)},
        "margin_metrics": {"ordinary_margin_zero_fraction": _fraction(sum(value == 0.0 for value in ordinary_margins), len(ordinary_margins)), "strict_positive_ordinary_margin_fraction": _fraction(len(positive_ordinary), len(ordinary_margins)), "positive_ordinary_margin_summary": _summary(positive_ordinary), "strict_distinct_margin_summary": _summary(strict_margins)},
        "conditioned_metrics": conditioned,
        "margin_conditioned_metrics": margin_conditioned,
        "online_metrics": {"learned_misses": learned_window_misses, "lru_misses": lru_window_misses, "learned_minus_lru": learned_window_misses - lru_window_misses},
        "outputs": {"summary_json": str(out_dir / "summary.json"), "provenance_json": str(out_dir / "provenance.json")},
    }
    _atomic_json(out_dir / "summary.json", summary)
    _atomic_json(out_dir / "provenance.json", {"git": summary["git"], "protocol": summary["protocol"], "trace": summary["trace"], "model": summary["model"], "hostname": socket.gethostname()})
    return summary


def _validate_summary(summary: Mapping[str, Any], family: str, capacity: int, trace_sha: str) -> None:
    if summary.get("status") != "COMPLETE":
        raise ValueError("unit is not complete")
    protocol = summary["protocol"]
    if protocol["horizon"] != 4 or protocol["history"] != [0, 10000] or protocol["score"] != [10000, 50000]:
        raise ValueError("protocol mismatch")
    if protocol["capacity"] != capacity or summary["trace"]["family"] != family or summary["trace"]["sha256"] != trace_sha:
        raise ValueError("unit identity mismatch")
    if summary["model"]["held_out_family"] != family or summary["model"]["objective"] != REGISTRY_OBJECTIVE:
        raise ValueError("model provenance mismatch")
    for value in (summary["decision_metrics"]["set_aware_agreement_fraction"], summary["decision_metrics"]["lexicographic_agreement_fraction"], summary["regret_metrics"]["zero_regret_fraction"], summary["regret_metrics"]["positive_regret_fraction"]):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError("fraction outside [0,1]")


def _run_unit(family: str, capacity: int, data_root: Path, root: Path, registry: Mapping[str, Any]) -> Dict[str, Any]:
    fold = _fold(family)
    trace_path = _trace_path(fold, data_root)
    trace_sha = _sha256(trace_path)
    if trace_sha != fold["test_trace_sha256"]:
        raise ValueError(f"trace hash mismatch for {family}")
    requests, pages, _source = load_trace_from_any(str(trace_path))
    model_record = registry["records"][family]
    model_path = Path(model_record["model_path"])
    final_dir = root / "units" / f"{family}_cap{capacity}"
    if (final_dir / "summary.json").exists():
        summary = json.loads((final_dir / "summary.json").read_text(encoding="utf-8"))
        _validate_summary(summary, family, capacity, trace_sha)
        return summary
    temporary = root / "units" / f".{family}_cap{capacity}.tmp-{os.getpid()}"
    if temporary.exists():
        raise RuntimeError(f"temporary unit exists: {temporary}")
    summary = _analyze_unit(requests=requests, pages=pages, family=family, capacity=capacity, trace_path=trace_path, trace_sha=trace_sha, fold=fold, model_record=model_record, model_path=model_path, out_dir=temporary)
    summary["outputs"] = _finalized_output_paths(summary.get("outputs", {}), final_dir)
    (temporary / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _validate_summary(summary, family, capacity, trace_sha)
    if family == "brightkite" and capacity == 64:
        metrics = summary["decision_metrics"]
        regret = summary["regret_metrics"]["summary"]
        if abs(metrics["set_aware_agreement_fraction"] - 0.9646579066606252) > 1e-12 or abs(regret["mean"] - 0.035342093339374714) > 1e-12:
            raise ValueError("Brightkite/cap64 historical agreement/regret regression mismatch")
    os.replace(temporary, final_dir)
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data-read-root", type=Path, default=REPO_ROOT.parent / "Augmented-caching")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    families = list(cfg["families"])
    capacities = [int(value) for value in cfg["capacities"]]
    registry = _verify_registry(Path(str(cfg["registry"])).resolve(), families)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.out_dir / "config_snapshot.json", cfg)
    manifest_path = args.out_dir / "unit_completion_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"status": "RUNNING", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source": {"head": _git(["rev-parse", "HEAD"]), "branch": _git(["branch", "--show-current"])}, "protocol": cfg, "model_registry": registry, "expected_units": 21, "completed_units": 0, "units": {}}
    for family in families:
        for capacity in capacities:
            key = f"{family}_cap{capacity}"
            summary = _run_unit(family, capacity, args.data_read_root, args.out_dir, registry)
            manifest["units"][key] = {"status": "COMPLETE", "family": family, "capacity": capacity, "summary": str(args.out_dir / "units" / key / "summary.json"), "trace_sha256": summary["trace"]["sha256"], "model_sha256": summary["model"]["model_sha256"]}
            manifest["completed_units"] = len(manifest["units"])
            _atomic_json(manifest_path, manifest)
            print(json.dumps({"event": "unit_complete", "unit": key, "completed_units": manifest["completed_units"]}, sort_keys=True), flush=True)
    if manifest["completed_units"] != 21:
        return
    cell_rows = []
    for family in families:
        for capacity in capacities:
            summary = json.loads((args.out_dir / "units" / f"{family}_cap{capacity}" / "summary.json").read_text(encoding="utf-8"))
            dm = summary["decision_metrics"]
            rm = summary["regret_metrics"]
            cell_rows.append({"family": family, "capacity": capacity, "status": "ok", "decision_count": dm["decision_count"], "set_aware_agreement_fraction": dm["set_aware_agreement_fraction"], "lexicographic_agreement_fraction": dm["lexicographic_agreement_fraction"], "set_aware_disagreement_count": dm["set_aware_disagreement_count"], "lexicographic_disagreement_count": dm["lexicographic_disagreement_count"], "unique_exact_winner_count": dm["unique_exact_winner_count"], "multiple_exact_optima_count": dm["multiple_exact_optima_count"], "all_candidates_tied_count": dm["all_candidates_tied_count"], "mean_target_regret": rm["summary"]["mean"], "median_target_regret": rm["summary"]["median"], "p90_target_regret": rm["summary"]["p90"], "p95_target_regret": rm["summary"]["p95"], "p99_target_regret": rm["summary"]["p99"], "zero_regret_fraction": rm["zero_regret_fraction"], "positive_regret_fraction": rm["positive_regret_fraction"], "learned_misses": summary["online_metrics"]["learned_misses"], "lru_misses": summary["online_metrics"]["lru_misses"], "learned_minus_lru": summary["online_metrics"]["learned_minus_lru"], "model_sha256": summary["model"]["model_sha256"]})
    _atomic_csv(args.out_dir / "cell_summary.csv", list(cell_rows[0]), cell_rows)
    _atomic_json(args.out_dir / "integrity_summary.json", {"status": "COMPLETE", "units": 21, "rows": 21, "unique_cell_keys": len({(row["family"], row["capacity"]) for row in cell_rows}), "model_count": len(registry["records"]), "model_selection_frozen": registry["MODEL_SELECTION_FROZEN"]})
    _atomic_json(args.out_dir / "provenance.json", {"git": manifest["source"], "protocol": cfg, "model_registry": registry, "hostname": socket.gethostname()})
    manifest["status"] = "COMPLETE"
    _atomic_json(manifest_path, manifest)


if __name__ == "__main__":
    main()
