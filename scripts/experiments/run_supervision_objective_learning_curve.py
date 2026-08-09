"""Local learning-curve diagnostic for same-target supervision:

1. scalar regression on eviction-loss labels L(q)
2. pairwise regret-derived supervision sign(L(A) - L(B))

Reads the immutable supervision-objective-ablation v1 scalar shards in
place, builds one deterministic nested training-decision ordering per
fold, derives the pairwise same-target rows from the exact same filtered
scalar rows, and evaluates both trained models on the held-out family via
the same run_policy() + score_window() reconstruction used by the frozen
objective-ablation evaluator.

This runner is resumable and wall-time aware. It writes rows
incrementally, checkpoints at the (held_out_family, fraction) unit level,
and stops cleanly before starting a new unit once the remaining time
budget falls below the observed average unit cost.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.external_baseline_common import (
    IncrementalCsvWriter,
    base_provenance,
    sha256_of_file,
    write_provenance_json,
)
from lafc.halp_model import HALPModel
from lafc.policies.supervision_objective_ablation_policy import (
    PairwiseObjectivePolicy,
    ScalarObjectivePolicy,
)
from lafc.reviewer_diagnostics import build_nested_fraction_subsets
from lafc.runner.run_policy import run_policy
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    build_pairwise_rows,
    iter_multi_label_candidate_rows,
)
from lafc.supervision_objective_ablation_train import FEATURES, train_pairwise_objective
from lafc.experiments.reviewer_fairness_common import (
    HISTORY_END,
    HISTORY_START,
    PROTOCOL_VERSION,
    SCORE_END,
    SCORE_START,
    score_window,
)

FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
DEFAULT_CONFIG = Path("configs/supervision_objective_learning_curve_v1.json")
CONDITIONS = ["eviction_loss_scalar", "eviction_loss_pairwise"]

NUMERIC_SCALAR_COLUMNS = (
    ["capacity", "horizon", "decision_t"]
    + [
        "eviction_loss_label",
        "next_arrival_label_raw",
        "next_arrival_label_censored",
        "next_arrival_censored_flag",
        "reuse_distance_label_raw",
        "reuse_distance_label_censored",
        "reuse_distance_censored_flag",
    ]
    + list(EVICT_VALUE_V1_FEATURE_COLUMNS)
)

FIELDNAMES = [
    "experiment_protocol_version",
    "protocol_id",
    "condition",
    "fraction",
    "held_out_family",
    "fold_id",
    "capacity",
    "trace",
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
    "train_decision_count",
    "train_candidate_row_count",
    "train_pair_count",
    "validation_decision_count",
    "validation_candidate_row_count",
    "validation_top1",
    "validation_mean_regret",
    "validation_pairwise_accuracy",
    "validation_mae",
    "validation_rmse",
    "seed",
    "scalar_model_family",
    "pairwise_label_source",
    "model_path",
    "model_sha256",
    "train_runtime_seconds",
    "eval_runtime_seconds",
    "status",
    "failure_reason",
]
KEY_FIELDS = ["condition", "fraction", "held_out_family", "capacity"]


class ProtocolBlocked(RuntimeError):
    pass


class TimeBudget:
    def __init__(self, max_wall_hours: float):
        self.start = time.time()
        self.max_seconds = max_wall_hours * 3600.0
        self.unit_costs: List[float] = []

    def remaining(self) -> float:
        return self.max_seconds - (time.time() - self.start)

    def avg_unit_cost(self) -> float:
        return float(np.mean(self.unit_costs)) if self.unit_costs else 0.0

    def can_start_new_unit(self) -> bool:
        if not self.unit_costs:
            return True
        return self.remaining() > self.avg_unit_cost()

    def record_unit(self, seconds: float) -> None:
        self.unit_costs.append(seconds)


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fraction_label(fraction: float) -> str:
    return f"{fraction:.2f}".rstrip("0").rstrip(".")


def _unit_id(family: str, fraction: float) -> str:
    return f"{family}|{fraction:.6f}"


def _abs_shard_path(dataset_repo_root: Path, shard_path: str) -> Path:
    p = Path(shard_path)
    if p.is_absolute():
        return p
    return dataset_repo_root / p


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _decision_subset_sha(decision_ids: Sequence[str]) -> str:
    h = hashlib.sha256()
    for decision_id in decision_ids:
        h.update(str(decision_id).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _build_scalar_estimator(name: str, seed: int):
    if name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))])
    if name == "random_forest":
        return RandomForestRegressor(
            n_estimators=80,
            max_depth=12,
            min_samples_leaf=4,
            random_state=seed,
            n_jobs=1,
        )
    if name == "hist_gb":
        return HistGradientBoostingRegressor(
            max_depth=6,
            learning_rate=0.05,
            max_iter=150,
            random_state=seed,
        )
    raise ValueError(f"Unsupported scalar model family: {name!r}")


def _load_fold(family: str) -> Dict[str, object]:
    return _load_json(FOLDS_DIR / f"{family}.json")


def _decision_ids_from_trace(
    *,
    requests,
    capacity: int,
    trace_name: str,
) -> List[str]:
    import collections

    order: "collections.OrderedDict[str, None]" = collections.OrderedDict()
    decision_ids: List[str] = []
    for t, req in enumerate(requests):
        pid = req.page_id
        if pid in order:
            order.move_to_end(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            continue
        decision_ids.append(f"{trace_name}|cap={capacity}|t={t}")
        lru_victim = next(iter(order))
        order.pop(lru_victim)
        order[pid] = None
    return decision_ids


def _trace_rows_for_decisions(
    *,
    trace_stats: Sequence[Mapping[str, object]],
    split: str,
    selected_decision_ids: set[str],
    capacities: Sequence[int],
    cfg: ObjectiveAblationConfig,
    trace_cache: Dict[str, tuple],
) -> Iterable[Dict[str, object]]:
    for item in trace_stats:
        if str(item["split"]) != split:
            continue
        trace_path = str(item["path"])
        if trace_path not in trace_cache:
            trace_cache[trace_path] = load_trace_from_any(trace_path)
        requests, _pages, _src = trace_cache[trace_path]
        for capacity in capacities:
            yield from iter_multi_label_candidate_rows(
                requests,
                int(capacity),
                str(item["trace_name"]),
                str(item["trace_family"]),
                cfg,
                selected_decision_ids=selected_decision_ids,
            )


def _load_rows_for_decisions(
    *,
    trace_stats: Sequence[Mapping[str, object]],
    split: str,
    selected_decision_ids: Sequence[str],
    capacities: Sequence[int],
    cfg: ObjectiveAblationConfig,
    trace_cache: Dict[str, tuple],
) -> List[Dict[str, object]]:
    selected = {str(decision_id) for decision_id in selected_decision_ids}
    rows = list(
        _trace_rows_for_decisions(
            trace_stats=trace_stats,
            split=split,
            selected_decision_ids=selected,
            capacities=capacities,
            cfg=cfg,
            trace_cache=trace_cache,
        )
    )
    found = {str(row["decision_id"]) for row in rows}
    missing = sorted(selected - found)
    if missing:
        raise ProtocolBlocked(
            f"Selected {len(selected)} decision ids for split={split}, but {len(missing)} could not be reconstructed from the verified traces."
        )
    return rows


def _validate_nested_subsets(subsets: Mapping[float, Tuple[str, ...]], fractions: Sequence[float]) -> None:
    previous: set[str] = set()
    for fraction in fractions:
        current = set(subsets[float(fraction)])
        if not previous.issubset(current):
            raise ProtocolBlocked(f"Nested-subset property violated at fraction={fraction}")
        previous = current


def _validate_pairwise_same_target_rows(rows: Sequence[Dict[str, object]], pairs: Sequence[Dict[str, object]]) -> None:
    row_decision_ids = {str(row["decision_id"]) for row in rows}
    for pair in pairs:
        if pair["pairwise_label_source"] != "regret":
            raise ProtocolBlocked("Pairwise same-target rows were not derived with source='regret'.")
        if str(pair["decision_id"]) not in row_decision_ids:
            raise ProtocolBlocked("Pairwise row references a decision id absent from the filtered scalar subset.")
        value_i = float(pair["value_i"])
        value_j = float(pair["value_j"])
        expected = 1 if value_i < value_j else 0
        if value_i == value_j:
            raise ProtocolBlocked("Regret-derived pairwise rows unexpectedly retained a tie.")
        if int(pair["label_i_preferred"]) != expected:
            raise ProtocolBlocked("Pairwise label does not match sign(L(A)-L(B)).")


def _common_ranking_metrics(rows: Sequence[Dict[str, object]], evict_scores: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        grouped[str(row["decision_id"])].append(idx)

    top1 = 0
    regrets: List[float] = []
    for idxs in grouped.values():
        chosen_idx = min(idxs, key=lambda i: (float(evict_scores[i]), str(rows[i]["candidate_page_id"])))
        best_idx = min(idxs, key=lambda i: (float(rows[i]["eviction_loss_label"]), str(rows[i]["candidate_page_id"])))
        top1 += int(str(rows[chosen_idx]["candidate_page_id"]) == str(rows[best_idx]["candidate_page_id"]))
        regrets.append(float(rows[chosen_idx]["eviction_loss_label"]) - float(rows[best_idx]["eviction_loss_label"]))
    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1": float(top1 / denom),
        "mean_regret": float(np.mean(regrets) if regrets else 0.0),
    }


def _pairwise_accuracy(pairs: Sequence[Dict[str, object]], model: HALPModel) -> float:
    if not pairs:
        return float("nan")
    correct = 0
    for pair in pairs:
        i_feats = np.asarray([[float(pair[f"i_{c}"]) for c in FEATURES]], dtype=float)
        j_feats = np.asarray([[float(pair[f"j_{c}"]) for c in FEATURES]], dtype=float)
        score_i = float(model.predict_rewards(i_feats)[0])
        score_j = float(model.predict_rewards(j_feats)[0])
        predicted_i = int(score_i > score_j)
        correct += int(predicted_i == int(pair["label_i_preferred"]))
    return float(correct / len(pairs))


def _fit_scalar_condition(
    *,
    family: str,
    train_rows: Sequence[Dict[str, object]],
    val_rows: Sequence[Dict[str, object]],
    model_family: str,
    seed: int,
) -> Tuple[EvictValueV1Model, Dict[str, float]]:
    estimator = _build_scalar_estimator(model_family, seed)
    x_train = np.asarray([[float(row[c]) for c in FEATURES] for row in train_rows], dtype=float)
    y_train = np.asarray([float(row["eviction_loss_label"]) for row in train_rows], dtype=float)
    x_val = np.asarray([[float(row[c]) for c in FEATURES] for row in val_rows], dtype=float)
    y_val = np.asarray([float(row["eviction_loss_label"]) for row in val_rows], dtype=float)

    estimator.fit(x_train, y_train)
    preds = np.asarray(estimator.predict(x_val), dtype=float)
    ranking = _common_ranking_metrics(val_rows, preds)
    metrics = {
        "validation_top1": ranking["top1"],
        "validation_mean_regret": ranking["mean_regret"],
        "validation_pairwise_accuracy": float("nan"),
        "validation_mae": float(mean_absolute_error(y_val, preds)),
        "validation_rmse": float(np.sqrt(mean_squared_error(y_val, preds))),
    }
    model = EvictValueV1Model(
        model_name=f"learning_curve_{family}_{model_family}",
        estimator=estimator,
        feature_columns=list(FEATURES),
    )
    return model, metrics


def _fit_pairwise_condition(
    *,
    train_rows: Sequence[Dict[str, object]],
    val_rows: Sequence[Dict[str, object]],
    max_pairs_per_decision: int,
    pairwise_sample_seed: int,
    seed: int,
) -> Tuple[HALPModel, Dict[str, float], int]:
    train_pairs = build_pairwise_rows(
        list(train_rows),
        source="regret",
        max_pairs_per_decision=max_pairs_per_decision,
        sample_seed=pairwise_sample_seed,
    )
    _validate_pairwise_same_target_rows(train_rows, train_pairs)
    result = train_pairwise_objective(
        objective="eviction_loss_pairwise",
        train_pairs=train_pairs,
        seed=seed,
    )
    val_pairs = build_pairwise_rows(
        list(val_rows),
        source="regret",
        max_pairs_per_decision=max_pairs_per_decision,
        sample_seed=pairwise_sample_seed,
    )
    _validate_pairwise_same_target_rows(val_rows, val_pairs)
    x_val = np.asarray([[float(row[c]) for c in FEATURES] for row in val_rows], dtype=float)
    rewards = np.asarray(result.model.predict_rewards(x_val), dtype=float)
    ranking = _common_ranking_metrics(val_rows, rewards)
    metrics = {
        "validation_top1": ranking["top1"],
        "validation_mean_regret": ranking["mean_regret"],
        "validation_pairwise_accuracy": _pairwise_accuracy(val_pairs, result.model),
        "validation_mae": float("nan"),
        "validation_rmse": float("nan"),
    }
    return result.model, metrics, result.n_train_pairs


def _evaluate_condition(
    *,
    condition: str,
    family: str,
    fold: Mapping[str, object],
    trace_sha256: str,
    trace_name: str,
    trace_cache: Dict[str, tuple],
    data_read_root: Path,
    capacities: Sequence[int],
    model_path: Path,
) -> List[Dict[str, object]]:
    if family not in trace_cache:
        test_trace_path = data_read_root / str(fold["test_trace_path"])
        trace_cache[family] = load_trace_from_any(str(test_trace_path))
    reqs, pages, _src = trace_cache[family]

    out_rows: List[Dict[str, object]] = []
    for capacity in capacities:
        if condition == "eviction_loss_scalar":
            policy = ScalarObjectivePolicy(model_path=str(model_path), direction="min")
        elif condition == "eviction_loss_pairwise":
            policy = PairwiseObjectivePolicy(model_path=str(model_path))
        else:
            raise ValueError(f"Unsupported condition: {condition}")
        t0 = time.time()
        result = run_policy(policy, reqs, pages, int(capacity))
        eval_seconds = time.time() - t0
        primary = score_window(result.events, SCORE_START, len(result.events))
        out_rows.append(
            {
                "trace": trace_name,
                "trace_sha256": trace_sha256,
                "history_start": HISTORY_START,
                "history_end": HISTORY_END,
                "score_start": primary.score_start,
                "score_end": primary.score_end,
                "history_requests": primary.history_requests,
                "scored_requests": primary.scored_requests,
                "hits": primary.hits,
                "misses": primary.misses,
                "miss_ratio": round(primary.miss_ratio, 6),
                "capacity": int(capacity),
                "eval_runtime_seconds": round(eval_seconds, 4),
            }
        )
    return out_rows


def _verify_dataset_and_fold(
    *,
    family: str,
    fold: Mapping[str, object],
    manifest: Mapping[str, object],
    dataset_repo_root: Path,
    data_read_root: Path,
) -> Tuple[List[Path], str, str]:
    if manifest["protocol_id"] != "supervision_objective_ablation_v1":
        raise ProtocolBlocked(f"{family}: source manifest protocol_id mismatch.")
    if manifest["held_out_family"] != family:
        raise ProtocolBlocked(f"{family}: source manifest held_out_family mismatch.")
    if manifest["fold_id"] != fold["fold_id"]:
        raise ProtocolBlocked(f"{family}: source manifest fold_id mismatch.")
    scalar_shards = [_abs_shard_path(dataset_repo_root, item["path"]) for item in manifest["scalar_shards"]]
    if not scalar_shards:
        raise ProtocolBlocked(f"{family}: no scalar shards listed in source manifest.")
    missing = [str(path) for path in scalar_shards if not path.exists()]
    if missing:
        raise ProtocolBlocked(f"{family}: missing scalar shards: {missing[:3]}")

    trace_stats = manifest["trace_stats"]
    expected = set(fold["training_families"]) | {fold["validation_family"]}
    seen = {str(item["trace_family"]) for item in trace_stats}
    if seen != expected:
        raise ProtocolBlocked(f"{family}: source trace families {seen} != expected {expected}")
    for item in trace_stats:
        path = Path(str(item["path"]))
        if not path.exists():
            raise ProtocolBlocked(f"{family}: trace path missing from source manifest: {path}")
        if sha256_of_file(path) != str(item["trace_sha256"]):
            raise ProtocolBlocked(f"{family}: trace hash mismatch for {path}")

    raw_test_trace_path = Path(str(fold["test_trace_path"]))
    test_trace_path = raw_test_trace_path if raw_test_trace_path.is_absolute() else (data_read_root / raw_test_trace_path)
    if not test_trace_path.exists():
        raise ProtocolBlocked(f"{family}: held-out trace missing at {test_trace_path}")
    test_sha = sha256_of_file(test_trace_path)
    if fold.get("test_trace_sha256") and test_sha != str(fold["test_trace_sha256"]):
        raise ProtocolBlocked(f"{family}: held-out trace hash mismatch.")
    return scalar_shards, str(fold["test_trace_name"]), test_sha


def _plan_fold(
    *,
    family: str,
    fold: Mapping[str, object],
    trace_stats: Sequence[Mapping[str, object]],
    fractions: Sequence[float],
    validation_fraction: float,
    capacities: Sequence[int],
    trace_cache: Dict[str, tuple],
) -> Dict[str, object]:
    train_decision_ids: List[str] = []
    for item in trace_stats:
        if str(item["split"]) != "train":
            continue
        trace_path = str(item["path"])
        if trace_path not in trace_cache:
            trace_cache[trace_path] = load_trace_from_any(trace_path)
        requests, _pages, _src = trace_cache[trace_path]
        for capacity in capacities:
            train_decision_ids.extend(
                _decision_ids_from_trace(
                    requests=requests,
                    capacity=int(capacity),
                    trace_name=str(item["trace_name"]),
                )
            )
    if not train_decision_ids:
        raise ProtocolBlocked(f"{family}: no training decision ids found in verified trace manifests.")
    train_subsets = build_nested_fraction_subsets(train_decision_ids, fractions, seed=0)
    _validate_nested_subsets(train_subsets, fractions)

    val_decision_ids: List[str] = []
    for item in trace_stats:
        if str(item["split"]) != "val":
            continue
        trace_path = str(item["path"])
        if trace_path not in trace_cache:
            trace_cache[trace_path] = load_trace_from_any(trace_path)
        requests, _pages, _src = trace_cache[trace_path]
        for capacity in capacities:
            val_decision_ids.extend(
                _decision_ids_from_trace(
                    requests=requests,
                    capacity=int(capacity),
                    trace_name=str(item["trace_name"]),
                )
            )
    if not val_decision_ids:
        raise ProtocolBlocked(f"{family}: no validation decision ids found in verified trace manifests.")
    val_subsets = build_nested_fraction_subsets(val_decision_ids, [validation_fraction], seed=0)
    validation_ids = val_subsets[float(validation_fraction)]

    return {
        "train_decision_ids": train_decision_ids,
        "train_subsets": train_subsets,
        "validation_decision_ids": validation_ids,
    }


def _build_fold_plans(
    *,
    config: Mapping[str, object],
    held_out_families: Sequence[str],
    fractions: Sequence[float],
    capacities: Sequence[int],
) -> Dict[str, Dict[str, object]]:
    dataset_repo_root = Path(str(config["dataset_repo_root"]))
    dataset_root = Path(str(config["dataset_root"]))
    validation_fraction = float(config["validation_decision_fraction"])
    plans: Dict[str, Dict[str, object]] = {}
    trace_cache: Dict[str, tuple] = {}
    for family in held_out_families:
        fold = _load_fold(family)
        manifest = _load_json(dataset_root / family / "manifest.json")
        scalar_shards, trace_name, trace_sha256 = _verify_dataset_and_fold(
            family=family,
            fold=fold,
            manifest=manifest,
            dataset_repo_root=dataset_repo_root,
            data_read_root=Path(str(config["data_read_root"])),
        )
        fold_plan = _plan_fold(
            family=family,
            fold=fold,
            trace_stats=manifest["trace_stats"],
            fractions=fractions,
            validation_fraction=validation_fraction,
            capacities=capacities,
            trace_cache=trace_cache,
        )
        plans[family] = {
            "fold": fold,
            "manifest": manifest,
            "scalar_shards": scalar_shards,
            "trace_stats": manifest["trace_stats"],
            "trace_name": trace_name,
            "trace_sha256": trace_sha256,
            **fold_plan,
        }
    return plans


def _read_state(state_path: Path) -> Dict[str, object]:
    if not state_path.exists():
        return {"completed_units": [], "unit_seconds": {}}
    return _load_json(state_path)


def _mark_completed_unit(state_path: Path, unit_id: str, seconds: float) -> None:
    state = _read_state(state_path)
    completed = list(state.get("completed_units", []))
    if unit_id not in completed:
        completed.append(unit_id)
    unit_seconds = dict(state.get("unit_seconds", {}))
    unit_seconds[unit_id] = round(seconds, 4)
    state["completed_units"] = completed
    state["unit_seconds"] = unit_seconds
    _atomic_write_json(state_path, state)


def _build_row(
    *,
    protocol_id: str,
    condition: str,
    fraction: float,
    family: str,
    fold_id: str,
    capacity_row: Mapping[str, object],
    train_decision_count: int,
    train_candidate_row_count: int,
    train_pair_count: int,
    validation_decision_count: int,
    validation_candidate_row_count: int,
    validation_metrics: Mapping[str, float],
    seed: int,
    scalar_model_family: str,
    model_path: Path,
    model_sha256: str,
    train_seconds: float,
) -> Dict[str, object]:
    return {
        "experiment_protocol_version": PROTOCOL_VERSION,
        "protocol_id": protocol_id,
        "condition": condition,
        "fraction": _fraction_label(fraction),
        "held_out_family": family,
        "fold_id": fold_id,
        "capacity": capacity_row["capacity"],
        "trace": capacity_row["trace"],
        "trace_sha256": capacity_row["trace_sha256"],
        "history_start": capacity_row["history_start"],
        "history_end": capacity_row["history_end"],
        "score_start": capacity_row["score_start"],
        "score_end": capacity_row["score_end"],
        "history_requests": capacity_row["history_requests"],
        "scored_requests": capacity_row["scored_requests"],
        "hits": capacity_row["hits"],
        "misses": capacity_row["misses"],
        "miss_ratio": capacity_row["miss_ratio"],
        "train_decision_count": train_decision_count,
        "train_candidate_row_count": train_candidate_row_count,
        "train_pair_count": train_pair_count,
        "validation_decision_count": validation_decision_count,
        "validation_candidate_row_count": validation_candidate_row_count,
        "validation_top1": round(float(validation_metrics["validation_top1"]), 6),
        "validation_mean_regret": round(float(validation_metrics["validation_mean_regret"]), 6),
        "validation_pairwise_accuracy": (
            "" if math.isnan(float(validation_metrics["validation_pairwise_accuracy"]))
            else round(float(validation_metrics["validation_pairwise_accuracy"]), 6)
        ),
        "validation_mae": (
            "" if math.isnan(float(validation_metrics["validation_mae"]))
            else round(float(validation_metrics["validation_mae"]), 6)
        ),
        "validation_rmse": (
            "" if math.isnan(float(validation_metrics["validation_rmse"]))
            else round(float(validation_metrics["validation_rmse"]), 6)
        ),
        "seed": seed,
        "scalar_model_family": scalar_model_family,
        "pairwise_label_source": "regret" if condition == "eviction_loss_pairwise" else "",
        "model_path": str(model_path),
        "model_sha256": model_sha256,
        "train_runtime_seconds": round(train_seconds, 4),
        "eval_runtime_seconds": capacity_row["eval_runtime_seconds"],
        "status": "ok",
        "failure_reason": "",
    }


def _run_unit(
    *,
    family: str,
    fraction: float,
    config: Mapping[str, object],
    fold: Mapping[str, object],
    trace_stats: Sequence[Mapping[str, object]],
    selected_train_ids: Sequence[str],
    selected_val_ids: Sequence[str],
    capacities: Sequence[int],
    models_dir: Path,
    out_dir: Path,
    writer: IncrementalCsvWriter,
    trace_name: str,
    trace_sha256: str,
    trace_cache: Dict[str, tuple],
    data_read_root: Path,
) -> float:
    start = time.time()
    seed = int(config["seed"])
    protocol_id = str(config["protocol_id"])
    train_rows = _load_rows_for_decisions(
        trace_stats=trace_stats,
        split="train",
        selected_decision_ids=selected_train_ids,
        capacities=capacities,
        cfg=ObjectiveAblationConfig(horizon=int(config["horizon"])),
        trace_cache=trace_cache,
    )
    val_rows = _load_rows_for_decisions(
        trace_stats=trace_stats,
        split="val",
        selected_decision_ids=selected_val_ids,
        capacities=capacities,
        cfg=ObjectiveAblationConfig(horizon=int(config["horizon"])),
        trace_cache=trace_cache,
    )
    train_decision_ids = {str(row["decision_id"]) for row in train_rows}
    if train_decision_ids != set(selected_train_ids):
        raise ProtocolBlocked(f"{family} fraction={fraction}: training decision subset mismatch after row load.")
    train_candidate_row_count = len(train_rows)
    validation_candidate_row_count = len(val_rows)
    validation_decision_count = len(selected_val_ids)

    audit_payload = {
        "protocol_id": protocol_id,
        "held_out_family": family,
        "fraction": _fraction_label(fraction),
        "train_decision_count": len(selected_train_ids),
        "train_decision_subset_sha256": _decision_subset_sha(selected_train_ids),
        "train_candidate_row_count": train_candidate_row_count,
        "validation_decision_count": validation_decision_count,
        "validation_decision_subset_sha256": _decision_subset_sha(selected_val_ids),
        "validation_candidate_row_count": validation_candidate_row_count,
        "same_example_guarantee": "train scalar rows and regret-derived pairwise rows share the exact same filtered decision ids",
    }

    fraction_dir = models_dir / family / f"fraction_{_fraction_label(fraction)}"
    fraction_dir.mkdir(parents=True, exist_ok=True)

    scalar_family = str(config["conditions"]["eviction_loss_scalar"]["fixed_model_family_by_fold"][family])
    scalar_t0 = time.time()
    scalar_model, scalar_metrics = _fit_scalar_condition(
        family=family,
        train_rows=train_rows,
        val_rows=val_rows,
        model_family=scalar_family,
        seed=seed,
    )
    scalar_train_seconds = time.time() - scalar_t0
    scalar_model_path = fraction_dir / "eviction_loss_scalar.pkl"
    scalar_model.save(scalar_model_path)
    scalar_hash = sha256_of_file(scalar_model_path)
    scalar_eval_rows = _evaluate_condition(
        condition="eviction_loss_scalar",
        family=family,
        fold=fold,
        trace_sha256=trace_sha256,
        trace_name=trace_name,
        trace_cache=trace_cache,
        data_read_root=data_read_root,
        capacities=capacities,
        model_path=scalar_model_path,
    )
    for capacity_row in scalar_eval_rows:
        key = {
            "condition": "eviction_loss_scalar",
            "fraction": _fraction_label(fraction),
            "held_out_family": family,
            "capacity": capacity_row["capacity"],
        }
        if writer.already_done(key):
            continue
        writer.write_row(
            _build_row(
                protocol_id=protocol_id,
                condition="eviction_loss_scalar",
                fraction=fraction,
                family=family,
                fold_id=str(fold["fold_id"]),
                capacity_row=capacity_row,
                train_decision_count=len(selected_train_ids),
                train_candidate_row_count=train_candidate_row_count,
                train_pair_count=0,
                validation_decision_count=validation_decision_count,
                validation_candidate_row_count=validation_candidate_row_count,
                validation_metrics=scalar_metrics,
                seed=seed,
                scalar_model_family=scalar_family,
                model_path=scalar_model_path,
                model_sha256=scalar_hash,
                train_seconds=scalar_train_seconds,
            )
        )

    pairwise_t0 = time.time()
    pairwise_model, pairwise_metrics, train_pair_count = _fit_pairwise_condition(
        train_rows=train_rows,
        val_rows=val_rows,
        max_pairs_per_decision=int(config["pairwise_max_pairs_per_decision"]),
        pairwise_sample_seed=int(config["pairwise_sample_seed"]),
        seed=seed,
    )
    pairwise_train_seconds = time.time() - pairwise_t0
    pairwise_model_path = fraction_dir / "eviction_loss_pairwise.pkl"
    pairwise_model.save(pairwise_model_path)
    pairwise_hash = sha256_of_file(pairwise_model_path)
    pairwise_eval_rows = _evaluate_condition(
        condition="eviction_loss_pairwise",
        family=family,
        fold=fold,
        trace_sha256=trace_sha256,
        trace_name=trace_name,
        trace_cache=trace_cache,
        data_read_root=data_read_root,
        capacities=capacities,
        model_path=pairwise_model_path,
    )
    for capacity_row in pairwise_eval_rows:
        key = {
            "condition": "eviction_loss_pairwise",
            "fraction": _fraction_label(fraction),
            "held_out_family": family,
            "capacity": capacity_row["capacity"],
        }
        if writer.already_done(key):
            continue
        writer.write_row(
            _build_row(
                protocol_id=protocol_id,
                condition="eviction_loss_pairwise",
                fraction=fraction,
                family=family,
                fold_id=str(fold["fold_id"]),
                capacity_row=capacity_row,
                train_decision_count=len(selected_train_ids),
                train_candidate_row_count=train_candidate_row_count,
                train_pair_count=train_pair_count,
                validation_decision_count=validation_decision_count,
                validation_candidate_row_count=validation_candidate_row_count,
                validation_metrics=pairwise_metrics,
                seed=seed,
                scalar_model_family="",
                model_path=pairwise_model_path,
                model_sha256=pairwise_hash,
                train_seconds=pairwise_train_seconds,
            )
        )

    audit_payload["train_pair_count"] = train_pair_count
    audit_payload["scalar_model_family"] = scalar_family
    audit_payload["scalar_model_sha256"] = scalar_hash
    audit_payload["pairwise_model_sha256"] = pairwise_hash
    audit_payload["scalar_train_runtime_seconds"] = round(scalar_train_seconds, 4)
    audit_payload["pairwise_train_runtime_seconds"] = round(pairwise_train_seconds, 4)
    audit_payload["capacities_evaluated"] = list(capacities)
    _atomic_write_json(
        out_dir / "unit_audits" / family / f"fraction_{_fraction_label(fraction)}.json",
        audit_payload,
    )
    return time.time() - start


def plan_units(
    *,
    config: Mapping[str, object],
    held_out_families: Sequence[str],
    fractions: Sequence[float],
    capacities: Sequence[int] | None = None,
) -> List[Dict[str, object]]:
    capacities_eff = list(capacities) if capacities is not None else [int(value) for value in config["capacities"]]
    fold_plans = _build_fold_plans(
        config=config,
        held_out_families=held_out_families,
        fractions=fractions,
        capacities=capacities_eff,
    )
    units: List[Dict[str, object]] = []
    for family in held_out_families:
        fold_plan = fold_plans[family]
        for fraction in fractions:
            selected_ids = fold_plan["train_subsets"][float(fraction)]
            units.append(
                {
                    "family": family,
                    "fraction": float(fraction),
                    "unit_id": _unit_id(family, float(fraction)),
                    "fold_id": fold_plan["fold"]["fold_id"],
                    "train_decision_count": len(selected_ids),
                    "validation_decision_count": len(fold_plan["validation_decision_ids"]),
                    "trace_name": fold_plan["trace_name"],
                    "trace_sha256": fold_plan["trace_sha256"],
                }
            )
    return units


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--held-out-families", default="")
    ap.add_argument("--fractions", default="")
    ap.add_argument("--capacities", default="")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--models-dir", type=Path, default=None)
    ap.add_argument("--max-wall-hours", type=float, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    config = _load_json(args.config)
    if str(config["protocol_id"]) != "supervision_objective_learning_curve_v1":
        raise ProtocolBlocked("Unexpected protocol_id in learning-curve config.")

    held_out_families = (
        [item.strip() for item in args.held_out_families.split(",") if item.strip()]
        if args.held_out_families
        else list(config["held_out_families"])
    )
    fractions = (
        [float(item.strip()) for item in args.fractions.split(",") if item.strip()]
        if args.fractions
        else [float(value) for value in config["fractions"]]
    )
    capacities = (
        [int(item.strip()) for item in args.capacities.split(",") if item.strip()]
        if args.capacities
        else [int(value) for value in config["capacities"]]
    )

    out_dir = args.out_dir or Path(str(config["output_dir"]))
    models_dir = args.models_dir or Path(str(config["models_dir"]))
    max_wall_hours = float(args.max_wall_hours if args.max_wall_hours is not None else config["max_wall_hours_default"])
    data_read_root = Path(str(config["data_read_root"]))
    dataset_repo_root = Path(str(config["dataset_repo_root"]))
    dataset_root = Path(str(config["dataset_root"]))

    fold_plans = _build_fold_plans(
        config=config,
        held_out_families=held_out_families,
        fractions=fractions,
        capacities=capacities,
    )
    units: List[Dict[str, object]] = []
    for family in held_out_families:
        fold_plan = fold_plans[family]
        for fraction in fractions:
            selected_ids = fold_plan["train_subsets"][float(fraction)]
            units.append(
                {
                    "family": family,
                    "fraction": float(fraction),
                    "unit_id": _unit_id(family, float(fraction)),
                    "fold_id": fold_plan["fold"]["fold_id"],
                    "train_decision_count": len(selected_ids),
                    "validation_decision_count": len(fold_plan["validation_decision_ids"]),
                    "trace_name": fold_plan["trace_name"],
                    "trace_sha256": fold_plan["trace_sha256"],
                }
            )
    expected_rows = len(units) * len(capacities) * len(CONDITIONS)
    print(f"[plan] units={len(units)} expected_rows={expected_rows} families={held_out_families} fractions={fractions}")
    for unit in units:
        print(
            f"[unit] family={unit['family']} fraction={_fraction_label(unit['fraction'])} "
            f"train_decisions={unit['train_decision_count']} val_decisions={unit['validation_decision_count']}"
        )
    if args.dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(out_dir / "protocol_snapshot.json", config)
    _atomic_write_json(
        out_dir / "provenance.json",
        {
            **base_provenance(),
            "protocol_id": config["protocol_id"],
            "expected_rows": expected_rows,
            "dataset_repo_root": str(dataset_repo_root),
            "dataset_root": str(dataset_root),
            "data_read_root": str(data_read_root),
            "held_out_families": held_out_families,
            "fractions": [_fraction_label(value) for value in fractions],
            "capacities": list(capacities),
            "max_wall_hours": max_wall_hours,
        },
    )

    writer = IncrementalCsvWriter(out_dir / "policy_comparison.csv", FIELDNAMES, KEY_FIELDS)
    state_path = out_dir / "campaign_state.json"
    state = _read_state(state_path)
    completed_units = set(state.get("completed_units", []))
    trace_cache: Dict[str, tuple] = {}
    budget = TimeBudget(max_wall_hours)

    for unit in units:
        unit_id = str(unit["unit_id"])
        if args.resume and unit_id in completed_units:
            print(f"[skip] unit={unit_id} already completed (resume)")
            continue
        if not budget.can_start_new_unit():
            print(
                f"[budget] remaining={budget.remaining():.0f}s < avg_unit_cost={budget.avg_unit_cost():.0f}s "
                f"-- stopping before starting unit={unit_id}"
            )
            break

        family = str(unit["family"])
        fraction = float(unit["fraction"])
        fold_plan = fold_plans[family]
        fold = fold_plan["fold"]
        scalar_shards = fold_plan["scalar_shards"]
        trace_name = fold_plan["trace_name"]
        trace_sha256 = fold_plan["trace_sha256"]
        selected_train_ids = list(fold_plan["train_subsets"][fraction])
        selected_val_ids = list(fold_plan["validation_decision_ids"])
        print(
            f"[start] family={family} fraction={_fraction_label(fraction)} "
            f"train_decisions={len(selected_train_ids)} val_decisions={len(selected_val_ids)}"
        )
        t0 = time.time()
        seconds = _run_unit(
            family=family,
            fraction=fraction,
            config=config,
            fold=fold,
            selected_train_ids=selected_train_ids,
            selected_val_ids=selected_val_ids,
            capacities=capacities,
            trace_stats=fold_plan["trace_stats"],
            models_dir=models_dir,
            out_dir=out_dir,
            writer=writer,
            trace_name=trace_name,
            trace_sha256=trace_sha256,
            trace_cache=trace_cache,
            data_read_root=data_read_root,
        )
        _mark_completed_unit(state_path, unit_id, seconds)
        budget.record_unit(time.time() - t0)
        print(
            f"[done] family={family} fraction={_fraction_label(fraction)} "
            f"seconds={seconds:.2f} avg_unit_cost={budget.avg_unit_cost():.2f}"
        )

    writer.close()
    write_provenance_json(
        out_dir / "provenance.json",
        {
            **base_provenance(),
            "protocol_id": config["protocol_id"],
            "expected_rows": expected_rows,
            "dataset_repo_root": str(dataset_repo_root),
            "dataset_root": str(dataset_root),
            "data_read_root": str(data_read_root),
            "held_out_families": held_out_families,
            "fractions": [_fraction_label(value) for value in fractions],
            "capacities": list(capacities),
            "max_wall_hours": max_wall_hours,
            "completed_units": _read_state(state_path).get("completed_units", []),
        },
    )


if __name__ == "__main__":
    main()
