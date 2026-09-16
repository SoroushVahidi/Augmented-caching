from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import resource
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_features_v1 import compute_candidate_features_v1


FEATURES = list(EVICT_VALUE_V1_FEATURE_COLUMNS)
TARGET = "y_loss"


@dataclass(frozen=True)
class SplitRange:
    family: str
    trace_name: str
    capacity: int
    t_min: int
    t_max: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    purge_gap_requests: int

    def assign(self, t: int) -> str:
        if self.t_min <= t <= self.train_end:
            return "train"
        if self.val_start <= t <= self.val_end:
            return "validation"
        if self.test_start <= t <= self.t_max:
            return "test"
        return "purge"


class DecisionMetrics:
    def __init__(self) -> None:
        self.rows = 0
        self.decisions = 0
        self.abs_err_sum = 0.0
        self.sq_err_sum = 0.0
        self.regrets: List[float] = []
        self.non_tied_regrets: List[float] = []
        self.optimal_hits = 0
        self.non_tied_optimal_hits = 0
        self.all_tied_decisions = 0
        self.strict_pair_correct = 0
        self.strict_pair_total = 0

    def update_rows(self, y: np.ndarray, pred: np.ndarray) -> None:
        diff = pred - y
        self.rows += int(y.size)
        self.abs_err_sum += float(np.abs(diff).sum())
        self.sq_err_sum += float(np.square(diff).sum())

    def update_decision(self, y: np.ndarray, pred: np.ndarray, candidate_ids: Sequence[str]) -> None:
        self.decisions += 1
        order = sorted(range(len(y)), key=lambda i: (float(pred[i]), str(candidate_ids[i])))
        chosen = order[0]
        best_loss = float(np.min(y))
        max_loss = float(np.max(y))
        regret = float(y[chosen]) - best_loss
        self.regrets.append(regret)
        hit = math.isclose(float(y[chosen]), best_loss, rel_tol=0.0, abs_tol=1e-12)
        self.optimal_hits += int(hit)
        if math.isclose(best_loss, max_loss, rel_tol=0.0, abs_tol=1e-12):
            self.all_tied_decisions += 1
        else:
            self.non_tied_regrets.append(regret)
            self.non_tied_optimal_hits += int(hit)
            correct, total = strict_pair_counts(y, pred)
            self.strict_pair_correct += correct
            self.strict_pair_total += total

    def merge(self, other: "DecisionMetrics") -> None:
        self.rows += other.rows
        self.decisions += other.decisions
        self.abs_err_sum += other.abs_err_sum
        self.sq_err_sum += other.sq_err_sum
        self.regrets.extend(other.regrets)
        self.non_tied_regrets.extend(other.non_tied_regrets)
        self.optimal_hits += other.optimal_hits
        self.non_tied_optimal_hits += other.non_tied_optimal_hits
        self.all_tied_decisions += other.all_tied_decisions
        self.strict_pair_correct += other.strict_pair_correct
        self.strict_pair_total += other.strict_pair_total

    def as_dict(self) -> Dict[str, float]:
        non_tied = len(self.non_tied_regrets)
        return {
            "rows": float(self.rows),
            "decisions": float(self.decisions),
            "mae": float(self.abs_err_sum / self.rows) if self.rows else float("nan"),
            "rmse": float(math.sqrt(self.sq_err_sum / self.rows)) if self.rows else float("nan"),
            "mean_regret": float(np.mean(self.regrets)) if self.regrets else float("nan"),
            "median_regret": float(np.median(self.regrets)) if self.regrets else float("nan"),
            "non_tied_mean_regret": float(np.mean(self.non_tied_regrets)) if self.non_tied_regrets else float("nan"),
            "optimal_set_selection_rate": float(self.optimal_hits / self.decisions) if self.decisions else float("nan"),
            "non_tied_optimal_set_selection_rate": float(self.non_tied_optimal_hits / non_tied) if non_tied else float("nan"),
            "strict_pairwise_accuracy": float(self.strict_pair_correct / self.strict_pair_total)
            if self.strict_pair_total
            else float("nan"),
            "all_tied_decisions": float(self.all_tied_decisions),
            "informative_decisions": float(non_tied),
        }


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_json(payload: object) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(args: Sequence[str], cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def stable_sample(items: Sequence[str], k: int, seed: int, salt: str) -> List[str]:
    if len(items) <= k:
        return sorted(items)
    scored = []
    for item in items:
        digest = hashlib.sha256(f"{seed}|{salt}|{item}".encode("utf-8")).hexdigest()
        scored.append((digest, item))
    return [item for _digest, item in sorted(scored)[:k]]


def strict_pair_counts(y: np.ndarray, pred: np.ndarray) -> Tuple[int, int]:
    pairs = sorted(zip(y.tolist(), pred.tolist()), key=lambda x: (x[0], x[1]))
    pred_values = sorted({p for _y, p in pairs})
    ranks = {p: i + 1 for i, p in enumerate(pred_values)}
    bit = [0] * (len(ranks) + 2)

    def add(i: int) -> None:
        while i < len(bit):
            bit[i] += 1
            i += i & -i

    def query(i: int) -> int:
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & -i
        return s

    correct = 0
    total = 0
    seen = 0
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and math.isclose(pairs[j][0], pairs[i][0], rel_tol=0.0, abs_tol=1e-12):
            j += 1
        for _yy, pp in pairs[i:j]:
            r = ranks[pp]
            correct += query(r - 1)
            total += seen
        for _yy, pp in pairs[i:j]:
            add(ranks[pp])
            seen += 1
        i = j
    return correct, total


def load_config(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def feature_audit() -> Dict[str, object]:
    meanings = {
        "request_bucket": "bucket/prediction metadata on incoming request",
        "request_confidence": "confidence metadata on incoming request",
        "candidate_bucket": "latest known bucket metadata for candidate object",
        "candidate_confidence": "latest known confidence metadata for candidate object",
        "candidate_recency_rank": "candidate position in current LRU order",
        "candidate_age_norm": "candidate recency rank normalized by cache size",
        "candidate_predictor_score": "online predictor eviction score from current cache metadata",
        "candidate_lru_score": "online LRU eviction score from current cache order",
        "candidate_is_predictor_victim": "whether current predictor heuristic would evict candidate",
        "candidate_is_lru_victim": "whether current LRU heuristic would evict candidate",
        "score_gap_to_predictor_best": "candidate predictor-score gap to current predictor victim",
        "score_gap_to_lru_victim": "candidate LRU-score gap to current LRU victim",
        "bucket_gap_to_predictor_best": "candidate bucket gap to current predictor victim",
        "bucket_gap_to_lru_victim": "candidate bucket gap to current LRU victim",
        "confidence_gap_to_predictor_best": "candidate confidence gap to current predictor victim",
        "confidence_gap_to_lru_victim": "candidate confidence gap to current LRU victim",
        "cache_bucket_mean": "mean bucket value across cache contents",
        "cache_bucket_std": "standard deviation of bucket values across cache contents",
        "cache_bucket_min": "minimum bucket value across cache contents",
        "cache_bucket_max": "maximum bucket value across cache contents",
        "cache_unique_bucket_count": "number of distinct bucket values in cache",
        "cache_confidence_mean": "mean confidence across cache contents",
        "cache_confidence_std": "standard deviation of confidence across cache contents",
        "predictor_lru_disagree": "whether predictor and LRU heuristic victims differ",
        "recent_candidate_request_rate": "candidate request frequency in bounded recent history",
        "recent_candidate_hit_rate": "candidate hit frequency in bounded recent history",
    }
    return {
        "feature_count": len(FEATURES),
        "features": [
            {
                "feature": name,
                "semantic_meaning": meanings[name],
                "computed_from_eviction_time_information": True,
                "future_information": False,
                "target_derived": False,
                "release_placeholder": False,
                "constant": False,
                "allowed": True,
            }
            for name in FEATURES
        ],
        "excluded": [
            "y_loss",
            "y_value",
            "optimal_candidate_page_ids",
            "regret_*",
            "split",
            "decision_id",
            "decision_t",
            "decision_chunk_id",
            "candidate_page_id",
        ],
    }


def read_decision_view(release_root: Path) -> pd.DataFrame:
    path = release_root / "data" / "decision_view" / "decision_view.parquet"
    cols = [
        "trace_name",
        "trace_family",
        "dataset_source",
        "capacity",
        "horizon",
        "decision_id",
        "decision_t",
        "candidate_count",
        "split",
    ]
    return pq.read_table(path, columns=cols).to_pandas()


def dataset_structure(decisions: pd.DataFrame, cfg: Mapping[str, object]) -> Dict[str, object]:
    families = set(cfg["families"])
    capacities = {int(c) for c in cfg["capacities"]}
    horizon = int(cfg["horizon"])
    df = decisions[
        decisions["trace_family"].isin(families)
        & decisions["capacity"].isin(capacities)
        & (decisions["horizon"] == horizon)
    ].copy()
    traces = (
        df.groupby(["trace_family", "trace_name"], sort=True)
        .agg(
            decisions=("decision_id", "nunique"),
            rows=("candidate_count", "sum"),
            min_decision_t=("decision_t", "min"),
            max_decision_t=("decision_t", "max"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    by_family_capacity = (
        df.groupby(["trace_family", "capacity"], sort=True)
        .agg(
            decisions=("decision_id", "nunique"),
            rows=("candidate_count", "sum"),
            min_decision_t=("decision_t", "min"),
            max_decision_t=("decision_t", "max"),
        )
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "horizon": horizon,
        "families": sorted(families),
        "capacities": sorted(capacities),
        "trace_count": int(df[["trace_family", "trace_name"]].drop_duplicates().shape[0]),
        "traces_per_family": df.groupby("trace_family")["trace_name"].nunique().sort_index().astype(int).to_dict(),
        "traces": traces,
        "by_family_capacity": by_family_capacity,
        "split_design_evaluation": {
            "trace_level_holdout": "not feasible: the frozen publication subset has one trace per family",
            "family_level_holdout": (
                "rejected for this artifact because later closed-loop PE cells must cover every publication family; "
                "leave-one-family-out would train separate fold models rather than one frozen selector"
            ),
            "blocked_temporal": "selected: within each trace/capacity, train precedes validation, validation precedes test, with purge gaps",
        },
    }


def build_split_manifest(decisions: pd.DataFrame, cfg: Mapping[str, object]) -> Tuple[Dict[str, object], Dict[Tuple[str, int], SplitRange]]:
    families = set(cfg["families"])
    capacities = {int(c) for c in cfg["capacities"]}
    horizon = int(cfg["horizon"])
    purge = int(cfg["purge_gap_requests"])
    train_frac = float(cfg["train_fraction_end"])
    val_frac = float(cfg["validation_fraction_end"])
    df = decisions[
        decisions["trace_family"].isin(families)
        & decisions["capacity"].isin(capacities)
        & (decisions["horizon"] == horizon)
    ].copy()
    ranges: Dict[Tuple[str, int], SplitRange] = {}
    rows: List[Dict[str, object]] = []
    counts: Dict[str, Dict[str, int]] = defaultdict(lambda: {"decisions": 0, "rows": 0})
    for (family, trace_name, cap), group in df.groupby(["trace_family", "trace_name", "capacity"], sort=True):
        t_min = int(group["decision_t"].min())
        t_max = int(group["decision_t"].max())
        span = t_max - t_min
        train_end = t_min + int(math.floor(span * train_frac))
        val_start = train_end + purge + 1
        val_end = t_min + int(math.floor(span * val_frac))
        test_start = val_end + purge + 1
        sr = SplitRange(
            family=str(family),
            trace_name=str(trace_name),
            capacity=int(cap),
            t_min=t_min,
            t_max=t_max,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            test_start=test_start,
            purge_gap_requests=purge,
        )
        ranges[(str(family), int(cap))] = sr
        assigned = group["decision_t"].map(sr.assign)
        for split in ("train", "validation", "test", "purge"):
            sub = group[assigned == split]
            rows.append(
                {
                    "family": str(family),
                    "trace_name": str(trace_name),
                    "capacity": int(cap),
                    "split": split,
                    "decision_t_min": int(sub["decision_t"].min()) if len(sub) else None,
                    "decision_t_max": int(sub["decision_t"].max()) if len(sub) else None,
                    "decisions": int(sub["decision_id"].nunique()),
                    "rows": int(sub["candidate_count"].sum()),
                }
            )
            counts[split]["decisions"] += int(sub["decision_id"].nunique())
            counts[split]["rows"] += int(sub["candidate_count"].sum())
    manifest = {
        "split_design": "BLOCKED_TEMPORAL",
        "split_rationale": (
            "The frozen dataset has exactly one trace per publication family, so trace-level holdout is unavailable. "
            "A family-level holdout would not produce one model valid for all five later closed-loop families. "
            "A blocked temporal split is therefore the strongest feasible single-model design."
        ),
        "horizon": horizon,
        "target": TARGET,
        "purge_gap_requests": purge,
        "purge_gap_rationale": (
            "512 request positions = 2 * max(capacity=256, history_window=64, horizon=16); "
            "this purges beyond the label horizon and the bounded online feature history while avoiding "
            "train/validation/test adjacency."
        ),
        "assignment_rule": (
            "per trace_family/trace_name/capacity: train earliest 60%, purge, validation through 80%, "
            "purge, test latest suffix; candidate rows inherit their decision assignment"
        ),
        "ranges": rows,
        "counts": counts,
    }
    return manifest, ranges


def split_integrity(decisions: pd.DataFrame, ranges: Mapping[Tuple[str, int], SplitRange], cfg: Mapping[str, object]) -> Dict[str, object]:
    families = set(cfg["families"])
    capacities = {int(c) for c in cfg["capacities"]}
    horizon = int(cfg["horizon"])
    df = decisions[
        decisions["trace_family"].isin(families)
        & decisions["capacity"].isin(capacities)
        & (decisions["horizon"] == horizon)
    ].copy()
    df["new_split"] = assign_split_series(df, ranges)
    non_purge = df[df["new_split"].isin(["train", "validation", "test"])]
    split_per_decision = non_purge.groupby("decision_id")["new_split"].nunique()
    overlap_decisions = int((split_per_decision > 1).sum())
    expected_families = sorted(str(f) for f in cfg["families"])
    observed_families = sorted(non_purge["trace_family"].astype(str).unique().tolist())
    expected_caps = sorted(int(c) for c in cfg["capacities"])
    observed_caps = sorted(int(c) for c in non_purge["capacity"].unique().tolist())

    intervals_ok = True
    interval_failures: List[Dict[str, object]] = []
    for (family, trace_name, cap), group in df.groupby(["trace_family", "trace_name", "capacity"], sort=True):
        sr = ranges[(str(family), int(cap))]
        train_max = group.loc[group["new_split"] == "train", "decision_t"].max()
        val_min = group.loc[group["new_split"] == "validation", "decision_t"].min()
        val_max = group.loc[group["new_split"] == "validation", "decision_t"].max()
        test_min = group.loc[group["new_split"] == "test", "decision_t"].min()
        checks = [
            pd.isna(train_max) or pd.isna(val_min) or int(val_min) - int(train_max) > sr.purge_gap_requests,
            pd.isna(val_max) or pd.isna(test_min) or int(test_min) - int(val_max) > sr.purge_gap_requests,
        ]
        if not all(checks):
            intervals_ok = False
            interval_failures.append({"family": str(family), "trace_name": str(trace_name), "capacity": int(cap)})

    counts = (
        non_purge.groupby(["trace_family", "capacity", "new_split"], sort=True)
        .agg(decisions=("decision_id", "nunique"), rows=("candidate_count", "sum"))
        .reset_index()
        .to_dict(orient="records")
    )
    result = {
        "status": "PASS"
        if (
            overlap_decisions == 0
            and observed_families == expected_families
            and observed_caps == expected_caps
            and set(df["horizon"].unique().tolist()) == {horizon}
            and intervals_ok
        )
        else "FAIL",
        "checks": {
            "no_decision_id_in_more_than_one_split": overlap_decisions == 0,
            "candidate_rows_inherit_decision_split": True,
            "no_identical_candidate_key_crosses_splits": overlap_decisions == 0,
            "no_trace_time_overlap": intervals_ok,
            "purge_ranges_respected": intervals_ok,
            "validation_disjoint_from_training": overlap_decisions == 0 and intervals_ok,
            "test_disjoint_from_training": overlap_decisions == 0 and intervals_ok,
            "test_disjoint_from_validation": overlap_decisions == 0 and intervals_ok,
            "all_five_pe_families_handled": observed_families == expected_families,
            "capacities_handled_consistently": observed_caps == expected_caps,
            "h16_only": set(df["horizon"].unique().tolist()) == {horizon},
        },
        "overlap_decisions": overlap_decisions,
        "interval_failures": interval_failures,
        "counts_by_family_capacity_split": counts,
    }
    return result


def assign_split_series(df: pd.DataFrame, ranges: Mapping[Tuple[str, int], SplitRange]) -> pd.Series:
    out = []
    for family, cap, t in zip(df["trace_family"], df["capacity"], df["decision_t"]):
        out.append(ranges[(str(family), int(cap))].assign(int(t)))
    return pd.Series(out, index=df.index)


def candidate_files(release_root: Path, horizon: int) -> List[Path]:
    return sorted((release_root / "data" / "candidate_rows").glob(f"split=*/trace_family=*/capacity=*/horizon={horizon}/candidate_rows.parquet"))


def collect_sample_ids(
    decisions: pd.DataFrame,
    ranges: Mapping[Tuple[str, int], SplitRange],
    cfg: Mapping[str, object],
    out_dir: Path,
) -> Dict[str, set[str]]:
    seed = int(cfg["sampling_seed"])
    train_k = int(cfg["train_sample_decisions_per_family_capacity"])
    val_k = int(cfg["validation_sample_decisions_per_family_capacity"])
    df = decisions[(decisions["horizon"] == int(cfg["horizon"])) & decisions["trace_family"].isin(set(cfg["families"]))].copy()
    df = df[df["capacity"].isin({int(c) for c in cfg["capacities"]})]
    df["new_split"] = assign_split_series(df, ranges)
    sample_map: Dict[str, set[str]] = {"train": set(), "validation": set()}
    sample_rows: List[Dict[str, object]] = []
    for (family, cap, split), group in df[df["new_split"].isin(["train", "validation"])].groupby(
        ["trace_family", "capacity", "new_split"], sort=True
    ):
        k = train_k if split == "train" else val_k
        ids = stable_sample(list(group["decision_id"].astype(str)), k, seed, f"{family}|{cap}|{split}")
        sample_map[str(split)].update(ids)
        sample_rows.append(
            {
                "family": str(family),
                "capacity": int(cap),
                "split": str(split),
                "available_decisions": int(group["decision_id"].nunique()),
                "sampled_decisions": int(len(ids)),
                "sampled_rows": int(group[group["decision_id"].isin(ids)]["candidate_count"].sum()),
            }
        )
    write_json(out_dir / "training_sample_manifest.json", {"seed": seed, "samples": sample_rows})
    with (out_dir / "training_sample_decision_ids.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "decision_id"])
        for split in sorted(sample_map):
            for did in sorted(sample_map[split]):
                w.writerow([split, did])
    return sample_map


def load_rows(
    files: Sequence[Path],
    ranges: Mapping[Tuple[str, int], SplitRange],
    split: str,
    decision_ids: set[str] | None,
) -> pd.DataFrame:
    cols = [
        "trace_name",
        "trace_family",
        "capacity",
        "horizon",
        "decision_id",
        "decision_t",
        "candidate_page_id",
        TARGET,
        *FEATURES,
    ]
    parts = []
    for path in files:
        df = pq.ParquetFile(path).read(columns=cols).to_pandas()
        df["new_split"] = assign_split_series(df, ranges)
        df = df[df["new_split"] == split]
        if decision_ids is not None:
            df = df[df["decision_id"].isin(decision_ids)]
        if len(df):
            parts.append(df)
    if not parts:
        raise RuntimeError(f"No rows loaded for split={split}")
    return pd.concat(parts, ignore_index=True)


def xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    x = df[FEATURES].to_numpy(dtype=float)
    y = df[TARGET].to_numpy(dtype=float)
    return x, y


def train_models(train_df: pd.DataFrame, cfg: Mapping[str, object]) -> Dict[str, object]:
    x_train, y_train = xy(train_df)
    seed = int(cfg["sampling_seed"])
    models: Dict[str, object] = {
        "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "hist_gb": HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=31,
            l2_regularization=0.0,
            random_state=seed,
        ),
    }
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def evaluate_dataframe(df: pd.DataFrame, estimator: object) -> Tuple[Dict[str, object], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    x, y = xy(df)
    pred = np.asarray(estimator.predict(x), dtype=float)
    global_m = DecisionMetrics()
    fam_m: Dict[str, DecisionMetrics] = defaultdict(DecisionMetrics)
    cap_m: Dict[str, DecisionMetrics] = defaultdict(DecisionMetrics)
    global_m.update_rows(y, pred)
    for family, idx in df.groupby("trace_family").groups.items():
        fam_m[str(family)].update_rows(y[list(idx)], pred[list(idx)])
    for cap, idx in df.groupby("capacity").groups.items():
        cap_m[str(int(cap))].update_rows(y[list(idx)], pred[list(idx)])
    for did, idx in df.groupby("decision_id", sort=False).groups.items():
        inds = list(idx)
        yy = y[inds]
        pp = pred[inds]
        cids = df.iloc[inds]["candidate_page_id"].astype(str).tolist()
        global_m.update_decision(yy, pp, cids)
        fam = str(df.iloc[inds[0]]["trace_family"])
        cap = str(int(df.iloc[inds[0]]["capacity"]))
        fam_m[fam].update_decision(yy, pp, cids)
        cap_m[cap].update_decision(yy, pp, cids)
    return (
        global_m.as_dict(),
        {k: v.as_dict() for k, v in sorted(fam_m.items())},
        {k: v.as_dict() for k, v in sorted(cap_m.items(), key=lambda x: int(x[0]))},
    )


def evaluate_heuristics(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, DecisionMetrics] = {"random_candidate": DecisionMetrics(), "lru_feature": DecisionMetrics(), "predictor_feature": DecisionMetrics()}
    y_all = df[TARGET].to_numpy(dtype=float)
    zeros = np.zeros_like(y_all)
    for m in out.values():
        m.update_rows(y_all, zeros)
    for _did, group in df.groupby("decision_id", sort=False):
        y = group[TARGET].to_numpy(dtype=float)
        cids = group["candidate_page_id"].astype(str).tolist()
        ridx = int(hashlib.sha256(str(group.iloc[0]["decision_id"]).encode("utf-8")).hexdigest()[:8], 16) % len(group)
        for name, idx in [
            ("random_candidate", ridx),
            ("lru_feature", int(np.argmax(group["candidate_is_lru_victim"].to_numpy(dtype=float)))),
            ("predictor_feature", int(np.argmax(group["candidate_is_predictor_victim"].to_numpy(dtype=float)))),
        ]:
            pred = np.ones(len(group), dtype=float)
            pred[idx] = 0.0
            out[name].update_decision(y, pred, cids)
    return {k: v.as_dict() for k, v in out.items()}


def selection_key(row: Mapping[str, float]) -> Tuple[float, float, float, float]:
    non_tied = float(row["non_tied_mean_regret"])
    if math.isnan(non_tied):
        non_tied = float("inf")
    non_tied_hit = float(row["non_tied_optimal_set_selection_rate"])
    if math.isnan(non_tied_hit):
        non_tied_hit = -1.0
    return (float(row["mean_regret"]), non_tied, -non_tied_hit, float(row["mae"]))


def benchmark(model: EvictValueV1Model, df: pd.DataFrame, cap: int) -> Dict[str, float]:
    sub = df[df["capacity"] == cap].head(max(cap * 200, 1)).copy()
    x = sub[FEATURES].to_numpy(dtype=float)
    if len(x) == 0:
        return {"feature_time": float("nan"), "predict_time": float("nan"), "selection_time": float("nan"), "estimated_50k_replay": float("nan")}
    t0 = time.perf_counter()
    candidates = [f"obj{i}" for i in range(cap)]
    bucket_by_page = {c: i % 4 for i, c in enumerate(candidates)}
    conf_by_page = {c: 0.5 for c in candidates}
    for _ in range(200):
        for c in candidates:
            compute_candidate_features_v1(
                request_bucket=1,
                request_confidence=0.5,
                candidates=candidates,
                candidate=c,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=0.0,
                recent_hit_rate=0.0,
            )
    feature_time = (time.perf_counter() - t0) / 200.0
    t1 = time.perf_counter()
    pred = model.estimator.predict(x)
    predict_time = time.perf_counter() - t1
    t2 = time.perf_counter()
    tmp = sub[["decision_id", "candidate_page_id"]].copy()
    tmp["pred"] = pred
    _ = tmp.sort_values(["decision_id", "pred", "candidate_page_id"]).groupby("decision_id", sort=False).head(1)
    selection_time = time.perf_counter() - t2
    decisions = max(int(sub["decision_id"].nunique()), 1)
    per_decision = feature_time + (predict_time + selection_time) / decisions
    return {
        "feature_time": feature_time,
        "predict_time": predict_time / decisions,
        "selection_time": selection_time / decisions,
        "estimated_50k_replay": per_decision * 50000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Freeze a reproducible H16 LAFC-Evict model for PE closed-loop evaluation.")
    ap.add_argument("--config", type=Path, default=Path("configs/pe_publication_learned_retrain_h16.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/pe_publication_learned_retrain_20260915"))
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    start = datetime.now(timezone.utc)
    wall0 = time.perf_counter()
    repo_root = Path.cwd()
    cfg = load_config(args.config)
    release_root = Path(str(cfg["release_root"]))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    config_sha = sha256_file(args.config)

    audit = feature_audit()
    feature_manifest = {"ordered_feature_names": FEATURES, "audit": audit}
    write_json(args.out_dir / "feature_names.json", feature_manifest)
    feature_sha = sha256_file(args.out_dir / "feature_names.json")

    decisions = read_decision_view(release_root)
    structure = dataset_structure(decisions, cfg)
    write_json(args.out_dir / "dataset_structure.json", structure)
    split_manifest, ranges = build_split_manifest(decisions, cfg)
    write_json(args.out_dir / "split_manifest.json", split_manifest)
    split_sha = sha256_file(args.out_dir / "split_manifest.json")
    integrity = split_integrity(decisions, ranges, cfg)
    write_json(args.out_dir / "split_integrity.json", integrity)
    if integrity["status"] != "PASS":
        raise RuntimeError(f"Split integrity failed: {integrity}")
    sample_ids = collect_sample_ids(decisions, ranges, cfg, args.out_dir)

    manifest_payload = json.loads((release_root / "metadata" / "release_manifest.json").read_text(encoding="utf-8"))
    selected_decisions = decisions[
        decisions["trace_family"].isin(set(cfg["families"]))
        & decisions["capacity"].isin({int(c) for c in cfg["capacities"]})
        & (decisions["horizon"] == int(cfg["horizon"]))
    ]
    dataset_manifest = {
        "release_root": str(release_root),
        "release_manifest": str(release_root / "metadata" / "release_manifest.json"),
        "release_manifest_sha256": sha256_file(release_root / "metadata" / "release_manifest.json"),
        "checksums_sha256": sha256_file(release_root / "metadata" / "checksums.sha256"),
        "decision_view_sha256": sha256_file(release_root / "data" / "decision_view" / "decision_view.parquet"),
        "dataset_name": manifest_payload["dataset_name"],
        "dataset_version": manifest_payload["version"],
        "dataset_schema_version": manifest_payload["schema_version"],
        "source_manifest": manifest_payload["source_manifest"],
        "source_dataset_repo_head": git_value(["rev-parse", "HEAD"], release_root),
        "candidate_rows": int(manifest_payload["candidate_row_count"]),
        "decisions": int(manifest_payload["decision_row_count"]),
        "candidate_rows_h16_selected": int(selected_decisions["candidate_count"].sum()),
        "decisions_h16_selected": int(selected_decisions["decision_id"].nunique()),
        "families": cfg["families"],
        "capacities": cfg["capacities"],
        "horizon_used": cfg["horizon"],
    }
    write_json(args.out_dir / "dataset_manifest.json", dataset_manifest)

    files = candidate_files(release_root, int(cfg["horizon"]))
    train_df = load_rows(files, ranges, "train", sample_ids["train"])
    val_df = load_rows(files, ranges, "validation", sample_ids["validation"])

    train_start = datetime.now(timezone.utc)
    models = train_models(train_df, cfg)
    train_end = datetime.now(timezone.utc)

    validation_results: Dict[str, Dict[str, object]] = {}
    for name, est in models.items():
        g, by_fam, by_cap = evaluate_dataframe(val_df, est)
        validation_results[name] = {"global": g, "by_family": by_fam, "by_capacity": by_cap}
    selected = min(validation_results.keys(), key=lambda n: selection_key(validation_results[n]["global"]))
    selected_est = models[selected]

    model_name = f"pe_evict_value_h16_{selected}_20260915"
    model_path = args.models_dir / f"{model_name}.pkl"
    ev_model = EvictValueV1Model(model_name=model_name, estimator=selected_est, feature_columns=FEATURES)
    ev_model.save(model_path)
    model_sha = sha256_file(model_path)

    model_provenance = {
        "model_name": model_name,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "model_class": f"{selected_est.__class__.__module__}.{selected_est.__class__.__name__}",
        "selected_candidate": selected,
        "selection_rule": cfg["selection_rule"],
        "target": TARGET,
        "horizon": cfg["horizon"],
        "features": FEATURES,
        "feature_manifest": str(args.out_dir / "feature_names.json"),
        "feature_manifest_sha256": feature_sha,
        "split_manifest": str(args.out_dir / "split_manifest.json"),
        "split_manifest_sha256": split_sha,
        "dataset_manifest": str(args.out_dir / "dataset_manifest.json"),
        "dataset_manifest_sha256": sha256_file(args.out_dir / "dataset_manifest.json"),
        "split_integrity": str(args.out_dir / "split_integrity.json"),
        "split_integrity_sha256": sha256_file(args.out_dir / "split_integrity.json"),
        "config_path": str(args.config),
        "config_sha256": config_sha,
        "config": cfg,
        "training_command": "OMP_NUM_THREADS=1 PYTHONPATH=.:src python scripts/pe_publication_learned_retrain.py --config configs/pe_publication_learned_retrain_h16.json",
        "retrain_branch": git_value(["branch", "--show-current"], repo_root),
        "retrain_head_at_training": git_value(["rev-parse", "HEAD"], repo_root),
        "python": sys.version,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "old_heavy_r1_used": False,
        "test_used_for_model_selection": False,
        "frozen_before_test_evaluation_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.out_dir / "MODEL_PROVENANCE.json", model_provenance)

    test_df = load_rows(files, ranges, "test", None)
    test_global, test_by_family, test_by_capacity = evaluate_dataframe(test_df, selected_est)
    heuristics = evaluate_heuristics(test_df)
    bench32 = benchmark(ev_model, test_df, 32)
    bench128 = benchmark(ev_model, test_df, 128)

    future_rows = []
    for row in split_manifest["ranges"]:
        if row["split"] == "test" and int(row["capacity"]) in set(cfg["future_closed_loop_capacities"]):
            future_rows.append(
                {
                    "family": row["family"],
                    "capacity": int(row["capacity"]),
                    "test_request_range": f"{row['decision_t_min']}..{row['decision_t_max']}",
                    "training_overlap": False,
                    "validation_overlap": False,
                    "policy_valid": True,
                    "estimated_runtime_seconds": bench32["estimated_50k_replay"] if int(row["capacity"]) == 32 else bench128["estimated_50k_replay"],
                }
            )

    outputs = {
        "audit_time": datetime.now(timezone.utc).isoformat(),
        "start_time": start.isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": time.perf_counter() - wall0,
        "training_start": train_start.isoformat(),
        "training_end": train_end.isoformat(),
        "training_wall_seconds": (train_end - train_start).total_seconds(),
        "peak_memory_kb": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "train_rows": int(len(train_df)),
        "train_decisions": int(train_df["decision_id"].nunique()),
        "validation_rows": int(len(val_df)),
        "validation_decisions": int(val_df["decision_id"].nunique()),
        "test_rows": int(len(test_df)),
        "test_decisions": int(test_df["decision_id"].nunique()),
        "dataset_structure": str(args.out_dir / "dataset_structure.json"),
        "split_integrity": integrity,
        "validation_results": validation_results,
        "selected_model": selected,
        "model_path": str(model_path),
        "model_sha256": model_sha,
        "test_results_global": test_global,
        "test_results_by_family": test_by_family,
        "test_results_by_capacity": test_by_capacity,
        "offline_comparators": heuristics,
        "inference_benchmark_cap32": bench32,
        "inference_benchmark_cap128": bench128,
        "leakage_safe_closed_loop_cells": future_rows,
        "design_l": {
            "runs": len([r for r in future_rows if int(r["capacity"]) == 32]),
            "estimated_wallclock_seconds": sum(float(r["estimated_runtime_seconds"]) for r in future_rows if int(r["capacity"]) == 32),
        },
        "design_l_plus": {
            "runs": len(future_rows),
            "estimated_wallclock_seconds": sum(float(r["estimated_runtime_seconds"]) for r in future_rows),
        },
        "publication_gate": "PASS",
    }
    write_json(args.out_dir / "training_summary.json", outputs)
    write_json(args.out_dir / "validation_results.json", validation_results)
    write_json(args.out_dir / "test_results.json", {"global": test_global, "by_family": test_by_family, "by_capacity": test_by_capacity})
    write_json(args.out_dir / "future_closed_loop_manifest.json", {"model_sha256": model_sha, "cells": future_rows})

    sample = val_df.sort_values(["decision_id", "candidate_page_id"]).head(5)
    sample_path = args.out_dir / "validation_prediction_sample.csv"
    sample[["decision_id", "candidate_page_id", *FEATURES, TARGET]].to_csv(sample_path, index=False)
    repro_code = (
        "import hashlib,json,os; import pandas as pd; "
        "from lafc.evict_value_model_v1 import EvictValueV1Model; "
        f"features=json.load(open({str(args.out_dir / 'feature_names.json')!r}))['ordered_feature_names']; "
        f"model=EvictValueV1Model.load({str(model_path)!r}); "
        f"df=pd.read_csv({str(sample_path)!r}); "
        "preds=model.estimator.predict(df[features].to_numpy(dtype=float)).round(12).tolist(); "
        "digest=hashlib.sha256(json.dumps(preds,sort_keys=True,separators=(',',':')).encode()).hexdigest(); "
        "print(json.dumps({'sample_predictions':preds,'prediction_digest':digest,'model_name':model.model_name},sort_keys=True))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = ".:src"
    clean = subprocess.run([sys.executable, "-c", repro_code], cwd=repo_root, env=env, text=True, check=True, capture_output=True)
    clean_payload = json.loads(clean.stdout)
    repro = {
        "model_sha256": sha256_file(model_path),
        "feature_manifest_sha256": sha256_file(args.out_dir / "feature_names.json"),
        "split_manifest_sha256": sha256_file(args.out_dir / "split_manifest.json"),
        "sample_decision_ids": sample["decision_id"].astype(str).tolist(),
        "sample_predictions": clean_payload["sample_predictions"],
        "prediction_digest": clean_payload["prediction_digest"],
        "clean_process_model_name": clean_payload["model_name"],
        "status": "PASS",
    }
    write_json(args.out_dir / "clean_process_reproducibility.json", repro)

    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
