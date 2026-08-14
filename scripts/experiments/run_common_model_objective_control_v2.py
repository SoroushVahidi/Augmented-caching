"""Corrected common-scorer objective control with feature-only deployment.

V2 preserves the V1 scalar protocol and fixes the V1 pairwise orientation bug.
The common scorer emits an eviction score: lower is selected for objectives with
``direction=min`` and higher is selected for objectives with ``direction=max``.
For ``objective_pairwise``, ``build_pairwise_rows(..., source="next_arrival")``
marks the sooner-reused candidate as ``label_i_preferred`` in the keep/reward
sense used by HALP-style models. This runner intentionally inverts that signal
for the common eviction-score convention, so the later-next-arrival candidate is
the higher-score side when deployment uses ``direction=max``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.reviewer_fairness_common import SCORE_END, SCORE_START
from lafc.oracle_diagnostics import compare_choice_to_exact_target
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    build_candidate_rows_for_full_cache_state,
    build_pairwise_rows,
    iter_multi_label_candidate_rows,
)
from lafc.types import PageId, Request

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT.parent / "Augmented-caching"
FOLDS = ROOT / "configs/fair_cross_family_v1/folds"
FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
CAPACITIES = [32, 64, 128]
FEATURES = list(EVICT_VALUE_V1_FEATURE_COLUMNS)
OBJECTIVES = [
    "objective_eviction_loss",
    "objective_next_arrival",
    "objective_reuse_distance",
    "objective_pairwise",
]


class CommonScorer:
    def __init__(self, hidden: int = 8, lr: float = 0.02, epochs: int = 40, l2: float = 1e-4, seed: int = 0):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.l2 = l2
        self.seed = seed
        self.fitted = False

    def fit(self, X, y=None, pairs=None, mode: str = "scalar"):
        X = np.asarray(X, float)
        self.mean = X.mean(0)
        self.std = X.std(0)
        self.std[self.std == 0] = 1
        X = (X - self.mean) / self.std
        d = X.shape[1]
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0, np.sqrt(2 / d), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, np.sqrt(2 / self.hidden), self.hidden)
        if mode == "scalar":
            y = np.asarray(y, float)
            self.y_mean = float(y.mean())
            self.y_std = float(y.std() or 1)
            y = (y - self.y_mean) / self.y_std
            for _ in range(self.epochs):
                z = X @ self.W1 + self.b1
                h = np.maximum(z, 0)
                out = h @ self.W2
                g = (out - y) / len(X)
                g2 = h.T @ g + self.l2 * self.W2
                gh = np.outer(g, self.W2)
                gz = gh * (z > 0)
                g1 = X.T @ gz + self.l2 * self.W1
                gb = gz.sum(0)
                self.W2 -= self.lr * g2
                self.W1 -= self.lr * g1
                self.b1 -= self.lr * gb
        else:
            A, B = pairs
            A = (A - self.mean) / self.std
            B = (B - self.mean) / self.std
            n = len(A)
            for _ in range(self.epochs):
                za = A @ self.W1 + self.b1
                ha = np.maximum(za, 0)
                ra = ha @ self.W2
                zb = B @ self.W1 + self.b1
                hb = np.maximum(zb, 0)
                rb = hb @ self.W2
                p = 1 / (1 + np.exp(-np.clip(ra - rb, -50, 50)))
                gd = (p - 1) / n
                g2 = (ha - hb).T @ gd + self.l2 * self.W2
                gha = np.outer(gd, self.W2)
                ghb = -gha
                gza = gha * (za > 0)
                gzb = ghb * (zb > 0)
                g1 = A.T @ gza + B.T @ gzb + self.l2 * self.W1
                gb = gza.sum(0) + gzb.sum(0)
                self.W2 -= self.lr * g2
                self.W1 -= self.lr * g1
                self.b1 -= self.lr * gb
            self.y_mean = 0.0
            self.y_std = 1.0
        self.fitted = True
        return self

    def score(self, X):
        X = np.asarray(X, float)
        X = (X - self.mean) / self.std
        return np.maximum(X @ self.W1 + self.b1, 0) @ self.W2 * self.y_std + self.y_mean

    def save(self, p: Path):
        np.savez(
            p,
            hidden=self.hidden,
            lr=self.lr,
            epochs=self.epochs,
            l2=self.l2,
            seed=self.seed,
            mean=self.mean,
            std=self.std,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            y_mean=self.y_mean,
            y_std=self.y_std,
        )

    @classmethod
    def load(cls, p: Path):
        z = np.load(p)
        m = cls(int(z["hidden"]), float(z["lr"]), int(z["epochs"]), float(z["l2"]), int(z["seed"]))
        for k in ("mean", "std", "W1", "b1", "W2", "y_mean", "y_std"):
            setattr(m, k, z[k])
        m.fitted = True
        return m


@dataclass(frozen=True)
class ReplayResult:
    hit_sequence: Tuple[bool, ...]
    victim_sequence: Tuple[Tuple[int, PageId], ...]
    diagnostics_count: int


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(p: Path, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=p.parent, delete=False, encoding="utf8") as f:
        json.dump(x, f, indent=2, sort_keys=True)
        f.write("\n")
        q = Path(f.name)
    os.replace(q, p)


def fold(family: str) -> Dict[str, object]:
    return json.loads((FOLDS / f"{family}.json").read_text())


def trace_for(family: str) -> Tuple[Dict[str, object], Path]:
    fd = fold(family)
    rel = Path(fd["test_trace_path"])
    p = rel if rel.is_absolute() else (ROOT / rel)
    if not p.exists():
        p = DATA_ROOT / rel
    return fd, p


def row_digest(rows: Sequence[Mapping[str, object]]) -> str:
    normalized = json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf8")).hexdigest()


def selected_rows_from_requests(
    requests: Sequence[Request],
    cap: int,
    n: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    ids: List[str] = []
    seen: set[str] = set()
    for row in iter_multi_label_candidate_rows(requests, cap, trace_name, trace_family, cfg):
        decision_id = str(row["decision_id"])
        if decision_id not in seen:
            if len(ids) >= n:
                break
            ids.append(decision_id)
            seen.add(decision_id)
        if decision_id in seen:
            rows.append(dict(row))
    return rows


def selected_rows(family: str, cap: int, n: int, cfg: ObjectiveAblationConfig) -> List[Dict[str, object]]:
    fd, p = trace_for(family)
    reqs, _, _ = load_trace_from_any(str(p))
    return selected_rows_from_requests(reqs, cap, n, str(fd["test_trace_name"]), family, cfg)


def selected_decision_ids(rows: Sequence[Mapping[str, object]]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for row in rows:
        decision_id = str(row["decision_id"])
        if decision_id not in seen:
            out.append(decision_id)
            seen.add(decision_id)
    return out


def x(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([[float(r[c]) for c in FEATURES] for r in rows], dtype=float)


def metrics(rows: Sequence[Mapping[str, object]], model: CommonScorer, label: str, direction: str) -> Tuple[float, int]:
    vals: Dict[str, List[Mapping[str, object]]] = {}
    for r in rows:
        vals.setdefault(str(r["decision_id"]), []).append(r)
    regrets: List[float] = []
    for items in vals.values():
        pred = model.score(x(items))
        idx = int(np.argmin(pred)) if direction == "min" else int(np.argmax(pred))
        true = [float(r[label]) for r in items]
        best = min(true) if direction == "min" else max(true)
        regrets.append((true[idx] - best) if direction == "min" else (best - true[idx]))
    return float(np.mean(regrets) if regrets else 0), len(vals)


def orient_pairwise_rows_for_eviction_score(pair_rows: Sequence[Mapping[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (A, B) where A should receive the higher common eviction score.

    Pair rows mark the keep/reward-preferred side for the source objective.
    For source="next_arrival", the keep-preferred side is reused sooner; the
    common control deploys pairwise with direction=max, so the later-reused
    side is the eviction-preferred side.
    """
    a_rows: List[List[float]] = []
    b_rows: List[List[float]] = []
    for p in pair_rows:
        i_feats = [float(p[f"i_{c}"]) for c in FEATURES]
        j_feats = [float(p[f"j_{c}"]) for c in FEATURES]
        if int(p["label_i_preferred"]) == 1:
            a_rows.append(j_feats)
            b_rows.append(i_feats)
        else:
            a_rows.append(i_feats)
            b_rows.append(j_feats)
    return np.asarray(a_rows, dtype=float), np.asarray(b_rows, dtype=float)


def train_one(
    train: Sequence[Mapping[str, object]],
    val: Sequence[Mapping[str, object]],
    obj: str,
    spec: Mapping[str, object],
    cfg: Mapping[str, object],
) -> Tuple[CommonScorer, Dict[str, object]]:
    label = str(spec["label"])
    direction = str(spec["direction"])
    arch = cfg["architecture"]
    model = CommonScorer(
        hidden=int(arch["hidden_units"]),
        lr=float(arch["lr"]),
        epochs=int(arch["epochs"]),
        l2=float(arch["l2"]),
        seed=int(cfg["seed"]),
    )
    if obj == "objective_pairwise":
        pairs = build_pairwise_rows(
            train,
            source=str(spec.get("pairwise_source", "next_arrival")),
            max_pairs_per_decision=int(cfg["pair_max_pairs_per_decision"]),
            sample_seed=int(cfg["seed"]),
        )
        A, B = orient_pairwise_rows_for_eviction_score(pairs)
        model.fit(A, pairs=(A, B), mode="pairwise")
        n_pairs = len(pairs)
    else:
        model.fit(x(train), [float(r[label]) for r in train], mode="scalar")
        n_pairs = 0
    vr, nd = metrics(val, model, label, direction)
    return model, {
        "validation_mean_regret": vr,
        "validation_decisions": nd,
        "train_rows": len(train),
        "train_pairs": n_pairs,
        "score_semantics": str(spec.get("score_semantics", "scalar_label_score")),
    }


def build_candidate_feature_rows_for_full_cache_state(
    *,
    request: Request,
    request_index: int,
    capacity: int,
    trace_name: str,
    trace_family: str,
    cache_order: Sequence[PageId],
    bucket_by_page: Mapping[PageId, int],
    confidence_by_page: Mapping[PageId, float],
    recent_req_hist: Sequence[PageId],
    recent_hit_hist: Sequence[PageId],
) -> List[Dict[str, object]]:
    all_candidates = list(cache_order)
    req_bucket = int(request.metadata.get("bucket", 0))
    req_conf = float(request.metadata.get("confidence", 0.5))
    decision_id = f"{trace_name}|cap={capacity}|t={request_index}"
    rows: List[Dict[str, object]] = []
    for candidate in all_candidates:
        req_rate = (sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)) if recent_req_hist else 0.0
        hit_rate = (sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)) if recent_hit_hist else 0.0
        row: Dict[str, object] = {
            "trace_name": trace_name,
            "trace_family": trace_family,
            "capacity": int(capacity),
            "decision_id": decision_id,
            "decision_t": int(request_index),
            "candidate_page_id": candidate,
        }
        row.update(
            compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=all_candidates,
                candidate=candidate,
                bucket_by_page=dict(bucket_by_page),
                confidence_by_page=dict(confidence_by_page),
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()
        )
        rows.append(row)
    return rows


def score_rows_once(rows: Sequence[Mapping[str, object]], model: CommonScorer) -> Dict[PageId, float]:
    scores = model.score(x(rows))
    return {str(r["candidate_page_id"]): float(scores[i]) for i, r in enumerate(rows)}


def choose_from_scores(rows: Sequence[Mapping[str, object]], scores: Mapping[PageId, float], direction: str) -> PageId:
    candidate_ids = [str(row["candidate_page_id"]) for row in rows]
    missing = [pid for pid in candidate_ids if pid not in scores]
    extra = [pid for pid in scores if pid not in candidate_ids]
    if missing or extra:
        raise ValueError(f"scores must cover exactly the candidates; missing={missing}, extra={extra}")
    pick = min if direction == "min" else max
    return pick(candidate_ids, key=lambda pid: (scores[pid], candidate_ids.index(pid)))


def replay_learned_policy(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
    model: CommonScorer,
    direction: str,
    diagnostics: bool = False,
) -> ReplayResult:
    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    hit_sequence: List[bool] = []
    victims: List[Tuple[int, PageId]] = []
    diagnostics_count = 0

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            hit_sequence.append(True)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue

        hit_sequence.append(False)
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        feature_rows = build_candidate_feature_rows_for_full_cache_state(
            request=req,
            request_index=t,
            capacity=capacity,
            trace_name=trace_name,
            trace_family=trace_family,
            cache_order=list(order.keys()),
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
        )
        scores = score_rows_once(feature_rows, model)
        chosen = choose_from_scores(feature_rows, scores, direction)
        if diagnostics:
            full_rows = build_candidate_rows_for_full_cache_state(
                requests=requests,
                request_index=t,
                capacity=capacity,
                trace_name=trace_name,
                trace_family=trace_family,
                cfg=cfg,
                cache_order=list(order.keys()),
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_req_hist=recent_req_hist,
                recent_hit_hist=recent_hit_hist,
            )
            compare_choice_to_exact_target(full_rows, "eviction_loss", chosen)
            diagnostics_count += 1
        victims.append((t, chosen))
        order.pop(chosen)
        order[pid] = None
        recent_req_hist.append(pid)

    return ReplayResult(tuple(hit_sequence), tuple(victims), diagnostics_count)


def bool_score(hit_sequence: Sequence[bool]) -> Tuple[int, float]:
    window = hit_sequence[SCORE_START:SCORE_END]
    misses = sum(1 for h in window if not h)
    return misses, misses / len(window)


def run_unit(args, cfg: Mapping[str, object]) -> Path:
    held = str(args.family)
    cap = int(args.capacity)
    if held not in cfg["families"]:
        raise ValueError(f"family {held!r} not in config")
    if cap not in cfg["capacities"]:
        raise ValueError(f"capacity {cap!r} not in config")
    unit = Path(args.out)
    summary_path = unit / "summary.json"
    if summary_path.exists() and not args.force:
        return summary_path

    unit.mkdir(parents=True, exist_ok=True)
    atomic_json(unit / "config_snapshot.json", cfg)
    fd, test_path = trace_for(held)
    if sha(test_path) != fd["test_trace_sha256"]:
        raise RuntimeError(f"trace hash mismatch {held}")
    test_reqs, _, _ = load_trace_from_any(str(test_path))
    train_fams = list(fd["training_families"])
    if held in train_fams or held == str(fd["validation_family"]):
        raise RuntimeError(f"held-out leakage detected for {held}")

    ocfg = ObjectiveAblationConfig(horizon=int(cfg["horizon"]))
    train_rows: List[Dict[str, object]] = []
    train_decision_ids: Dict[str, List[str]] = {}
    for fam in train_fams:
        fam_rows = selected_rows(fam, cap, int(cfg["train_decisions_per_family"]), ocfg)
        train_decision_ids[fam] = selected_decision_ids(fam_rows)
        train_rows += fam_rows
    val_fam = str(fd["validation_family"])
    val_rows = selected_rows(val_fam, cap, int(cfg["validation_decisions"]), ocfg)
    val_decision_ids = selected_decision_ids(val_rows)

    unit_rows: List[Dict[str, object]] = []
    objective_names = list(args.objectives or OBJECTIVES)
    for obj in objective_names:
        spec = cfg["objectives"][obj]
        model, stat = train_one(train_rows, val_rows, obj, spec, cfg)
        mp = unit / f"{obj}.npz"
        model.save(mp)
        replay = replay_learned_policy(
            requests=test_reqs,
            capacity=cap,
            trace_name=str(fd["test_trace_name"]),
            trace_family=held,
            cfg=ocfg,
            model=model,
            direction=str(spec["direction"]),
            diagnostics=bool(args.diagnostics),
        )
        misses, ratio = bool_score(replay.hit_sequence)
        unit_rows.append(
            {
                "objective": obj,
                "held_out_family": held,
                "capacity": cap,
                "misses": misses,
                "miss_ratio": ratio,
                "delta_vs_lru": None,
                "validation_mean_regret": stat["validation_mean_regret"],
                "model_sha256": sha(mp),
                "trace_sha256": sha(test_path),
                "seed": int(cfg["seed"]),
                "diagnostics_count": replay.diagnostics_count,
                "victim_sequence_sha256": row_digest(
                    [{"decision_t": t, "candidate_page_id": p} for t, p in replay.victim_sequence]
                ),
            }
        )

    metadata = {
        "status": "COMPLETE",
        "protocol_id": cfg["protocol_id"],
        "source_head": os.popen("git rev-parse HEAD").read().strip(),
        "family": held,
        "capacity": cap,
        "trace_sha256": sha(test_path),
        "training_families": train_fams,
        "validation_family": val_fam,
        "train_decision_ids": train_decision_ids,
        "validation_decision_ids": val_decision_ids,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "train_rows_sha256": row_digest(train_rows),
        "validation_rows_sha256": row_digest(val_rows),
        "objectives": objective_names,
        "diagnostics": bool(args.diagnostics),
    }
    atomic_json(unit / "metadata.json", metadata)
    atomic_json(summary_path, {"status": "COMPLETE", "rows": unit_rows, "metadata": metadata})
    return summary_path


def run_campaign(args, cfg: Mapping[str, object]):
    out = Path(args.out)
    rows: List[Dict[str, object]] = []
    for family in cfg["families"]:
        for cap in cfg["capacities"]:
            unit_dir = out / "units" / f"{family}_cap{cap}"
            unit_args = argparse.Namespace(
                family=family,
                capacity=cap,
                out=unit_dir,
                force=args.force,
                diagnostics=args.diagnostics,
                objectives=args.objectives,
            )
            summary_path = run_unit(unit_args, cfg)
            rows.extend(json.loads(summary_path.read_text())["rows"])
    out.mkdir(parents=True, exist_ok=True)
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    atomic_json(
        out / "completion_manifest.json",
        {
            "status": "COMPLETE",
            "expected_units": 21,
            "completed_units": len({(r["held_out_family"], r["capacity"]) for r in rows}),
            "expected_rows": 84,
            "rows": len(rows),
            "source_head": os.popen("git rev-parse HEAD").read().strip(),
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/common_model_objective_control_v2.json")
    ap.add_argument("--out", type=Path, default=ROOT / "analysis/common_model_objective_control_v2")
    ap.add_argument("--family", choices=FAMILIES)
    ap.add_argument("--capacity", type=int, choices=CAPACITIES)
    ap.add_argument("--objectives", nargs="+", choices=OBJECTIVES)
    ap.add_argument("--diagnostics", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.family or args.capacity:
        if not (args.family and args.capacity):
            raise SystemExit("--family and --capacity must be supplied together")
        run_unit(args, cfg)
    else:
        run_campaign(args, cfg)


if __name__ == "__main__":
    main()
