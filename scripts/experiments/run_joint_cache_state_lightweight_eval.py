from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.experiments.joint_cache_state_dataset import _history_summary
from lafc.experiments.joint_cache_state_model import JointSoftmaxConfig, JointSoftmaxVictimModel
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import load_trace
from lafc.types import PageId, Request

GLOBAL_COLUMNS = [
    "incoming_bucket",
    "incoming_confidence",
    "history_len",
    "hit_history_len",
    "unique_ratio",
    "repeat_rate",
    "hit_ratio",
]


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s:
            rows.append(json.loads(s))
    return rows


def _state_global_features(row: Dict[str, object]) -> Dict[str, float]:
    incoming = dict(row.get("incoming_request", {}))
    hist = dict(row.get("history_summary", {}))
    return {
        "incoming_bucket": float(incoming.get("bucket", 0.0)),
        "incoming_confidence": float(incoming.get("confidence", 0.5)),
        "history_len": float(hist.get("history_len", 0.0)),
        "hit_history_len": float(hist.get("hit_history_len", 0.0)),
        "unique_ratio": float(hist.get("unique_ratio", 0.0)),
        "repeat_rate": float(hist.get("repeat_rate", 0.0)),
        "hit_ratio": float(hist.get("hit_ratio", 0.0)),
    }


def _candidate_loss_map(row: Dict[str, object]) -> Dict[str, float]:
    return {str(x["candidate_page_id"]): float(x["loss"]) for x in row.get("candidate_losses", [])}


def _offline_metrics(rows: Sequence[Dict[str, object]], choose_fn) -> Dict[str, float]:
    top1 = 0
    regrets: List[float] = []
    for row in rows:
        pred = str(choose_fn(row))
        oracle = str(row["oracle_victim"])
        losses = _candidate_loss_map(row)
        if pred not in losses or oracle not in losses:
            continue
        top1 += int(pred == oracle)
        regrets.append(float(losses[pred] - losses[oracle]))
    n = len(regrets)
    return {
        "decisions": float(n),
        "top1": float(top1 / n) if n else 0.0,
        "mean_regret": float(np.mean(regrets) if regrets else 0.0),
    }


def _fit_pairwise(rows: Sequence[Dict[str, object]]) -> HistGradientBoostingClassifier:
    xs: List[List[float]] = []
    ys: List[int] = []
    for row in rows:
        g = _state_global_features(row)
        losses = _candidate_loss_map(row)
        cands = list(row.get("candidate_features", []))
        for i in range(len(cands)):
            for j in range(i + 1, len(cands)):
                a = cands[i]
                b = cands[j]
                aid = str(a["candidate_page_id"])
                bid = str(b["candidate_page_id"])
                la = float(losses.get(aid, np.inf))
                lb = float(losses.get(bid, np.inf))
                if not np.isfinite(la) or not np.isfinite(lb) or la == lb:
                    continue
                diff = [float(a[c]) - float(b[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
                glob = [float(g[col]) for col in GLOBAL_COLUMNS]
                x = diff + glob
                y = 1 if la < lb else 0
                xs.append(x)
                ys.append(y)
                xs.append([-v for v in diff] + glob)
                ys.append(1 - y)
    if not xs:
        raise ValueError("No pairwise samples from joint decision rows")
    clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=250, random_state=7)
    clf.fit(np.asarray(xs, dtype=float), np.asarray(ys, dtype=int))
    return clf


def _choose_pairwise(row: Dict[str, object], clf: HistGradientBoostingClassifier) -> str:
    g = _state_global_features(row)
    cands = list(row.get("candidate_features", []))
    utilities: Dict[str, float] = {}
    for i, a in enumerate(cands):
        aid = str(a["candidate_page_id"])
        total = 0.0
        cnt = 0
        for j, b in enumerate(cands):
            if i == j:
                continue
            diff = [float(a[c]) - float(b[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
            x = np.asarray([diff + [float(g[col]) for col in GLOBAL_COLUMNS]], dtype=float)
            total += float(clf.predict_proba(x)[0][1])
            cnt += 1
        utilities[aid] = total / max(cnt, 1)
    return max(utilities.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _choose_joint(row: Dict[str, object], model: JointSoftmaxVictimModel) -> str:
    probs = model.predict_proba(
        candidate_features=list(row.get("candidate_features", [])),
        global_features=_state_global_features(row),
        feature_columns=EVICT_VALUE_V1_FEATURE_COLUMNS,
        global_columns=GLOBAL_COLUMNS,
    )
    return max(probs.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _online_features(
    *,
    candidates: List[PageId],
    incoming: Request,
    bucket_by_page: Dict[PageId, int],
    conf_by_page: Dict[PageId, float],
    recent_req_hist: Sequence[PageId],
    recent_hit_hist: Sequence[PageId],
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    req_bucket = int(incoming.metadata.get("bucket", 0))
    req_conf = float(incoming.metadata.get("confidence", 0.5))
    feats: List[Dict[str, object]] = []
    for c in candidates:
        rr = float(sum(1 for x in recent_req_hist if x == c) / len(recent_req_hist)) if recent_req_hist else 0.0
        hr = float(sum(1 for x in recent_hit_hist if x == c) / len(recent_hit_hist)) if recent_hit_hist else 0.0
        base = compute_candidate_features_v1(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            candidate=c,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_request_rate=rr,
            recent_hit_rate=hr,
        ).as_dict()
        row: Dict[str, object] = {"candidate_page_id": str(c)}
        for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
            row[col] = float(base[col])
        feats.append(row)

    hist = _history_summary(recent_req_hist=recent_req_hist, recent_hit_hist=recent_hit_hist)
    g = {
        "incoming_bucket": float(req_bucket),
        "incoming_confidence": float(req_conf),
        "history_len": float(hist["history_len"]),
        "hit_history_len": float(hist["hit_history_len"]),
        "unique_ratio": float(hist["unique_ratio"]),
        "repeat_rate": float(hist["repeat_rate"]),
        "hit_ratio": float(hist["hit_ratio"]),
    }
    return feats, g


def _replay_custom(requests: Sequence[Request], capacity: int, choose_victim_fn) -> int:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: collections.deque[PageId] = collections.deque(maxlen=64)
    recent_hit_hist: collections.deque[PageId] = collections.deque(maxlen=64)
    misses = 0

    for req in requests:
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue

        misses += 1
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        cand_feats, glob = _online_features(
            candidates=candidates,
            incoming=req,
            bucket_by_page=bucket_by_page,
            conf_by_page=conf_by_page,
            recent_req_hist=list(recent_req_hist),
            recent_hit_hist=list(recent_hit_hist),
        )
        victim = str(choose_victim_fn(cand_feats, glob))
        order.pop(victim, None)
        order[pid] = None
        recent_req_hist.append(pid)

    return misses


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train/evaluate lightweight joint cache-state victim model")
    ap.add_argument("--decision-dataset", default="data/derived/experiments/joint_cache_state_decision_v1/decision_states.jsonl")
    ap.add_argument("--dataset-summary", default="analysis/joint_cache_state_decision_v1/dataset_summary.json")
    ap.add_argument("--out-dir", default="analysis/joint_cache_state_model_light")
    args = ap.parse_args()

    rows = _read_jsonl(Path(args.decision_dataset))
    train = [r for r in rows if str(r.get("split")) == "train"]
    val = [r for r in rows if str(r.get("split")) == "val"]
    test = [r for r in rows if str(r.get("split")) == "test"]

    joint_train_rows = []
    for r in train:
        x = dict(r)
        x.update(_state_global_features(r))
        joint_train_rows.append(x)

    joint_model = JointSoftmaxVictimModel(JointSoftmaxConfig())
    joint_model.fit(joint_train_rows, feature_columns=EVICT_VALUE_V1_FEATURE_COLUMNS, global_columns=GLOBAL_COLUMNS)

    pair_clf = _fit_pairwise(train)

    model_rows: List[Dict[str, object]] = []
    for split, split_rows in [("train", train), ("val", val), ("test", test)]:
        ev1 = _offline_metrics(split_rows, lambda r: min(r["candidate_features"], key=lambda c: (float(c["candidate_predictor_score"]), str(c["candidate_page_id"])))["candidate_page_id"])
        pair = _offline_metrics(split_rows, lambda r: _choose_pairwise(r, pair_clf))
        joint = _offline_metrics(split_rows, lambda r: _choose_joint({**r, **_state_global_features(r)}, joint_model))
        for model_name, met in [("evict_value_v1_lightweight", ev1), ("history_pairwise_light", pair), ("joint_softmax", joint)]:
            model_rows.append({"split": split, "model": model_name, **met})

    summary_cfg = json.loads(Path(args.dataset_summary).read_text(encoding="utf-8"))
    trace_paths = [str(x) for x in summary_cfg.get("trace_paths", [])]
    capacities = [int(x) for x in summary_cfg.get("capacities", [])]

    replay_rows: List[Dict[str, object]] = []
    for trace_path in trace_paths:
        reqs, pages = load_trace(trace_path)
        for cap in capacities:
            base_res = run_policy(EvictValueV1Policy(scorer_mode="lightweight"), reqs, pages, cap)
            pair_miss = _replay_custom(
                reqs,
                cap,
                lambda cand_feats, glob: max(
                    {str(c["candidate_page_id"]): sum(float(pair_clf.predict_proba(np.asarray([[float(c[col]) - float(o[col]) for col in EVICT_VALUE_V1_FEATURE_COLUMNS] + [glob[g] for g in GLOBAL_COLUMNS]], dtype=float))[0][1]) for o in cand_feats if o is not c) / max(len(cand_feats) - 1, 1) for c in cand_feats}.items(),
                    key=lambda kv: (kv[1], kv[0]),
                )[0],
            )
            joint_miss = _replay_custom(
                reqs,
                cap,
                lambda cand_feats, glob: max(
                    joint_model.predict_proba(
                        candidate_features=cand_feats,
                        global_features=glob,
                        feature_columns=EVICT_VALUE_V1_FEATURE_COLUMNS,
                        global_columns=GLOBAL_COLUMNS,
                    ).items(),
                    key=lambda kv: (kv[1], kv[0]),
                )[0],
            )
            replay_rows.extend(
                [
                    {"trace": trace_path, "capacity": cap, "model": "evict_value_v1_lightweight", "misses": int(base_res.total_misses)},
                    {"trace": trace_path, "capacity": cap, "model": "history_pairwise_light", "misses": int(pair_miss)},
                    {"trace": trace_path, "capacity": cap, "model": "joint_softmax", "misses": int(joint_miss)},
                ]
            )

    by_model: Dict[str, List[float]] = {}
    for r in replay_rows:
        by_model.setdefault(str(r["model"]), []).append(float(r["misses"]))
    replay_mean = {m: float(mean(v)) for m, v in by_model.items()}
    joint_vs_pair = replay_mean.get("joint_softmax", 1e9) - replay_mean.get("history_pairwise_light", 1e9)
    promise = "not_better_than_pairwise" if joint_vs_pair >= 0 else "promising_but_needs_medium_scale_follow_up"

    out_dir = Path(args.out_dir)
    _write_csv(out_dir / "model_comparison.csv", model_rows)
    _write_csv(out_dir / "downstream_replay.csv", replay_rows)

    summary = {
        "dataset": args.decision_dataset,
        "train_decisions": len(train),
        "val_decisions": len(val),
        "test_decisions": len(test),
        "offline": model_rows,
        "downstream_replay_mean_misses": replay_mean,
        "joint_minus_pairwise_mean_misses": joint_vs_pair,
        "assessment": promise,
        "note": "Conservative readout on compact in-repo traces only; no canonical heavy_r1 pipeline changes.",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Lightweight joint cache-state model follow-up",
        "",
        "- Data source: existing joint decision-state dataset (no format changes).",
        "- Models compared: evict_value_v1 lightweight baseline, history-aware pairwise-light, joint softmax victim model.",
        f"- Replay mean misses: {replay_mean}",
        f"- Conservative assessment: **{promise}**.",
        "",
        "## Interpretation",
    ]
    if promise == "not_better_than_pairwise":
        lines.append("On this compact subset, joint softmax does not beat the simpler history-aware pairwise direction; medium-scale follow-up is not yet justified.")
    else:
        lines.append("Joint softmax shows early promise versus pairwise-light on this compact subset and could justify a careful medium-scale follow-up.")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
