from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.types import PageId, Request


@dataclass(frozen=True)
class JointCacheStateDatasetConfig:
    horizon: int = 8
    history_window: int = 64


def _split_for_decision(decision_id: str) -> str:
    h = int(hashlib.sha256(decision_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if h < 70:
        return "train"
    if h < 85:
        return "val"
    return "test"


def _rate(hist: Sequence[PageId], pid: PageId) -> float:
    if not hist:
        return 0.0
    return float(sum(1 for x in hist if x == pid) / len(hist))


def _history_summary(
    *,
    recent_req_hist: Sequence[PageId],
    recent_hit_hist: Sequence[PageId],
) -> Dict[str, object]:
    req_len = len(recent_req_hist)
    hit_len = len(recent_hit_hist)
    uniq = len(set(recent_req_hist)) if recent_req_hist else 0
    repeat_rate = (float(req_len - uniq) / float(req_len)) if req_len else 0.0

    req_counts = collections.Counter(str(x) for x in recent_req_hist)
    top_pages = [
        {"page_id": pid, "count": int(cnt)}
        for pid, cnt in sorted(req_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:3]
    ]

    return {
        "history_len": req_len,
        "hit_history_len": hit_len,
        "unique_ratio": (float(uniq) / float(req_len)) if req_len else 0.0,
        "repeat_rate": repeat_rate,
        "hit_ratio": (float(hit_len) / float(req_len)) if req_len else 0.0,
        "top_recent_pages": top_pages,
    }


def _serialize_candidate_features(
    *,
    candidates: List[PageId],
    request_bucket: int,
    request_confidence: float,
    bucket_by_page: Dict[PageId, int],
    confidence_by_page: Dict[PageId, float],
    recent_req_hist: Sequence[PageId],
    recent_hit_hist: Sequence[PageId],
) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for candidate in candidates:
        feats = compute_candidate_features_v1(
            request_bucket=request_bucket,
            request_confidence=request_confidence,
            candidates=candidates,
            candidate=candidate,
            bucket_by_page=bucket_by_page,
            confidence_by_page=confidence_by_page,
            recent_request_rate=_rate(recent_req_hist, candidate),
            recent_hit_rate=_rate(recent_hit_hist, candidate),
        ).as_dict()
        row: Dict[str, object] = {"candidate_page_id": str(candidate)}
        for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
            row[col] = float(feats[col])
        out.append(row)
    return out


def build_joint_cache_state_examples(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    cfg: JointCacheStateDatasetConfig,
) -> List[Dict[str, object]]:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)

    rows: List[Dict[str, object]] = []
    for t, req in enumerate(requests):
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

        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))
        candidate_features = _serialize_candidate_features(
            candidates=candidates,
            request_bucket=req_bucket,
            request_confidence=req_conf,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_req_hist=list(recent_req_hist),
            recent_hit_hist=list(recent_hit_hist),
        )

        future = requests[t + 1 :]
        losses: List[Dict[str, object]] = []
        for c in candidates:
            after = [p for p in candidates if p != c] + [pid]
            losses.append(
                {
                    "candidate_page_id": str(c),
                    "loss": float(_simulate_lru_misses(after, future[: cfg.horizon], capacity=capacity)),
                }
            )

        best = min(losses, key=lambda x: (float(x["loss"]), str(x["candidate_page_id"])))
        decision_id = f"{trace_name}|cap={capacity}|t={t}"

        rows.append(
            {
                "decision_id": decision_id,
                "split": _split_for_decision(decision_id),
                "trace": trace_name,
                "capacity": int(capacity),
                "t": int(t),
                "horizon": int(cfg.horizon),
                "incoming_request": {
                    "page_id": str(pid),
                    "bucket": req_bucket,
                    "confidence": req_conf,
                },
                "cache_residents": [str(x) for x in candidates],
                "history_summary": _history_summary(recent_req_hist=list(recent_req_hist), recent_hit_hist=list(recent_hit_hist)),
                "candidate_features": candidate_features,
                "candidate_losses": losses,
                "oracle_victim": str(best["candidate_page_id"]),
                "oracle_loss": float(best["loss"]),
            }
        )

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)

    return rows


def to_jsonl_lines(rows: Iterable[Dict[str, object]]) -> List[str]:
    return [json.dumps(r, sort_keys=True) for r in rows]


__all__ = [
    "JointCacheStateDatasetConfig",
    "build_joint_cache_state_examples",
    "to_jsonl_lines",
]
