from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Sequence, Tuple

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.offline.general_caching_approx import GeneralCachingLPApproxSolver
from lafc.offline.validation import validate_uniform_paging_inputs
from lafc.types import Page, PageId, Request


@dataclass(frozen=True)
class OfflineTeacherLabelConfig:
    horizon: int = 32
    history_window: int = 64
    include_pairwise_ties: bool = False


def _simulate_exact_belady_suffix(
    future_reqs: Sequence[Request],
    initial_cache: Sequence[PageId],
    capacity: int,
) -> float:
    """Exact suffix cost in uniform paging with a fixed initial cache state."""
    if not future_reqs:
        return 0.0

    pos: Dict[PageId, Deque[int]] = collections.defaultdict(collections.deque)
    for i, req in enumerate(future_reqs):
        pos[req.page_id].append(i)

    cache = set(initial_cache)
    misses = 0
    for i, req in enumerate(future_reqs):
        pid = req.page_id
        pos[pid].popleft()
        if pid in cache:
            continue
        misses += 1
        if len(cache) >= capacity:
            victim = max(
                sorted(cache),
                key=lambda q: (pos[q][0] if pos[q] else math.inf, q),
            )
            cache.remove(victim)
        cache.add(pid)
    return float(misses)


def _compute_candidate_feature_rows(
    *,
    candidates: List[PageId],
    req_bucket: int,
    req_conf: float,
    bucket_by_page: Dict[PageId, int],
    conf_by_page: Dict[PageId, float],
    recent_req_hist: Deque[PageId],
    recent_hit_hist: Deque[PageId],
) -> Dict[PageId, Dict[str, float]]:
    out: Dict[PageId, Dict[str, float]] = {}
    for candidate in candidates:
        req_rate = (sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)) if recent_req_hist else 0.0
        hit_rate = (sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)) if recent_hit_hist else 0.0
        out[candidate] = compute_candidate_features_v1(
            request_bucket=req_bucket,
            request_confidence=req_conf,
            candidates=candidates,
            candidate=candidate,
            bucket_by_page=bucket_by_page,
            confidence_by_page=conf_by_page,
            recent_request_rate=req_rate,
            recent_hit_rate=hit_rate,
        ).as_dict()
    return out


def _rank_from_scores(scores: Dict[PageId, float]) -> Dict[PageId, int]:
    ordered = sorted(scores.items(), key=lambda x: (x[1], str(x[0])))
    return {pid: i + 1 for i, (pid, _v) in enumerate(ordered)}


def _simulate_no_insert_suffix_cost(
    future_reqs: Sequence[Request],
    initial_cache: Sequence[PageId],
    pages: Dict[PageId, Page],
) -> float:
    cache = set(initial_cache)
    cost = 0.0
    for req in future_reqs:
        if req.page_id in cache:
            continue
        cost += float(pages[req.page_id].weight)
    return cost


def _uniform_exact_applicable(
    requests: Sequence[Request],
    pages: Dict[PageId, Page],
    page_sizes: Dict[PageId, float],
) -> bool:
    try:
        validate_uniform_paging_inputs(requests, pages, mode="strict")
    except Exception:
        return False
    sizes = {float(page_sizes[pid]) for pid in page_sizes}
    return len(sizes) == 1


def build_offline_teacher_candidate_rows(
    *,
    requests: Sequence[Request],
    pages: Dict[PageId, Page],
    page_sizes: Dict[PageId, float],
    capacity: float,
    trace_name: str,
    trace_family: str,
    cfg: OfflineTeacherLabelConfig,
) -> List[Dict[str, object]]:
    """Build candidate-level supervision rows using offline teacher costs.

    Computational shortcut:
    - per decision + candidate, evaluate teacher on a finite suffix horizon
      (`cfg.horizon`) instead of full trace tail for tractability.
    """

    use_exact = _uniform_exact_applicable(requests, pages, page_sizes)
    teacher_type = "exact_teacher_belady" if use_exact else "approx_teacher_lp"
    approx_solver = GeneralCachingLPApproxSolver()

    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    cache_size = 0.0
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

        pid_size = float(page_sizes[pid])
        if pid_size > float(capacity):
            recent_req_hist.append(pid)
            continue
        if cache_size + pid_size <= float(capacity):
            order[pid] = None
            cache_size += pid_size
            recent_req_hist.append(pid)
            continue

        free_needed = pid_size - (float(capacity) - cache_size)
        candidates = [p for p in order.keys() if float(page_sizes[p]) + 1e-12 >= free_needed]
        if not candidates:
            recent_req_hist.append(pid)
            continue
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))
        features = _compute_candidate_feature_rows(
            candidates=candidates,
            req_bucket=req_bucket,
            req_conf=req_conf,
            bucket_by_page=bucket_by_page,
            conf_by_page=conf_by_page,
            recent_req_hist=recent_req_hist,
            recent_hit_hist=recent_hit_hist,
        )

        decision_id = f"{trace_name}|c{capacity}|t{t}|h{cfg.horizon}|teacher"
        future = requests[t + 1 : t + 1 + cfg.horizon]

        teacher_cost: Dict[PageId, float] = {}
        heuristic_proxy_loss: Dict[PageId, float] = {}

        for candidate in candidates:
            forced_cache = [p for p in order.keys() if p != candidate] + [pid]
            if use_exact:
                teacher_cost[candidate] = _simulate_exact_belady_suffix(
                    future_reqs=future,
                    initial_cache=forced_cache,
                    capacity=int(capacity),
                )
            else:
                if not future:
                    teacher_cost[candidate] = 0.0
                else:
                    try:
                        result = approx_solver.solve(
                            requests=future,
                            pages=pages,
                            page_sizes=page_sizes,
                            capacity=float(capacity),
                            allow_bypass=True,
                            initial_cache=forced_cache,
                        )
                        teacher_cost[candidate] = float(result.total_cost)
                    except ValueError:
                        teacher_cost[candidate] = _simulate_no_insert_suffix_cost(
                            future_reqs=future,
                            initial_cache=forced_cache,
                            pages=pages,
                        )

            after = [p for p in order.keys() if p != candidate] + [pid]
            heuristic_proxy_loss[candidate] = float(
                _simulate_lru_misses(after, future, capacity=int(capacity))
            )

        best = min(teacher_cost.values()) if teacher_cost else 0.0
        ranks = _rank_from_scores(teacher_cost)
        best_heur = min(heuristic_proxy_loss.values()) if heuristic_proxy_loss else 0.0

        for candidate in candidates:
            regret = float(teacher_cost[candidate] - best)
            row: Dict[str, object] = {
                "decision_id": decision_id,
                "trace": trace_name,
                "family": trace_family,
                "request_t": t,
                "capacity": capacity,
                "horizon": cfg.horizon,
                "teacher_type": teacher_type,
                "label_source": "offline_teacher",
                "request_page_id": pid,
                "candidate_page_id": candidate,
                "cache_contents": "|".join(order.keys()),
                "candidate_count": len(candidates),
                "teacher_cost": float(teacher_cost[candidate]),
                "teacher_best": float(regret == 0.0),
                "teacher_regret": regret,
                "teacher_rank": int(ranks[candidate]),
                "heuristic_proxy_loss": float(heuristic_proxy_loss[candidate]),
                "heuristic_proxy_best": float(heuristic_proxy_loss[candidate] == best_heur),
                "teacher_minus_heuristic": float(teacher_cost[candidate] - heuristic_proxy_loss[candidate]),
            }
            row.update(features[candidate])
            rows.append(row)

        lru_victim = candidates[0]
        order.pop(lru_victim)
        cache_size -= float(page_sizes[lru_victim])
        order[pid] = None
        cache_size += pid_size
        recent_req_hist.append(pid)

    return rows


def build_offline_teacher_pairwise_rows(
    candidate_rows: Iterable[Dict[str, object]],
    *,
    include_ties: bool = False,
) -> List[Dict[str, object]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in candidate_rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)

    rows: List[Dict[str, object]] = []
    for decision_id, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: str(r["candidate_page_id"]))
        for i in range(len(items_sorted)):
            for j in range(i + 1, len(items_sorted)):
                left = items_sorted[i]
                right = items_sorted[j]
                reg_i = float(left["teacher_regret"])
                reg_j = float(right["teacher_regret"])
                if reg_i == reg_j and not include_ties:
                    continue
                label = 1 if reg_i < reg_j else 0
                pair: Dict[str, object] = {
                    "decision_id": decision_id,
                    "trace": left["trace"],
                    "family": left["family"],
                    "request_t": left["request_t"],
                    "capacity": left["capacity"],
                    "horizon": left["horizon"],
                    "teacher_type": left["teacher_type"],
                    "candidate_i_page_id": left["candidate_page_id"],
                    "candidate_j_page_id": right["candidate_page_id"],
                    "teacher_regret_i": reg_i,
                    "teacher_regret_j": reg_j,
                    "teacher_regret_diff": reg_i - reg_j,
                    "label_i_better": int(label),
                    "is_tie": float(reg_i == reg_j),
                }
                for col in EVICT_VALUE_V1_FEATURE_COLUMNS:
                    fi = float(left[col])
                    fj = float(right[col])
                    pair[f"i_{col}"] = fi
                    pair[f"j_{col}"] = fj
                    pair[f"delta_{col}"] = fi - fj
                rows.append(pair)
    return rows


__all__ = [
    "OfflineTeacherLabelConfig",
    "build_offline_teacher_candidate_rows",
    "build_offline_teacher_pairwise_rows",
]
