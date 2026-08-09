from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Callable, Deque, Dict, List, Mapping, Sequence, Tuple

from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    _build_distinct_suffix_counts,
    _build_occurrence_index,
    build_candidate_rows_for_full_cache_state,
)
from lafc.types import PageId, Request

EXACT_ORACLE_WELL_DEFINED = "EXACT_ORACLE_WELL_DEFINED"
EXACT_ORACLE_REQUIRES_CLARIFICATION = "EXACT_ORACLE_REQUIRES_CLARIFICATION"
NOT_MEANINGFUL = "NOT_MEANINGFUL"

MINIMIZE = "min"
MAXIMIZE = "max"

DecisionScorer = Callable[[Sequence[Mapping[str, object]]], Mapping[PageId, float]]


@dataclass(frozen=True)
class ExactOracleObjectiveSpec:
    canonical_name: str
    label_column: str
    optimize: str
    status: str
    explanation: str


@dataclass(frozen=True)
class ExactOracleDecision:
    decision_id: str
    request_t: int
    candidate_values: Dict[PageId, float]
    optimal_candidates: Tuple[PageId, ...]
    exact_candidate: PageId
    chosen_candidate: PageId
    exact_value: float
    chosen_value: float
    target_regret: float
    agrees_with_exact: bool


@dataclass(frozen=True)
class OracleReplaySummary:
    policy_name: str
    objective: str
    trace_name: str
    trace_family: str
    capacity: int
    horizon: int
    total_hits: int
    total_misses: int
    hit_sequence: Tuple[bool, ...]
    decisions: Tuple[ExactOracleDecision, ...]


def get_exact_oracle_objective_specs() -> Dict[str, ExactOracleObjectiveSpec]:
    return {
        "eviction_loss": ExactOracleObjectiveSpec(
            canonical_name="eviction_loss",
            label_column="eviction_loss_label",
            optimize=MINIMIZE,
            status=EXACT_ORACLE_WELL_DEFINED,
            explanation=(
                "Finite-horizon H-step eviction-loss target with LRU continuation, "
                "matching the current frozen training-label definition exactly."
            ),
        ),
        "next_arrival": ExactOracleObjectiveSpec(
            canonical_name="next_arrival",
            label_column="next_arrival_label_censored",
            optimize=MAXIMIZE,
            status=EXACT_ORACLE_WELL_DEFINED,
            explanation=(
                "Evict the candidate with the largest censored next-arrival label, "
                "matching the scalar next-arrival objective."
            ),
        ),
        "reuse_distance": ExactOracleObjectiveSpec(
            canonical_name="reuse_distance",
            label_column="reuse_distance_label_censored",
            optimize=MAXIMIZE,
            status=EXACT_ORACLE_WELL_DEFINED,
            explanation=(
                "Evict the candidate with the largest censored reuse-distance label, "
                "matching the scalar reuse-distance objective."
            ),
        ),
        "objective_pairwise": ExactOracleObjectiveSpec(
            canonical_name="objective_pairwise",
            label_column="next_arrival_label_censored",
            optimize=MAXIMIZE,
            status=EXACT_ORACLE_REQUIRES_CLARIFICATION,
            explanation=(
                "The frozen pairwise objective is trained from next-arrival ordering, "
                "so an exact multi-candidate oracle is only defined here via that "
                "underlying scalar source label."
            ),
        ),
    }


def get_exact_oracle_objective_spec(objective: str) -> ExactOracleObjectiveSpec:
    alias_map = {
        "objective_eviction_loss": "eviction_loss",
        "objective_next_arrival": "next_arrival",
        "objective_reuse_distance": "reuse_distance",
        "eviction_loss_scalar": "eviction_loss",
        "eviction_loss_pairwise": "eviction_loss",
    }
    key = alias_map.get(objective, objective)
    specs = get_exact_oracle_objective_specs()
    if key not in specs:
        raise ValueError(f"Unsupported exact-oracle objective: {objective!r}")
    return specs[key]


def _candidate_values(
    candidate_rows: Sequence[Mapping[str, object]],
    spec: ExactOracleObjectiveSpec,
) -> Dict[PageId, float]:
    return {str(row["candidate_page_id"]): float(row[spec.label_column]) for row in candidate_rows}


def _best_value(values: Mapping[PageId, float], optimize: str) -> float:
    if optimize == MINIMIZE:
        return min(values.values())
    if optimize == MAXIMIZE:
        return max(values.values())
    raise ValueError(f"Unsupported optimization direction: {optimize!r}")


def _regret(chosen_value: float, best_value: float, optimize: str) -> float:
    if optimize == MINIMIZE:
        return chosen_value - best_value
    if optimize == MAXIMIZE:
        return best_value - chosen_value
    raise ValueError(f"Unsupported optimization direction: {optimize!r}")


def select_exact_target_candidate(
    candidate_rows: Sequence[Mapping[str, object]],
    objective: str,
) -> Tuple[PageId, Dict[PageId, float]]:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")
    spec = get_exact_oracle_objective_spec(objective)
    values = _candidate_values(candidate_rows, spec)
    best = _best_value(values, spec.optimize)
    optimal = tuple(sorted(pid for pid, value in values.items() if value == best))
    return optimal[0], values


def compare_choice_to_exact_target(
    candidate_rows: Sequence[Mapping[str, object]],
    objective: str,
    chosen_candidate: PageId,
) -> ExactOracleDecision:
    if not candidate_rows:
        raise ValueError("candidate_rows must not be empty")
    spec = get_exact_oracle_objective_spec(objective)
    values = _candidate_values(candidate_rows, spec)
    if chosen_candidate not in values:
        raise ValueError(f"Chosen candidate {chosen_candidate!r} is not a valid eviction candidate")
    chosen_value = values[chosen_candidate]
    decision_id = str(candidate_rows[0]["decision_id"])
    request_t = int(candidate_rows[0]["decision_t"])
    for row in candidate_rows:
        if int(row["decision_t"]) != request_t or str(row["decision_id"]) != decision_id:
            raise ValueError("candidate_rows must come from a single decision")
    exact_candidate, _ = select_exact_target_candidate(candidate_rows, objective)
    best = _best_value(values, spec.optimize)
    optimal = tuple(sorted(pid for pid, value in values.items() if value == best))
    return ExactOracleDecision(
        decision_id=decision_id,
        request_t=request_t,
        candidate_values=values,
        optimal_candidates=optimal,
        exact_candidate=exact_candidate,
        chosen_candidate=chosen_candidate,
        exact_value=best,
        chosen_value=chosen_value,
        target_regret=float(_regret(chosen_value, best, spec.optimize)),
        agrees_with_exact=(chosen_candidate in optimal),
    )


def _run_replay(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
    objective: str,
    policy_name: str,
    choose_candidate: Callable[[Sequence[Mapping[str, object]]], PageId],
) -> OracleReplaySummary:
    order: "collections.OrderedDict[PageId, None]" = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    hits = 0
    misses = 0
    hit_sequence: List[bool] = []
    decisions: List[ExactOracleDecision] = []
    occurrence_index = _build_occurrence_index(requests)
    distinct_suffix_counts = _build_distinct_suffix_counts(requests)

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        if pid in order:
            order.move_to_end(pid)
            hits += 1
            hit_sequence.append(True)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue

        misses += 1
        hit_sequence.append(False)
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidate_rows = build_candidate_rows_for_full_cache_state(
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
            occurrence_index=occurrence_index,
            distinct_suffix_counts=distinct_suffix_counts,
        )
        chosen_candidate = choose_candidate(candidate_rows)
        decision = compare_choice_to_exact_target(candidate_rows, objective, chosen_candidate)
        decisions.append(decision)
        order.pop(chosen_candidate)
        order[pid] = None
        recent_req_hist.append(pid)

    return OracleReplaySummary(
        policy_name=policy_name,
        objective=get_exact_oracle_objective_spec(objective).canonical_name,
        trace_name=trace_name,
        trace_family=trace_family,
        capacity=capacity,
        horizon=int(cfg.horizon),
        total_hits=hits,
        total_misses=misses,
        hit_sequence=tuple(hit_sequence),
        decisions=tuple(decisions),
    )


def replay_exact_target_policy(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
    objective: str,
) -> OracleReplaySummary:
    return _run_replay(
        requests=requests,
        capacity=capacity,
        trace_name=trace_name,
        trace_family=trace_family,
        cfg=cfg,
        objective=objective,
        policy_name=f"exact_target_oracle[{get_exact_oracle_objective_spec(objective).canonical_name}]",
        choose_candidate=lambda rows: select_exact_target_candidate(rows, objective)[0],
    )


def replay_score_driven_policy(
    *,
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    trace_family: str,
    cfg: ObjectiveAblationConfig,
    objective: str,
    scorer: DecisionScorer,
    policy_name: str = "score_driven_policy",
) -> OracleReplaySummary:
    spec = get_exact_oracle_objective_spec(objective)

    def choose_candidate(candidate_rows: Sequence[Mapping[str, object]]) -> PageId:
        scores = {str(pid): float(score) for pid, score in scorer(candidate_rows).items()}
        candidate_ids = [str(row["candidate_page_id"]) for row in candidate_rows]
        missing = [pid for pid in candidate_ids if pid not in scores]
        extra = [pid for pid in scores if pid not in candidate_ids]
        if missing or extra:
            raise ValueError(
                f"Scorer must return exactly one score per candidate; missing={missing}, extra={extra}"
            )
        if spec.optimize == MINIMIZE:
            return min(candidate_ids, key=lambda pid: (scores[pid], candidate_ids.index(pid)))
        return max(candidate_ids, key=lambda pid: (scores[pid], candidate_ids.index(pid)))

    return _run_replay(
        requests=requests,
        capacity=capacity,
        trace_name=trace_name,
        trace_family=trace_family,
        cfg=cfg,
        objective=objective,
        policy_name=policy_name,
        choose_candidate=choose_candidate,
    )


def summarize_decision_diagnostics(decisions: Sequence[ExactOracleDecision]) -> Dict[str, float]:
    if not decisions:
        return {
            "decision_count": 0.0,
            "agreement_rate": 0.0,
            "mean_target_regret": 0.0,
            "non_optimal_fraction": 0.0,
        }
    agreement_count = sum(1 for decision in decisions if decision.agrees_with_exact)
    regret_sum = sum(float(decision.target_regret) for decision in decisions)
    non_optimal = sum(1 for decision in decisions if not decision.agrees_with_exact)
    count = len(decisions)
    return {
        "decision_count": float(count),
        "agreement_rate": float(agreement_count / count),
        "mean_target_regret": float(regret_sum / count),
        "non_optimal_fraction": float(non_optimal / count),
    }


__all__ = [
    "EXACT_ORACLE_REQUIRES_CLARIFICATION",
    "EXACT_ORACLE_WELL_DEFINED",
    "NOT_MEANINGFUL",
    "ExactOracleDecision",
    "ExactOracleObjectiveSpec",
    "OracleReplaySummary",
    "compare_choice_to_exact_target",
    "get_exact_oracle_objective_spec",
    "get_exact_oracle_objective_specs",
    "replay_exact_target_policy",
    "replay_score_driven_policy",
    "select_exact_target_candidate",
    "summarize_decision_diagnostics",
]
