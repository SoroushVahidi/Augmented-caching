from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


@dataclass(frozen=True)
class PreferenceCycleMetrics:
    candidate_count: int
    edge_count: int
    has_cycle: bool
    cycle_triplet_count: int
    strongly_connected_component_sizes: Tuple[int, ...]


def build_nested_fraction_subsets(
    decision_ids: Iterable[str],
    fractions: Sequence[float],
    *,
    seed: int = 0,
) -> Dict[float, Tuple[str, ...]]:
    """Build deterministic nested decision-id subsets for learning curves.

    The subsets are order-independent with respect to the input iterable:
    every decision id receives a stable hash rank, then each fraction takes
    a prefix of that single global order. This guarantees:

    - determinism for a fixed ``seed`` and decision-id set,
    - nested subsets across fractions,
    - exact same-example membership for every objective that reuses the same
      selected decision ids.
    """

    unique_ids = sorted({str(decision_id) for decision_id in decision_ids})
    if not unique_ids:
        return {float(fraction): tuple() for fraction in fractions}

    normalized: List[float] = []
    for fraction in fractions:
        frac = float(fraction)
        if not (0.0 < frac <= 1.0):
            raise ValueError(f"fractions must lie in (0, 1], got {fraction!r}")
        normalized.append(frac)

    ordered = sorted(unique_ids, key=lambda value: _stable_rank(seed=seed, value=value))
    total = len(ordered)
    subsets: Dict[float, Tuple[str, ...]] = {}
    for frac in normalized:
        count = min(total, max(1, int(math.ceil(frac * total))))
        subsets[frac] = tuple(ordered[:count])
    return subsets


def filter_rows_by_decision_subset(
    rows: Sequence[Mapping[str, object]],
    allowed_decision_ids: Iterable[str],
) -> List[Dict[str, object]]:
    allowed = {str(decision_id) for decision_id in allowed_decision_ids}
    return [dict(row) for row in rows if str(row["decision_id"]) in allowed]


def compute_preference_cycle_metrics(
    candidates: Sequence[str],
    preferred_edges: Iterable[tuple[str, str]],
) -> PreferenceCycleMetrics:
    unique_candidates = tuple(dict.fromkeys(str(candidate) for candidate in candidates))
    adjacency = {candidate: set() for candidate in unique_candidates}
    edge_count = 0

    for src, dst in preferred_edges:
        src_s = str(src)
        dst_s = str(dst)
        if src_s == dst_s or src_s not in adjacency or dst_s not in adjacency:
            continue
        if dst_s not in adjacency[src_s]:
            adjacency[src_s].add(dst_s)
            edge_count += 1

    scc_sizes = tuple(sorted((len(component) for component in _tarjan_scc(adjacency) if len(component) > 1), reverse=True))
    cycle_triplet_count = 0
    for a, b, c in combinations(unique_candidates, 3):
        if (
            (b in adjacency[a] and c in adjacency[b] and a in adjacency[c]) or
            (c in adjacency[a] and b in adjacency[c] and a in adjacency[b])
        ):
            cycle_triplet_count += 1

    return PreferenceCycleMetrics(
        candidate_count=len(unique_candidates),
        edge_count=edge_count,
        has_cycle=bool(scc_sizes),
        cycle_triplet_count=cycle_triplet_count,
        strongly_connected_component_sizes=scc_sizes,
    )


def _stable_rank(*, seed: int, value: str) -> tuple[str, str]:
    payload = f"{seed}:{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), value


def _tarjan_scc(adjacency: Mapping[str, set[str]]) -> List[Tuple[str, ...]]:
    index = 0
    stack: List[str] = []
    stack_members = set()
    index_by_node: Dict[str, int] = {}
    lowlink: Dict[str, int] = {}
    components: List[Tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        index_by_node[node] = index
        lowlink[node] = index
        index += 1
        stack.append(node)
        stack_members.add(node)

        for neighbor in adjacency[node]:
            if neighbor not in index_by_node:
                strongconnect(neighbor)
                lowlink[node] = min(lowlink[node], lowlink[neighbor])
            elif neighbor in stack_members:
                lowlink[node] = min(lowlink[node], index_by_node[neighbor])

        if lowlink[node] == index_by_node[node]:
            component: List[str] = []
            while True:
                popped = stack.pop()
                stack_members.remove(popped)
                component.append(popped)
                if popped == node:
                    break
            components.append(tuple(component))

    for candidate in adjacency:
        if candidate not in index_by_node:
            strongconnect(candidate)
    return components


__all__ = [
    "PreferenceCycleMetrics",
    "build_nested_fraction_subsets",
    "compute_preference_cycle_metrics",
    "filter_rows_by_decision_subset",
]
