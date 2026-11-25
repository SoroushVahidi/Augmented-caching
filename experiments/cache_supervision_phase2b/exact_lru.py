"""All forced-eviction labels with a shared deterministic LRU continuation."""

from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import islice
from typing import Hashable, Iterable, Sequence


@dataclass(frozen=True)
class LabelSnapshot:
    requests: int
    reference_misses: int
    advantage: tuple[int, ...]
    active: int

    @property
    def misses(self) -> tuple[int, ...]:
        return tuple(self.reference_misses - a for a in self.advantage)


@dataclass
class RolloutDetails:
    """Optional observation metadata from the same reference trajectory."""
    distinct_targets: tuple[int, ...] = ()
    distinct_snapshots: dict[int, LabelSnapshot] = field(default_factory=dict)
    fixed_distinct: dict[int, int] = field(default_factory=dict)
    attained_distinct: dict[int, int] = field(default_factory=dict)
    prevented: list[int] = field(default_factory=list)
    additional: list[int] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)
    reasons: list[int] = field(default_factory=list)


def rollout(
    residents: Sequence[Hashable],
    incoming: Hashable,
    future: Iterable[Hashable],
    horizons: Iterable[int] = (),
    *,
    stop_when_resolved: bool = False,
    details: RolloutDetails | None = None,
) -> tuple[dict[int, LabelSnapshot], LabelSnapshot]:
    """Use oldest-first residents and exclude the forced admission from labels.

    With no horizons, consume the observed suffix. With horizons, stop at the
    largest requested horizon and truncate snapshots at the observed trace end.
    Early resolution is available for relative labels. In that mode the final
    absolute miss count only covers the consumed prefix.
    """
    if len(residents) == 0:
        raise ValueError("Expected at least one resident")
    reference = OrderedDict.fromkeys(residents)
    if len(reference) != len(residents) or incoming in reference:
        raise ValueError("Expected unique residents and a nonresident incoming object")
    boundary, _ = reference.popitem(last=False)
    active = {page: index for index, page in enumerate(reference, start=1)}
    reference[incoming] = None
    advantage = [0] * len(residents)
    count = misses = boundary_hits = 0
    requested = set(horizons)
    if any(h < 0 for h in requested):
        raise ValueError("Horizons must be nonnegative")
    last_requested = max(requested, default=0)
    snapshots = {}
    if details is not None:
        if not stop_when_resolved or details.lengths:
            raise ValueError("Observation details require a fresh tracker and resolution mode")
        if any(d <= 0 for d in details.distinct_targets):
            raise ValueError("Distinct-object horizons must be positive")
        pending = set(details.distinct_targets)
        seen = set()
        details.prevented = [0] * len(residents)
        details.additional = [0] * len(residents)
        details.lengths = [0] * len(residents)
        details.reasons = [1] * len(residents)

        def resolve(index, additional, reason):
            details.prevented[index] = boundary_hits
            details.additional[index] = additional
            details.lengths[index] = count
            details.reasons[index] = reason


    def snapshot():
        values = advantage.copy()
        for index in active.values():
            values[index] = boundary_hits
        return LabelSnapshot(count, misses, tuple(values), len(active))

    if 0 in requested:
        snapshots[0] = snapshot()
        if details is not None:
            details.fixed_distinct[0] = 0
    if not stop_when_resolved or active or last_requested or (details is not None and pending):
        requests = islice(future, last_requested) if requested and not stop_when_resolved else iter(future)
        for count, page in enumerate(requests, start=1):
            if page in reference:
                index = active.pop(page, None)
                if index is not None:
                    advantage[index] = boundary_hits - 1
                    if details is not None:
                        resolve(index, 1, 1)
                reference.move_to_end(page)
            else:
                misses += 1
                if page == boundary:
                    boundary_hits += 1
                boundary, _ = reference.popitem(last=False)
                index = active.pop(boundary, None)
                if index is not None:
                    advantage[index] = boundary_hits
                    if details is not None:
                        resolve(index, 0, 1)
                reference[page] = None
            if details is not None:
                # Retain distinct IDs only while a requested snapshot needs them.
                if pending or count <= last_requested:
                    seen.add(page)
                if len(seen) in pending:
                    target = len(seen)
                    details.distinct_snapshots[target] = snapshot()
                    details.attained_distinct[target] = target
                    pending.remove(target)
            if count in requested:
                snapshots[count] = snapshot()
                if details is not None:
                    details.fixed_distinct[count] = len(seen)
            if (stop_when_resolved and not active and count >= last_requested
                    and (details is None or not pending)):
                break
    final = snapshots.get(count) or snapshot()
    for horizon in requested:
        snapshots.setdefault(horizon, final)
        if details is not None:
            details.fixed_distinct.setdefault(horizon, len(seen))
    if details is not None:
        for target in pending:
            details.distinct_snapshots[target] = final
            details.attained_distinct[target] = len(seen)
        for index in active.values():
            resolve(index, 0, 2)
    return snapshots, final
