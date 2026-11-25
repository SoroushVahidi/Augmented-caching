"""Causal request and resident observations for unit-size object caches."""

from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
import math

import numpy as np

WINDOWS = (16, 64, 256, 1024)
GAPS = 16
GHOST_LIMIT = 2048
GLOBAL_NAMES = (
    "request_age", "request_count", "request_prior_gap", "known", "hit",
    "unique_rate64", "hit_rate16", "hit_rate64", "hit_rate256", "miss_rate64",
    "hit_rate_shift", "occupancy", "log_capacity", "eviction_rate64",
)
STATIC_NAMES = (
    "age", "count", "gap_mean", "gap_variance", "gap_min", "gap_max",
    "recency_rank", "frequency_rank", "decayed_popularity", "evictions",
    "reinsertions", "age_since_eviction", "known_eviction", "has_gap",
    *(f"frequency_{w}" for w in WINDOWS),
    *(f"gap_{i}" for i in range(1, GAPS + 1)),
    *(f"known_gap_{i}" for i in range(1, GAPS + 1)),
)


@dataclass(slots=True)
class ObjectHistory:
    last: int
    count: int = 0
    gaps: deque = field(default_factory=lambda: deque(maxlen=GAPS))
    mean: float = 0.0
    m2: float = 0.0
    minimum: int = 0
    maximum: int = 0
    popularity: float = 0.0
    evictions: int = 0
    reinsertions: int = 0
    last_eviction: int = -1

    def observe(self, t, hit):
        if self.count:
            gap = t - self.last
            self.gaps.append(gap)
            delta = gap - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (gap - self.mean)
            self.minimum = min(self.minimum, gap) if self.count > 1 else gap
            self.maximum = max(self.maximum, gap)
            self.popularity *= math.exp(-gap / 64)
            if not hit and self.last_eviction >= 0:
                self.reinsertions += 1
        self.popularity += 1
        self.count += 1
        self.last = t


class Window:
    def __init__(self, size):
        self.size = size
        self.items = deque()
        self.count = Counter()

    def add(self, item):
        if len(self.items) == self.size:
            previous = self.items.popleft()
            self.count[previous] -= 1
            if not self.count[previous]:
                del self.count[previous]
        self.items.append(item)
        self.count[item] += 1

    def rate(self, item):
        return self.count[item] / max(1, len(self.items))


class CausalHistory:
    def __init__(self, capacity, ghost_limit=GHOST_LIMIT):
        self.capacity = capacity
        self.ghost_limit = ghost_limit
        self.residents = {}
        self.ghosts = OrderedDict()
        self.requests = {size: Window(size) for size in WINDOWS}
        self.hits = {size: Window(size) for size in WINDOWS[:3]}
        self.evictions = Window(64)
        self.peak_ghosts = 0

    def before(self, pid, t, hit):
        entry = self.residents.get(pid) or self.ghosts.get(pid)
        rates = [self.hits[w].rate(True) for w in WINDOWS[:3]]
        age = t - entry.last if entry else 0
        gap = entry.gaps[-1] if entry and entry.gaps else 0
        return np.array([
            math.log1p(age), math.log1p(entry.count if entry else 0), math.log1p(gap),
            float(entry is not None), float(hit), len(self.requests[64].count) / max(1, len(self.requests[64].items)),
            *rates, self.hits[64].rate(False), rates[0] - rates[-1],
            len(self.residents) / self.capacity, math.log1p(self.capacity), self.evictions.rate(True),
        ], dtype=np.float32)

    def candidates(self, residents, t):
        entries = [self.residents[pid] for pid in residents]
        counts = np.array([e.count for e in entries])
        ranks = np.searchsorted(np.sort(counts), counts) / max(1, len(entries) - 1)
        x = np.zeros((len(entries), len(STATIC_NAMES)), dtype=np.float32)
        for i, (pid, e) in enumerate(zip(residents, entries)):
            age = t - e.last
            x[i, :18] = (
                math.log1p(age), math.log1p(e.count), math.log1p(e.mean),
                math.log1p(max(0, e.m2) / max(1, e.count - 1)), math.log1p(e.minimum),
                math.log1p(e.maximum), i / max(1, len(entries) - 1), ranks[i],
                math.log1p(e.popularity * math.exp(-age / 64)), math.log1p(e.evictions),
                math.log1p(e.reinsertions), math.log1p(t - e.last_eviction) if e.last_eviction >= 0 else 0,
                e.last_eviction >= 0, bool(e.gaps), *(self.requests[w].rate(pid) for w in WINDOWS),
            )
            n = len(e.gaps)
            if n:
                x[i, 18:18+n] = [math.log1p(gap) for gap in reversed(e.gaps)]
                x[i, 18+GAPS:18+GAPS+n] = 1.0
        return x

    def after(self, pid, t, hit, victim):
        incoming = None if hit else self.ghosts.pop(pid, None)
        if victim is not None:
            previous = self.residents.pop(victim)
            previous.evictions += 1
            previous.last_eviction = t
            if self.ghost_limit:
                self.ghosts[victim] = previous
                if len(self.ghosts) > self.ghost_limit:
                    self.ghosts.popitem(last=False)
        if hit:
            entry = self.residents[pid]
        else:
            entry = incoming if incoming is not None else ObjectHistory(last=t)
            self.residents[pid] = entry
        entry.observe(t, hit)
        for window in self.requests.values():
            window.add(pid)
        for window in self.hits.values():
            window.add(hit)
        self.evictions.add(victim is not None)
        self.peak_ghosts = max(self.peak_ghosts, len(self.ghosts))


def object_sequence(x, length):
    """Right-aligned chronological values and observation indicators."""
    return x[..., 18:18+length].flip(-1), x[..., 18+GAPS:18+GAPS+length].flip(-1)
