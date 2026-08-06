"""HALP (Heuristic Aided Learned Preference Eviction Policy, Song et al.,
NSDI 2023) policy.

Independent reimplementation of the published algorithm (see
`docs/halp_method_spec.md` for the full, per-decision fidelity
classification), adapted to this repository's unit-size, offline-replay
paging setting. No official HALP code is public; this is not a port of any
released implementation.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from lafc.policies.base import BasePolicy
from lafc.halp_features import ObjectMeta, compute_halp_feature_row
from lafc.halp_model import HALPModel
from lafc.types import CacheEvent, Page, PageId, Request


class _LRUList:
    """Explicit doubly-linked list over page ids for LRU order."""

    def __init__(self) -> None:
        self._next: Dict[PageId, PageId] = {}
        self._prev: Dict[PageId, PageId] = {}
        self.head: Optional[PageId] = None
        self.tail: Optional[PageId] = None

    def __len__(self) -> int:
        return len(self._next)

    def __contains__(self, key: PageId) -> bool:
        return key in self._next

    def append(self, key: PageId) -> None:
        if self.head is None:
            self.head = self.tail = key
            self._next[key] = key
            self._prev[key] = key
        else:
            self._next[self.tail] = key
            self._prev[key] = self.tail
            self._next[key] = self.head
            self._prev[self.head] = key
            self.tail = key

    def remove(self, key: PageId) -> None:
        if len(self._next) == 1:
            self.head = self.tail = None
            del self._next[key]
            del self._prev[key]
            return
        nxt, prv = self._next[key], self._prev[key]
        if key == self.head:
            self.head = nxt
        if key == self.tail:
            self.tail = prv
        self._next[prv] = nxt
        self._prev[nxt] = prv
        del self._next[key]
        del self._prev[key]

    def move_to_tail(self, key: PageId) -> None:
        if key == self.tail:
            return
        self.remove(key)
        self.append(key)

    def next_of(self, key: PageId) -> Optional[PageId]:
        return self._next.get(key)


class HALPConfig:
    """Configuration class for the HALP policy."""

    def __init__(
        self,
        training_trigger: int = 10000,
        hidden_units: int = 8,
        alpha: float = 1e-4,
        lr: float = 0.05,
        n_epochs: int = 300,
        seed: int = 0,
    ):
        self.training_trigger = training_trigger
        self.hidden_units = hidden_units
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.seed = seed


class HALPPolicy(BasePolicy):
    """Heuristic Aided Learned Preference (HALP) Policy.

    Augments LRU by choosing an eviction victim from a shortlist of the 8 oldest
    pages using a trained pairwise Bradley-Terry preference model.
    """

    name: str = "halp"

    def __init__(self, config: Optional[HALPConfig] = None):
        self._config = config or HALPConfig()

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._model = HALPModel(
            hidden_units=self._config.hidden_units,
            alpha=self._config.alpha,
            lr=self._config.lr,
            n_epochs=self._config.n_epochs,
            seed=self._config.seed,
        )
        self._model_trained = False
        self._actual_next: Dict[PageId, float] = {}
        self._in_cache_meta: Dict[PageId, ObjectMeta] = {}
        self._lru = _LRUList()
        self._recorded_events: List[Dict[str, object]] = []

        # Diagnostics counters
        self._n_cold_start_evictions = 0
        self._n_model_ranked_evictions = 0

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        t = int(request.t)

        self._actual_next[pid] = request.actual_next

        was_hit = self.in_cache(pid)
        if was_hit:
            self._lru.move_to_tail(pid)
            meta = self._in_cache_meta[pid]
            meta.record_request(t)
            self._record_hit()
            return CacheEvent(
                t=request.t,
                page_id=pid,
                hit=True,
                cost=0.0,
                diagnostics={"mode": "hit"},
            )

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        step_diag: Dict[str, object] = {"mode": "direct_admit"}

        if self._cache.is_full():
            if t >= self._config.training_trigger and not self._model_trained:
                self._train()

            candidates = self._sample_candidates(k=8)

            if not self._model_trained:
                evicted = self._lru.head
                candidate_features = []
                for cand in candidates:
                    meta_cand = self._in_cache_meta[cand]
                    feat = compute_halp_feature_row(meta_cand, t)
                    candidate_features.append(feat)
                self._recorded_events.append(
                    {
                        "timestamp": t,
                        "candidates": list(candidates),
                        "features": candidate_features,
                    }
                )
                self._n_cold_start_evictions += 1
                step_diag = {"mode": "cold_start_lru"}
            else:
                candidate_features = []
                for cand in candidates:
                    meta_cand = self._in_cache_meta[cand]
                    feat = compute_halp_feature_row(meta_cand, t)
                    candidate_features.append(feat)

                rewards = self._model.predict_rewards(np.array(candidate_features))

                best_idx = 0
                lowest_reward = rewards[0]
                for idx in range(1, len(candidates)):
                    r = rewards[idx]
                    if r < lowest_reward:
                        lowest_reward = r
                        best_idx = idx
                    elif r == lowest_reward:
                        if candidates[idx] > candidates[best_idx]:
                            best_idx = idx

                evicted = candidates[best_idx]
                self._n_model_ranked_evictions += 1
                step_diag = {"mode": "model_ranked"}

            self._evict_physical(evicted)
            self._evict(evicted)

        self._admit(pid, t)
        self._add(pid)

        return CacheEvent(
            t=request.t,
            page_id=pid,
            hit=False,
            cost=1.0,
            evicted=evicted,
            diagnostics=step_diag,
        )

    def _sample_candidates(self, k: int) -> List[PageId]:
        candidates: List[PageId] = []
        curr = self._lru.head
        while curr is not None and len(candidates) < k:
            candidates.append(curr)
            curr = self._lru.next_of(curr)
            if curr == self._lru.head:
                break
        return candidates

    def _evict_physical(self, victim: PageId) -> None:
        self._in_cache_meta.pop(victim)
        self._lru.remove(victim)

    def _admit(self, pid: PageId, t: int) -> None:
        meta = ObjectMeta(key=pid, past_timestamp=t)
        self._in_cache_meta[pid] = meta
        self._lru.append(pid)

    def _train(self) -> None:
        X_pref = []
        X_non_pref = []

        for event in self._recorded_events:
            candidates = event["candidates"]
            features = event["features"]

            next_arrivals = [
                self._actual_next.get(cand, math.inf) for cand in candidates
            ]

            num_cand = len(candidates)
            for i in range(num_cand):
                for j in range(i + 1, num_cand):
                    na_i = next_arrivals[i]
                    na_j = next_arrivals[j]

                    if na_i < na_j:
                        X_pref.append(features[i])
                        X_non_pref.append(features[j])
                    elif na_j < na_i:
                        X_pref.append(features[j])
                        X_non_pref.append(features[i])

        if len(X_pref) > 0:
            X_pref = np.array(X_pref)
            X_non_pref = np.array(X_non_pref)
            self._model.fit(X_pref, X_non_pref)

        self._model_trained = True

    def diagnostics_summary(self) -> Dict[str, float]:
        return {
            "n_cold_start_evictions": float(self._n_cold_start_evictions),
            "n_model_ranked_evictions": float(self._n_model_ranked_evictions),
            "model_trained": float(1.0 if self._model_trained else 0.0),
        }
