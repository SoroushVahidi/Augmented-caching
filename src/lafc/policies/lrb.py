"""LRB: Learning Relaxed Belady for content distribution network caching.

Reference
---------
Song, Berger, Li, Lloyd. "Learning Relaxed Belady for Content Distribution
Network Caching." NSDI 2020.
Official implementation: https://github.com/sunnyszy/lrb (BSD-2-Clause),
commit ``9e8b4423383c01c4528deb447f152f0437a37c3a`` (pinned 2026-08-06).

See ``docs/lrb_method_spec.md`` for the full paper-and-code-grounded method
specification with every design decision classified as exact-from-paper,
exact-from-code, a required adaptation, or an optional deviation.

Paper-faithful core
--------------------
- Online candidate-level eviction: on a full-cache miss, samples
  ``sample_rate`` (default 64) cached objects uniformly at random and evicts
  the one with the largest model-predicted ``log1p(time-to-next-request)``.
- Delayed-label training: a randomly sampled object's training label matures
  only once it is (a) re-requested, (b) force-evicted for exceeding the
  sliding ``memory_window``, or (c) its post-eviction "ghost" metadata times
  out of the window -- never from ground-truth future information.
- GBDT (LightGBM) regressor, retrained from scratch whenever the matured
  training buffer reaches ``batch_size``.
- Documented cold-start fallback: before any model has been trained, or
  whenever the LRU-tail object's age exceeds ``memory_window``, evicts via
  plain LRU with no model call (this is the official design, not an ad hoc
  addition -- see paper Section 4.1, "our implementation uses LRU as a
  fallback until sufficient training data is available").

Unit-size specialization (required adaptation -- Strategy A)
--------------------------------------------------------------
This repository's manuscript evaluation is standard unweighted paging
(unit miss cost, capacity measured in object slots). The official LRB
targets variable-sized, byte-capacity CDN caches. Every ``ObjectMeta.size``
here is therefore held at a constant ``1.0``; the size feature and the
``objective="object_miss_ratio"`` size-multiplication step are both kept
structurally (not dropped), but are numerically inert under this
specialization. This is "LRB under unit-size specialization," not a
reproduction of the paper's byte-cache CDN experiments.

Structural adaptation forced by this repository's simulator contract
----------------------------------------------------------------------
``BasePolicy``/``CacheState`` require eviction to happen strictly *before*
inserting the newly-fetched page (``CacheState.add`` raises if the cache is
already full). The official C++ implementation instead inserts first and
evicts after, which means the just-fetched page is technically eligible to
be sampled (and, in principle, immediately evicted) as one of its own
``sample_rate`` candidates. Evict-before-add structurally excludes the
just-fetched page from candidacy in this port -- consistent with this
repository's existing convention (see ``offline_belady.py``'s explicit
``exclude`` parameter) but a real, documented behavioral difference from the
literal reference.
"""

from __future__ import annotations

import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lafc.lrb_features import (
    BASE_EDC_WINDOW,
    DEFAULT_MAX_N_PAST_TIMESTAMPS,
    N_EDC_FEATURE,
    ObjectMeta,
    compute_lrb_feature_row,
    edc_windows,
    hash_edc_table,
    label_from_future_interval,
    n_feature_count,
)
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


@dataclass(frozen=True)
class LRBConfig:
    """LRB hyperparameters.

    Defaults for ``sample_rate``, ``num_iterations``, ``num_leaves``,
    ``learning_rate``, ``feature_fraction``, ``bagging_fraction``, and
    ``bagging_freq`` are exact-from-code (official ``lrb.h``). ``memory_window``
    and ``batch_size`` defaults are this repository's own illustrative
    adaptation: the official code's defaults (67,108,864 and 131,072
    respectively) are CDN-scale constants tuned for multi-million-request
    traces and never fire within this repository's 50,000-request/trace,
    32-128-slot evaluation protocol. They are validation-tunable, not fixed
    paper constants -- see ``scripts/experiments/run_lrb_external_baseline.py``
    and ``docs/lrb_method_spec.md``.
    """

    sample_rate: int = 64
    memory_window: int = 4096
    batch_size: int = 2048
    max_n_past_timestamps: int = DEFAULT_MAX_N_PAST_TIMESTAMPS
    num_iterations: int = 32
    num_leaves: int = 32
    learning_rate: float = 0.1
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    objective: str = "object_miss_ratio"
    min_data_in_leaf: Optional[int] = None
    min_data_in_bin: Optional[int] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be >= 1, got {self.sample_rate}")
        if self.memory_window <= 0:
            raise ValueError(f"memory_window must be >= 1, got {self.memory_window}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_n_past_timestamps < 1:
            raise ValueError(
                f"max_n_past_timestamps must be >= 1, got {self.max_n_past_timestamps}"
            )
        if self.objective not in ("object_miss_ratio", "byte_miss_ratio"):
            raise ValueError(
                f"objective must be 'object_miss_ratio' or 'byte_miss_ratio', got {self.objective!r}"
            )


class LRBPolicy(BasePolicy):
    """Learning Relaxed Belady (Song et al., NSDI 2020) -- see module docstring."""

    name: str = "lrb"

    def __init__(self, config: Optional[LRBConfig] = None):
        self._config = config or LRBConfig()

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # lightgbm is an optional dependency of this repository (pip install
        # 'lafc[lrb]'); deferring the import here keeps every other policy
        # and the CLI's POLICY_REGISTRY importable without it installed.
        from lafc.lrb_model import DEFAULT_TRAINING_PARAMS, LRBModel

        self._model_cls = LRBModel
        cfg = self._config

        self._rng = random.Random(cfg.seed)
        self._windows = edc_windows(BASE_EDC_WINDOW, N_EDC_FEATURE)
        self._hash_edc = hash_edc_table(cfg.memory_window, BASE_EDC_WINDOW)
        self._max_n_past_distances = cfg.max_n_past_timestamps - 1
        self._n_features = n_feature_count(cfg.max_n_past_timestamps, 0, N_EDC_FEATURE)

        self._training_params = dict(DEFAULT_TRAINING_PARAMS)
        self._training_params.update(
            {
                "num_iterations": cfg.num_iterations,
                "num_leaves": cfg.num_leaves,
                "learning_rate": cfg.learning_rate,
                "feature_fraction": cfg.feature_fraction,
                "bagging_fraction": cfg.bagging_fraction,
                "bagging_freq": cfg.bagging_freq,
                "seed": cfg.seed,
                "bagging_seed": cfg.seed,
                "feature_fraction_seed": cfg.seed,
            }
        )
        if cfg.min_data_in_leaf is not None:
            self._training_params["min_data_in_leaf"] = cfg.min_data_in_leaf
        if cfg.min_data_in_bin is not None:
            self._training_params["min_data_in_bin"] = cfg.min_data_in_bin

        self._model = None  # type: Optional["LRBModel"]

        self._in_cache_meta: Dict[PageId, ObjectMeta] = {}
        self._ghost_meta: Dict[PageId, ObjectMeta] = {}
        self._ghost_expiry: Dict[int, PageId] = {}
        self._lru_queue: "OrderedDict[PageId, None]" = OrderedDict()

        self._pending_rows: List[List[float]] = []
        self._pending_labels: List[float] = []

        self._is_sampling = False
        self._n_retrain = 0
        self._n_force_eviction = 0
        self._n_cold_start_evictions = 0
        self._n_age_forced_evictions = 0
        self._n_model_ranked_evictions = 0
        self._n_candidates_sampled_total = 0

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        t = int(request.t)

        self._forget(t)

        was_hit = pid in self._in_cache_meta
        was_ghost = (not was_hit) and (pid in self._ghost_meta)

        if was_hit or was_ghost:
            meta = self._in_cache_meta[pid] if was_hit else self._ghost_meta[pid]
            self._mature_pending(meta, t)
            old_ghost_slot = (
                meta.past_timestamp % self._config.memory_window if was_ghost else None
            )
            meta.record_request(
                t,
                max_n_past_distances=self._max_n_past_distances,
                windows=self._windows,
                hash_edc=self._hash_edc,
            )
            if was_hit:
                self._lru_queue.move_to_end(pid)
            elif old_ghost_slot is not None and self._ghost_expiry.get(old_ghost_slot) == pid:
                # The official code re-inserts a fresh ghost-expiry slot here
                # before admit() immediately erases it again; skipped as a
                # provably-equivalent simplification (forget() cannot fire
                # between these two steps of the same request).
                del self._ghost_expiry[old_ghost_slot]

        if self._is_sampling:
            self._sample_for_training(t)

        if was_hit:
            self._record_hit()
            return CacheEvent(
                t=request.t, page_id=pid, hit=True, cost=0.0, diagnostics={"mode": "hit"}
            )

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        step_diag: Dict[str, object] = {"mode": "direct_admit", "candidate_count": 0}
        if self._cache.is_full():
            self._is_sampling = True
            evicted, step_diag = self._choose_and_evict(t)
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

    # ------------------------------------------------------------------
    # Memory-window bookkeeping ("forget" -- official LRBCache::forget)
    # ------------------------------------------------------------------

    def _forget(self, t: int) -> None:
        slot = t % self._config.memory_window
        pid = self._ghost_expiry.pop(slot, None)
        if pid is None:
            return
        meta = self._ghost_meta.pop(pid, None)
        if meta is None:
            return
        self._mature_pending(meta, t, forced_future_distance=float(2 * self._config.memory_window))

    # ------------------------------------------------------------------
    # Unlabeled-sample generation ("sample" -- official LRBCache::sample)
    # ------------------------------------------------------------------

    def _sample_for_training(self, t: int) -> None:
        pool = list(self._in_cache_meta.keys()) + list(self._ghost_meta.keys())
        if not pool:
            return
        key = self._rng.choice(pool)
        meta = self._in_cache_meta.get(key)
        if meta is None:
            meta = self._ghost_meta.get(key)
        if meta is not None:
            meta.sample_times.append(t)

    # ------------------------------------------------------------------
    # Delayed-label maturation
    # ------------------------------------------------------------------

    def _mature_pending(
        self,
        meta: ObjectMeta,
        current_seq: int,
        forced_future_distance: Optional[float] = None,
    ) -> None:
        if not meta.sample_times:
            return
        for sample_time in meta.sample_times:
            future_distance = (
                forced_future_distance
                if forced_future_distance is not None
                else float(current_seq - sample_time)
            )
            row = compute_lrb_feature_row(
                meta,
                sample_time,
                memory_window=self._config.memory_window,
                max_n_past_timestamps=self._config.max_n_past_timestamps,
                windows=self._windows,
                hash_edc=self._hash_edc,
            )
            self._pending_rows.append(row)
            self._pending_labels.append(label_from_future_interval(future_distance))
        meta.sample_times = []
        if len(self._pending_labels) >= self._config.batch_size:
            self._train()

    def _train(self) -> None:
        self._n_retrain += 1
        model = self._model_cls()
        model.train(
            self._pending_rows,
            self._pending_labels,
            params=self._training_params,
            n_features=self._n_features,
        )
        self._model = model
        self._pending_rows = []
        self._pending_labels = []

    # ------------------------------------------------------------------
    # Eviction-candidate selection ("rank"/"evict" -- official LRBCache)
    # ------------------------------------------------------------------

    def _choose_and_evict(self, t: int) -> Tuple[PageId, Dict[str, object]]:
        lru_tail_pid = next(iter(self._lru_queue))
        lru_tail_meta = self._in_cache_meta[lru_tail_pid]
        lru_tail_age = t - lru_tail_meta.past_timestamp

        if self._model is None:
            victim = lru_tail_pid
            self._n_cold_start_evictions += 1
            step_diag: Dict[str, object] = {"mode": "cold_start_lru", "candidate_count": 1}
        elif lru_tail_age >= self._config.memory_window:
            victim = lru_tail_pid
            self._n_age_forced_evictions += 1
            step_diag = {"mode": "age_forced_lru", "candidate_count": 1}
        else:
            pool = list(self._in_cache_meta.keys())
            k = min(self._config.sample_rate, len(pool))
            sampled = self._rng.sample(pool, k=k)
            rows = [
                compute_lrb_feature_row(
                    self._in_cache_meta[key],
                    t,
                    memory_window=self._config.memory_window,
                    max_n_past_timestamps=self._config.max_n_past_timestamps,
                    windows=self._windows,
                    hash_edc=self._hash_edc,
                )
                for key in sampled
            ]
            scores = self._model.predict(rows)
            if self._config.objective == "object_miss_ratio":
                scores = [
                    s * self._in_cache_meta[key].size for s, key in zip(scores, sampled)
                ]
            # Evict the largest-scored candidate (farthest predicted reuse);
            # deterministic tie-break by smallest page_id.
            victim = min(zip(scores, sampled), key=lambda sp: (-sp[0], sp[1]))[1]
            self._n_model_ranked_evictions += 1
            self._n_candidates_sampled_total += k
            step_diag = {"mode": "model_ranked", "candidate_count": k}

        self._finalize_eviction(victim, t)
        return victim, step_diag

    def _finalize_eviction(self, victim: PageId, t: int) -> None:
        meta = self._in_cache_meta.pop(victim)
        self._lru_queue.pop(victim, None)
        age = t - meta.past_timestamp
        if age >= self._config.memory_window:
            forced_future_distance = float(age + self._config.memory_window)
            self._mature_pending(meta, t, forced_future_distance=forced_future_distance)
            self._n_force_eviction += 1
        else:
            self._ghost_meta[victim] = meta
            self._ghost_expiry[meta.past_timestamp % self._config.memory_window] = victim

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def _admit(self, pid: PageId, t: int) -> None:
        if pid in self._ghost_meta:
            meta = self._ghost_meta.pop(pid)
        elif pid in self._in_cache_meta:
            meta = self._in_cache_meta[pid]
        else:
            meta = ObjectMeta(key=pid, past_timestamp=t, size=1.0)
        self._in_cache_meta[pid] = meta
        self._lru_queue[pid] = None
        self._lru_queue.move_to_end(pid)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics_summary(self) -> Dict[str, float]:
        cfg = self._config
        return {
            "sample_rate": float(cfg.sample_rate),
            "memory_window": float(cfg.memory_window),
            "batch_size": float(cfg.batch_size),
            "max_n_past_timestamps": float(cfg.max_n_past_timestamps),
            "num_iterations": float(cfg.num_iterations),
            "num_leaves": float(cfg.num_leaves),
            "learning_rate": float(cfg.learning_rate),
            "seed": float(cfg.seed),
            "objective_is_object_miss_ratio": float(cfg.objective == "object_miss_ratio"),
            "n_retrain": float(self._n_retrain),
            "model_trained": float(self._model is not None),
            "n_in_cache_meta": float(len(self._in_cache_meta)),
            "n_ghost_meta": float(len(self._ghost_meta)),
            "n_pending_rows": float(len(self._pending_rows)),
            "n_force_eviction": float(self._n_force_eviction),
            "n_cold_start_evictions": float(self._n_cold_start_evictions),
            "n_age_forced_evictions": float(self._n_age_forced_evictions),
            "n_model_ranked_evictions": float(self._n_model_ranked_evictions),
            "n_candidates_sampled_total": float(self._n_candidates_sampled_total),
        }
