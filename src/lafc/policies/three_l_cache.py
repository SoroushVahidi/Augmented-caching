"""3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for Caches.

Reference
---------
Zhou, Niu, Xiong, Fang, Wang. "3L-Cache: Low Overhead and Precise
Learning-based Eviction Policy for Caches." FAST 2025.
Official implementation: https://github.com/optiq-lab/3L-Cache (GPL-3.0),
commit ``134cd159b635cdab75419a4281bed1a330fef31f`` (pinned 2026-08-06).
"3L" = Low overhead + Lowest object miss ratio + Lowest byte miss ratio
(three "L"s) -- not a "three-layer" architecture.

See ``docs/three_l_cache_method_spec.md`` for the full paper-and-code-grounded
method specification with every design decision classified.

Paper-faithful core
--------------------
- Bidirectional sampling: eviction candidates are drawn both from newly
  admitted objects ("sampling from the head", ``_quick_demotion``) and from
  long-resident objects via a *persistent* scan pointer walking the LRU
  order once per lap ("sampling from the tail", ``_tail_scan_round``).
- Batched, heap-based eviction: candidates are scored once per resampling
  round (not once per miss) and pushed onto a shared min-heap; a run of
  evictions is drawn from that heap before the next resampling round.
  Stale heap entries (objects re-requested or already evicted since being
  scored) are detected via a side ``pred_map`` and silently skipped.
- Delayed-label training with a *dynamic* window-exit label (a running
  empirical maximum wait time, frozen at each retrain), not LRB's fixed
  ``2*window`` constant.
- Auto-tuning of five parameters (``h_sw, f, x, Q``, and the closed-form
  ``n``) from accumulated eviction-outcome statistics, exactly as specified
  in the paper's Table 2 -- can be disabled (``auto_tune=False``) to use the
  paper's own documented "default value" ablation variant instead.
- Documented cold-start fallback: plain LRU before any model is trained
  (paper Section 4.2.1, footnote 3).

Unit-size specialization (required adaptation -- Strategy B)
--------------------------------------------------------------
This repository's manuscript evaluation is standard unweighted paging.
Every ``ObjectMeta.size`` here is held at a constant 1.0. ``objective``
defaults to ``"byte_miss_ratio"`` (the official code's own default) because,
under unit size, that branch is the mechanistically complete one (it keeps
predictions gathered at different times on one heap comparable via an
elapsed-time offset); ``object_miss_ratio`` in the reference omits that
offset. See docs/three_l_cache_method_spec.md for the full disclosure.

Structural adaptation forced by this repository's simulator contract
----------------------------------------------------------------------
Same as LRB: ``BasePolicy``/``CacheState`` require eviction strictly before
insertion, so the just-fetched page can never be sampled as its own
eviction candidate here, unlike the official C++ (insert-then-evict).
"""

from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from lafc.policies.base import BasePolicy
from lafc.three_l_cache_features import (
    N_FEATURE,
    ObjectMeta,
    compute_three_l_cache_feature_row,
    label_from_future_interval,
    score_to_reuse_time,
)
from lafc.types import CacheEvent, Page, PageId, Request


class _LRUList:
    """Explicit doubly-linked list over page ids (LRU order, head = LRU end,
    tail = MRU end). Mirrors the official code's index-based circular
    ``CacheUpdateQueue`` with page ids as node identity instead of array
    indices -- a data-structure simplification (dict vs. index-recycled
    array), not a behavioral change; the official code's head-rotation
    optimization for ``re_request(head)`` is algebraically equivalent to a
    plain remove+append on a linear list (both yield the same final ring
    topology), verified by inspection during the port.
    """

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


@dataclass(frozen=True)
class ThreeLCacheConfig:
    """3L-Cache hyperparameters.

    GBM hyperparameters and ``sample_subsampling_prob`` are exact-from-code
    defaults (official ``TLCache.h``/``TLCache.cpp``). ``batch_size``'s
    official default (65,536) is, like LRB's, CDN-scale and never fires
    within this repository's 50,000-request/trace protocol; it is
    validation-tunable here, not a fixed paper constant -- see
    ``scripts/experiments/run_three_l_cache_comparison.py`` and
    docs/three_l_cache_method_spec.md. ``h_sw, f, x, Q`` are *initial*
    values for auto-tuned state (paper Table 2); set ``auto_tune=False`` to
    freeze them at these initial values instead (the paper's own "default
    value" ablation variant, Section 5.4).
    """

    batch_size: int = 4096
    num_iterations: int = 16
    num_leaves: int = 32
    learning_rate: float = 0.1
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    objective: str = "byte_miss_ratio"
    sample_subsampling_prob: float = 0.25
    eviction_rate: int = 2  # fixed internal constant in the reference, not auto-tuned
    initial_hsw: int = 2
    initial_f: int = 1
    initial_x: int = 1
    initial_q: int = 2
    max_hsw: int = 6
    auto_tune: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.eviction_rate <= 0:
            raise ValueError(f"eviction_rate must be >= 1, got {self.eviction_rate}")
        if self.initial_hsw < 2:
            raise ValueError(f"initial_hsw must be >= 2, got {self.initial_hsw}")
        if not (0.0 <= self.sample_subsampling_prob <= 1.0):
            raise ValueError("sample_subsampling_prob must be in [0, 1]")
        if self.objective not in ("byte_miss_ratio", "object_miss_ratio"):
            raise ValueError(
                f"objective must be 'byte_miss_ratio' or 'object_miss_ratio', got {self.objective!r}"
            )


class ThreeLCachePolicy(BasePolicy):
    """3L-Cache (Zhou et al., FAST 2025) -- see module docstring."""

    name: str = "three_l_cache"

    def __init__(self, config: Optional[ThreeLCacheConfig] = None):
        self._config = config or ThreeLCacheConfig()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        from lafc.three_l_cache_model import DEFAULT_TRAINING_PARAMS, ThreeLCacheModel

        self._model_cls = ThreeLCacheModel
        cfg = self._config
        self._rng = random.Random(cfg.seed)

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

        self._model = None  # type: Optional["ThreeLCacheModel"]

        self._in_cache_meta: Dict[PageId, ObjectMeta] = {}
        # _ghost_meta maps key -> the *current* ghost ObjectMeta for that key.
        # _ghost_order is a FIFO of (key, meta) slots, one per demotion event,
        # mirroring the official code's deque-of-Meta-*copies* (not a dict):
        # if a key is demoted, promoted back, then demoted again before its
        # first FIFO slot is consumed, that first slot must be recognized as
        # stale (tombstoned) rather than matched against the second meta.
        # Object identity (`is`) reproduces the official per-slot Meta-copy
        # independence exactly.
        self._ghost_meta: Dict[PageId, ObjectMeta] = {}
        self._ghost_order: List[Tuple[PageId, ObjectMeta]] = []
        self._lru = _LRUList()
        self._is_sampling = False

        self._pending_rows: List[List[float]] = []
        self._pending_labels: List[float] = []

        # Bidirectional-sampling / batched-eviction state.
        self._new_obj_keys: List[PageId] = []
        self._new_obj_size: float = 0.0
        self._scan_pointer: Optional[PageId] = None
        self._scan_length: int = 0
        self._initial_queue_length: int = 0
        self._pred_heap: List[Tuple[float, PageId]] = []
        self._pred_map: Dict[PageId, float] = {}
        self._evict_nums: int = 0
        self._spointer_timestamp: int = 0
        self._origin_seq: int = 0
        self._max_eviction_boundary: List[float] = [0.0, 0.0]

        # Auto-tuned parameters (paper Table 2 / TLCache.cpp).
        self._hsw = cfg.initial_hsw
        self._f = cfg.initial_f
        self._x = cfg.initial_x
        self._q = cfg.initial_q
        self._eviction_distribution = [0, 0, 0, 0]
        self._object_distribution_n_eviction = [0] * 16
        self._is_window_full = False
        self._n_req = 0
        self._n_hit = 0
        self._n_window_hit = 0

        self._n_retrain = 0
        self._n_cold_start_evictions = 0
        self._n_model_ranked_evictions = 0
        self._n_stale_heap_pops = 0
        self._n_resample_rounds = 0

    # ------------------------------------------------------------------
    # Main algorithm step
    # ------------------------------------------------------------------

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        t = int(request.t)

        was_hit = pid in self._in_cache_meta
        was_ghost = (not was_hit) and (pid in self._ghost_meta)

        if self._is_window_full:
            self._n_req += 1

        if was_hit or was_ghost:
            meta = self._in_cache_meta[pid] if was_hit else self._ghost_meta[pid]
            if self._is_window_full:
                if was_hit:
                    self._n_hit += 1
                self._n_window_hit += 1
            self._maybe_mature_pending(meta, t)
            meta.record_request(t)
            if was_hit:
                if self._scan_pointer == pid:
                    self._scan_pointer = self._lru.next_of(pid)
                self._pred_map.pop(pid, None)
                self._lru.move_to_tail(pid)

        if self._is_sampling:
            self._sample_for_training(t)

        self._erase_out_cache(t)

        if was_hit:
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0, diagnostics={"mode": "hit"})

        self._record_miss(1.0)
        evicted: Optional[PageId] = None
        step_diag: Dict[str, object] = {"mode": "direct_admit"}
        if self._cache.is_full():
            evicted, step_diag = self._choose_victim(t)
            self._evict_physical(evicted)
            self._evict(evicted)

        self._admit(pid, t)
        self._add(pid)

        return CacheEvent(
            t=request.t, page_id=pid, hit=False, cost=1.0, evicted=evicted, diagnostics=step_diag,
        )

    # ------------------------------------------------------------------
    # Ghost-window bookkeeping ("erase_out_cache" -- official TLCacheCache)
    # ------------------------------------------------------------------

    def _erase_out_cache(self, t: int) -> None:
        max_out_cache_size = len(self._in_cache_meta) * (self._hsw - 1) + 2
        if len(self._ghost_order) < max_out_cache_size:
            return
        if not self._is_window_full:
            self._is_window_full = True
        key, slot_meta = self._ghost_order.pop(0)
        if self._ghost_meta.get(key) is not slot_meta:
            return  # tombstoned: this key was promoted (and possibly re-demoted) since this slot was created
        meta = self._ghost_meta.pop(key)
        age = t - meta.past_timestamp
        future_interval = self._max_eviction_boundary[0] + age
        if age > self._max_eviction_boundary[1]:
            self._max_eviction_boundary[1] = float(age)
        self._maybe_mature_pending(meta, t, forced_future_distance=future_interval)

    # ------------------------------------------------------------------
    # Unlabeled-sample generation ("sample" -- official TLCacheCache::sample)
    # ------------------------------------------------------------------

    def _sample_for_training(self, t: int) -> None:
        pool_size = len(self._in_cache_meta) + len(self._ghost_meta)
        if pool_size == 0:
            return
        idx = self._rng.randrange(pool_size)
        if idx < len(self._in_cache_meta):
            meta = list(self._in_cache_meta.values())[idx]
        else:
            meta = list(self._ghost_meta.values())[idx - len(self._in_cache_meta)]
        if meta.sample_time == 0:
            meta.sample_time = t

    # ------------------------------------------------------------------
    # Delayed-label maturation
    # ------------------------------------------------------------------

    def _maybe_mature_pending(
        self, meta: ObjectMeta, t: int, forced_future_distance: Optional[float] = None
    ) -> None:
        if meta.sample_time == 0:
            return
        # Both the re-request path and the window-exit path apply the same
        # probabilistic gate in the official code (TLCache.cpp:93, 139):
        # a pending sample only actually becomes a training row 25% of the
        # time, or always before any model has been trained.
        take = (self._model is None) or (self._rng.random() < self._config.sample_subsampling_prob)
        if take:
            future_distance = (
                forced_future_distance
                if forced_future_distance is not None
                else float(t - meta.sample_time)
            )
            row = compute_three_l_cache_feature_row(meta, meta.sample_time)
            self._pending_rows.append(row)
            self._pending_labels.append(label_from_future_interval(future_distance))
        meta.sample_time = 0
        if len(self._pending_labels) >= self._config.batch_size and self._evict_nums <= 0:
            self._train(t)

    def _train(self, t: int) -> None:
        self._n_retrain += 1
        model = self._model_cls()
        model.train(
            self._pending_rows,
            self._pending_labels,
            params=self._training_params,
            n_features=N_FEATURE,
        )
        self._model = model
        self._pending_rows = []
        self._pending_labels = []

        self._pred_map.clear()
        self._pred_heap = []
        self._max_eviction_boundary[0] = self._max_eviction_boundary[1]
        self._origin_seq = t

        if self._config.auto_tune and self._n_req > 1_000_000 and self._is_window_full:
            if self._n_hit > 0 and (self._hsw - 1) > 0:
                if (self._n_window_hit - self._n_hit) / (self._n_hit * (self._hsw - 1)) > 0.01:
                    denom = self._n_window_hit - self._n_hit
                    if denom > 0 and (self._hsw - 1) < (self._n_req - self._n_hit) / denom:
                        self._hsw += 1
                        self._is_window_full = False
                    self._hsw = min(self._hsw, self._config.max_hsw)
            self._n_hit = 0
            self._n_window_hit = 0
            self._n_req = 0

    # ------------------------------------------------------------------
    # Bidirectional sampling + batched eviction
    # ------------------------------------------------------------------

    def _quick_demotion(self) -> List[PageId]:
        """"Sampling from the head": new/recently-admitted objects."""
        sampled: List[PageId] = []
        threshold = len(self._in_cache_meta) * self._q / 10.0
        cap_sample = int(self._sample_rate() * 1.5)
        i = 0
        while (
            self._new_obj_size > threshold
            and len(sampled) < cap_sample
            and i < len(self._new_obj_keys)
        ):
            key = self._new_obj_keys[i]
            if key in self._in_cache_meta:
                self._new_obj_size -= self._in_cache_meta[key].size
                sampled.append(key)
            i += 1
        del self._new_obj_keys[:i]
        if not self._new_obj_keys:
            self._new_obj_size = 0.0
        return sampled

    def _sample_rate(self) -> int:
        L = self._initial_queue_length
        formula = (L * 0.01 + self._config.eviction_rate) if L > 2 else 2
        return int(min(1024, formula))

    def _tail_scan_round(self, already_sampled: int) -> List[PageId]:
        """"Sampling from the tail": long-resident/unpopular objects via a
        persistent scan pointer that continues across calls."""
        if self._scan_pointer is None:
            # First-ever call: start scanning from the current LRU head (the
            # oldest resident object), matching "sampling from the tail"'s
            # documented intent. The official code instead defaults
            # `samplepointer = 0` (a raw array index), which by the time of
            # first use points at whichever object an unrelated swap-removal
            # eviction happens to have relocated into slot 0 -- an
            # implementation artifact of the array-based storage with no
            # equivalent meaning in this key-based port; the LRU head is the
            # well-defined, clearly-intended starting position.
            self._scan_pointer = self._lru.head
        sampled: List[PageId] = []
        sample_rate = self._sample_rate()
        idx_row = 0
        while idx_row < sample_rate and (already_sampled + len(sampled)) < self._initial_queue_length:
            if self._scan_pointer is None:
                break
            meta = self._in_cache_meta[self._scan_pointer]
            freq = meta.freq - 1
            if self._eviction_distribution[3] == 0 and self._scan_length > self._initial_queue_length * self._x / 100:
                self._eviction_distribution[2] = self._eviction_distribution[0]
                self._eviction_distribution[3] = self._eviction_distribution[1]
                self._eviction_distribution[1] = 0
                self._eviction_distribution[0] = 0
            if freq < self._f or self._scan_length <= self._initial_queue_length * self._x / 100 + self._config.eviction_rate:
                sampled.append(self._scan_pointer)
                idx_row += 1
            self._scan_length += 1
            next_ptr = self._lru.next_of(self._scan_pointer)
            if self._scan_length >= self._initial_queue_length:
                self._complete_lap()
                idx_row = 0
                self._scan_pointer = self._lru.head
                self._scan_length = 0
                if self._config.objective == "object_miss_ratio":
                    continue
                self._auto_tune_f_x_q()
                continue
            self._scan_pointer = next_ptr
        if sampled:
            self._spointer_timestamp = self._in_cache_meta[sampled[-1]].past_timestamp
            self._eviction_distribution[1] += sample_rate
        return sampled

    def _complete_lap(self) -> None:
        self._initial_queue_length = len(self._in_cache_meta)
        self._pred_map.clear()
        self._pred_heap = []

    def _auto_tune_f_x_q(self) -> None:
        if not self._config.auto_tune:
            self._eviction_distribution = [0, 0, 0, 0]
            self._object_distribution_n_eviction = [0] * 16
            return
        total = sum(self._object_distribution_n_eviction)
        if total > 0:
            cumulative = 0
            for i in range(16):
                cumulative += self._object_distribution_n_eviction[i]
                if cumulative >= 0.99 * total:
                    if i == 0:
                        self._f = 1
                    else:
                        bucket_count = self._object_distribution_n_eviction[i]
                        frac = (0.99 * total + bucket_count - cumulative) / bucket_count
                        self._f = int((2 ** (i - 1)) + math.ceil(((2**i) - (2 ** (i - 1))) * frac))
                    break
        ed = self._eviction_distribution
        if ed[2] * ed[1] > ed[0] * ed[3]:
            self._x += 1
        elif self._x > 1:
            self._x -= 1
        if (ed[0] + ed[2]) > len(self._new_obj_keys):
            self._q += 1
        elif self._q > 1:
            self._q = max(1, self._q // 2)
        self._eviction_distribution = [0, 0, 0, 0]
        self._object_distribution_n_eviction = [0] * 16

    def _rank(self, t: int) -> List[PageId]:
        if self._initial_queue_length == 0:
            self._initial_queue_length = len(self._in_cache_meta)
        sampled = self._quick_demotion()
        if self._new_obj_size < len(self._in_cache_meta) * self._q / 10.0:
            sampled.extend(self._tail_scan_round(len(sampled)))
        self._n_resample_rounds += 1
        self._prediction(sampled, t)
        return sampled

    def _prediction(self, sampled: List[PageId], current: int) -> None:
        if not sampled:
            return
        rows = [
            compute_three_l_cache_feature_row(self._in_cache_meta[k], current) for k in sampled
        ]
        scores = self._model.predict(rows)
        for key, log_score in zip(sampled, scores):
            size = self._in_cache_meta[key].size
            if self._config.objective == "byte_miss_ratio":
                value = score_to_reuse_time(log_score) + (current - self._origin_seq)
            else:
                value = size * score_to_reuse_time(log_score)
            heapq.heappush(self._pred_heap, (-value, key))
            self._pred_map[key] = value

    def _choose_victim(self, t: int) -> Tuple[PageId, Dict[str, object]]:
        lru_head = self._lru.head
        if self._model is None:
            self._n_cold_start_evictions += 1
            return lru_head, {"mode": "cold_start_lru"}

        if self._evict_nums <= 0 or not self._pred_heap:
            sampled = self._rank(t)
            self._evict_nums = max(1, len(sampled) // self._config.eviction_rate) if sampled else 1

        while self._pred_heap:
            neg_value, key = heapq.heappop(self._pred_heap)
            value = -neg_value
            if self._pred_map.get(key) == value:
                meta = self._in_cache_meta[key]
                bucket = min(15, int(math.log2(max(1, meta.freq))))
                self._object_distribution_n_eviction[bucket] += 1
                if meta.past_timestamp <= self._spointer_timestamp:
                    self._eviction_distribution[0] += 1
                self._n_model_ranked_evictions += 1
                return key, {"mode": "model_ranked"}
            self._n_stale_heap_pops += 1
        # Heap exhausted without a valid entry (should not normally happen
        # once trained and the cache is non-empty): fall back to LRU head,
        # matching the reference's own eventual behavior of always having a
        # concrete victim to evict.
        self._n_cold_start_evictions += 1
        return lru_head, {"mode": "heap_exhausted_lru_fallback"}

    def _evict_physical(self, victim: PageId) -> None:
        self._is_sampling = True
        self._evict_nums -= 1
        meta = self._in_cache_meta.pop(victim)
        if self._scan_pointer == victim:
            self._scan_pointer = self._lru.next_of(victim)
        self._pred_map.pop(victim, None)
        self._lru.remove(victim)
        self._ghost_meta[victim] = meta
        self._ghost_order.append((victim, meta))

    # ------------------------------------------------------------------
    # Admission
    # ------------------------------------------------------------------

    def _admit(self, pid: PageId, t: int) -> None:
        if pid in self._ghost_meta:
            meta = self._ghost_meta.pop(pid)  # tombstones the FIFO entry
            meta.size = 1.0
        else:
            meta = ObjectMeta(key=pid, past_timestamp=t, size=1.0)
        self._in_cache_meta[pid] = meta
        self._lru.append(pid)
        if self._model is not None:
            self._new_obj_size += meta.size
            self._new_obj_keys.append(pid)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics_summary(self) -> Dict[str, float]:
        cfg = self._config
        return {
            "batch_size": float(cfg.batch_size),
            "num_iterations": float(cfg.num_iterations),
            "num_leaves": float(cfg.num_leaves),
            "learning_rate": float(cfg.learning_rate),
            "seed": float(cfg.seed),
            "objective_is_byte_miss_ratio": float(cfg.objective == "byte_miss_ratio"),
            "auto_tune": float(cfg.auto_tune),
            "n_retrain": float(self._n_retrain),
            "model_trained": float(self._model is not None),
            "n_in_cache_meta": float(len(self._in_cache_meta)),
            "n_ghost_meta": float(len(self._ghost_meta)),
            "n_pending_rows": float(len(self._pending_rows)),
            "n_cold_start_evictions": float(self._n_cold_start_evictions),
            "n_model_ranked_evictions": float(self._n_model_ranked_evictions),
            "n_stale_heap_pops": float(self._n_stale_heap_pops),
            "n_resample_rounds": float(self._n_resample_rounds),
            "hsw": float(self._hsw),
            "f": float(self._f),
            "x": float(self._x),
            "q": float(self._q),
        }
