"""CACHEUS (Rodriguez et al., FAST 2021) external-baseline adapter.

This is an **official-source wrapper**, not a reimplementation: the
decision engine is the authors' own `Cacheus` class from
github.com/sylab/cacheus, executed unmodified via
`lafc.cacheus_official_loader`. This module only adapts the official
class's `(cache_size, window_size, **kwargs)` / `.request(oblock, ts) ->
(CacheOp, evicted_oblock)` interface to this repository's `BasePolicy`
interface and mirrors its decisions into this repository's own
`CacheState` for bookkeeping consistency with every other policy here.

See `docs/cacheus_method_spec.md` for the full, per-decision fidelity
classification and `docs/cacheus_provenance.md` for licensing/provenance.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from lafc.cacheus_official_loader import EXPECTED_COMMIT, load_official_classes
from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request

# The official Cacheus.__init__ unconditionally calls `np.random.seed(123)`
# -- a hardcoded constant, not a parameter. There is no seed argument
# anywhere in the official constructor or `.request()`. This repository
# does not add one (would be an algorithm-changing patch); this constant
# documents what the official source actually does, for provenance/
# diagnostics recording. See docs/cacheus_method_spec.md, "Expert victim
# proposal and selection" and docs/cacheus_provenance.md's RNG audit.
OFFICIAL_RNG_SEED = 123


class CacheusConfig:
    """Configuration for the CACHEUS adapter.

    Defaults match the official `Cacheus.__init__` defaults exactly (see
    docs/cacheus_method_spec.md, "Hyperparameters"): `initial_weight=0.5`,
    `history_size=cache_size // 2` (computed internally by the official
    class, not overridden here unless explicitly requested), and
    `learning_rate=sqrt(2*ln(2)/cache_size)` (also computed internally).

    `window_size` is a non-algorithmic, visualization-only parameter (see
    method spec): the official `run.py` harness sets it to 100 whenever
    cache sizes are given as absolute integers (as ours are), so 100 is
    used here as the matching default.
    """

    def __init__(
        self,
        window_size: int = 100,
        initial_weight: Optional[float] = None,
        history_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
    ):
        self.window_size = window_size
        self.initial_weight = initial_weight
        self.history_size = history_size
        self.learning_rate = learning_rate

    def official_kwargs(self) -> Dict[str, object]:
        kwargs: Dict[str, object] = {}
        if self.initial_weight is not None:
            kwargs["initial_weight"] = self.initial_weight
        if self.history_size is not None:
            kwargs["history_size"] = self.history_size
        if self.learning_rate is not None:
            kwargs["learning_rate"] = self.learning_rate
        return kwargs


class CacheusPolicy(BasePolicy):
    """Wraps the official `Cacheus` (SR-LRU + CR-LFU expert combination)."""

    name: str = "cacheus"

    def __init__(self, config: Optional[CacheusConfig] = None):
        self._config = config or CacheusConfig()

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        if capacity < 2:
            # Official Cacheus.__init__ sets history_size = cache_size // 2;
            # at capacity 1 this is 0, and the official addToHistory()
            # unconditionally evicts-from-history-before-inserting once
            # len(history) == history_size, which is true (0 == 0) on the
            # very first history write to an empty deque -- an upstream
            # AttributeError ('NoneType' has no attribute 'key'/'value'),
            # confirmed empirically against the pinned commit. Not patched
            # here (would be an algorithm-changing patch to third-party
            # code); surfaced explicitly instead of propagating a cryptic
            # third-party stack trace. See docs/cacheus_method_spec.md,
            # "Cold-start behavior" / known upstream limitations.
            raise ValueError(
                f"CacheusPolicy: capacity={capacity} is not supported. The "
                "official Cacheus source (sylab/cacheus, pinned commit "
                f"{EXPECTED_COMMIT}) crashes at capacity 1 "
                "because history_size = capacity // 2 == 0 triggers an "
                "AttributeError in its own addToHistory(); this is an "
                "upstream limitation, not an adapter bug. Minimum "
                "supported capacity is 2."
            )
        # Isolation, not seeding: the official Cacheus.__init__ itself
        # deterministically resets numpy's *global* RNG state to a
        # hardcoded seed (123) -- that already makes each freshly
        # constructed instance's own decisions independent of whatever ran
        # before it in this process (see OFFICIAL_RNG_SEED docstring
        # above). What is NOT handled upstream is the reverse direction:
        # protecting *other* code in the same process from being affected
        # by CACHEUS having consumed/reset the global numpy RNG stream.
        # Snapshotting here and restoring in restore_global_rng_state()
        # (called by run_policy() after the simulation loop) closes that
        # gap without touching any official algorithmic code -- the
        # official constructor's own np.random.seed(123) call and all of
        # Cacheus's own random draws during the run are completely
        # untouched and unaffected by this save/restore.
        self._numpy_state_before_cacheus = np.random.get_state()

        Cacheus, _LRU = load_official_classes()
        self._official = Cacheus(
            capacity, self._config.window_size, **self._config.official_kwargs()
        )
        self._n_expert_disagreements = 0
        self._n_history_hits_lru = 0
        self._n_history_hits_lfu = 0

    def restore_global_rng_state(self) -> None:
        """Restore numpy's global RNG state to what it was before this
        policy instance constructed the official Cacheus object.

        Purely an isolation measure for code outside this policy; has no
        effect on any decision CACHEUS itself already made (those are
        final, already recorded in `self._official`'s state and in the
        CacheEvents already returned). Safe to call multiple times or not
        at all; a no-op if reset() was never called.
        """
        state = getattr(self, "_numpy_state_before_cacheus", None)
        if state is not None:
            np.random.set_state(state)

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        was_hit = self.in_cache(pid)

        # Track official-side ghost-history hits (delayed feedback events)
        # before calling request(), purely for diagnostics -- the official
        # class's own __contains__ only reflects the physical cache, same
        # as this repository's self.in_cache().
        in_lru_hist = pid in self._official.lru_hist
        in_lfu_hist = pid in self._official.lfu_hist

        op, evicted = self._official.request(pid, request.t)
        # CacheOp.HIT == 0 (see algs/lib/cacheop.py); compared by name to
        # avoid importing the official enum type into this module.
        hit = op.name == "HIT"
        if hit != was_hit:
            raise RuntimeError(
                f"CacheusPolicy adapter desync at t={request.t}, page={pid}: "
                f"official Cacheus reported hit={hit} but this repository's "
                f"CacheState reported hit={was_hit}. This indicates a bug in "
                "the adapter's state mirroring, not in the official "
                "algorithm; see docs/cacheus_method_spec.md."
            )

        cost = 0.0
        if not hit:
            cost = self._pages[pid].weight
            self._record_miss(cost)
            if evicted is not None:
                self._evict(evicted)
                if in_lru_hist:
                    self._n_history_hits_lru += 1
                elif in_lfu_hist:
                    self._n_history_hits_lfu += 1
            self._add(pid)
        else:
            self._record_hit()

        return CacheEvent(
            t=request.t, page_id=pid, hit=hit, cost=cost, evicted=evicted,
            diagnostics={"mode": "hit" if hit else "official_cacheus"},
        )

    def diagnostics_summary(self) -> Dict[str, float]:
        w = getattr(self._official, "W", None)
        lr = getattr(self._official, "learning_rate", None)
        return {
            "final_weight_srlru": float(w[0]) if w is not None else float("nan"),
            "final_weight_crlfu": float(w[1]) if w is not None else float("nan"),
            "final_learning_rate": float(lr.learning_rate) if lr is not None else float("nan"),
            "n_history_hits_lru": float(self._n_history_hits_lru),
            "n_history_hits_lfu": float(self._n_history_hits_lfu),
            "dem_count": float(getattr(self._official, "dem_count", float("nan"))),
            "nor_count": float(getattr(self._official, "nor_count", float("nan"))),
            "official_rng_seed": float(OFFICIAL_RNG_SEED),
        }
