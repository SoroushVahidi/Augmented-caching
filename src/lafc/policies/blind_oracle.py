"""
Blind Oracle caching policy.

The Blind Oracle is the prediction-following baseline from:

    Antoniadis, Coester, Eliáš, Polak, Simon.
    "Online Metric Algorithms with Untrusted Predictions."
    ICML 2020.

In the caching context, the Blind Oracle simply trusts the predictor
completely: on a miss it evicts the cached page whose *predicted* next
arrival is farthest in the future (Belady's algorithm run on predictions
instead of actual arrivals).

Setting: unweighted paging (unit fetch costs), cache of k pages.

Properties:
- **1-consistent**: optimal when predictions are perfect (η = 0).
- **Not robust**: competitive ratio can be Ω(n) under adversarial predictions.

Prediction interface (aligned with :class:`~lafc.types.Request`):
    ``request.predicted_next`` — the predicted next time index at which
    ``request.page_id`` will be requested again.  ``math.inf`` means
    "predicted to never be requested again."

This class is essentially equivalent to
:class:`~lafc.policies.advice_trusting.AdviceTrustingPolicy` but is
documented in the ICML 2020 context and is used as the trust-phase
sub-routine inside :class:`~lafc.policies.trust_and_doubt.TrustAndDoubtPolicy`.

Paper-to-code mapping
----------------------
Section 2 (caching setting), Algorithm FTP → this class.
The FTP (Follow-The-Prediction) abstraction from Section 3 specialises
to the Blind Oracle for caching: follow the predictor's recommended
eviction order.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class BlindOraclePolicy(BasePolicy):
    """Blind Oracle: evict the page with the largest predicted next-arrival time.

    Faithfully follows the predictor's advice with no robustness mechanism.
    Consistent (optimal under perfect predictions); not robust.

    Parameters
    ----------
    None.  The policy is stateless beyond the cache and predicted_next dict.
    """

    name: str = "blind_oracle"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        # Stores the most recently received predicted_next for each page.
        # Initialised to math.inf: "assume page is never needed again"
        # until we receive a prediction for it.
        #
        # INTERPRETATION NOTE: For pages whose prediction has not yet been
        # received, math.inf means they are treated as first-to-evict
        # candidates.  This is consistent with the ICML 2020 paper's
        # "Follow-The-Prediction" framing: absent advice, assume far future.
        self._predicted_next: Dict[PageId, float] = {
            pid: math.inf for pid in pages
        }

    def _choose_eviction_candidate(self, exclude: PageId) -> PageId:
        """Return the cached page with the largest predicted next arrival.

        Excludes *exclude* (the page currently being fetched, which is not
        yet in cache but should not be accidentally evicted).

        Tie-breaking: alphabetical on page_id for determinism.
        """
        candidates = [
            q for q in self._cache.current_cache() if q != exclude
        ]
        if not candidates:
            raise RuntimeError(
                f"No eviction candidate (cache={self._cache.current_cache()}, "
                f"exclude={exclude})"
            )
        return max(candidates, key=lambda q: (self._predicted_next.get(q, math.inf), q))

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        # Update prediction for this page regardless of hit/miss.
        # Section 2 of ICML 2020: prediction τ_t is revealed at time t.
        self._predicted_next[pid] = request.predicted_next

        if self.in_cache(pid):
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)
        evicted: Optional[PageId] = None

        if self._cache.is_full():
            evicted = self._choose_eviction_candidate(exclude=pid)
            self._evict(evicted)

        self._add(pid)

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)
