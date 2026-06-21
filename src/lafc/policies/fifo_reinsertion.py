"""
FIFO-Reinsertion eviction algorithm (unweighted paging), a.k.a. CLOCK /
Second-Chance.

Reference
---------
Corbato, F. J. "A Paging Experiment with the Multics System." In Honor of
Philip M. Morse, MIT Press, 1969, pp. 217-228 (the original CLOCK /
second-chance algorithm).

Zhang, Yang, Yue, Vigfusson, Rashmi. "SIEVE is Simpler than LRU: an
Efficient Turn-Key Eviction Algorithm for Web Caches." NSDI '24, Section
3.1, p. 1232: names "FIFO-Reinsertion" as the standard term for this
CLOCK/Second-Chance family and contrasts it directly with SIEVE ("those
algorithms keep the hand fixed at the tail and move retained (survived)
objects to the head on a second-chance pass").

Yang, Zhang, Qiu, Yue, Rashmi. "FIFO Queues are All You Need for Cache
Eviction." SOSP '23. Frames FIFO-Reinsertion within the "lazy promotion /
passive demotion" taxonomy: only resident, re-accessed objects are
promoted (lazy promotion), and reinsertion mixes survivors back in with
newly-inserted objects in plain FIFO order (passive demotion) — the direct
contrast to SIEVE's "lazy promotion / quick demotion".

Verification
------------
See ``reports/kbs_halp_fifo_source_verification.md`` Section 2 for the
verbatim source quotes establishing this is the standard, unambiguous
reading of "FIFO-Reinsertion" intended by Reviewer #3 (who names it in the
same sentence as SIEVE).

============================================================
ALGORITHM, ADAPTED TO THIS REPO'S BasePolicy INTERFACE
============================================================

Identical FIFO-queue-plus-visited-bit state to ``sieve.py``. The only
behavioral difference from SIEVE is what happens to a *retained* (visited)
object encountered during an eviction scan: FIFO-Reinsertion physically
moves it to the head of the queue (mixing it with newly-inserted objects),
whereas SIEVE leaves it in place and advances a separate hand pointer
instead. Because reinsertion always restores the scanned object to the
head, the tail of the queue is always the next eviction candidate — no
persistent hand pointer is needed across calls (unlike SIEVE).

  1. Hit: x.visited <- 1. No reordering.
  2. Miss, cache full (eviction scan): look at the object o currently at
     the tail (oldest / least-recently-inserted-or-reinserted).
       while o.visited = 1:
         o.visited <- 0
         move o to the head of the queue (reinsertion / "second chance")
         o <- new tail
       Evict o (o.visited = 0).
  3. Insertion: new object goes to the head with visited <- 0.
"""

from __future__ import annotations

from typing import Dict, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class FIFOReinsertionPolicy(BasePolicy):
    """FIFO-Reinsertion / CLOCK / Second-Chance caching policy (unweighted).

    Maintained state
    -----------------
      _order : dict[PageId, None]
          Resident pages in insertion order (oldest-first as the *first*
          key / tail, newest-or-most-recently-reinserted as the *last* key
          / head), used as an ordered set for O(1) membership, O(1)
          append-at-head, and O(1) pop-from-tail.
      _visited : dict[PageId, bool]
          Visited bit per resident page.
    """

    name: str = "fifo_reinsertion"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: Dict[PageId, None] = {}
        self._visited: Dict[PageId, bool] = {}

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id

        if self.in_cache(pid):
            # Cache hit: set visited bit only. Do not move the object.
            self._visited[pid] = True
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)

        # Cache miss.
        cost = self._pages[pid].weight
        self._record_miss(cost)

        evicted: Optional[PageId] = None
        if self._cache.is_full():
            evicted = self._evict_one()
            del self._order[evicted]
            del self._visited[evicted]
            self._evict(evicted)

        # Insert x at the head of the queue; x.visited <- 0.
        self._add(pid)
        self._order[pid] = None
        self._visited[pid] = False

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)

    # ------------------------------------------------------------------
    # Eviction scan: reinsert visited survivors at the head, evict the
    # first unvisited object found at the tail.
    # ------------------------------------------------------------------

    def _evict_one(self) -> PageId:
        while True:
            tail_pid = next(iter(self._order))
            if self._visited[tail_pid]:
                self._visited[tail_pid] = False
                del self._order[tail_pid]
                self._order[tail_pid] = None  # reinsert at head (second chance)
            else:
                return tail_pid
