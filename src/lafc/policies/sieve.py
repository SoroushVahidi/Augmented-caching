"""
SIEVE eviction algorithm (unweighted paging).

Reference
---------
Zhang, Yang, Yue, Vigfusson, Rashmi.
"SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for
Web Caches." NSDI '24, pp. 1229-1246.

Verification
------------
See ``reports/kbs_sieve_source_verification.md`` for the verbatim Algorithm 1
pseudocode (quoted from the official USENIX PDF) and an independent
cross-check against the official reference C implementation
(``libCacheSim/cache/eviction/Sieve.c``). This module implements that
algorithm directly, with no semantic deviation.

============================================================
PAPER-TO-CODE IMPLEMENTATION NOTE
============================================================

Algorithm 1 (paper notation) maintains one doubly-linked queue ``T`` and one
"hand" pointer ``p``. Each resident object carries one ``visited`` bit.

  1: if x is in T then                          (Cache Hit)
  2:   x.visited <- 1
  3: else
  4:   if |T| = C then                          (Cache Miss, Cache Full)
  5:     o <- p
  6:     if o is NULL then
  7:       o <- tail of T
  8:     while o.visited = 1 do
  9:       o.visited <- 0
 10:       o <- o.prev
 11:       if o is NULL then
 12:         o <- tail of T
 13:     p <- o.prev
 14:     Discard o in T                          (Eviction)
 15:   Insert x in the head of T                 (Insertion)
 16:   x.visited <- 0

Mapping to this module:
- ``T`` is represented by ``self._order``, a ``dict`` whose insertion order
  gives newest-to-oldest order from the *last* key to the *first* key (i.e.
  Python dict insertion order is oldest-first; the paper's "head" is the
  most-recently-inserted end, which is the *last* key in ``self._order``).
- ``self._visited`` is the per-key visited bit (``True``/``False``), absent
  keys are treated as not resident.
- ``self._hand`` is the hand pointer ``p``: either ``None`` (paper's NULL,
  meaning "start scanning from the tail/oldest end") or a resident
  ``PageId`` (the object immediately preceding, toward the head, the most
  recently evicted object).
- A cache hit sets ``visited[pid] = True`` and does **nothing else** — no
  reordering, unlike LRU's ``move_to_end``. This is the one rule most likely
  to be implemented incorrectly if copy-pasting from an LRU-style policy.
- A full-cache miss scans from the hand (or the oldest/tail end if the hand
  is unset) walking toward the head, clearing ``visited`` bits as it passes,
  wrapping from past-the-head back to the tail if necessary, and evicting
  the first object found with ``visited = False``. The hand is then updated
  to that evicted object's predecessor (toward the head).
- A newly inserted object goes to the head (the end of ``self._order``) with
  ``visited`` initialized to ``False``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from lafc.policies.base import BasePolicy
from lafc.types import CacheEvent, Page, PageId, Request


class SievePolicy(BasePolicy):
    """SIEVE caching policy (unweighted), faithful to NSDI'24 Algorithm 1.

    Maintained state
    -----------------
      _order : dict[PageId, None]
          Resident pages in insertion order (oldest-first / tail-first as
          the *first* key, newest/head-most as the *last* key), used as an
          ordered set for O(1) membership + O(1) append-at-head.
      _visited : dict[PageId, bool]
          Visited bit per resident page.
      _hand : Optional[PageId]
          The SIEVE "hand" pointer ``p``. ``None`` means unset (paper's
          NULL) — the next full-cache-miss scan starts at the tail.
    """

    name: str = "sieve"

    def reset(self, capacity: int, pages: Dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._order: Dict[PageId, None] = {}
        self._visited: Dict[PageId, bool] = {}
        self._hand: Optional[PageId] = None

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

        # Insert x at the head of T; x.visited <- 0.
        self._add(pid)
        self._order[pid] = None
        self._visited[pid] = False

        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)

    # ------------------------------------------------------------------
    # Eviction scan (Algorithm 1, lines 5-14)
    # ------------------------------------------------------------------

    def _residents_oldest_to_newest(self) -> List[PageId]:
        # dict insertion order: first key = oldest (tail), last key = newest (head).
        return list(self._order.keys())

    def _evict_one(self) -> PageId:
        """Run the SIEVE hand scan and return the page to evict.

        Walks from the hand (or the tail if unset) toward the head,
        clearing visited bits along the way and wrapping past the head
        back to the tail, until it finds an object with ``visited = False``.
        Updates ``self._hand`` to that object's predecessor (toward the
        head) before returning.
        """
        residents = self._residents_oldest_to_newest()
        n = len(residents)
        index_of = {pid: i for i, pid in enumerate(residents)}

        # o <- p; if o is NULL then o <- tail of T.
        if self._hand is not None and self._hand in index_of:
            idx = index_of[self._hand]
        else:
            idx = 0  # tail = oldest = first key

        while self._visited[residents[idx]]:
            self._visited[residents[idx]] = False
            # o <- o.prev: move from tail-side toward head-side.
            idx += 1
            if idx >= n:
                idx = 0  # wrap past the head back to the tail

        victim = residents[idx]

        # p <- o.prev (the position toward the head from the victim).
        prev_idx = idx + 1
        self._hand = residents[prev_idx] if prev_idx < n else None

        return victim
