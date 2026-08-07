"""Tests for the reviewer-fairness harness's core primitive: lossless
windowed-metric reconstruction from a full-stream `run_policy()` result.

Uses a small, fully hand-traceable LRU trace where every hit/miss in the
scored suffix can be verified by inspection, not just by re-running code.
"""

from __future__ import annotations

import pytest

from lafc.experiments.reviewer_fairness_common import (
    COMMON_SCHEMA_FIELDS,
    score_window,
    validate_common_row,
)
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import run_policy
from lafc.types import Page, Request


def _lru_trace():
    # capacity=2. Requests: A B A C A D A E A F (10 requests, 0-indexed).
    # LRU trace: A(miss,admit) B(miss,admit,eA/B) A(hit) C(miss,evict B)
    #            A(hit) D(miss,evict C) A(hit) E(miss,evict D) A(hit)
    #            F(miss,evict E)
    # hit/miss sequence (index:page:hit) --
    #  0:A miss  1:B miss  2:A hit  3:C miss  4:A hit
    #  5:D miss  6:A hit  7:E miss  8:A hit  9:F miss
    page_ids = ["A", "B", "A", "C", "A", "D", "A", "E", "A", "F"]
    pages = {p: Page(page_id=p, weight=1.0) for p in set(page_ids)}
    requests = [Request(t=i, page_id=p) for i, p in enumerate(page_ids)]
    return requests, pages


def test_score_window_matches_hand_derived_suffix():
    requests, pages = _lru_trace()
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)

    # History = indices [0, 4): A miss, B miss, A hit, C miss.
    # Scored suffix = indices [4, 10): A hit, D miss, A hit, E miss, A hit, F miss.
    #   -> 3 hits, 3 misses, 6 scored requests.
    w = score_window(result.events, score_start=4, score_end=10)
    assert w.history_requests == 4
    assert w.scored_requests == 6
    assert w.hits == 3
    assert w.misses == 3
    assert w.miss_ratio == pytest.approx(0.5)

    # Full-stream (deployment) count for the same run, independently: 6
    # misses total (A,B,C,D,E,F each miss once = 6 misses out of 10).
    assert result.total_misses == 6


def test_score_window_full_range_matches_full_stream_result():
    requests, pages = _lru_trace()
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    w = score_window(result.events, score_start=0, score_end=len(result.events))
    assert w.misses == result.total_misses
    assert w.hits == result.total_hits
    assert w.scored_requests == len(requests)
    assert w.history_requests == 0


def test_score_window_rejects_out_of_range_window():
    requests, pages = _lru_trace()
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)
    with pytest.raises(ValueError):
        score_window(result.events, score_start=-1, score_end=5)
    with pytest.raises(ValueError):
        score_window(result.events, score_start=5, score_end=len(result.events) + 1)
    with pytest.raises(ValueError):
        score_window(result.events, score_start=8, score_end=3)  # start > end


def test_score_window_history_does_not_affect_state_after_it():
    # Confirms the state-preserving property the whole protocol depends
    # on: the cache at the score boundary reflects genuinely having
    # processed the history (not a reset-to-empty cache at score_start).
    # capacity=2, history = [A, B] (both admitted, cache now {A,B} at
    # score_start=2), scored suffix = [A] (must be a hit -- A was never
    # evicted).
    page_ids = ["A", "B", "A"]
    pages = {p: Page(page_id=p, weight=1.0) for p in set(page_ids)}
    requests = [Request(t=i, page_id=p) for i, p in enumerate(page_ids)]
    result = run_policy(LRUPolicy(), requests, pages, capacity=2)

    w = score_window(result.events, score_start=2, score_end=3)
    assert w.scored_requests == 1
    assert w.hits == 1
    assert w.misses == 0


def test_common_schema_validation_rejects_missing_field():
    row = {f: 0 for f in COMMON_SCHEMA_FIELDS}
    validate_common_row(row)  # should not raise

    del row["trace_sha256"]
    with pytest.raises(ValueError, match="trace_sha256"):
        validate_common_row(row)
