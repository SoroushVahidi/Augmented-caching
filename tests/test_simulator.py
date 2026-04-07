"""
Tests for request trace loading and CacheState.
"""

from __future__ import annotations

import math
import pytest

from lafc.simulator.cache_state import CacheState
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import Page


# ---------------------------------------------------------------------------
# build_requests_from_lists
# ---------------------------------------------------------------------------


def test_actual_next_filled_basic():
    """actual_next should point to the next occurrence of the same page."""
    requests, _ = build_requests_from_lists(
        ["A", "B", "A", "B", "A"],
        {"A": 1.0, "B": 2.0},
    )
    assert requests[0].actual_next == 2  # A at t=0 → next A at t=2
    assert requests[1].actual_next == 3  # B at t=1 → next B at t=3
    assert requests[2].actual_next == 4  # A at t=2 → next A at t=4
    assert requests[3].actual_next == math.inf  # B at t=3 → no more B
    assert requests[4].actual_next == math.inf  # A at t=4 → no more A


def test_actual_next_single_page():
    requests, _ = build_requests_from_lists(["X", "X", "X"], {"X": 3.0})
    assert requests[0].actual_next == 1
    assert requests[1].actual_next == 2
    assert requests[2].actual_next == math.inf


def test_predictions_set_when_provided():
    preds = [5.0, 6.0, 7.0]
    requests, _ = build_requests_from_lists(
        ["A", "B", "C"], {"A": 1.0, "B": 1.0, "C": 1.0}, predictions=preds
    )
    for i, req in enumerate(requests):
        assert req.predicted_next == preds[i]


def test_no_predictions_defaults_to_inf():
    requests, _ = build_requests_from_lists(["A", "B"], {"A": 1.0, "B": 1.0})
    for req in requests:
        assert math.isinf(req.predicted_next)


def test_missing_weight_raises():
    with pytest.raises(KeyError):
        build_requests_from_lists(["A", "B"], {"A": 1.0})  # B has no weight


def test_zero_weight_raises():
    with pytest.raises(ValueError):
        build_requests_from_lists(["A"], {"A": 0.0})


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        build_requests_from_lists(["A"], {"A": -1.0})


def test_predictions_length_mismatch_raises():
    with pytest.raises(ValueError):
        build_requests_from_lists(["A", "B"], {"A": 1.0, "B": 1.0}, predictions=[1.0])


def test_pages_dict_contains_all_pages():
    _, pages = build_requests_from_lists(["A", "B", "A"], {"A": 1.0, "B": 2.0})
    assert set(pages.keys()) == {"A", "B"}
    assert pages["A"].weight == 1.0
    assert pages["B"].weight == 2.0


def test_request_t_index():
    requests, _ = build_requests_from_lists(["A", "B", "C"], {"A": 1.0, "B": 1.0, "C": 1.0})
    for i, req in enumerate(requests):
        assert req.t == i


# ---------------------------------------------------------------------------
# load_trace
# ---------------------------------------------------------------------------


def test_load_trace_example_json(tmp_path):
    """load_trace should round-trip correctly with a minimal JSON file."""
    import json
    trace = {
        "requests": ["P", "Q", "P"],
        "weights":  {"P": 2.0, "Q": 3.0},
        "predictions": [2, 9999, 9999],
    }
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))

    requests, pages = load_trace(str(p))
    assert len(requests) == 3
    assert requests[0].predicted_next == 2.0
    assert requests[0].actual_next == 2.0
    assert pages["P"].weight == 2.0


def test_load_trace_no_predictions(tmp_path):
    import json
    trace = {"requests": ["A", "B", "A"], "weights": {"A": 1.0, "B": 1.0}}
    p = tmp_path / "trace.json"
    p.write_text(json.dumps(trace))
    requests, _ = load_trace(str(p))
    assert all(math.isinf(req.predicted_next) for req in requests)


# ---------------------------------------------------------------------------
# CacheState
# ---------------------------------------------------------------------------


def _make_pages(*page_ids, weight=1.0):
    return {pid: Page(page_id=pid, weight=weight) for pid in page_ids}


def test_cache_state_basic():
    pages = _make_pages("A", "B", "C")
    cs = CacheState(capacity=2, pages=pages)
    assert cs.size() == 0
    assert not cs.is_full()

    cs.add("A")
    assert cs.in_cache("A")
    assert cs.size() == 1

    cs.add("B")
    assert cs.is_full()

    with pytest.raises(ValueError):
        cs.add("C")  # cache full


def test_cache_state_evict():
    pages = _make_pages("A", "B")
    cs = CacheState(capacity=2, pages=pages)
    cs.add("A")
    cs.add("B")
    cs.evict("A")
    assert not cs.in_cache("A")
    assert cs.size() == 1


def test_cache_state_evict_unknown_raises():
    pages = _make_pages("A")
    cs = CacheState(capacity=2, pages=pages)
    with pytest.raises(KeyError):
        cs.evict("A")  # not in cache


def test_cache_state_current_cache():
    pages = _make_pages("A", "B", "C")
    cs = CacheState(capacity=3, pages=pages)
    cs.add("A")
    cs.add("C")
    snap = cs.current_cache()
    assert snap == frozenset({"A", "C"})


def test_cache_state_unknown_page_raises():
    pages = _make_pages("A")
    cs = CacheState(capacity=2, pages=pages)
    with pytest.raises(KeyError):
        cs.add("Z")  # unknown page
