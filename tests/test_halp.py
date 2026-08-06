"""Tests for the HALP (Song et al., NSDI 2023) policy.

Verifies feature row extraction, Bradley-Terry pairwise preference model training,
cold-start LRU evictions, model-ranked evictions, deterministic tie-breaking,
and future-information leakage isolation.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from lafc.halp_features import ObjectMeta, compute_halp_feature_row
from lafc.halp_model import HALPModel
from lafc.policies.halp import HALPConfig, HALPPolicy
from lafc.runner.run_policy import run_policy, POLICY_REGISTRY
from lafc.types import Request


def _req(t: int, page_id: str, actual_next: float = math.inf) -> Request:
    return Request(t=int(t), page_id=page_id, actual_next=actual_next)


def test_halp_features_and_meta():
    meta = ObjectMeta(key="A", past_timestamp=10)
    meta.record_request(15)  # delta 5
    meta.record_request(22)  # delta 7
    
    row = compute_halp_feature_row(meta, sample_timestamp=30)
    assert len(row) == 5
    assert row[0] == pytest.approx(30 - 22)  # age
    assert row[1] == pytest.approx(3.0)     # frequency
    assert row[2] == pytest.approx(7.0)     # Delta 1
    assert row[3] == pytest.approx(5.0)     # Delta 2
    assert math.isnan(row[4])               # Delta 3 unobserved


def test_halp_pairwise_model_training():
    model = HALPModel(hidden_units=4, seed=42)

    # Train data: A is always preferred over B
    # Feature format: [Age, Freq, D1, D2, D3]
    X_pref = np.array([
        [1.0, 10.0, 2.0, 2.0, 2.0],
        [2.0, 15.0, 1.0, 1.0, 1.0],
        [1.5, 12.0, 1.5, 1.5, 1.5],
        [1.2, 11.0, 1.8, 1.8, 1.8],
    ])
    X_non_pref = np.array([
        [10.0, 1.0, 20.0, 20.0, 20.0],
        [12.0, 2.0, 15.0, 15.0, 15.0],
        [11.0, 1.5, 18.0, 18.0, 18.0],
        [10.5, 1.2, 19.0, 19.0, 19.0],
    ])

    model.fit(X_pref, X_non_pref)

    # A should get a significantly higher predicted reward than B
    score_pref = model.predict_rewards(X_pref)
    score_non_pref = model.predict_rewards(X_non_pref)
    assert np.all(score_pref > score_non_pref)


def test_halp_model_cold_start_returns_zero_rewards():
    model = HALPModel(hidden_units=4, seed=42)
    X = np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0, 2.0]])
    rewards = model.predict_rewards(X)
    assert np.all(rewards == 0.0)


def test_halp_policy_cold_start_to_training_transition():
    # Setup policy with capacity 3, training_trigger at t=5
    cfg = HALPConfig(training_trigger=5, seed=42)
    policy = HALPPolicy(cfg)
    
    # Define a sequence of requests.
    # At t=1, 2, 3: admit A, B, C. Cache is full.
    # At t=4: miss on D. Evicts oldest in LRU (A). Cold-start mode.
    # At t=5: miss on E. Trigger training, evict based on learned scores.
    requests = [
        _req(1, "A", actual_next=6),
        _req(2, "B", actual_next=8),
        _req(3, "C", actual_next=7),
        _req(4, "D", actual_next=9), # evicts A (LRU oldest)
        _req(5, "E", actual_next=10), # t=5 triggers training. Evicts based on pairwise scores
    ]
    
    pages = {p: None for p in ["A", "B", "C", "D", "E"]}
    policy.reset(capacity=3, pages=pages)
    
    events = [policy.on_request(r) for r in requests]
    
    # Verify t=4 (cold start): A is evicted
    assert events[3].hit is False
    assert events[3].evicted == "A"
    assert events[3].diagnostics["mode"] == "cold_start_lru"
    
    # Verify t=5 (learned mode): Model is trained and eviction uses predicted reward
    assert events[4].hit is False
    assert events[4].diagnostics["mode"] == "model_ranked"
    assert policy._model_trained is True


def test_no_future_leakage_from_next_arrival():
    # Run HALP once on a trace.
    cfg = HALPConfig(training_trigger=5, seed=42)
    policy = HALPPolicy(cfg)
    
    requests_1 = [
        _req(1, "A", actual_next=6),
        _req(2, "B", actual_next=8),
        _req(3, "C", actual_next=7),
        _req(4, "D", actual_next=9),
        _req(5, "E", actual_next=10),
    ]
    pages = {p: None for p in ["A", "B", "C", "D", "E"]}
    
    policy.reset(capacity=3, pages=pages)
    events_1 = [policy.on_request(r) for r in requests_1]
    
    # Now, run again with completely mutated actual_next times during the evaluation phase (t >= 5)
    # The evaluation phase must be independent of actual_next, producing identical eviction targets!
    requests_2 = [
        _req(1, "A", actual_next=6),
        _req(2, "B", actual_next=8),
        _req(3, "C", actual_next=7),
        _req(4, "D", actual_next=9),
        _req(5, "E", actual_next=9999), # Completely changed actual_next at t=5!
    ]
    
    policy.reset(capacity=3, pages=pages)
    events_2 = [policy.on_request(r) for r in requests_2]
    
    # Assert exact behavior matching
    assert [e.evicted for e in events_1] == [e.evicted for e in events_2]
    assert [e.diagnostics["mode"] for e in events_1] == [e.diagnostics["mode"] for e in events_2]


def test_halp_registered_in_policy_registry():
    assert "halp" in POLICY_REGISTRY
    assert isinstance(POLICY_REGISTRY["halp"], HALPPolicy)


def test_halp_run_policy_integration_populates_diagnostics():
    # Regression test: run_policy's HALP branch must guard against
    # extra_diagnostics defaulting to None (every other policy branch does
    # `result.extra_diagnostics = result.extra_diagnostics or {}` first).
    requests = [
        _req(t, page_id, actual_next=t + 5)
        for t, page_id in enumerate(["A", "B", "C", "D", "A", "B", "E", "A"], start=1)
    ]
    pages = {p: None for p in ["A", "B", "C", "D", "E"]}

    cfg = HALPConfig(training_trigger=3, seed=0)
    result = run_policy(HALPPolicy(cfg), requests, pages, capacity=2)

    assert result.extra_diagnostics is not None
    assert "halp" in result.extra_diagnostics
    summary = result.extra_diagnostics["halp"]["summary"]
    assert "n_cold_start_evictions" in summary
    assert "n_model_ranked_evictions" in summary
    assert "model_trained" in summary


def test_halp_capacity_one():
    cfg = HALPConfig(training_trigger=100, seed=42)
    policy = HALPPolicy(cfg)
    pages = {p: None for p in ["A", "B"]}
    policy.reset(capacity=1, pages=pages)

    r1 = policy.on_request(_req(1, "A"))
    r2 = policy.on_request(_req(2, "B"))
    r3 = policy.on_request(_req(3, "A"))

    assert r1.hit is False and r1.evicted is None
    assert r2.hit is False and r2.evicted == "A"
    assert r3.hit is False and r3.evicted == "B"


def test_halp_repeated_requests_are_hits():
    cfg = HALPConfig(training_trigger=100, seed=42)
    policy = HALPPolicy(cfg)
    pages = {p: None for p in ["A", "B"]}
    policy.reset(capacity=2, pages=pages)

    r1 = policy.on_request(_req(1, "A"))
    r2 = policy.on_request(_req(2, "A"))

    assert r1.hit is False
    assert r2.hit is True


def test_halp_never_reaccessed_object_evictable():
    # An object with actual_next == inf (never seen again) must still be a
    # valid eviction candidate and must not crash feature/label construction.
    cfg = HALPConfig(training_trigger=5, seed=42)
    policy = HALPPolicy(cfg)
    pages = {p: None for p in ["A", "B", "C"]}
    policy.reset(capacity=2, pages=pages)

    requests = [
        _req(1, "A", actual_next=math.inf),  # never reaccessed
        _req(2, "B", actual_next=5),
        _req(3, "C", actual_next=6),  # cold-start: evicts LRU head (A)
    ]
    events = [policy.on_request(r) for r in requests]
    assert events[2].evicted == "A"
    assert events[2].diagnostics["mode"] == "cold_start_lru"


class _FakeTiedModel:
    def predict_rewards(self, X):
        return np.zeros(len(X))


def test_halp_deterministic_tie_break_prefers_larger_page_id():
    cfg = HALPConfig(training_trigger=5, seed=42)
    policy = HALPPolicy(cfg)
    pages = {p: None for p in ["A", "B", "C", "D"]}
    policy.reset(capacity=3, pages=pages)

    for r in [
        _req(1, "A", actual_next=6),
        _req(2, "B", actual_next=8),
        _req(3, "C", actual_next=7),
    ]:
        policy.on_request(r)

    # Force learned mode with a model that ties every candidate's reward.
    policy._model = _FakeTiedModel()
    policy._model_trained = True

    event = policy.on_request(_req(4, "D", actual_next=9))

    # A, B, C all tie at reward 0.0; deterministic tie-break selects the
    # lexicographically largest page_id among the shortlist.
    assert event.diagnostics["mode"] == "model_ranked"
    assert event.evicted == max(["A", "B", "C"])
