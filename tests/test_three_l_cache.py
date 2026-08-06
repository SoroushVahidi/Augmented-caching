"""Tests for the 3L-Cache (Zhou et al., FAST 2025) policy.

Where a test asserts an exact numeric value, it is either (a) derived by
hand from the recurrences documented in ``src/lafc/three_l_cache_features.py``
and the official implementation (``TLCache.h``/``TLCache.cpp``, commit
``134cd159b635cdab75419a4281bed1a330fef31f``), computed independently in the
test, or (b) captured once from a real deterministic run and frozen as a
regression baseline (see ``test_frozen_regression_trace``).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from lafc.three_l_cache_features import (
    N_FEATURE,
    ObjectMeta,
    compute_three_l_cache_feature_row,
    label_from_future_interval,
    score_to_reuse_time,
)
from lafc.policies.three_l_cache import ThreeLCacheConfig, ThreeLCachePolicy
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.policies.lru import LRUPolicy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Request


def _req(t: int, page_id: str) -> Request:
    return Request(t=t, page_id=page_id, predicted_next=math.inf, actual_next=math.inf)


# ---------------------------------------------------------------------------
# Feature module: official-parity checks
# ---------------------------------------------------------------------------


def test_feature_row_layout_and_values():
    meta = ObjectMeta(key="A", past_timestamp=0)
    meta.record_request(3)  # distance 3
    meta.record_request(10)  # distance 7
    row = compute_three_l_cache_feature_row(meta, sample_timestamp=15)
    assert len(row) == N_FEATURE == 6
    assert row[0] == pytest.approx(15 - 10)  # age
    assert row[1] == pytest.approx(7)  # most recent delta
    assert row[2] == pytest.approx(3)
    assert math.isnan(row[3])  # only 2 deltas observed, 3rd slot unfilled
    assert row[4] == pytest.approx(1.0)  # size, constant under unit-size specialization
    assert row[5] == pytest.approx(3.0)  # frequency: 3 requests total (initial + 2 updates)


def test_frequency_caps_and_deltas_most_recent_first():
    meta = ObjectMeta(key="A", past_timestamp=0)
    t = 0
    for _ in range(5):
        t += 1
        meta.record_request(t)
    assert len(meta.past_distances) == 3  # capped at MAX_N_PAST_DISTANCES
    assert meta.past_distances == [1, 1, 1]
    assert meta.freq == 6


def test_label_and_score_inverse_transform():
    assert label_from_future_interval(0.0) == pytest.approx(math.log1p(0.0))
    assert label_from_future_interval(15.0) == pytest.approx(math.log1p(15.0))
    # score_to_reuse_time uses plain exp(), NOT expm1() -- a real, documented
    # asymmetry in the official code (log1p forward / exp backward).
    assert score_to_reuse_time(math.log1p(15.0)) == pytest.approx(math.exp(math.log1p(15.0)))
    assert score_to_reuse_time(math.log1p(15.0)) != pytest.approx(15.0)


def test_record_request_rejects_non_increasing_timestamp():
    meta = ObjectMeta(key="A", past_timestamp=5)
    with pytest.raises(ValueError):
        meta.record_request(5)
    with pytest.raises(ValueError):
        meta.record_request(4)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        ThreeLCacheConfig(batch_size=0)
    with pytest.raises(ValueError):
        ThreeLCacheConfig(eviction_rate=0)
    with pytest.raises(ValueError):
        ThreeLCacheConfig(initial_hsw=1)
    with pytest.raises(ValueError):
        ThreeLCacheConfig(sample_subsampling_prob=1.5)
    with pytest.raises(ValueError):
        ThreeLCacheConfig(objective="not_a_real_objective")


def test_capacity_zero_rejected():
    _, pages = build_requests_from_lists(["A", "B"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig())
    with pytest.raises(ValueError):
        policy.reset(0, pages)


# ---------------------------------------------------------------------------
# Core online behavior
# ---------------------------------------------------------------------------


def test_reset_and_empty_cache_state():
    _, pages = build_requests_from_lists(["A"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig())
    policy.reset(3, pages)
    assert policy.current_cache() == frozenset()
    assert policy._in_cache_meta == {}
    assert policy._ghost_meta == {}


def test_hit_and_miss_bookkeeping():
    requests, pages = build_requests_from_lists(["A", "A", "B", "A"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig())
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert [e.hit for e in events] == [False, True, False, True]
    assert policy.total_cost() == 2.0


def test_miss_with_available_space_no_eviction():
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig())
    policy.reset(3, pages)
    events = [policy.on_request(r) for r in requests]
    assert all(e.evicted is None for e in events)
    assert policy.current_cache() == frozenset({"A", "B", "C"})


def test_capacity_one_every_new_page_evicts_previous():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig())
    policy.reset(1, pages)
    events = [policy.on_request(r) for r in requests]
    assert [e.evicted for e in events] == [None, "A", "B", "C"]
    assert policy.current_cache() == frozenset({"D"})


def test_repeated_requests_single_page_capacity_one():
    requests, pages = build_requests_from_lists(["A"] * 10)
    policy = ThreeLCachePolicy(ThreeLCacheConfig())
    policy.reset(1, pages)
    events = [policy.on_request(r) for r in requests]
    assert sum(1 for e in events if not e.hit) == 1
    assert sum(1 for e in events if e.hit) == 9
    assert all(e.evicted is None for e in events)


def test_cold_start_lru_fallback_before_any_model_trained():
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=100000))
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert events[2].diagnostics["mode"] == "cold_start_lru"
    assert events[2].evicted == "A"  # LRU head: A requested first, never re-touched
    assert policy.diagnostics_summary()["model_trained"] == 0.0


def test_cache_lru_and_meta_stay_in_sync_with_base_cache_state():
    import random as pyrandom

    _rng = pyrandom.Random(0)
    ids = [_rng.choice("ABCDEFGH") for _ in range(300)]
    requests, pages = build_requests_from_lists(ids)
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=30, seed=1))
    policy.reset(4, pages)
    for req in requests:
        policy.on_request(req)
        assert (
            set(policy.current_cache())
            == set(policy._in_cache_meta.keys())
            == set(policy._lru._next.keys())
        )


# ---------------------------------------------------------------------------
# Bidirectional sampling / batched heap eviction
# ---------------------------------------------------------------------------


def test_tail_scan_pointer_initializes_on_first_use():
    """Regression for a bug found during implementation: the scan pointer
    started as None and was never seeded before first use, silently
    producing zero tail-scan candidates on the first resampling round."""
    requests, pages = build_requests_from_lists(["A", "B", "C"] * 20)
    # batch_size=1 would train LightGBM on a single row, which combined with
    # bagging_fraction=0.8 can round down to zero sampled rows internally
    # and raise -- an unrelated LightGBM edge case, not a policy defect.
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=5, sample_subsampling_prob=1.0, seed=0))
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert policy._model is not None
    first_model_ranked = next(e for e in events if e.diagnostics.get("mode") == "model_ranked")
    assert first_model_ranked is not None
    assert policy._scan_pointer is not None


def test_reproducible_candidate_selection_same_seed():
    import random as pyrandom

    _rng = pyrandom.Random(3)
    ids = [_rng.choice("ABCDEF") for _ in range(400)]
    requests, pages = build_requests_from_lists(ids)

    def run(seed: int):
        policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=20, seed=seed))
        policy.reset(3, pages)
        return [(e.hit, e.evicted) for e in (policy.on_request(r) for r in requests)]

    assert run(11) == run(11)
    assert run(11) != run(12)


def test_heap_staleness_skips_re_requested_candidate():
    """Directly exercises the pred_map staleness check: a candidate scored
    into the heap, then re-requested (hit) before being evicted, must be
    skipped when popped rather than incorrectly evicted."""
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=100000, seed=0))
    policy.reset(3, pages)
    for r in requests[:3]:
        policy.on_request(r)  # A, B, C admitted, no eviction yet (capacity=3)

    # Force a trained model and manually populate the heap with predictions
    # for A and B (A "farther" than B), then re-request A (a hit) so its
    # pred_map entry is invalidated before eviction is attempted.
    class _StubModel:
        def predict(self, rows):
            return [10.0, 5.0]  # A -> larger predicted value than B

    policy._model = _StubModel()
    policy._prediction(["A", "B"], 3)
    assert "A" in policy._pred_map and "B" in policy._pred_map

    policy.on_request(_req(3, "A"))  # hit: erases A's pred_map entry (stale heap tuple remains)

    ev = policy.on_request(requests[3])  # D forces eviction among the stale heap + B
    assert policy.diagnostics_summary()["n_stale_heap_pops"] >= 1.0
    assert ev.evicted == "B"  # A's stale entry was skipped; B is the only valid candidate left


def test_deterministic_tie_break_smallest_key_wins():
    """heapq orders (-value, key) tuples; for equal predicted values the
    smallest page_id is popped first -- a deterministic, well-defined rule
    since the reference's own tie-break (implicit in heap comparison order)
    is not otherwise specified for ties.
    """
    requests, pages = build_requests_from_lists(["Z", "M", "A", "D"])
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=100000, seed=0))
    policy.reset(3, pages)
    for r in requests[:3]:
        policy.on_request(r)

    class _ConstantModel:
        def predict(self, rows):
            return [0.0] * len(rows)

    policy._model = _ConstantModel()
    policy._prediction(["Z", "M", "A"], 3)
    ev = policy.on_request(requests[3])
    assert ev.evicted == "A"  # smallest key among tied predicted values


# ---------------------------------------------------------------------------
# Delayed-label maturation / no future leakage
# ---------------------------------------------------------------------------


def test_delayed_label_on_rerequest_exact_value_and_snapshot_timing():
    requests, pages = build_requests_from_lists(["A"])
    policy = ThreeLCachePolicy(
        ThreeLCacheConfig(batch_size=1_000_000, sample_subsampling_prob=1.0, seed=0)
    )
    policy.reset(3, pages)
    policy.on_request(requests[0])  # t=0: A admitted, past_timestamp=0

    policy._in_cache_meta["A"].sample_time = 2  # simulate an unlabeled sample at t=2

    policy.on_request(_req(7, "A"))  # re-request at t=7: label = log1p(7-2)

    assert policy._pending_labels == [pytest.approx(math.log1p(7 - 2))]
    assert policy._pending_rows[0][0] == pytest.approx(2.0)  # age snapshot at sample time, not "now"


def test_delayed_label_window_exit_dynamic_boundary():
    """The window-exit label is max_eviction_boundary[0] (frozen at the last
    retrain) + this object's own wait -- not a fixed constant like LRB's
    2*window (TLCache.cpp:48-49, 140-142)."""
    # capacity=1: each new distinct page forces exactly one eviction, so a
    # run of distinct filler pages grows the ghost FIFO deterministically.
    filler = [f"F{i}" for i in range(10)]
    requests, pages = build_requests_from_lists(["A"] + filler)
    policy = ThreeLCachePolicy(
        ThreeLCacheConfig(batch_size=1_000_000, sample_subsampling_prob=1.0, seed=0)
    )
    policy.reset(1, pages)
    policy.on_request(requests[0])  # t=0: A admitted, past_timestamp=0
    # Nonzero sentinel marking "a sample is pending"; the window-exit label
    # formula does not depend on the sample_time's value, only on
    # meta.past_timestamp (see docs/three_l_cache_method_spec.md).
    policy._in_cache_meta["A"].sample_time = 1

    # Force the frozen boundary to a known nonzero value, as if a prior
    # training round had observed a max wait of 9.
    policy._max_eviction_boundary = [9.0, 0.0]

    ev = policy.on_request(requests[1])  # t=1: F0 evicts A -> ghosted
    assert ev.evicted == "A"
    assert "A" in policy._ghost_meta

    t_exit = None
    for i, req in enumerate(requests[2:], start=2):
        policy.on_request(req)
        if "A" not in policy._ghost_meta:
            t_exit = i
            break
    assert t_exit is not None, "A's ghost slot never expired within the filler sequence"

    # future_interval = frozen boundary (9) + A's own wait (t_exit - past_timestamp(0))
    expected_label = math.log1p(9.0 + (t_exit - 0))
    assert policy._pending_labels[-1] == pytest.approx(expected_label)
    assert "A" not in policy._ghost_meta


def test_no_future_leakage_from_actual_next_or_predicted_next():
    import random as pyrandom

    _rng = pyrandom.Random(5)
    ids = [_rng.choice("ABCDE") for _ in range(300)]
    requests, pages = build_requests_from_lists(ids)

    def run(reqs):
        policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=15, seed=2))
        policy.reset(3, pages)
        return [(e.hit, e.evicted) for e in (policy.on_request(r) for r in reqs)]

    baseline = run(requests)
    corrupted = [
        Request(t=r.t, page_id=r.page_id, predicted_next=-999.0, actual_next=-999.0)
        for r in requests
    ]
    assert run(corrupted) == baseline


# ---------------------------------------------------------------------------
# Training schedule / auto-tuning
# ---------------------------------------------------------------------------


def test_training_trigger_at_batch_size_gated_by_evict_nums():
    requests, pages = build_requests_from_lists(["A"])
    policy = ThreeLCachePolicy(
        ThreeLCacheConfig(batch_size=2, sample_subsampling_prob=1.0, seed=0)
    )
    policy.reset(2, pages)
    policy.on_request(requests[0])

    meta = policy._in_cache_meta["A"]
    policy._evict_nums = 1  # simulate "mid-batch": retrain must NOT fire yet
    meta.sample_time = 0
    meta.sample_time = 0
    policy._pending_rows = [[0.0] * N_FEATURE, [0.0] * N_FEATURE]
    policy._pending_labels = [0.1, 0.2]
    policy._maybe_mature_pending(ObjectMeta(key="Z", past_timestamp=0, freq=1), 5)  # no-op (sample_time=0... )
    # Directly invoke maturation path with a real pending sample to trigger the check.
    meta.sample_time = 1
    policy._maybe_mature_pending(meta, 5)
    assert policy._n_retrain == 0  # gated: evict_nums > 0

    policy._evict_nums = 0
    meta.sample_time = 1
    policy._maybe_mature_pending(meta, 6)
    assert policy._n_retrain == 1
    assert policy._model is not None
    assert policy._pending_rows == []


def test_auto_tune_disabled_freezes_f_x_q():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"] * 20)
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=5, auto_tune=False, seed=0))
    policy.reset(2, pages)
    for r in requests:
        policy.on_request(r)
    diag = policy.diagnostics_summary()
    assert diag["f"] == 1.0
    assert diag["x"] == 1.0
    assert diag["q"] == 2.0
    assert diag["auto_tune"] == 0.0


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_summary_keys_and_types():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"] * 10)
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=8, seed=0))
    policy.reset(2, pages)
    for r in requests:
        policy.on_request(r)
    summary = policy.diagnostics_summary()
    expected_keys = {
        "batch_size", "num_iterations", "num_leaves", "learning_rate", "seed",
        "objective_is_byte_miss_ratio", "auto_tune", "n_retrain", "model_trained",
        "n_in_cache_meta", "n_ghost_meta", "n_pending_rows", "n_cold_start_evictions",
        "n_model_ranked_evictions", "n_stale_heap_pops", "n_resample_rounds",
        "hsw", "f", "x", "q",
    }
    assert set(summary.keys()) == expected_keys
    assert all(isinstance(v, float) for v in summary.values())
    assert summary["n_retrain"] >= 1.0
    assert summary["model_trained"] == 1.0


# ---------------------------------------------------------------------------
# Integration: registry, CLI, run_policy schema, comparison with LRU
# ---------------------------------------------------------------------------


def test_three_l_cache_registered_in_policy_registry():
    assert "three_l_cache" in POLICY_REGISTRY
    assert POLICY_REGISTRY["three_l_cache"].name == "three_l_cache"


def test_run_policy_produces_expected_schema_and_diagnostics():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"] * 10)
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=8, seed=0))
    result = run_policy(policy, requests, pages, capacity=2)
    assert result.policy_name == "three_l_cache"
    assert result.total_hits + result.total_misses == len(requests)
    assert result.extra_diagnostics is not None
    assert "three_l_cache" in result.extra_diagnostics
    assert result.extra_diagnostics["three_l_cache"]["summary"]["n_retrain"] >= 1.0


def test_run_policy_deterministic_repeat_same_seed():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D", "E"] * 10)

    def run():
        policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=10, seed=9))
        result = run_policy(policy, requests, pages, capacity=3)
        return [(e.hit, e.evicted) for e in result.events]

    assert run() == run()


def test_run_policy_different_seeds_recorded_distinctly():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D", "E"] * 10)
    p1 = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=10, seed=1))
    p2 = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=10, seed=2))
    r1 = run_policy(p1, requests, pages, capacity=3)
    r2 = run_policy(p2, requests, pages, capacity=3)
    assert r1.extra_diagnostics["three_l_cache"]["summary"]["seed"] == 1.0
    assert r2.extra_diagnostics["three_l_cache"]["summary"]["seed"] == 2.0


def test_three_l_cache_vs_lru_same_trace_both_valid():
    import random as pyrandom

    _rng = pyrandom.Random(4)
    ids = [_rng.choice("ABCDEFGH") for _ in range(400)]
    requests, pages = build_requests_from_lists(ids)

    lru_result = run_policy(LRUPolicy(), requests, pages, capacity=4)
    tlc_result = run_policy(
        ThreeLCachePolicy(ThreeLCacheConfig(batch_size=20, seed=0)), requests, pages, capacity=4
    )
    assert lru_result.total_hits + lru_result.total_misses == len(requests)
    assert tlc_result.total_hits + tlc_result.total_misses == len(requests)


def test_cli_smoke_run(tmp_path: Path):
    out_dir = tmp_path / "three_l_cache_smoke_out"
    result = subprocess.run(
        [
            sys.executable, "-m", "lafc.runner.run_policy",
            "--policy", "three_l_cache",
            "--trace", "data/example_unweighted.json",
            "--capacity", "3",
            "--three-l-cache-batch-size", "4",
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Policy:      three_l_cache" in result.stdout
    assert any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# Regression: frozen deterministic trace
# ---------------------------------------------------------------------------


def test_frozen_regression_trace():
    """A fixed 24-request trace, capacity=3, batch_size=4, seed=42. Expected
    hit/evicted/mode sequence captured once from this exact deterministic
    configuration (algorithm as implemented in
    src/lafc/policies/three_l_cache.py, itself hand-verified against
    TLCache.h/TLCache.cpp above) and frozen here as a regression baseline;
    any future change to the algorithm, RNG usage, or LightGBM
    parameterization that changes this sequence must be treated as a
    behavior change requiring re-review, not silently re-frozen.
    """
    page_ids = [
        "A", "B", "C", "D", "A", "B", "E", "A", "C", "B", "D", "A",
        "B", "C", "E", "A", "B", "D", "C", "A", "B", "E", "A", "B",
    ]
    requests, pages = build_requests_from_lists(page_ids)
    policy = ThreeLCachePolicy(ThreeLCacheConfig(batch_size=4, seed=42))
    policy.reset(3, pages)

    hits, evicted, modes = [], [], []
    for req in requests:
        ev = policy.on_request(req)
        hits.append(ev.hit)
        evicted.append(ev.evicted)
        modes.append(ev.diagnostics.get("mode"))

    expected_hits = [
        False, False, False, False, False, False, False, True, False, False,
        False, False, False, True, False, False, False, True, False, False,
        False, True, False, False,
    ]
    expected_evicted = [
        None, None, None, "A", "B", "C", "D", None, "B", "E", "A", "B",
        "A", None, "B", "C", "A", None, "B", "C", "A", None, "B", "A",
    ]
    expected_modes = [
        "direct_admit", "direct_admit", "direct_admit", "cold_start_lru",
        "cold_start_lru", "cold_start_lru", "cold_start_lru", "hit",
        "cold_start_lru", "cold_start_lru", "cold_start_lru", "model_ranked",
        "model_ranked", "hit", "model_ranked", "model_ranked", "model_ranked",
        "hit", "model_ranked", "model_ranked", "model_ranked", "hit",
        "model_ranked", "model_ranked",
    ]

    assert hits == expected_hits
    assert evicted == expected_evicted
    assert modes == expected_modes
    assert sum(1 for h in hits if not h) == 20

    summary = policy.diagnostics_summary()
    assert summary["n_retrain"] == 1.0
    assert summary["n_cold_start_evictions"] == 7.0
    assert summary["n_model_ranked_evictions"] == 10.0
