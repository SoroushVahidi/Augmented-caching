"""Tests for the LRB (Learning Relaxed Belady, NSDI 2020) policy.

Where a test asserts an exact numeric value, the value is either (a) derived
by hand from the recurrences documented in ``src/lafc/lrb_features.py`` and
the official implementation (``lrb.h``/``lrb.cpp``, commit
``9e8b4423383c01c4528deb447f152f0437a37c3a``), computed independently in the
test rather than by re-invoking the code under test, or (b) captured once
from a real deterministic run and frozen as a regression baseline (see
``test_frozen_regression_trace`` for the derivation note).
"""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from lafc.lrb_features import (
    N_EDC_FEATURE,
    ObjectMeta,
    compute_lrb_feature_row,
    compute_n_within,
    edc_windows,
    hash_edc_table,
    label_from_future_interval,
    n_feature_count,
)
from lafc.policies.lrb import LRBConfig, LRBPolicy
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.policies.lru import LRUPolicy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Request


class _FakeRng:
    """Deterministic stand-in for random.Random used to isolate sampling logic."""

    def sample(self, population, k):
        return list(population)[:k]

    def choice(self, population):
        return list(population)[0]


class _ConstantModel:
    """Stub model that scores every candidate identically, to isolate tie-break logic."""

    def predict(self, rows):
        return [0.0] * len(rows)


def _req(t: int, page_id: str) -> Request:
    return Request(t=t, page_id=page_id, predicted_next=math.inf, actual_next=math.inf)


# ---------------------------------------------------------------------------
# Feature module: official-parity checks (hand-derived against lrb.h/lrb.cpp)
# ---------------------------------------------------------------------------


def test_edc_recurrence_matches_official_metaextra():
    """MetaExtra constructor seeds edc[i] = hash_edc[idx] + 1; update() applies
    edc[i] = edc[i] * hash_edc[idx] + 1 (lrb.h:64-86)."""
    windows = edc_windows()
    memory_window = 10000
    hash_edc = hash_edc_table(memory_window)  # hash_edc[i] = 0.5**i

    meta = ObjectMeta(key="A", past_timestamp=0)
    assert meta.edc is None  # mirrors _extra == nullptr for one-hit-wonders

    # Second request at t=5: distance=5, falls in edc_windows[0]=1024's first bucket.
    meta.record_request(5, max_n_past_distances=31, windows=windows, hash_edc=hash_edc)
    idx0 = min(5 // windows[0], len(hash_edc) - 1)
    expected_edc0 = hash_edc[idx0] + 1.0  # constructor formula, independent of update()
    assert meta.edc is not None
    assert meta.edc[0] == pytest.approx(expected_edc0)
    assert meta.past_distances == [5]

    # Third request at t=9: distance=4; update() recurrence, computed independently here.
    meta.record_request(9, max_n_past_distances=31, windows=windows, hash_edc=hash_edc)
    idx_new = min(4 // windows[0], len(hash_edc) - 1)
    expected_edc0_v2 = expected_edc0 * hash_edc[idx_new] + 1.0
    assert meta.edc[0] == pytest.approx(expected_edc0_v2)
    assert meta.past_distances == [4, 5]  # most-recent-first


def test_past_distances_capped_and_most_recent_first():
    windows = edc_windows()
    hash_edc = hash_edc_table(100000)
    meta = ObjectMeta(key="A", past_timestamp=0)
    t = 0
    for i in range(1, 6):
        t += 1
        meta.record_request(t, max_n_past_distances=3, windows=windows, hash_edc=hash_edc)
    # cap=3: only the 3 most recent deltas (all equal to 1 here) are retained.
    assert len(meta.past_distances) == 3


def test_n_within_matches_official_loop_semantics():
    # distances most-recent-first; running sum: 4 (<6 -> count), 9 (>=6 -> stop counting)
    # but the loop keeps iterating (lrb.cpp's `else break;` is commented out).
    assert compute_n_within([4, 5, 100], memory_window=6) == 1
    assert compute_n_within([], memory_window=6) == 0
    assert compute_n_within([1, 1, 1], memory_window=100) == 3


def test_feature_row_layout_and_values():
    windows = edc_windows()
    memory_window = 50
    hash_edc = hash_edc_table(memory_window)
    meta = ObjectMeta(key="A", past_timestamp=0)
    meta.record_request(3, max_n_past_distances=31, windows=windows, hash_edc=hash_edc)  # distance=3
    meta.record_request(10, max_n_past_distances=31, windows=windows, hash_edc=hash_edc)  # distance=7

    row = compute_lrb_feature_row(
        meta, sample_timestamp=15, memory_window=memory_window,
        max_n_past_timestamps=32, windows=windows, hash_edc=hash_edc,
    )
    n = n_feature_count(32, 0, N_EDC_FEATURE)
    assert len(row) == n == 44
    assert row[0] == pytest.approx(15 - 10)  # age = sample_timestamp - past_timestamp
    assert row[1] == pytest.approx(7)  # most recent delta
    assert row[2] == pytest.approx(3)  # second most recent delta
    assert math.isnan(row[3])  # unfilled delta slot -> NaN (LightGBM "missing")
    assert row[32] == pytest.approx(1.0)  # size, constant under unit-size specialization
    assert row[33] == pytest.approx(compute_n_within([7, 3], memory_window))
    # EDC block starts at index 34; spot-check the first one independently.
    age = 15 - 10
    idx0 = min(age // windows[0], len(hash_edc) - 1)
    assert row[34] == pytest.approx(meta.edc[0] * hash_edc[idx0])


def test_feature_row_one_hit_wonder_edc_uses_plain_decay():
    """For an object with only 1 past request (meta.edc is None), the EDC
    feature is hash_edc[idx] directly, not edc*hash_edc[idx] (lrb.h:444-448)."""
    windows = edc_windows()
    memory_window = 1000
    hash_edc = hash_edc_table(memory_window)
    meta = ObjectMeta(key="A", past_timestamp=0)
    row = compute_lrb_feature_row(
        meta, sample_timestamp=5, memory_window=memory_window,
        max_n_past_timestamps=32, windows=windows, hash_edc=hash_edc,
    )
    idx0 = min(5 // windows[0], len(hash_edc) - 1)
    assert row[34] == pytest.approx(hash_edc[idx0])


def test_label_transform_is_log1p():
    assert label_from_future_interval(0.0) == pytest.approx(math.log1p(0.0))
    assert label_from_future_interval(99.0) == pytest.approx(math.log1p(99.0))


def test_record_request_rejects_non_increasing_timestamp():
    meta = ObjectMeta(key="A", past_timestamp=5)
    with pytest.raises(ValueError):
        meta.record_request(5, max_n_past_distances=31, windows=edc_windows(), hash_edc=hash_edc_table(100))
    with pytest.raises(ValueError):
        meta.record_request(4, max_n_past_distances=31, windows=edc_windows(), hash_edc=hash_edc_table(100))


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        LRBConfig(sample_rate=0)
    with pytest.raises(ValueError):
        LRBConfig(memory_window=0)
    with pytest.raises(ValueError):
        LRBConfig(batch_size=0)
    with pytest.raises(ValueError):
        LRBConfig(max_n_past_timestamps=0)
    with pytest.raises(ValueError):
        LRBConfig(objective="not_a_real_objective")


def test_capacity_zero_rejected():
    _, pages = build_requests_from_lists(["A", "B"])
    policy = LRBPolicy(LRBConfig())
    with pytest.raises(ValueError):
        policy.reset(0, pages)


# ---------------------------------------------------------------------------
# Core online behavior
# ---------------------------------------------------------------------------


def test_reset_and_empty_cache_state():
    _, pages = build_requests_from_lists(["A"])
    policy = LRBPolicy(LRBConfig())
    policy.reset(3, pages)
    assert policy.current_cache() == frozenset()
    assert policy._in_cache_meta == {}
    assert policy._ghost_meta == {}


def test_hit_and_miss_bookkeeping():
    requests, pages = build_requests_from_lists(["A", "A", "B", "A"])
    policy = LRBPolicy(LRBConfig())
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert [e.hit for e in events] == [False, True, False, True]
    assert policy.total_cost() == 2.0  # two misses, unit cost each


def test_miss_with_available_space_no_eviction():
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    policy = LRBPolicy(LRBConfig())
    policy.reset(3, pages)
    events = [policy.on_request(r) for r in requests]
    assert all(e.evicted is None for e in events)
    assert policy.current_cache() == frozenset({"A", "B", "C"})


def test_capacity_one_every_new_page_evicts_previous():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"])
    policy = LRBPolicy(LRBConfig())
    policy.reset(1, pages)
    events = [policy.on_request(r) for r in requests]
    assert [e.evicted for e in events] == [None, "A", "B", "C"]
    assert policy.current_cache() == frozenset({"D"})


def test_repeated_requests_single_page_capacity_one():
    requests, pages = build_requests_from_lists(["A"] * 10)
    policy = LRBPolicy(LRBConfig())
    policy.reset(1, pages)
    events = [policy.on_request(r) for r in requests]
    assert sum(1 for e in events if not e.hit) == 1
    assert sum(1 for e in events if e.hit) == 9
    assert all(e.evicted is None for e in events)


def test_cold_start_lru_fallback_before_any_model_trained():
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    policy = LRBPolicy(LRBConfig(batch_size=100000))
    policy.reset(2, pages)
    events = [policy.on_request(r) for r in requests]
    assert events[2].diagnostics["mode"] == "cold_start_lru"
    assert events[2].evicted == "A"  # LRU tail: A requested first, never re-touched
    assert policy.diagnostics_summary()["model_trained"] == 0.0


def test_cache_lru_and_meta_stay_in_sync_with_base_cache_state():
    import random as pyrandom

    _rng = pyrandom.Random(0)
    ids = [_rng.choice("ABCDEFGH") for _ in range(200)]
    requests, pages = build_requests_from_lists(ids)
    policy = LRBPolicy(LRBConfig(sample_rate=5, memory_window=30, batch_size=12, seed=1))
    policy.reset(4, pages)
    for req in requests:
        policy.on_request(req)
        assert (
            set(policy.current_cache())
            == set(policy._in_cache_meta.keys())
            == set(policy._lru_queue.keys())
        )


# ---------------------------------------------------------------------------
# Candidate sampling
# ---------------------------------------------------------------------------


def test_candidate_sampling_fixed_rng_exact_set():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"])
    policy = LRBPolicy(LRBConfig(sample_rate=2))
    policy.reset(3, pages)
    for r in requests[:3]:
        policy.on_request(r)  # A, B, C admitted
    policy._model = _ConstantModel()
    policy._rng = _FakeRng()  # sample() returns the first k of the pool, in insertion order
    ev = policy.on_request(requests[3])  # D forces eviction among {A, B, C}
    assert ev.diagnostics["mode"] == "model_ranked"
    assert ev.diagnostics["candidate_count"] == 2  # min(sample_rate=2, pool=3)
    # _FakeRng.sample returns pool[:2] = ["A", "B"] (insertion order); all tied
    # at score 0.0 -> deterministic tie-break picks the smallest page_id, "A".
    assert ev.evicted == "A"


def test_candidate_count_smaller_than_sample_rate():
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    policy = LRBPolicy(LRBConfig(sample_rate=10))
    policy.reset(2, pages)
    policy.on_request(requests[0])
    policy.on_request(requests[1])
    policy._model = _ConstantModel()
    ev = policy.on_request(requests[2])  # cache full with only 2 candidates < sample_rate=10
    assert ev.diagnostics["candidate_count"] == 2


def test_deterministic_tie_break_by_page_id():
    requests, pages = build_requests_from_lists(["Z", "M", "A", "X"])
    policy = LRBPolicy(LRBConfig(sample_rate=10))
    policy.reset(3, pages)
    for r in requests[:3]:
        policy.on_request(r)  # Z, M, A admitted
    policy._model = _ConstantModel()  # every candidate scores 0.0 -> pure tie-break
    ev = policy.on_request(requests[3])
    assert ev.evicted == "A"  # smallest page_id among {Z, M, A}


def test_reproducible_candidate_selection_same_seed():
    import random as pyrandom

    _rng = pyrandom.Random(3)
    ids = [_rng.choice("ABCDEF") for _ in range(150)]
    requests, pages = build_requests_from_lists(ids)

    def run(seed: int):
        policy = LRBPolicy(LRBConfig(sample_rate=3, memory_window=20, batch_size=8, seed=seed))
        policy.reset(3, pages)
        return [(e.hit, e.evicted) for e in (policy.on_request(r) for r in requests)]

    assert run(11) == run(11)
    assert run(11) != run(12)


# ---------------------------------------------------------------------------
# Delayed-label maturation / no future leakage
# ---------------------------------------------------------------------------


def test_delayed_label_on_rerequest_exact_value_and_snapshot_timing():
    requests, pages = build_requests_from_lists(["A"])
    policy = LRBPolicy(LRBConfig(sample_rate=10, memory_window=1000, batch_size=1_000_000, seed=0))
    policy.reset(3, pages)
    policy.on_request(requests[0])  # t=0: A admitted, past_timestamp=0

    # Simulate an unlabeled sample recorded for A at t=2 (what _sample_for_training does).
    policy._in_cache_meta["A"].sample_times.append(2)

    # Re-request A at t=7: label matures using the *observed* gap since the
    # sample was taken (7 - 2 = 5), and the feature row reflects the object's
    # state AS OF the sample time (age = 2 - 0 = 2), not "now" (7 - 0 = 7).
    policy.on_request(_req(7, "A"))

    assert policy._pending_labels == [pytest.approx(math.log1p(7 - 2))]
    assert policy._pending_rows[0][0] == pytest.approx(2.0)


def test_finalize_eviction_young_object_is_ghosted_not_dropped():
    requests, pages = build_requests_from_lists(["A"])
    memory_window = 5
    policy = LRBPolicy(LRBConfig(sample_rate=10, memory_window=memory_window, batch_size=1_000_000, seed=0))
    policy.reset(2, pages)
    policy.on_request(requests[0])
    policy._finalize_eviction("A", 2)  # age = 2 - 0 = 2 < memory_window
    assert "A" in policy._ghost_meta
    assert "A" not in policy._in_cache_meta


def test_finalize_eviction_aged_object_is_dropped_with_forced_label():
    requests, pages = build_requests_from_lists(["A"])
    memory_window = 5
    policy = LRBPolicy(LRBConfig(sample_rate=10, memory_window=memory_window, batch_size=1_000_000, seed=0))
    policy.reset(2, pages)
    policy.on_request(requests[0])  # A: past_timestamp=0
    policy._in_cache_meta["A"].sample_times.append(0)  # pretend sampled at t=0

    policy._finalize_eviction("A", 9)  # age = 9 - 0 = 9 >= memory_window=5

    assert "A" not in policy._in_cache_meta
    assert "A" not in policy._ghost_meta  # dropped, not ghosted
    assert policy._n_force_eviction == 1
    # future_distance = age + memory_window = 9 + 5 = 14 (lrb.cpp:554)
    assert policy._pending_labels == [pytest.approx(math.log1p(9 + 5))]


def test_choose_and_evict_age_forced_branch_end_to_end():
    memory_window = 3
    requests, pages = build_requests_from_lists(["A", "B", "C"])
    policy = LRBPolicy(LRBConfig(sample_rate=10, memory_window=memory_window, batch_size=1_000_000, seed=0))
    policy.reset(2, pages)
    policy.on_request(requests[0])  # t=0 A
    policy.on_request(requests[1])  # t=1 B
    policy._model = _ConstantModel()  # pretend already trained, to skip cold-start branch

    ev = policy.on_request(_req(10, "C"))  # age(A) = 10 - 0 = 10 >= memory_window=3

    assert ev.evicted == "A"
    assert ev.diagnostics["mode"] == "age_forced_lru"
    assert "A" not in policy._ghost_meta


def test_never_reaccessed_label_via_ghost_timeout():
    memory_window = 5
    requests, pages = build_requests_from_lists(["A", "B"])
    policy = LRBPolicy(LRBConfig(sample_rate=10, memory_window=memory_window, batch_size=1_000_000, seed=0))
    policy.reset(1, pages)
    policy._sample_for_training = lambda t: None  # isolate the ghost-timeout maturation path
    policy.on_request(requests[0])  # t=0 A admitted
    policy._in_cache_meta["A"].sample_times.append(0)  # pretend sampled at t=0

    ev = policy.on_request(requests[1])  # t=1 B: evicts A (age=1 < window) -> ghosted
    assert ev.evicted == "A"
    assert "A" in policy._ghost_meta

    for t in range(2, 5):
        policy.on_request(_req(t, "B"))
    assert "A" in policy._ghost_meta  # not yet expired

    policy.on_request(_req(5, "B"))  # slot 5 % 5 == 0 -> A's ghost entry expires
    assert "A" not in policy._ghost_meta
    assert policy._pending_labels == [pytest.approx(math.log1p(2 * memory_window))]


def test_no_future_leakage_from_actual_next_or_predicted_next():
    import random as pyrandom

    _rng = pyrandom.Random(5)
    ids = [_rng.choice("ABCDE") for _ in range(120)]
    requests, pages = build_requests_from_lists(ids)

    def run(reqs):
        policy = LRBPolicy(LRBConfig(sample_rate=3, memory_window=15, batch_size=6, seed=2))
        policy.reset(3, pages)
        return [(e.hit, e.evicted) for e in (policy.on_request(r) for r in reqs)]

    baseline = run(requests)

    corrupted = [
        Request(t=r.t, page_id=r.page_id, predicted_next=-999.0, actual_next=-999.0)
        for r in requests
    ]
    assert run(corrupted) == baseline


# ---------------------------------------------------------------------------
# Training schedule / model refresh
# ---------------------------------------------------------------------------


def test_training_trigger_at_batch_size_and_buffer_clears():
    requests, pages = build_requests_from_lists(["A"])
    policy = LRBPolicy(LRBConfig(sample_rate=4, memory_window=1000, batch_size=3, seed=0))
    policy.reset(2, pages)
    policy.on_request(requests[0])

    meta = policy._in_cache_meta["A"]
    meta.sample_times = [0, 0]
    policy._mature_pending(meta, 5)
    assert policy._n_retrain == 0
    assert len(policy._pending_rows) == 2

    meta_b = ObjectMeta(key="B", past_timestamp=0)
    meta_b.sample_times = [0]
    policy._mature_pending(meta_b, 5)
    assert policy._n_retrain == 1
    assert policy._model is not None
    assert policy._pending_rows == []
    assert policy._pending_labels == []


def test_model_refresh_replaces_booster_not_incremental():
    requests, pages = build_requests_from_lists(["A"])
    policy = LRBPolicy(LRBConfig(sample_rate=4, memory_window=1000, batch_size=3, seed=0))
    policy.reset(2, pages)
    policy.on_request(requests[0])

    meta = policy._in_cache_meta["A"]
    meta.sample_times = [0, 0, 0]
    policy._mature_pending(meta, 5)
    first_model = policy._model
    assert first_model is not None

    meta.sample_times = [0, 0, 0]
    policy._mature_pending(meta, 20)
    assert policy._n_retrain == 2
    assert policy._model is not first_model  # full refit, not incremental update


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def test_diagnostics_summary_keys_and_types():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"] * 5)
    policy = LRBPolicy(LRBConfig(sample_rate=3, memory_window=10, batch_size=4, seed=0))
    policy.reset(2, pages)
    for r in requests:
        policy.on_request(r)
    summary = policy.diagnostics_summary()
    expected_keys = {
        "sample_rate", "memory_window", "batch_size", "max_n_past_timestamps",
        "num_iterations", "num_leaves", "learning_rate", "seed",
        "objective_is_object_miss_ratio", "n_retrain", "model_trained",
        "n_in_cache_meta", "n_ghost_meta", "n_pending_rows", "n_force_eviction",
        "n_cold_start_evictions", "n_age_forced_evictions",
        "n_model_ranked_evictions", "n_candidates_sampled_total",
    }
    assert set(summary.keys()) == expected_keys
    assert all(isinstance(v, float) for v in summary.values())
    assert summary["n_retrain"] >= 1.0
    assert summary["model_trained"] == 1.0


# ---------------------------------------------------------------------------
# Integration: registry, CLI, run_policy schema, comparison with LRU
# ---------------------------------------------------------------------------


def test_lrb_registered_in_policy_registry():
    assert "lrb" in POLICY_REGISTRY
    assert POLICY_REGISTRY["lrb"].name == "lrb"


def test_run_policy_produces_expected_schema_and_diagnostics():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D"] * 6)
    policy = LRBPolicy(LRBConfig(sample_rate=3, memory_window=10, batch_size=4, seed=0))
    result = run_policy(policy, requests, pages, capacity=2)
    assert result.policy_name == "lrb"
    assert result.total_hits + result.total_misses == len(requests)
    assert result.extra_diagnostics is not None
    assert "lrb" in result.extra_diagnostics
    assert "summary" in result.extra_diagnostics["lrb"]
    assert result.extra_diagnostics["lrb"]["summary"]["n_retrain"] >= 1.0


def test_run_policy_deterministic_repeat_same_seed():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D", "E"] * 8)

    def run():
        policy = LRBPolicy(LRBConfig(sample_rate=3, memory_window=12, batch_size=5, seed=9))
        result = run_policy(policy, requests, pages, capacity=3)
        return [(e.hit, e.evicted) for e in result.events]

    assert run() == run()


def test_run_policy_different_seeds_recorded_distinctly():
    requests, pages = build_requests_from_lists(["A", "B", "C", "D", "E"] * 8)
    p1 = LRBPolicy(LRBConfig(sample_rate=3, memory_window=12, batch_size=5, seed=1))
    p2 = LRBPolicy(LRBConfig(sample_rate=3, memory_window=12, batch_size=5, seed=2))
    r1 = run_policy(p1, requests, pages, capacity=3)
    r2 = run_policy(p2, requests, pages, capacity=3)
    assert r1.extra_diagnostics["lrb"]["summary"]["seed"] == 1.0
    assert r2.extra_diagnostics["lrb"]["summary"]["seed"] == 2.0


def test_lrb_vs_lru_same_trace_both_valid():
    import random as pyrandom

    _rng = pyrandom.Random(4)
    ids = [_rng.choice("ABCDEFGH") for _ in range(300)]
    requests, pages = build_requests_from_lists(ids)

    lru_result = run_policy(LRUPolicy(), requests, pages, capacity=4)
    lrb_result = run_policy(
        LRBPolicy(LRBConfig(sample_rate=4, memory_window=40, batch_size=16, seed=0)),
        requests, pages, capacity=4,
    )
    assert lru_result.total_hits + lru_result.total_misses == len(requests)
    assert lrb_result.total_hits + lrb_result.total_misses == len(requests)
    assert lru_result.total_misses <= len(requests)
    assert lrb_result.total_misses <= len(requests)


def test_cli_smoke_run(tmp_path: Path):
    out_dir = tmp_path / "lrb_smoke_out"
    result = subprocess.run(
        [
            sys.executable, "-m", "lafc.runner.run_policy",
            "--policy", "lrb",
            "--trace", "data/example_unweighted.json",
            "--capacity", "3",
            "--lrb-sample-rate", "4",
            "--lrb-memory-window", "20",
            "--lrb-batch-size", "8",
            "--output-dir", str(out_dir),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Policy:      lrb" in result.stdout
    assert (out_dir / "metrics.json").exists() or (out_dir / "summary.json").exists() or any(out_dir.iterdir())


# ---------------------------------------------------------------------------
# Regression: frozen deterministic trace
# ---------------------------------------------------------------------------


def test_frozen_regression_trace():
    """A fixed 24-request trace, capacity=3, sample_rate=3, memory_window=12,
    batch_size=4, seed=42. Expected hit/evicted/mode sequence captured once
    from this exact deterministic configuration (algorithm as implemented in
    src/lafc/policies/lrb.py, itself hand-verified against lrb.h/lrb.cpp
    above) and frozen here as a regression baseline; any future change to the
    algorithm, RNG usage, or LightGBM parameterization that changes this
    sequence must be treated as a behavior change requiring re-review, not
    silently re-frozen.
    """
    page_ids = [
        "A", "B", "C", "D", "A", "B", "E", "A", "C", "B", "D", "A",
        "B", "C", "E", "A", "B", "D", "C", "A", "B", "E", "A", "B",
    ]
    requests, pages = build_requests_from_lists(page_ids)
    policy = LRBPolicy(LRBConfig(sample_rate=3, memory_window=12, batch_size=4, seed=42))
    policy.reset(3, pages)

    hits = []
    evicted = []
    modes = []
    for req in requests:
        ev = policy.on_request(req)
        hits.append(ev.hit)
        evicted.append(ev.evicted)
        modes.append(ev.diagnostics.get("mode"))

    expected_hits = [
        False, False, False, False, False, False, False, True, False, False,
        False, False, False, False, True, False, False, True, False, False,
        False, True, False, False,
    ]
    expected_evicted = [
        None, None, None, "A", "B", "C", "D", None, "B", "A", "B", "C",
        "A", "B", None, "C", "A", None, "B", "C", "A", None, "B", "A",
    ]
    expected_modes = [
        "direct_admit", "direct_admit", "direct_admit", "cold_start_lru",
        "cold_start_lru", "cold_start_lru", "cold_start_lru", "hit",
        "cold_start_lru", "model_ranked", "model_ranked", "model_ranked",
        "model_ranked", "model_ranked", "hit", "model_ranked", "model_ranked",
        "hit", "model_ranked", "model_ranked", "model_ranked", "hit",
        "model_ranked", "model_ranked",
    ]

    assert hits == expected_hits
    assert evicted == expected_evicted
    assert modes == expected_modes
    assert sum(1 for h in hits if not h) == 20

    summary = policy.diagnostics_summary()
    assert summary["n_retrain"] == 4.0
    assert summary["n_cold_start_evictions"] == 5.0
    assert summary["n_model_ranked_evictions"] == 12.0
    assert summary["n_force_eviction"] == 0.0
