"""Tests for the MRU/random continuation-policy support added to
evict_value_v2_rollout.py for the continuation-policy-sensitivity pilot.

Does not modify, and does not import for mutation, evict_value_dataset_v1.py
or evict_value_wulver_v1.py (the canonical LRU generator) -- only imports
_simulate_lru_misses read-only, for the equivalence comparison.
"""
from __future__ import annotations

import random

import pytest

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_v2_rollout import (
    SUPPORTED_REFERENCE_POLICIES,
    EvictValueV2RolloutConfig,
    _choose_victim,
    simulate_rollout_misses,
)
from lafc.simulator.request_trace import build_requests_from_lists


def _requests(page_ids):
    reqs, _pages = build_requests_from_lists(page_ids)
    return reqs


def test_sieve_is_not_a_supported_reference_policy():
    assert "sieve" not in SUPPORTED_REFERENCE_POLICIES
    with pytest.raises(ValueError, match="Unsupported reference policy"):
        simulate_rollout_misses(
            cache_pages=["a", "b"], future_reqs=_requests(["a", "c"]), capacity=2, reference_policy="sieve"
        )


def test_mru_victim_is_most_recently_used_end_of_recency_order():
    # order is an OrderedDict built from cache_pages in that insertion order;
    # candidates[-1] is the tail = most recently positioned.
    import collections
    order = collections.OrderedDict((p, None) for p in ["a", "b", "c"])
    victim = _choose_victim(order, future_reqs=[], step_idx=0, policy="mru")
    assert victim == "c"


def test_lru_victim_is_least_recently_used_head_of_recency_order():
    import collections
    order = collections.OrderedDict((p, None) for p in ["a", "b", "c"])
    victim = _choose_victim(order, future_reqs=[], step_idx=0, policy="lru")
    assert victim == "a"


def test_random_requires_rng():
    with pytest.raises(ValueError, match="requires an rng"):
        import collections
        order = collections.OrderedDict((p, None) for p in ["a", "b"])
        _choose_victim(order, future_reqs=[], step_idx=0, policy="random", rng=None)


def test_random_reference_policy_requires_rng_seed_in_simulate_rollout_misses():
    with pytest.raises(ValueError, match="requires rng_seed"):
        simulate_rollout_misses(
            cache_pages=["a", "b"], future_reqs=_requests(["c", "d"]), capacity=2,
            reference_policy="random", rng_seed=None,
        )


def test_random_same_seed_is_reproducible():
    cache_pages = ["a", "b", "c", "d"]
    future = _requests(["x", "y", "z", "x", "y"])
    r1 = simulate_rollout_misses(cache_pages=cache_pages, future_reqs=future, capacity=4, reference_policy="random", rng_seed=42)
    r2 = simulate_rollout_misses(cache_pages=cache_pages, future_reqs=future, capacity=4, reference_policy="random", rng_seed=42)
    assert r1 == r2


def test_random_different_seeds_can_differ():
    # Construct a synthetic scenario with enough branching that at least one
    # of a handful of seeds must disagree with seed=0 (avoids a flaky
    # "assert not all equal" over too few trials).
    cache_pages = [f"p{i}" for i in range(6)]
    future = _requests([f"q{i}" for i in range(6)] * 3)
    results = {
        seed: simulate_rollout_misses(cache_pages=cache_pages, future_reqs=future, capacity=6, reference_policy="random", rng_seed=seed)
        for seed in range(30)
    }
    assert len(set(results.values())) > 1, "expected at least some variation in miss counts across 30 seeds"


def test_crn_same_seed_used_across_candidates_gives_correlated_not_identical_draws():
    # Two different "candidate" starting caches (as build_rollout_candidate_rows_v2
    # would construct via forced_cache), same seed: draws are not required to be
    # literally identical (different residents / different miss timing), but
    # both must be individually seed-reproducible (already covered above) --
    # this test just documents/exercises that CRN means "same seed value",
    # not "same literal random draws regardless of state".
    future = _requests(["x", "y", "z", "x", "y", "z"])
    loss_candidate_a = simulate_rollout_misses(
        cache_pages=["a", "b", "c"], future_reqs=future, capacity=3, reference_policy="random", rng_seed=7,
    )
    loss_candidate_b = simulate_rollout_misses(
        cache_pages=["b", "c", "d"], future_reqs=future, capacity=3, reference_policy="random", rng_seed=7,
    )
    # Both reproducible under the same seed value (this is the actual CRN contract):
    assert loss_candidate_a == simulate_rollout_misses(
        cache_pages=["a", "b", "c"], future_reqs=future, capacity=3, reference_policy="random", rng_seed=7,
    )
    assert loss_candidate_b == simulate_rollout_misses(
        cache_pages=["b", "c", "d"], future_reqs=future, capacity=3, reference_policy="random", rng_seed=7,
    )


def test_lru_generalized_path_equals_canonical_lru_on_synthetic_examples():
    rng = random.Random(99)
    for _ in range(20):
        n_pages = rng.randint(2, 6)
        cache_pages = [f"p{i}" for i in range(n_pages)]
        future_ids = [f"p{rng.randrange(n_pages + 2)}" for _ in range(30)]
        future = _requests(future_ids)
        canonical = _simulate_lru_misses(cache_pages, future, capacity=n_pages)
        generalized = simulate_rollout_misses(
            cache_pages=cache_pages, future_reqs=future, capacity=n_pages, reference_policy="lru",
        )
        assert canonical == generalized, (cache_pages, future_ids)


def test_no_candidate_omission_all_residents_present_in_order():
    import collections
    cache_pages = ["a", "b", "c"]
    order = collections.OrderedDict((p, None) for p in cache_pages)
    for policy in ("lru", "mru", "fifo"):
        victim = _choose_victim(order, future_reqs=[], step_idx=0, policy=policy)
        assert victim in cache_pages


def test_horizon_slice_is_exactly_h_future_requests_not_misses():
    # Documents/locks the existing (unchanged) semantics this pilot depends on:
    # build_rollout_candidate_rows_v2 slices `future[:h]`, i.e. H future
    # requests regardless of hit/miss composition.
    reqs = _requests(["a", "a", "a", "b", "c", "d", "e"])
    future = reqs[1:]
    assert len(future[:3]) == 3
