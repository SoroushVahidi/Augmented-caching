"""Tests for the CACHEUS external-baseline adapter.

CACHEUS's own SR-LRU / CR-LFU / expert-weighting algorithmic correctness is
the authors' own code (github.com/sylab/cacheus), executed unmodified via
`lafc.cacheus_official_loader` -- it is not re-derived or re-verified here.
What this file verifies is the **adapter**: that this repository drives the
official class correctly (request order, hit/miss interpretation, capacity
semantics, no future-information leakage, deterministic replay, CLI/registry
wiring), and a cross-simulator parity check using the official repository's
own `LRU` class as a ground truth for "is the harness wired correctly."

All tests that touch the official source are skipped (not failed) when the
external clone hasn't been fetched, so `pytest tests/` stays green in a
fresh checkout. Run:

    python scripts/setup/fetch_cacheus_official.py

to enable them.
"""

from __future__ import annotations

import math
import random as py_random

import numpy as np
import pytest

from lafc.cacheus_official_loader import EXPECTED_COMMIT, EXTERNAL_CODE_DIR, load_official_classes
from lafc.policies.cacheus import OFFICIAL_RNG_SEED, CacheusConfig, CacheusPolicy
from lafc.policies.lru import LRUPolicy
from lafc.runner.run_policy import POLICY_REGISTRY, run_policy
from lafc.types import Page, Request

_OFFICIAL_SOURCE_PRESENT = (EXTERNAL_CODE_DIR / "algs" / "cacheus.py").exists()
requires_official_source = pytest.mark.skipif(
    not _OFFICIAL_SOURCE_PRESENT,
    reason="Official CACHEUS source not fetched; run "
    "scripts/setup/fetch_cacheus_official.py",
)


def _trace(page_ids, weight=1.0):
    pages = {p: Page(page_id=p, weight=weight) for p in set(page_ids)}
    requests = [Request(t=i, page_id=p) for i, p in enumerate(page_ids)]
    return requests, pages


def _expert_disagreement_trace():
    # Two "hot" pages (high frequency, but not necessarily Q-recency-oldest)
    # interleaved with a stream of unique "cold" pages (Q churn) -- designed
    # to make the Q-LRU-oldest and LFU-heap-min eviction proposals disagree,
    # exercising Cacheus.getChoice()'s np.random.rand() draw and, via ghost-
    # history hits, adjustWeights(). Empirically verified (not asserted
    # blind) to produce n_history_hits_{lru,lfu} > 0 and final weights that
    # move away from the uniform 0.5/0.5 prior at capacity=4.
    page_ids = []
    cold_counter = 0
    for i in range(300):
        if i % 3 == 0:
            page_ids.append("hot_a")
        elif i % 5 == 0:
            page_ids.append("hot_b")
        else:
            page_ids.append(f"cold_{cold_counter}")
            cold_counter += 1
    return _trace(page_ids)


# ---------------------------------------------------------------------------
# Cross-simulator LRU parity (validates the harness wiring, independent of
# CACHEUS-specific algorithmic behavior).
# ---------------------------------------------------------------------------


@requires_official_source
@pytest.mark.parametrize(
    "page_ids,capacity",
    [
        (["a", "b", "c", "d", "a", "b", "e", "a", "c", "d"], 3),
        (["a"] * 5 + ["b"] * 5 + ["a", "b", "c", "d", "e"] * 4, 4),
        ([str(i % 7) for i in range(200)], 5),
    ],
)
def test_official_lru_matches_repository_lru(page_ids, capacity):
    _Cacheus, LRU = load_official_classes()

    official = LRU(capacity, 100)
    official_hits = []
    for i, pid in enumerate(page_ids):
        op, _evicted = official.request(pid, i)
        official_hits.append(op.name == "HIT")

    requests, pages = _trace(page_ids)
    result = run_policy(LRUPolicy(), requests, pages, capacity)
    repo_hits = [e.hit for e in result.events]

    assert official_hits == repo_hits, (
        "Official sylab/cacheus LRU and this repository's native LRU "
        "disagree on hit/miss sequence -- request order, capacity "
        "semantics, or hit/miss interpretation differ. This must pass "
        "before CACHEUS results can be trusted."
    )
    assert sum(1 for h in official_hits if not h) == result.total_misses


# ---------------------------------------------------------------------------
# Adapter wiring
# ---------------------------------------------------------------------------


@requires_official_source
def test_cacheus_hit_and_miss_with_free_space():
    requests, pages = _trace(["a", "b", "a"])
    result = run_policy(CacheusPolicy(), requests, pages, capacity=3)
    assert [e.hit for e in result.events] == [False, False, True]
    assert result.total_misses == 2


@requires_official_source
def test_cacheus_full_cache_eviction():
    requests, pages = _trace(["a", "b", "c", "d"])
    result = run_policy(CacheusPolicy(), requests, pages, capacity=2)
    assert result.events[2].evicted is not None
    assert result.events[2].evicted in ("a", "b")


@requires_official_source
def test_cacheus_capacity_one_raises_explicit_error():
    # Confirmed upstream limitation in the official source (history_size =
    # capacity // 2 == 0 crashes addToHistory with an AttributeError). The
    # adapter surfaces a clear, explicit ValueError instead of letting that
    # third-party crash propagate. See docs/cacheus_method_spec.md.
    policy = CacheusPolicy()
    with pytest.raises(ValueError, match="capacity=1 is not supported"):
        policy.reset(capacity=1, pages={"a": Page(page_id="a", weight=1.0)})


@requires_official_source
def test_cacheus_capacity_two_minimum_supported():
    requests, pages = _trace(["a", "b", "c", "a", "b", "c"] * 3)
    result = run_policy(CacheusPolicy(), requests, pages, capacity=2)
    assert result.total_misses > 0


@requires_official_source
def test_cacheus_repeated_requests_are_hits():
    requests, pages = _trace(["a", "a", "a", "a"])
    result = run_policy(CacheusPolicy(), requests, pages, capacity=2)
    assert [e.hit for e in result.events] == [False, True, True, True]


@requires_official_source
def test_cacheus_deterministic_repeated_run():
    page_ids = [str(i % 11) for i in range(300)]
    requests, pages = _trace(page_ids)

    result_1 = run_policy(CacheusPolicy(), requests, pages, capacity=5)
    requests_2, _ = _trace(page_ids)
    result_2 = run_policy(CacheusPolicy(), requests_2, pages, capacity=5)

    assert [e.hit for e in result_1.events] == [e.hit for e in result_2.events]
    assert [e.evicted for e in result_1.events] == [e.evicted for e in result_2.events]
    assert result_1.total_misses == result_2.total_misses


@requires_official_source
def test_cacheus_no_future_information_leakage():
    # The adapter never reads request.actual_next / request.predicted_next
    # at all (it only forwards page_id and t to the official request()
    # call) -- mutating them must have zero effect on decisions.
    page_ids = [str(i % 9) for i in range(150)]

    requests_1 = [Request(t=i, page_id=p, actual_next=math.inf) for i, p in enumerate(page_ids)]
    requests_2 = [
        Request(t=i, page_id=p, actual_next=float(i + 1) if i % 3 == 0 else math.inf)
        for i, p in enumerate(page_ids)
    ]
    pages = {p: Page(page_id=p, weight=1.0) for p in set(page_ids)}

    result_1 = run_policy(CacheusPolicy(), requests_1, pages, capacity=4)
    result_2 = run_policy(CacheusPolicy(), requests_2, pages, capacity=4)

    assert [e.hit for e in result_1.events] == [e.hit for e in result_2.events]
    assert [e.evicted for e in result_1.events] == [e.evicted for e in result_2.events]


@requires_official_source
def test_cacheus_run_policy_integration_populates_diagnostics():
    page_ids = [str(i % 6) for i in range(60)]
    requests, pages = _trace(page_ids)
    result = run_policy(CacheusPolicy(), requests, pages, capacity=3)

    assert result.extra_diagnostics is not None
    assert "cacheus" in result.extra_diagnostics
    summary = result.extra_diagnostics["cacheus"]["summary"]
    for key in (
        "final_weight_srlru", "final_weight_crlfu", "final_learning_rate",
        "n_history_hits_lru", "n_history_hits_lfu", "dem_count", "nor_count",
        "official_rng_seed",
    ):
        assert key in summary
    # Official W always sums to 1.0 (adjustWeights normalizes).
    assert summary["final_weight_srlru"] == pytest.approx(
        1.0 - summary["final_weight_crlfu"], abs=1e-4
    )
    assert summary["official_rng_seed"] == OFFICIAL_RNG_SEED


@requires_official_source
def test_cacheus_registered_in_policy_registry():
    assert "cacheus" in POLICY_REGISTRY
    assert isinstance(POLICY_REGISTRY["cacheus"], CacheusPolicy)


def test_cacheus_official_source_missing_raises_clear_error(monkeypatch):
    # Simulate a fresh checkout that hasn't run the fetch script, regardless
    # of whether this session actually has the clone -- this test does not
    # need `requires_official_source` since it deliberately points the
    # loader at a nonexistent path.
    import lafc.cacheus_official_loader as loader_mod

    monkeypatch.setattr(loader_mod, "EXTERNAL_CODE_DIR", loader_mod.REPO_ROOT / "does_not_exist")
    with pytest.raises(loader_mod.CacheusOfficialSourceMissing, match="fetch_cacheus_official"):
        loader_mod.load_official_classes()


# ---------------------------------------------------------------------------
# RNG audit: the official Cacheus.__init__ hardcodes np.random.seed(123),
# consumed via bare np.random.rand()/np.random.choice() calls against
# numpy's *global* RNG state (confirmed by grepping the fetched source, not
# assumed -- see docs/cacheus_method_spec.md and docs/cacheus_provenance.md).
# These tests prove that this repository's wrapper does not depend on, and
# is not corrupted by, that global-state usage.
# ---------------------------------------------------------------------------


@requires_official_source
def test_cacheus_deterministic_despite_prior_random_state_perturbation():
    # Simulate "unrelated code ran first and left the global RNG streams in
    # an arbitrary state" -- must have zero effect on CACHEUS's own
    # decisions, because the official Cacheus.__init__ unconditionally
    # reseeds numpy's global state to 123 itself.
    requests_a, pages = _expert_disagreement_trace()
    py_random.seed(999)
    for _ in range(37):
        py_random.random()
    np.random.seed(999)
    np.random.rand(41)

    result_1 = run_policy(CacheusPolicy(), requests_a, pages, capacity=4)

    py_random.seed(42424242)
    for _ in range(101):
        py_random.random()
    np.random.seed(1)
    np.random.rand(500)

    requests_b, _ = _expert_disagreement_trace()
    result_2 = run_policy(CacheusPolicy(), requests_b, pages, capacity=4)

    assert [e.hit for e in result_1.events] == [e.hit for e in result_2.events]
    assert [e.evicted for e in result_1.events] == [e.evicted for e in result_2.events]
    assert result_1.total_misses == result_2.total_misses
    s1 = result_1.extra_diagnostics["cacheus"]["summary"]
    s2 = result_2.extra_diagnostics["cacheus"]["summary"]
    assert s1["final_weight_srlru"] == s2["final_weight_srlru"]
    assert s1["n_history_hits_lru"] == s2["n_history_hits_lru"]
    assert s1["n_history_hits_lfu"] == s2["n_history_hits_lfu"]


@requires_official_source
def test_cacheus_expert_disagreement_trace_exercises_rng_and_stays_deterministic():
    # Confirms the trace actually exercises the random expert-selection
    # path (not just the deterministic agreement branch): at least one
    # ghost-history hit occurred (proof a weight update happened, which
    # only follows a disagreement decision), and weights moved away from
    # the uniform 0.5/0.5 prior.
    requests, pages = _expert_disagreement_trace()
    result = run_policy(CacheusPolicy(), requests, pages, capacity=4)
    summary = result.extra_diagnostics["cacheus"]["summary"]

    assert summary["n_history_hits_lru"] + summary["n_history_hits_lfu"] > 0
    assert summary["final_weight_srlru"] != pytest.approx(0.5, abs=1e-6)

    requests_2, _ = _expert_disagreement_trace()
    result_2 = run_policy(CacheusPolicy(), requests_2, pages, capacity=4)
    summary_2 = result_2.extra_diagnostics["cacheus"]["summary"]
    assert summary["final_weight_srlru"] == summary_2["final_weight_srlru"]
    assert [e.evicted for e in result.events] == [e.evicted for e in result_2.events]


@requires_official_source
def test_cacheus_restores_global_numpy_state_after_run():
    # Isolation test (the reverse direction from the two tests above):
    # after CacheusPolicy has run inside run_policy(), numpy's global RNG
    # state must be exactly what it was before CACHEUS touched it, so
    # unrelated code sharing this process draws the same sequence it would
    # have drawn had CACHEUS never run.
    np.random.seed(7)
    control_draws = np.random.rand(20).copy()

    np.random.seed(7)
    requests, pages = _expert_disagreement_trace()
    run_policy(CacheusPolicy(), requests, pages, capacity=4)
    post_cacheus_draws = np.random.rand(20)

    assert np.array_equal(control_draws, post_cacheus_draws), (
        "numpy global RNG state was not restored after CacheusPolicy ran -- "
        "downstream code in this process would see different randomness "
        "than if CACHEUS had never run."
    )


def test_cacheus_official_rng_seed_is_not_configurable():
    # Documents, via the public API surface, that CacheusConfig has no seed
    # parameter: the official source hardcodes it and this repository does
    # not add one (would be an algorithm-changing patch). Fails loudly if
    # a `seed` kwarg is ever silently added without updating the docs.
    import inspect

    params = set(inspect.signature(CacheusConfig.__init__).parameters) - {"self"}
    assert "seed" not in params
    assert OFFICIAL_RNG_SEED == 123


# ---------------------------------------------------------------------------
# Upstream source integrity (section 5): commit pin, dirty-tree detection,
# and per-file hash verification against what was recorded at fetch time.
# ---------------------------------------------------------------------------


@requires_official_source
def test_cacheus_integrity_verification_passes_on_clean_fetch():
    from lafc.cacheus_official_loader import verify_official_source_integrity

    report = verify_official_source_integrity()
    assert report["resolved_commit"] == EXPECTED_COMMIT
    assert report["clean"] is True
    assert len(report["tracked_file_sha256"]) > 0


@requires_official_source
def test_cacheus_integrity_verification_detects_tampering(tmp_path):
    from lafc.cacheus_official_loader import (
        EXTERNAL_CLONE_ROOT,
        CacheusIntegrityError,
        verify_official_source_integrity,
    )

    target = EXTERNAL_CLONE_ROOT / "code" / "algs" / "cacheus.py"
    original = target.read_bytes()
    try:
        target.write_bytes(original + b"\n# tampered by test\n")
        with pytest.raises(CacheusIntegrityError, match="unexpected local changes"):
            verify_official_source_integrity()
    finally:
        target.write_bytes(original)
        # Confirm the fixture actually restores a passing state, so a bug
        # in the restore itself wouldn't be silently masked.
        verify_official_source_integrity()


def test_cacheus_integrity_verification_fails_clearly_when_clone_absent(monkeypatch):
    import lafc.cacheus_official_loader as loader_mod

    monkeypatch.setattr(loader_mod, "EXTERNAL_CLONE_ROOT", loader_mod.REPO_ROOT / "does_not_exist")
    with pytest.raises(loader_mod.CacheusIntegrityError, match="fetch_cacheus_official"):
        loader_mod.verify_official_source_integrity()


@requires_official_source
def test_cacheus_config_official_kwargs_only_includes_overrides():
    cfg = CacheusConfig()
    assert cfg.official_kwargs() == {}
    cfg2 = CacheusConfig(initial_weight=0.7, history_size=10, learning_rate=0.2)
    assert cfg2.official_kwargs() == {
        "initial_weight": 0.7, "history_size": 10, "learning_rate": 0.2,
    }
