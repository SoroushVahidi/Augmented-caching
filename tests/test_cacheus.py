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

import pytest

from lafc.cacheus_official_loader import EXTERNAL_CODE_DIR, load_official_classes
from lafc.policies.cacheus import CacheusConfig, CacheusPolicy
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
    ):
        assert key in summary
    # Official W always sums to 1.0 (adjustWeights normalizes).
    assert summary["final_weight_srlru"] == pytest.approx(
        1.0 - summary["final_weight_crlfu"], abs=1e-4
    )


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


@requires_official_source
def test_cacheus_config_official_kwargs_only_includes_overrides():
    cfg = CacheusConfig()
    assert cfg.official_kwargs() == {}
    cfg2 = CacheusConfig(initial_weight=0.7, history_size=10, learning_rate=0.2)
    assert cfg2.official_kwargs() == {
        "initial_weight": 0.7, "history_size": 10, "learning_rate": 0.2,
    }
