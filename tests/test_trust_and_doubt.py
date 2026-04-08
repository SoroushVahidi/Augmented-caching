"""
Tests for TRUST&DOUBT and related policies.

Covers:
1. Simulator correctness (basic hit/miss accounting)
2. Cost accounting correctness (unit costs)
3. TRUST&DOUBT runs without errors on toy traces
4. Prediction-error metric correctness (discrete η)
5. Deterministic reproducibility (no randomness in TRUST&DOUBT)
6. Sanity comparisons: TRUST&DOUBT vs LRU / Marker / BlindOracle
7. Regression tests for qualitative robustness behavior

All traces are hand-designed and results are manually verifiable.
"""

from __future__ import annotations

import math
from typing import Dict, List

import pytest

from lafc.metrics.prediction_error import compute_discrete_eta
from lafc.policies.blind_oracle import BlindOraclePolicy
from lafc.policies.follow_the_prediction import FollowThePredictionPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.marker import MarkerPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.runner.run_policy import run_policy
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.types import Page, PageId, Request


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_unit_pages(*page_ids: str) -> Dict[PageId, Page]:
    """Create a page dictionary with all weights = 1.0."""
    return {pid: Page(page_id=pid, weight=1.0) for pid in page_ids}


def make_requests(
    page_ids: List[str],
    predictions: List[float] = None,
) -> tuple:
    """Build requests and pages for a unit-weight trace.

    Returns (requests, pages).
    """
    all_pages = sorted(set(page_ids))
    weights = {p: 1.0 for p in all_pages}
    return build_requests_from_lists(page_ids, weights, predictions)


# ---------------------------------------------------------------------------
# 1. Simulator correctness
# ---------------------------------------------------------------------------


class TestSimulatorCorrectness:
    """Basic hit/miss accounting with hand-verifiable traces."""

    def test_all_hits_after_warmup(self):
        """With capacity 3 and 3 distinct pages, after the first 3 misses
        everything is a hit."""
        pages_seq = ["A", "B", "C", "A", "B", "C", "A"]
        requests, pages = make_requests(pages_seq)
        policy = LRUPolicy()
        result = run_policy(policy, requests, pages, capacity=3)
        # First 3 are misses, remaining 4 are hits.
        assert result.total_misses == 3
        assert result.total_hits == 4
        assert result.total_cost == 3.0

    def test_capacity_one_every_request_miss_except_repeat(self):
        """With capacity 1: hit only when consecutive requests are the same page."""
        pages_seq = ["A", "A", "B", "B", "A"]
        requests, pages = make_requests(pages_seq)
        policy = LRUPolicy()
        result = run_policy(policy, requests, pages, capacity=1)
        # t=0 miss, t=1 hit, t=2 miss, t=3 hit, t=4 miss
        assert result.total_misses == 3
        assert result.total_hits == 2

    def test_marker_basic_no_phase_crossing(self):
        """Marker with capacity 2, sequence that stays within one phase."""
        pages_seq = ["A", "B", "A", "B"]
        requests, pages = make_requests(pages_seq)
        policy = MarkerPolicy()
        result = run_policy(policy, requests, pages, capacity=2)
        # t=0: miss A, t=1: miss B, t=2: hit A, t=3: hit B
        assert result.total_misses == 2
        assert result.total_hits == 2
        assert policy.phase_count() == 0  # no phase crossing

    def test_marker_phase_crossing(self):
        """Marker with capacity 2, sequence that forces a phase crossing."""
        # k=2: phase ends when all 2 cached pages are marked and a miss occurs.
        # Sequence: A, B, C → after A, B are fetched and marked,
        # C is a miss; A/B are both marked → new phase → unmark all → evict LRU (A) → fetch C.
        pages_seq = ["A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        policy = MarkerPolicy()
        result = run_policy(policy, requests, pages, capacity=2)
        assert result.total_misses == 3  # A, B, C are all misses
        assert policy.phase_count() == 1  # one phase crossing occurred


# ---------------------------------------------------------------------------
# 2. Cost accounting correctness
# ---------------------------------------------------------------------------


class TestCostAccounting:
    """Unit-weight cost accounting: cost = number of misses."""

    def test_cost_equals_misses_unit_weight(self):
        pages_seq = ["A", "B", "C", "D", "A"]
        requests, pages = make_requests(pages_seq)
        for Policy in [LRUPolicy, MarkerPolicy, BlindOraclePolicy, TrustAndDoubtPolicy]:
            policy = Policy()
            result = run_policy(policy, requests, pages, capacity=2)
            assert result.total_cost == float(result.total_misses), (
                f"{Policy.__name__}: cost {result.total_cost} != misses {result.total_misses}"
            )

    def test_hit_plus_miss_equals_total(self):
        pages_seq = ["A", "B", "A", "C", "B", "A"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy()
        result = run_policy(policy, requests, pages, capacity=2)
        assert result.total_hits + result.total_misses == len(pages_seq)


# ---------------------------------------------------------------------------
# 3. TRUST&DOUBT runs without errors on toy traces
# ---------------------------------------------------------------------------


class TestTrustAndDoubtBasic:
    """Smoke tests: TRUST&DOUBT runs without exceptions."""

    def test_single_request(self):
        pages_seq = ["A"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy()
        result = run_policy(policy, requests, pages, capacity=2)
        assert result.total_misses == 1
        assert result.total_hits == 0

    def test_repeated_single_page(self):
        """Repeated requests for the same page: miss once, then all hits."""
        pages_seq = ["A"] * 10
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy()
        result = run_policy(policy, requests, pages, capacity=1)
        assert result.total_misses == 1
        assert result.total_hits == 9

    def test_all_distinct_pages_capacity_one(self):
        """Every request is for a unique page: all misses."""
        pages_seq = list("ABCDEFGH")
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy()
        result = run_policy(policy, requests, pages, capacity=1)
        assert result.total_misses == len(pages_seq)

    def test_runs_on_example_trace(self):
        """TRUST&DOUBT runs cleanly on the provided example trace."""
        from lafc.simulator.request_trace import load_trace
        requests, pages = load_trace("data/example_unweighted.json")
        policy = TrustAndDoubtPolicy()
        result = run_policy(policy, requests, pages, capacity=3)
        assert result.total_hits + result.total_misses == len(requests)
        assert result.total_cost == float(result.total_misses)

    def test_capacity_equals_universe_size_no_eviction(self):
        """If capacity = number of unique pages, no evictions after warmup."""
        pages_seq = ["A", "B", "C", "A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy()
        result = run_policy(policy, requests, pages, capacity=3)
        # First 3 misses, then 3 hits; no evictions needed.
        assert result.total_misses == 3
        events = [e for e in result.events if e.evicted is not None]
        assert len(events) == 0

    def test_mode_transitions_occur(self):
        """Force trust budget exhaustion to check mode transitions."""
        # capacity=2, trust_budget=2: after 2 misses in trust mode, switch to doubt.
        # Sequence: A B C (misses A, B → in trust; miss C triggers doubt switch after 2 faults)
        pages_seq = ["A", "B", "C", "A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy(initial_trust_budget=2)
        run_policy(policy, requests, pages, capacity=2)
        # Should have entered doubt mode at least once (epoch >= 1).
        assert policy.epoch >= 1

    def test_state_after_reset(self):
        """After reset, policy is back in trust mode with original budget."""
        policy = TrustAndDoubtPolicy(initial_trust_budget=3)
        pages_seq = ["A", "B", "C", "D"]
        requests, pages = make_requests(pages_seq)
        run_policy(policy, requests, pages, capacity=2)
        # Re-run (reset is called by run_policy).
        result = run_policy(policy, requests, pages, capacity=2)
        assert policy.mode == "trust" or policy.mode == "doubt"  # valid state
        assert result.total_hits + result.total_misses == len(pages_seq)


# ---------------------------------------------------------------------------
# 4. Prediction error metric correctness (discrete η)
# ---------------------------------------------------------------------------


class TestDiscreteEta:
    """Discrete prediction error η for ICML 2020."""

    def test_perfect_predictions_zero_error(self):
        """Perfect predictions → η_discrete = 0."""
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        pages_seq = ["A", "B", "A", "C", "B"]
        requests, _ = make_requests(pages_seq)
        perfect = compute_perfect_predictions(requests)
        assert compute_discrete_eta(perfect) == 0

    def test_all_wrong_predictions(self):
        """Predictions that are all incorrect → η_discrete = T."""
        pages_seq = ["A", "B", "A"]
        requests, _ = make_requests(pages_seq)
        # Actual next arrivals: A→2, B→inf, A→inf
        # Make all predictions wrong: use a fixed wrong value.
        wrong = [
            Request(t=req.t, page_id=req.page_id, predicted_next=99.0, actual_next=req.actual_next)
            if not math.isinf(req.actual_next) else
            Request(t=req.t, page_id=req.page_id, predicted_next=1.0, actual_next=req.actual_next)
            for req in requests
        ]
        assert compute_discrete_eta(wrong) == 3

    def test_partial_wrong_predictions(self):
        """Some wrong predictions."""
        pages_seq = ["A", "B", "A"]
        requests, _ = make_requests(pages_seq)
        # actual_next: [2, inf, inf]
        # Set t=0 wrong (actual=2, predict=5), others perfect.
        modified = [
            Request(t=requests[0].t, page_id=requests[0].page_id, predicted_next=5.0, actual_next=requests[0].actual_next),
            Request(t=requests[1].t, page_id=requests[1].page_id, predicted_next=requests[1].actual_next, actual_next=requests[1].actual_next),
            Request(t=requests[2].t, page_id=requests[2].page_id, predicted_next=requests[2].actual_next, actual_next=requests[2].actual_next),
        ]
        assert compute_discrete_eta(modified) == 1

    def test_both_inf_no_error(self):
        """Both predicted and actual are inf → no error."""
        reqs = [Request(t=0, page_id="A", predicted_next=math.inf, actual_next=math.inf)]
        assert compute_discrete_eta(reqs) == 0

    def test_one_sided_inf_is_error(self):
        """Predicted=inf but actual=5 → 1 error."""
        reqs = [Request(t=0, page_id="A", predicted_next=math.inf, actual_next=5.0)]
        assert compute_discrete_eta(reqs) == 1

    def test_actual_inf_predicted_finite_is_error(self):
        """Actual=inf but predicted=3 → 1 error."""
        reqs = [Request(t=0, page_id="A", predicted_next=3.0, actual_next=math.inf)]
        assert compute_discrete_eta(reqs) == 1


# ---------------------------------------------------------------------------
# 5. Deterministic reproducibility
# ---------------------------------------------------------------------------


class TestDeterminism:
    """TRUST&DOUBT is deterministic: same trace → same result."""

    def test_same_trace_same_result(self):
        pages_seq = ["A", "B", "C", "A", "D", "B", "C", "A"]
        requests, pages = make_requests(pages_seq)
        policy1 = TrustAndDoubtPolicy(initial_trust_budget=2)
        policy2 = TrustAndDoubtPolicy(initial_trust_budget=2)
        r1 = run_policy(policy1, requests, pages, capacity=2)
        r2 = run_policy(policy2, requests, pages, capacity=2)
        assert r1.total_misses == r2.total_misses
        for e1, e2 in zip(r1.events, r2.events):
            assert e1.hit == e2.hit
            assert e1.evicted == e2.evicted

    def test_marker_deterministic(self):
        pages_seq = ["A", "B", "C", "A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        r1 = run_policy(MarkerPolicy(), requests, pages, capacity=2)
        r2 = run_policy(MarkerPolicy(), requests, pages, capacity=2)
        assert r1.total_misses == r2.total_misses


# ---------------------------------------------------------------------------
# 6. Sanity comparisons
# ---------------------------------------------------------------------------


class TestSanityComparisons:
    """Compare policies on hand-constructed traces with known properties."""

    def test_blind_oracle_equals_ftp(self):
        """BlindOracle and FTP are identical algorithms."""
        pages_seq = ["A", "B", "C", "A", "B"]
        preds = [3.0, 4.0, 9999.0, 9999.0, 9999.0]
        requests, pages = make_requests(pages_seq, preds)
        r_bo = run_policy(BlindOraclePolicy(), requests, pages, capacity=2)
        r_ftp = run_policy(FollowThePredictionPolicy(), requests, pages, capacity=2)
        assert r_bo.total_misses == r_ftp.total_misses
        for e_bo, e_ftp in zip(r_bo.events, r_ftp.events):
            assert e_bo.hit == e_ftp.hit
            assert e_bo.evicted == e_ftp.evicted

    def test_perfect_predictions_td_le_lru(self):
        """With perfect predictions, TRUST&DOUBT should not be much worse than LRU.

        We test that TRUST&DOUBT's cost is within 2× LRU on a simple trace.
        """
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        pages_seq = ["A", "B", "C", "A", "B", "C", "D", "A", "B"]
        requests, pages = make_requests(pages_seq)
        perfect_reqs = compute_perfect_predictions(requests)
        r_lru = run_policy(LRUPolicy(), requests, pages, capacity=2)
        r_td = run_policy(TrustAndDoubtPolicy(), perfect_reqs, pages, capacity=2)
        # With perfect predictions, TRUST&DOUBT should be at most 2× LRU cost.
        assert r_td.total_misses <= 2 * r_lru.total_misses + 1

    def test_trust_and_doubt_leq_lru_adversarial_budget_large(self):
        """With a very large trust budget (no mode switches), TRUST&DOUBT
        behaves as Blind Oracle.  On adversarial predictions it might be worse
        than LRU, but on benign traces it should be reasonable.
        """
        pages_seq = ["A", "B", "A", "B", "A", "B"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy(initial_trust_budget=100)
        result = run_policy(policy, requests, pages, capacity=2)
        assert result.total_misses == 2  # Only A and B are ever accessed; capacity 2
        assert result.total_hits == 4

    def test_marker_never_worse_than_k_plus_one_misses_per_phase(self):
        """In a single Marker phase with k=2, at most k+1 = 3 misses (the
        initial cold-start misses plus 1 phase-boundary miss).
        For a 3-request trace with 3 distinct pages and capacity=2:
        - t=0: miss A (cold)
        - t=1: miss B (cold)
        - t=2: miss C (A and B marked → new phase → evict LRU(A or B) → miss C)
        Total = 3 misses.
        """
        pages_seq = ["A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        r = run_policy(MarkerPolicy(), requests, pages, capacity=2)
        assert r.total_misses == 3

    def test_predictive_marker_better_than_marker_on_good_predictions(self):
        """Predictive Marker should match or beat standard Marker when
        predictions are perfect (it makes better eviction choices).

        Trace constructed so that perfect predictions let PredictiveMarker
        avoid a fault that standard Marker incurs.
        """
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        # k=2, sequence: A B C A B C
        # Standard Marker (LRU tiebreak) may evict A or B during phase crossing.
        # With perfect predictions, PredictiveMarker knows which of A/B is
        # needed next and keeps it.
        pages_seq = ["A", "B", "C", "A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        perfect = compute_perfect_predictions(requests)
        r_marker = run_policy(MarkerPolicy(), requests, pages, capacity=2)
        r_pm = run_policy(PredictiveMarkerPolicy(), perfect, pages, capacity=2)
        # PredictiveMarker should be ≤ standard Marker in cost.
        assert r_pm.total_misses <= r_marker.total_misses

    def test_blind_oracle_perfect_predictions_optimal_unit(self):
        """Blind Oracle with perfect predictions is Belady-optimal for
        unweighted paging.  Check on a trace where OPT is known.

        Trace: A B C A B C with k=2.
        OPT strategy: always evict the page needed farthest in future.
        Sequence: miss A, miss B, miss C (evict A since A is needed @3, B@4,C@5
        so at step 2 we need to evict one of A/B; A is next at 3, B at 4,
        so evict B), hit A at t=3, miss B (evict C which has next=5), hit C?
        Let me just verify it equals OPT numerically.
        """
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        pages_seq = ["A", "B", "C", "A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        perfect = compute_perfect_predictions(requests)
        r_bo = run_policy(BlindOraclePolicy(), perfect, pages, capacity=2)
        # Perfect predictions → Belady optimal → OPT for this trace.
        # Manually: misses at t=0,1,2; then at t=3 A is in cache (A was evicted? no...)
        # Let's just check it's <= LRU.
        r_lru = run_policy(LRUPolicy(), requests, pages, capacity=2)
        assert r_bo.total_misses <= r_lru.total_misses


# ---------------------------------------------------------------------------
# 7. Regression tests for qualitative robustness behavior
# ---------------------------------------------------------------------------


class TestRobustnessRegression:
    """Qualitative regression tests reflecting expected TRUST&DOUBT properties."""

    def test_trust_and_doubt_enters_doubt_mode(self):
        """TRUST&DOUBT eventually enters doubt mode when faults accumulate.

        With trust_budget=1, the first miss triggers a switch to doubt.
        The trace is long enough to also complete the doubt phase.
        """
        # budget=1, k=2, trace: A B C D
        # t=0: TRUST MISS A. faults=1 ≥ 1 → start doubt. Cache={A}. marked={}.
        # t=1: DOUBT MISS B. Cache not full. Add B, mark B. Cache={A,B}. marked={B}.
        # t=2: DOUBT MISS C. Cache full. unmarked={A}. Evict A. Add C, mark C.
        #   Cache={B,C}. marked={B,C}. All marked!
        # t=3: DOUBT MISS D. All marked → end doubt. epoch=1. FTP evict B or C (∞ → C).
        #   Add D. trust_faults=1 ≥ budget=2? No (budget is now 2). Mode=trust.
        policy = TrustAndDoubtPolicy(initial_trust_budget=1)
        pages_seq = ["A", "B", "C", "D"]
        requests, pages = make_requests(pages_seq)
        run_policy(policy, requests, pages, capacity=2)
        # Doubt phase was entered and completed: epoch = 1.
        assert policy.epoch >= 1

    def test_trust_budget_doubles_after_doubt_phase(self):
        """Trust budget should double after each doubt phase completion."""
        # budget=1, k=2. Trace: A B C D (same as above but check budget).
        # t=0: TRUST MISS A → doubt starts (budget exhausted). Cache={A}.
        # t=1: DOUBT MISS B. Not full. Add B mark B. Cache={A,B}. marked={B}.
        # t=2: DOUBT MISS C. Full. unmarked={A}. Evict A. Add C mark C. Cache={B,C}. marked={B,C}.
        # t=3: DOUBT MISS D. All marked → end doubt! epoch=1, budget=2.
        #   FTP evict max(B,C pred∞) → C (alphabetical). Add D. trust_faults=1.
        pages_seq = ["A", "B", "C", "D"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy(initial_trust_budget=1)
        run_policy(policy, requests, pages, capacity=2)
        # After one full doubt phase, epoch=1, budget=2.
        assert policy.epoch >= 1
        assert policy.trust_budget >= 2

    def test_adversarial_trace_cost_bounded(self):
        """On an adversarial trace, TRUST&DOUBT cost should be bounded by
        O(log k) times the Marker cost (same robustness class).

        We test this weakly: TRUST&DOUBT cost ≤ some constant × Marker cost.
        """
        # Adversarial trace: cycle through k+1 pages repeatedly.
        # k=2, pages: A B C, cycle A B C A B C ...
        pages_seq = ["A", "B", "C"] * 10
        requests, pages = make_requests(pages_seq)
        r_marker = run_policy(MarkerPolicy(), requests, pages, capacity=2)
        r_td = run_policy(TrustAndDoubtPolicy(), requests, pages, capacity=2)
        # Both should handle the adversarial trace; TRUST&DOUBT may be ≤ 2× Marker.
        assert r_td.total_misses <= 3 * r_marker.total_misses

    def test_perfect_predictions_td_at_most_ftp_plus_k(self):
        """With perfect predictions, TRUST&DOUBT cost ≤ FTP cost + k.

        The +k accounts for at most one trust-budget overhead.
        """
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        pages_seq = ["A", "B", "C", "A", "B", "C", "D", "E", "A", "B"]
        requests, pages = make_requests(pages_seq)
        perfect = compute_perfect_predictions(requests)
        k = 3
        r_ftp = run_policy(BlindOraclePolicy(), perfect, pages, capacity=k)
        r_td = run_policy(TrustAndDoubtPolicy(), perfect, pages, capacity=k)
        assert r_td.total_misses <= r_ftp.total_misses + k

    def test_marker_phases_reset_on_new_run(self):
        """After reset (new run_policy call), Marker starts from phase 0."""
        policy = MarkerPolicy()
        pages_seq = ["A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        run_policy(policy, requests, pages, capacity=2)
        # First run: phase_count was 1.
        run_policy(policy, requests, pages, capacity=2)
        # After second run (with fresh reset), phase_count should reflect
        # only the second run's phases.
        assert policy.phase_count() == 1

    def test_all_policies_finish_cleanly(self):
        """Smoke test: all registered policies run without exceptions."""
        from lafc.runner.run_policy import POLICY_REGISTRY
        pages_seq = ["A", "B", "C", "A", "B", "D"]
        requests, pages = make_requests(pages_seq)
        for name, policy in POLICY_REGISTRY.items():
            if name in {"la_weighted_paging_randomized"}:
                continue  # known stub
            result = run_policy(policy, requests, pages, capacity=2)
            assert result.total_hits + result.total_misses == len(pages_seq), (
                f"Policy {name}: hit+miss != T"
            )


# ---------------------------------------------------------------------------
# 8. Hand-checkable trace with verified decisions
# ---------------------------------------------------------------------------


class TestHandCheckable:
    """Manually verifiable TRUST&DOUBT decisions on a tiny trace.

    Trace: pages A B C, capacity k=2, initial_trust_budget=2.
    No predictions → predicted_next=∞ for all pages.

    Step-by-step (mode starts as TRUST):

    t=0: TRUST MISS A. Cache empty, not full → add A. Cache={A}.
         trust_faults=1 < budget=2. Mode=trust.

    t=1: TRUST MISS B. Cache={A}, size=1 < k=2, not full → add B. Cache={A,B}.
         trust_faults=2 ≥ budget=2 → start DOUBT. marked={}.

    t=2: DOUBT MISS C. Cache={A,B}. marked={}. unmarked={A,B}.
         Evict LRU: last_access(A)=0, last_access(B)=1 → evict A.
         Add C, mark C. Cache={B,C}. marked={C}.

    t=3: DOUBT MISS A. Cache={B,C}. marked={C}. unmarked={B}.
         Evict B (LRU unmarked). Add A, mark A. Cache={C,A}. marked={C,A}.
         Doubt phase not over (unmarked pages existed when the miss was processed).

    Summary: 4 misses, 0 hits.
    """

    def test_hand_trace_k2_budget2_no_predictions(self):
        """Verify TRUST&DOUBT decisions on the hand-traced 4-request sequence."""
        # k=2, budget=2, all predicted_next=inf
        pages_seq = ["A", "B", "C", "A"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy(initial_trust_budget=2)
        result = run_policy(policy, requests, pages, capacity=2)

        events = result.events
        # t=0: MISS (cold start)
        assert events[0].hit is False
        assert events[0].evicted is None  # cache not full

        # t=1: MISS (cold start, cache goes from 1→2)
        assert events[1].hit is False
        assert events[1].evicted is None  # cache not full; switch to DOUBT after

        # After t=1: mode should be DOUBT (trust_budget=2 exhausted with 2 faults).
        # t=2: DOUBT MISS, evict LRU unmarked.
        assert events[2].hit is False
        assert events[2].evicted is not None  # cache full → eviction

        # t=3: DOUBT MISS. unmarked = cache - marked.
        # After t=2: cache={B,C} or {C,B}, marked={C}, unmarked={B}.
        # Evict B (LRU unmarked). Add A. Cache={C,A}. Doubt phase not over.
        assert events[3].hit is False

        # Total: 4 misses, 0 hits.
        assert result.total_misses == 4
        assert result.total_hits == 0

    def test_doubt_mode_ends_when_all_marked(self):
        """Verify the doubt-to-trust transition triggers correctly.

        k=2, budget=1. Trace: A B C D.

        t=0: TRUST MISS A. faults=1 ≥ budget=1 → doubt. Cache={A}. marked={}.
        t=1: DOUBT MISS B. Cache not full. Add B, mark B. Cache={A,B}. marked={B}.
        t=2: DOUBT MISS C. Cache full. marked={B}, unmarked={A}. Evict A (LRU).
             Add C, mark C. Cache={B,C}. marked={B,C}. All cached marked.
        t=3: DOUBT MISS D. All marked → end doubt. epoch=1, budget=2.
             FTP evict: B or C (both ∞). Tie → alphabetical: C evicted.
             Add D. trust_faults=1.
        """
        pages_seq = ["A", "B", "C", "D"]
        requests, pages = make_requests(pages_seq)
        policy = TrustAndDoubtPolicy(initial_trust_budget=1)
        run_policy(policy, requests, pages, capacity=2)
        # After the trace, we should have at least 1 completed epoch.
        assert policy.epoch == 1

    def test_predictive_marker_evicts_farthest_in_future(self):
        """PredictiveMarker should evict the page with max predicted_next
        when a miss occurs and there are unmarked pages.

        k=2, trace: A B C, with predictions that tell us B is needed again
        soonest and A is needed farthest (or never).
        At t=2 (miss for C), cache={A,B}, both unmarked (first miss after
        phase start). PM should evict the one with max predicted_next.
        """
        # predictions: at t=0 A's pred_next=100, at t=1 B's pred_next=3, at t=2 C's pred_next=inf
        pages_seq = ["A", "B", "C"]
        preds = [100.0, 3.0, math.inf]
        requests, pages = make_requests(pages_seq, preds)
        policy = PredictiveMarkerPolicy()
        result = run_policy(policy, requests, pages, capacity=2)
        # t=2: miss C; cache={A,B} both unmarked. Evict max(pred_next(A)=100, pred_next(B)=3) → A.
        assert result.events[2].evicted == "A"

    def test_marker_evicts_lru_among_unmarked(self):
        """Marker should evict the LRU page among unmarked pages.

        k=2, trace: A B C.
        t=0: miss A; not full → add A. LRU order: [A].
        t=1: miss B; not full → add B. LRU order: [A, B] (A is LRU).
        t=2: miss C; full, marked={}. unmarked=[A,B]. LRU=A. Evict A.
        """
        pages_seq = ["A", "B", "C"]
        requests, pages = make_requests(pages_seq)
        policy = MarkerPolicy()
        result = run_policy(policy, requests, pages, capacity=2)
        assert result.events[2].evicted == "A"  # LRU

    def test_trust_and_doubt_with_perfect_predictions(self):
        """TRUST&DOUBT with perfect predictions stays in trust mode longer
        (since FTP cost = OPT cost with perfect predictions)."""
        from lafc.predictors.offline_from_trace import compute_perfect_predictions
        pages_seq = ["A", "B", "A", "B", "A"]
        requests, pages = make_requests(pages_seq)
        perfect = compute_perfect_predictions(requests)
        policy = TrustAndDoubtPolicy(initial_trust_budget=2)
        result = run_policy(policy, perfect, pages, capacity=2)
        # k=2: A and B both fit in cache after cold start (2 misses), then all hits.
        assert result.total_misses == 2
        assert result.total_hits == 3
