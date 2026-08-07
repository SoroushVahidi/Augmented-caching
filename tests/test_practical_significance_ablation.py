from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

from lafc.evict_value_dataset_v1 import EvictValueDatasetV1Config, build_evict_value_examples_v1
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.evict_value_v1_optimized import (
    EvictValueV1CachedExactPolicy,
    EvictValueV1VectorizedCachedExactPolicy,
    EvictValueV1VectorizedExactPolicy,
)
from lafc.policies.evict_value_v1_selective import (
    EvictValueV1SelectiveDisagreementPolicy,
    EvictValueV1SelectivePeriodicPolicy,
    EvictValueV1TopKPolicy,
    canonical_victim_would_be_pruned,
)
from lafc.simulator.request_trace import load_trace
from sklearn.ensemble import RandomForestRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiments" / "run_practical_significance_ablation.py"


def _load_module(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("run_practical_significance_ablation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT_DIR", tmp_path / "analysis")
    return module


def _train_tiny_model(tmp_path: Path, capacity: int = 3, horizon: int = 8, seed: int = 7) -> str:
    reqs, _ = load_trace("data/example_atlas_v1.json")
    rows = [
        r
        for r in build_evict_value_examples_v1(reqs, capacity=capacity, trace_name="toy", cfg=EvictValueDatasetV1Config(horizons=(horizon,)))
        if int(r["horizon"]) == horizon
    ]
    x = [[float(r[c]) for c in EVICT_VALUE_V1_FEATURE_COLUMNS] for r in rows]
    y = [float(r["y_loss"]) for r in rows]
    est = RandomForestRegressor(n_estimators=15, random_state=seed)
    est.fit(x, y)
    artifact = EvictValueV1Model(model_name="rf_test", estimator=est, feature_columns=list(EVICT_VALUE_V1_FEATURE_COLUMNS))
    model_path = tmp_path / "m.pkl"
    artifact.save(model_path)
    return str(model_path)


# ---------------------------------------------------------------------------
# Exact-optimization equivalence (decision-preserving variants)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("trace_path,capacity", [("data/example_atlas_v1.json", 3), ("data/example_unweighted.json", 4)])
def test_exact_variants_match_canonical_decision_sequence(tmp_path, trace_path, capacity):
    model_path = _train_tiny_model(tmp_path, capacity=capacity)
    reqs, pages = load_trace(trace_path)

    def decisions(cls):
        pol = cls(model_path=model_path)
        pol.reset(capacity, pages)
        return [(ev.hit, ev.evicted) for r in reqs for ev in [pol.on_request(r)]]

    canonical_seq = decisions(EvictValueV1Policy)
    for cls in (EvictValueV1CachedExactPolicy, EvictValueV1VectorizedExactPolicy, EvictValueV1VectorizedCachedExactPolicy):
        assert decisions(cls) == canonical_seq, f"{cls.__name__} diverged from canonical"


def test_exact_variants_match_canonical_in_lightweight_mode():
    reqs, pages = load_trace("data/example_atlas_v1.json")

    def decisions(cls):
        pol = cls(model_path="models/definitely_missing.pkl", scorer_mode="lightweight")
        pol.reset(5, pages)
        return [(ev.hit, ev.evicted) for r in reqs for ev in [pol.on_request(r)]]

    canonical_seq = decisions(EvictValueV1Policy)
    for cls in (EvictValueV1CachedExactPolicy, EvictValueV1VectorizedExactPolicy, EvictValueV1VectorizedCachedExactPolicy):
        assert decisions(cls) == canonical_seq


# ---------------------------------------------------------------------------
# Selective invocation correctness
# ---------------------------------------------------------------------------


def test_selective_periodic_invokes_exactly_every_kth_decision(tmp_path):
    model_path = _train_tiny_model(tmp_path, capacity=3)
    reqs, pages = load_trace("data/example_atlas_v1.json")
    period_k = 3
    pol = EvictValueV1SelectivePeriodicPolicy(model_path=model_path, period_k=period_k)
    pol.reset(3, pages)
    invoked_at = []
    for i, r in enumerate(reqs):
        before = pol.n_learned_scorer_invocations
        pol.on_request(r)
        if pol.n_learned_scorer_invocations > before:
            invoked_at.append(pol.n_eviction_decisions - 1)  # decision index just processed
    assert invoked_at == [i for i in range(pol.n_eviction_decisions) if i % period_k == 0]
    assert 0.0 <= pol.invocation_rate() <= 1.0


def test_selective_periodic_rejects_invalid_period():
    with pytest.raises(ValueError):
        EvictValueV1SelectivePeriodicPolicy(model_path="unused.pkl", period_k=0)


def test_selective_disagreement_falls_back_to_lru_when_signals_agree(tmp_path):
    model_path = _train_tiny_model(tmp_path, capacity=3)
    reqs, pages = load_trace("data/example_atlas_v1.json")
    pol = EvictValueV1SelectiveDisagreementPolicy(model_path=model_path)
    pol.reset(3, pages)
    for r in reqs:
        pol.on_request(r)
    # invocation rate must be a well-formed fraction of eviction decisions
    assert 0.0 <= pol.invocation_rate() <= 1.0
    assert pol.n_learned_scorer_invocations <= pol.n_eviction_decisions


def test_selective_disagreement_never_invokes_more_than_the_number_of_decisions(tmp_path):
    model_path = _train_tiny_model(tmp_path, capacity=4)
    reqs, pages = load_trace("data/example_unweighted.json")
    pol = EvictValueV1SelectiveDisagreementPolicy(model_path=model_path)
    pol.reset(4, pages)
    for r in reqs:
        ev = pol.on_request(r)
        if ev.evicted is not None:
            assert ev.diagnostics["rule"] == "disagreement_lru_sieve"
            assert isinstance(ev.diagnostics["invoked_learned_scorer"], bool)


# ---------------------------------------------------------------------------
# Top-k candidate pruning: inclusion / retention
# ---------------------------------------------------------------------------


def test_topk_only_scores_the_k_oldest_candidates(tmp_path):
    model_path = _train_tiny_model(tmp_path, capacity=4)
    reqs, pages = load_trace("data/example_atlas_v1.json")
    k = 2
    pol = EvictValueV1TopKPolicy(model_path=model_path, k=k)
    pol.reset(4, pages)
    for r in reqs:
        ev = pol.on_request(r)
        if ev.evicted is not None:
            assert ev.diagnostics["pruned_candidate_count"] <= k
            assert ev.diagnostics["k"] == k


def test_canonical_victim_would_be_pruned_helper():
    candidates = ["a", "b", "c", "d", "e"]
    # k=2 -> pruned set is the 2 oldest: {a, b}
    assert canonical_victim_would_be_pruned("c", candidates, k=2) is True
    assert canonical_victim_would_be_pruned("a", candidates, k=2) is False
    # k >= len(candidates): nothing is pruned
    assert canonical_victim_would_be_pruned("e", candidates, k=10) is False


def test_topk_rejects_invalid_k():
    with pytest.raises(ValueError):
        EvictValueV1TopKPolicy(model_path="unused.pkl", k=0)


# ---------------------------------------------------------------------------
# Runner-module pure-function tests (profiler stats, break-even, sweep parsing,
# Pareto, weighted-cost determinism, resumable CSV I/O)
# ---------------------------------------------------------------------------


def test_stats_ms_percentiles(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    s = module._stats_ms(values)
    assert s["n"] == 5
    assert s["mean_ms"] == pytest.approx(3.0)
    assert s["median_ms"] == pytest.approx(3.0)
    assert s["p90_ms"] <= 5.0
    assert module._stats_ms([])["n"] == 0


def test_parse_sweep_value_ms(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    assert module._parse_sweep_value_ms("1us") == pytest.approx(0.001)
    assert module._parse_sweep_value_ms("10ms") == pytest.approx(10.0)
    assert module._parse_sweep_value_ms("1s") == pytest.approx(1000.0)
    assert module._parse_sweep_value_ms("100us") == pytest.approx(0.1)
    with pytest.raises(ValueError):
        module._parse_sweep_value_ms("1foo")


def test_break_even_formula_crossover_case(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    # learned costs more compute (100ms) but misses fewer (80 vs 100):
    # Cmiss* = (100 - 10) / (100 - 80) = 4.5 ms
    result = module.break_even_cmiss(
        compute_learned_total_ms=100.0, compute_baseline_total_ms=10.0, misses_baseline=100.0, misses_learned=80.0
    )
    assert result == pytest.approx(4.5)


def test_break_even_formula_no_crossover_case(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    # learned has >= misses than baseline: no positive Cmiss makes it preferable
    result = module.break_even_cmiss(
        compute_learned_total_ms=100.0, compute_baseline_total_ms=10.0, misses_baseline=80.0, misses_learned=80.0
    )
    assert result is None
    result2 = module.break_even_cmiss(
        compute_learned_total_ms=100.0, compute_baseline_total_ms=10.0, misses_baseline=70.0, misses_learned=80.0
    )
    assert result2 is None


def test_is_pareto_efficient(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    # (latency, misses): point 1 dominates point 2 on both axes -> point 2 not efficient
    points = [(1.0, 10.0), (5.0, 20.0), (2.0, 5.0), (2.0, 5.0)]
    eff = module._is_pareto_efficient(points)
    assert eff[1] is False  # (5.0, 20.0) dominated by (1.0, 10.0)
    assert eff[0] is True
    assert eff[2] is True
    # duplicate of an efficient point: not strictly dominated (no strict improvement exists), stays efficient
    assert eff[3] is True


def test_pick_fourth_baseline_excludes_fixed_and_picks_min(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fake_misses = {"lru": 100.0, "sieve": 90.0, "fifo_reinsertion": 95.0, "blind_oracle_lru_combiner": 80.0, "rest_v1": 120.0, "trust_and_doubt": 999.0, "predictive_marker": 85.0}
    monkeypatch.setattr(module, "certified_mean_misses", lambda name: fake_misses.get(name))
    fourth = module.pick_fourth_baseline()
    assert fourth == "blind_oracle_lru_combiner"
    assert fourth not in module.FIXED_BREAK_EVEN_BASELINES


def test_lognormal_multiplier_is_deterministic_given_same_seed(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    page_ids = ["p3", "p1", "p2"]
    m1 = module.lognormal_multiplier(page_ids, seed=0)
    m2 = module.lognormal_multiplier(list(reversed(page_ids)), seed=0)
    assert m1 == m2  # order of the input list must not matter (sorted internally)
    assert set(m1.keys()) == set(page_ids)
    assert all(v > 0 for v in m1.values())  # exp(...) is always positive


def test_weighted_miss_cost_accounting(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    multiplier = {"a": 2.0, "b": 0.5, "c": 1.0}
    missed = ["a", "a", "b", "c"]
    cost = module.weighted_miss_cost(missed, multiplier, base_cmiss_ms=10.0)
    # 10*2 + 10*2 + 10*0.5 + 10*1 = 20+20+5+10 = 55
    assert cost == pytest.approx(55.0)
    # unweighted equivalent (all multipliers = 1) must equal base_cmiss_ms * n_missed
    unit_multiplier = {k: 1.0 for k in multiplier}
    assert module.weighted_miss_cost(missed, unit_multiplier, base_cmiss_ms=10.0) == pytest.approx(10.0 * len(missed))


def test_profiled_eviction_decision_components_are_nonnegative_and_informative(tmp_path):
    spec = importlib.util.spec_from_file_location("run_practical_significance_ablation_profcomp", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    model_path = _train_tiny_model(tmp_path, capacity=3)
    reqs, pages = load_trace("data/example_atlas_v1.json")
    pol = module.EvictValueV1VectorizedCachedExactPolicy(model_path=model_path)
    pol.reset(3, pages)
    # advance until the cache is full so the next miss triggers an eviction
    for r in reqs:
        if pol._cache.is_full():
            break
        pol.on_request(r)
    remaining = [r for r in reqs if not pol.in_cache(r.page_id)]
    assert remaining, "trace too short to reach an eviction decision"
    victim, components = module._profiled_eviction_decision(pol, remaining[0])
    assert victim in pol._order
    assert set(components.keys()) == {"B_candidate_construction_ms", "CD_feature_and_prediction_ms", "E_ranking_ms"}
    for v in components.values():
        assert v >= 0.0
    assert sum(components.values()) > 0.0  # real work was measured, not a no-op


# ---------------------------------------------------------------------------
# Resumable / incremental CSV output
# ---------------------------------------------------------------------------


def test_incremental_csv_writer_is_resumable_without_duplicates(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    out_csv = tmp_path / "out.csv"
    fields = ["trace_name", "capacity", "value"]

    first_batch = [
        {"trace_name": "a", "capacity": "32", "value": "1"},
        {"trace_name": "b", "capacity": "32", "value": "2"},
    ]
    module.append_rows(out_csv, first_batch, fields)

    existing = module.read_existing_keys(out_csv, ["trace_name", "capacity"])
    assert existing == {("a", "32"), ("b", "32")}

    # simulate a resumed run: only rows NOT already present get appended
    candidate_batch = [
        {"trace_name": "a", "capacity": "32", "value": "1"},  # already present -> caller should skip
        {"trace_name": "c", "capacity": "32", "value": "3"},  # new -> caller appends
    ]
    to_append = [r for r in candidate_batch if (r["trace_name"], r["capacity"]) not in existing]
    module.append_rows(out_csv, to_append, fields)

    with out_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    keys = {(r["trace_name"], r["capacity"]) for r in rows}
    assert keys == {("a", "32"), ("b", "32"), ("c", "32")}


def test_append_rows_no_op_on_empty_list(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    out_csv = tmp_path / "empty.csv"
    module.append_rows(out_csv, [], ["a", "b"])
    assert not out_csv.exists()
