"""Causal closed-loop replay using the existing policy state transition."""
import csv
import resource
from time import perf_counter_ns

import joblib
import numpy as np
from numba import njit
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import torch

from data import END, HISTORY, PROTOCOL, REPO, historical, historical_misses, load_cell, matrix, sha256, trace
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.sieve import SievePolicy
from metrics import oldest_within
from models import load, predict
from training import fold_output, reference_estimator, sklearn_predict, write_json

CLASSICAL = {"lru": LRUPolicy, "sieve": SievePolicy, "fifo_reinsertion": FIFOReinsertionPolicy}
FAIRNESS = REPO / "analysis/reviewer_fairness"
SELECTED = REPO / "analysis/reviewer_fairness_cross_family_v1/evict_value_v1_final_42_20260810/policy_comparison.csv"


@njit(cache=True, boundscheck=True)
def tree_scores(x, roots, feature, threshold, left, right, value, intercept, divisor):
    result = np.empty(len(x), dtype=np.float64)
    for i in range(len(x)):
        total = intercept
        for root in roots:
            node = root
            while left[node] >= 0:
                if x[i, feature[node]] <= threshold[node]:
                    node = left[node]
                else:
                    node = right[node]
            total += value[node]
        result[i] = total / divisor
    return result


def cpu_estimator(estimator, sample):
    """Pack fitted numeric sklearn trees once. Fail closed on batch prediction drift."""
    if isinstance(estimator, HistGradientBoostingRegressor):
        if estimator.is_categorical_ is not None and estimator.is_categorical_.any():
            raise ValueError("Categorical HGB trees are outside the matched protocol")
        trees = []
        for (tree,) in estimator._predictors:
            n = tree.nodes
            leaf = n["is_leaf"].astype(bool)
            trees.append((n["feature_idx"], n["num_threshold"], np.where(leaf, -1, n["left"].astype(np.int64)),
                          n["right"].astype(np.int64), n["value"]))
        intercept, divisor, dtype = float(estimator._baseline_prediction.item()), 1, np.float64
    elif isinstance(estimator, RandomForestRegressor):
        trees = [(t.tree_.feature, t.tree_.threshold, t.tree_.children_left,
                  t.tree_.children_right, t.tree_.value[:, 0, 0]) for t in estimator.estimators_]
        intercept, divisor, dtype = 0.0, len(trees), np.float32
    else:
        return lambda x: sklearn_predict(estimator, x)[1][0]
    roots, features, thresholds, left, right, values = [], [], [], [], [], []
    offset = 0
    for f, threshold, l, r, v in trees:
        roots.append(offset)
        features.append(f)
        thresholds.append(threshold)
        left.append(np.where(l < 0, -1, l + offset))
        right.append(r + offset)
        values.append(v)
        offset += len(f)
    args = (np.array(roots, dtype=np.int64), np.concatenate(features).astype(np.int64),
            np.concatenate(thresholds), np.concatenate(left).astype(np.int64),
            np.concatenate(right).astype(np.int64), np.concatenate(values), intercept, divisor)

    def score(features):
        return tree_scores(np.ascontiguousarray(features[0], dtype=dtype), *args)

    for group in sample:
        expected = sklearn_predict(estimator, group[None])[1][0]
        actual = score(group[None])
        np.testing.assert_array_equal(actual, expected)
    return score


class BatchPolicy(EvictValueV1Policy):
    def __init__(self, predictor, margin=0.0):
        super().__init__(scorer_mode="lightweight", history_window=64)
        self.predictor = predictor
        self.margin = margin
        self.score_ns = 0
        self.victim_ns = 0

    def _choose_victim(self, request):
        started = perf_counter_ns()
        candidates = list(self._order)
        x = matrix(self, request, candidates)
        before_score = perf_counter_ns()
        scores = np.asarray(self.predictor(x[None]))
        after_score = perf_counter_ns()
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("Expected one finite score per eviction candidate")
        index = oldest_within(scores, self.margin)
        self.score_ns += after_score - before_score
        self.victim_ns += perf_counter_ns() - started
        return candidates[index], {}


def run_policy(policy, requests, pages, capacity, history, end):
    policy.reset(capacity, pages)
    for request in requests[:history]:
        policy.on_request(request)
    if isinstance(policy, BatchPolicy):
        policy.score_ns = 0
        policy.victim_ns = 0
    hits = evictions = 0
    scored = requests[history:end]
    start = perf_counter_ns()
    for request in scored:
        event = policy.on_request(request)
        hits += int(event.hit)
        evictions += int(event.evicted is not None)
    duration = perf_counter_ns() - start
    misses = len(scored) - hits
    return dict(hits=hits, misses=misses, evictions=evictions, requests=len(scored), miss_ratio=misses / len(scored),
                replay_seconds=duration / 1e9, mean_us_per_request=duration / len(scored) / 1e3,
                mean_us_per_scoring_call=policy.score_ns / evictions / 1e3
                if isinstance(policy, BatchPolicy) and evictions else None,
                mean_us_per_victim_selection=policy.victim_ns / evictions / 1e3
                if isinstance(policy, BatchPolicy) and evictions else None,
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run_replay(out, family, horizon, capacity, settings, data_root, model_root):
    dest = fold_output(out, horizon, family)
    target = dest / f"replay-{capacity}.json"
    if target.exists():
        return
    torch.set_num_threads(1)
    import json
    models = json.loads((dest / "offline.json").read_text())["models"]
    history, end, margin = settings["history"], settings["end"], settings["margin"]
    exact_window = (history, end) == (HISTORY, END)
    requests, pages = trace(data_root, family)
    common = dict(family=family, capacity=capacity, horizon=horizon, history_start=0, score_start=history,
                  score_end=end, trace_sha256=PROTOCOL[family]["sha256"])
    results = []
    for name, factory in CLASSICAL.items():
        measurements = run_policy(factory(), requests, pages, capacity, history, end)
        if exact_window:
            expected = historical_misses(FAIRNESS / f"policy_comparison_{name}.csv", family, capacity)
            if expected != measurements["misses"]:
                raise ValueError(f"{name} baseline disagrees with final KBS for {family}/{capacity}")
        results.append(dict(common, method=name, seed=0, strategy="argmin", origin="measured",
                            margin=0.0, **measurements))
    if exact_window and horizon == 4:
        for name in ("lrb", "three_l_cache", "cacheus", "halp"):
            with (FAIRNESS / f"policy_comparison_{name}.csv").open(newline="") as stream:
                misses = historical(csv.DictReader(stream), family, capacity)
            results.append(dict(common, method=name, seed=0, strategy="argmin", origin="historical_matched",
                                requests=end - history, misses=misses, hits=end - history - misses,
                                miss_ratio=misses / (end - history), margin=0.0))
    cached_x, _, _ = load_cell(out / "data", family, capacity)
    sample = cached_x[np.linspace(0, len(cached_x) - 1, min(24, len(cached_x)), dtype=int)]
    for model in models:
        name, seed, fmt = model["method"], model["seed"], model["format"]
        path = dest / model["file"]
        if fmt == "torch":
            if sha256(path) != model["sha256"]:
                raise ValueError(f"Changed neural checkpoint {path}")
            network, _ = load(path, family, horizon)
            predictor = lambda x: predict(network, x)[1][0]
        elif fmt == "original":
            estimator, _ = reference_estimator(model_root, family, name)
            predictor = cpu_estimator(estimator, sample)
        else:
            if sha256(path) != model["sha256"]:
                raise ValueError(f"Changed refit checkpoint {path}")
            predictor = cpu_estimator(joblib.load(path), sample)
        predictor(sample[:1])
        strategies = [("argmin", 0.0)]
        if name in settings["replay_margin_methods"] and seed == settings["seeds"][0]:
            strategies.append(("margin_lru", margin))
        for strategy, tolerance in strategies:
            measurements = run_policy(BatchPolicy(predictor, tolerance), requests, pages, capacity, history, end)
            if name == "selected_kbs" and exact_window:
                expected = historical_misses(SELECTED, family, capacity)
                if expected != measurements["misses"]:
                    raise ValueError(f"Selected KBS replay drift for {family}/{capacity}: {measurements['misses']} != {expected}")
            results.append(dict(common, method=name, seed=seed, strategy=strategy, origin="measured",
                                margin=tolerance, checkpoint_bytes=model["size_bytes"],
                                checkpoint_sha256=model["sha256"], **measurements))
    write_json(target, results)
    print(f"Replayed {family}/{capacity}, H={horizon}, {len(results)} rows", flush=True)
