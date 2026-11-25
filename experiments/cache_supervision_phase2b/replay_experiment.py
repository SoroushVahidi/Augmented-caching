"""Validation-only selection and independent cache trajectories for Phase 2B."""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import json
import math
import os
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch

from diagnostics import CELLS, WORK
from verify import OUT, SEED, write_csv, write_json
from sources import CAPACITIES, NOVEMBER, PROTOCOL, TRACES, digest, stream_for
from data import END, FEATURES, HISTORY, fold, historical_misses, matrix, trace
from math_core import next_positions, shortlist_hindsight
from models import load, predict
from replay import CLASSICAL, FAIRNESS
from lafc.policies.evict_value_v1 import EvictValueV1Policy

GRID = (0., .025, .05, .1, .2, .5)
STUDENT_GRID = (0., .025, .05, .1, .2, None)
SEEDS = (7, 19)
CONFIG = {"ridge": {"alpha": 10.},
          "hist_gradient_boosting": {"max_iter": 30, "max_depth": 3, "min_samples_leaf": 32,
                                     "learning_rate": .1, "l2_regularization": 10.,
                                     "early_stopping": False, "random_state": SEED}}
NEURAL = {"approximate_tie": GRID + (None,), "rounding": GRID[1:] + (None,), "baseline_gate": GRID + (None,)}


def choose_neural(scores, kind, threshold):
    best = int(scores.argmin())
    if kind == "exact_argmin":
        return best
    if kind == "baseline_gate":
        return best if float(scores[0]) > float(scores[best]) + threshold else 0
    if kind == "rounding":
        return int(np.rint(scores.astype(np.float64) / threshold).argmin())
    return int(np.argmax(scores <= scores[best] + threshold))


def choose_student(scores, threshold):
    best = int(scores.argmax())
    return best if scores[best] - scores[0] > threshold else 0


class FallbackPolicy(EvictValueV1Policy):
    def __init__(self, selector, counts=False):
        super().__init__(scorer_mode="lightweight", history_window=64)
        self.selector = selector
        self.use_counts = counts

    def reset(self, capacity, pages):
        super().reset(capacity, pages)
        self.overrides = 0
        self.counts = {}
        self.ghosts = OrderedDict()
        self.feature_seconds = 0.
        self.scoring_seconds = 0.

    def on_request(self, request):
        event = super().on_request(request)
        if self.use_counts:
            prior = 0 if event.hit else self.ghosts.pop(request.page_id, 0)
            if event.evicted is not None:
                self.ghosts[event.evicted] = self.counts.pop(event.evicted)
                if len(self.ghosts) > 2048:
                    self.ghosts.popitem(last=False)
            if event.hit:
                self.counts[request.page_id] += 1
            else:
                self.counts[request.page_id] = 1 + prior
        return event

    def _choose_victim(self, request):
        old_to_new = list(self._order)
        chosen = self.selector(self, request, old_to_new)
        self.overrides += chosen != 0
        return old_to_new[chosen], {}


def neural_selector(scorer, kind, value):
    def select(policy, request, candidates):
        start = process_time()
        x = matrix(policy, request, candidates)
        before_score = process_time()
        scores = scorer(x)
        policy.scoring_seconds += process_time() - before_score
        policy.feature_seconds += before_score - start
        return choose_neural(scores, kind, value)
    return select


def student_selector(scorer, threshold):
    def select(policy, request, candidates):
        start = process_time()
        counts = np.asarray([math.log1p(policy.counts[c]) for c in candidates], dtype=np.float32)
        order = np.lexsort((np.arange(len(candidates)), counts))
        shortlist = np.r_[0, order[order != 0][:7]]
        x = matrix(policy, request, candidates)[shortlist]
        before_score = process_time()
        scores = scorer(x)
        policy.scoring_seconds += process_time() - before_score
        policy.feature_seconds += before_score - start
        return int(shortlist[choose_student(scores, threshold)])
    return select


def random_selector(rate, fold_index, cap, seed):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, fold_index, cap, seed]))

    def select(policy, request, candidates):
        return int(rng.integers(1, len(candidates))) if rng.random() < rate else 0
    return select


def replay_policy(factory, req, pages, capacity):
    policy = factory()
    policy.reset(capacity, pages)
    for request in req[:HISTORY]:
        policy.on_request(request)
    if isinstance(policy, FallbackPolicy):
        policy.overrides = 0
        policy.scoring_seconds = policy.feature_seconds = 0.
    misses = evictions = 0
    wall, cpu = perf_counter(), process_time()
    for request in req[HISTORY:END]:
        event = policy.on_request(request)
        misses += not event.hit
        evictions += event.evicted is not None
    cpu = process_time() - cpu
    seconds = perf_counter() - wall
    overrides = getattr(policy, "overrides", 0)
    return dict(misses=misses, requests=END-HISTORY, evictions=evictions, overrides=overrides,
                override_frequency=overrides / max(1, evictions), cpu_seconds=cpu, seconds=seconds,
                cpu_us_per_request=cpu * 1e6 / (END-HISTORY),
                feature_cpu_seconds=getattr(policy, "feature_seconds", 0.),
                score_cpu_seconds=getattr(policy, "scoring_seconds", 0.),
                process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def neural_checkpoint(heldout, seed, val):
    root = NOVEMBER / "runs/campaign/h8" / heldout
    models = json.loads((root / "offline.json").read_text())["models"]
    spec = next((m for m in models if m["method"] == "deepsets" and m["seed"] == seed), None)
    if spec is None:
        return None
    path = root / spec["file"]
    assert digest(path) == spec["sha256"]
    net, provenance = load(path, heldout, 8)
    assert provenance["test"] == heldout
    assert set(provenance["train"]) == set(PROTOCOL[heldout]["train"])
    assert provenance["validation"] == val
    return net, spec


def scorer_for(net, cap):
    @lru_cache(maxsize=4096)
    def scores(encoded):
        x = np.frombuffer(encoded, dtype=np.float32).copy().reshape(1, cap, len(FEATURES))
        return predict(net, x)[1][0]

    def score(x):
        return scores(np.ascontiguousarray(x, dtype=np.float32).tobytes())
    return score, scores.cache_info


def run_neural(net, spec, requests, pages, cap, kind, value, scorer, cache_info):
    before = cache_info()
    if value is None:
        measurements = replay_policy(CLASSICAL["lru"], requests, pages, cap)
    else:
        select = neural_selector(scorer, kind, value)
        measurements = replay_policy(lambda: FallbackPolicy(select), requests, pages, cap)
    after = cache_info()
    return dict(checkpoint_sha256=spec["sha256"], checkpoint_bytes=spec["size_bytes"],
                score_cache_hits=after.hits-before.hits, score_cache_misses=after.misses-before.misses,
                **measurements)


def load_groups(families, capacities=CAPACITIES):
    groups = {}
    for family in families:
        for cap in capacities:
            with np.load(WORK / f"{family}-{cap}.npz", allow_pickle=False) as dataset:
                groups[family, cap] = {key: dataset[key] for key in dataset.files}
    return groups


def diagnostics(advantage, chosen):
    values = advantage[np.arange(len(chosen)), chosen]
    mask = chosen != 0
    return dict(states=len(chosen), overrides=int(mask.sum()), beneficial=int((values[mask]>0).sum()),
                harmful=int((values[mask]<0).sum()), neutral=int((values[mask]==0).sum()),
                net_advantage=int(values.sum()))


def fit_students(heldout, train, val, groups):
    inputs = np.concatenate([groups[f,c]["x"] for f in train for c in CAPACITIES]).reshape(-1, len(FEATURES))
    scaler = StandardScaler().fit(inputs)
    train_x = scaler.transform(inputs)
    va_x = np.concatenate([groups[val,c]["x"] for c in CAPACITIES])
    va_y = np.concatenate([groups[val,c]["resolved"] for c in CAPACITIES])
    val_x = scaler.transform(va_x.reshape(-1, len(FEATURES)))
    candidates = np.arange(len(va_x))
    outputs = {}
    rows = []
    for architecture, params in CONFIG.items():
        for target in ("h8", "resolved"):
            y = np.concatenate([groups[f,c][target] for f in train for c in CAPACITIES]).ravel()
            start_cpu, start_wall = process_time(), perf_counter()
            if architecture == "ridge":
                model = Ridge(**params).fit(train_x, y)
                weights = model.coef_ / scaler.scale_
                offset = float(model.intercept_ - scaler.mean_ @ weights)
                def score(x, w=weights, b=offset):
                    return x @ w + b
            else:
                model = HistGradientBoostingRegressor(**params).fit(train_x, y)
                def score(x, m=model):
                    return m.predict(scaler.transform(x))
            predictions = model.predict(val_x).reshape(len(va_x), 8)
            np.testing.assert_allclose(score(va_x[0]), predictions[0], atol=1e-12)
            fit_cpu, fit_wall = process_time()-start_cpu, perf_counter()-start_wall
            best = predictions.argmax(axis=1)
            margin = predictions[candidates,best] - predictions[:,0]
            scores = []
            for value in STUDENT_GRID:
                choices = np.zeros(len(va_y), dtype=np.int64) if value is None else np.where(margin>value,best,0)
                measures = diagnostics(va_y, choices)
                row = dict(fold=heldout, family=val, policy=f"student_{architecture}_{target}",
                           strategy="threshold_on_archived_states", threshold=value, fit_cpu_seconds=fit_cpu,
                           fit_wall_seconds=fit_wall, train_groups=len(inputs)//8, **measures)
                rows.append(row)
                scores.append((row, value))
            selected, value = max(scores, key=lambda pair: (pair[0]["net_advantage"],-pair[0]["overrides"],
                                                          float("inf") if pair[1] is None else pair[1]))
            selected["selected"] = True
            outputs[f"student_{architecture}_{target}"] = (score, value, fit_cpu, fit_wall)
    return outputs, rows


def family_runs(heldout, stage):
    torch.set_num_threads(1)
    train, val = fold(heldout)
    fold_index = list(PROTOCOL).index(heldout)
    groups = load_groups((heldout, val, *train))
    if stage == "neural":
        vr, vp = trace(TRACES, val)
    outer_req, outer_pages = trace(TRACES, heldout)
    outer = []
    validation = []
    override = []
    checkpoints = {}
    student = {}
    if stage == "students":
        student, valid_rows = fit_students(heldout, train, val, groups)
        validation.extend(valid_rows)

    # Finish all validation selections before attempting any outer replay.
    selected = {}
    for seed in SEEDS if stage == "neural" else ():
        loaded = neural_checkpoint(heldout, seed, val)
        if loaded is None:
            continue
        net, spec = loaded
        checkpoints[seed] = (net, spec)
        validation_map = {}
        for cap in CAPACITIES:
            score, cache = scorer_for(net, cap)
            for kind, grid in NEURAL.items():
                for value in grid:
                    # Exact-arithmetic epsilon and gating coincide except when
                    # epsilon chooses an older nonbaseline candidate than argmin.
                    # Each rule takes its own cache trajectory.
                    if value not in validation_map:
                        validation_map[value] = {}
                    key = (kind, cap)
                    if value is None:
                        if "lru" not in validation_map[value]:
                            validation_map[value]["lru"] = {}
                        baseline = validation_map[value]["lru"]
                        if cap not in baseline:
                            baseline[cap] = run_neural(net, spec, vr, vp, cap, kind, value, score, cache)
                        result = baseline[cap]
                    else:
                        result = run_neural(net, spec, vr, vp, cap, kind, value, score, cache)
                    row = dict(fold=heldout, family=val, capacity=cap, policy="deepsets_h8",
                               seed=seed, strategy=kind, threshold=value, **result)
                    validation_map[value][key] = row
                    validation.append(row)
            print(f"Validation {heldout=} {val=} {seed=} {cap=}", flush=True)
        selected[seed] = {}
        for kind, grid in NEURAL.items():
            value = min(grid, key=lambda h: (sum(validation_map[h][kind,c]["misses"] for c in CAPACITIES),
                                             sum(validation_map[h][kind,c]["overrides"] for c in CAPACITIES),
                                             -float("inf") if h is None else -h))
            selected[seed][kind] = value
            for cap in CAPACITIES:
                validation_map[value][kind,cap]["selected"] = True
    output = WORK / f"{heldout}-{stage}-selection.json"
    output.write_text(json.dumps({"neural": selected,
                                  "students": {name: model[1] for name, model in student.items()}}, indent=2))

    if stage == "students":
        stream, objects = stream_for(heldout)
        nxt = next_positions(stream, objects)
    for cap in CAPACITIES:
        nonmodel = dict(fold=heldout, family=heldout, capacity=cap, trace_sha256=PROTOCOL[heldout]["sha256"])
        block = groups[heldout,cap]
        for name, factory in CLASSICAL.items() if stage == "students" else ():
            result = replay_policy(factory, outer_req, outer_pages, cap)
            expected = historical_misses(FAIRNESS / f"policy_comparison_{name}.csv", heldout, cap)
            assert result["misses"] == expected, (name, heldout, cap, result["misses"], expected)
            outer.append(dict(nonmodel, policy=name, seed=0, **result))
        for k in (8,cap) if stage == "students" else ():
            begin, cpu = perf_counter(), process_time()
            misses = int(shortlist_hindsight(stream,nxt,cap,k,HISTORY))
            outer.append(dict(nonmodel, policy="oracle_belady" if k==cap else "oracle_lru_tail_k8_next_use",
                              seed=0, misses=misses, requests=END-HISTORY, seconds=perf_counter()-begin,
                              cpu_seconds=process_time()-cpu))
        for seed,(net,spec) in checkpoints.items():
            score, cache = scorer_for(net, cap)
            strategies = {"exact_argmin": ("exact_argmin", 0.), "near_tie_0.05": ("approximate_tie", .05),
                          **{f"calibrated_{kind}": (kind, value) for kind,value in selected[seed].items()}}
            for name,(kind,value) in strategies.items():
                measured = run_neural(net, spec, outer_req, outer_pages, cap, kind, value, score, cache)
                outer.append(dict(nonmodel, policy=f"deepsets_h8_{name}", seed=seed, threshold=value, **measured))
                if value is None:
                    picks = np.zeros(len(block["full_x"]), dtype=np.int64)
                else:
                    picks = np.asarray([choose_neural(score(x), kind, value) for x in block["full_x"]])
                override.append(dict(nonmodel, policy=f"deepsets_h8_{name}", seed=seed,
                                     **diagnostics(block["full_advantage"], picks)))
                if name in {"near_tie_0.05", "calibrated_approximate_tie"}:
                    rate = measured["override_frequency"]
                    random = random_selector(rate, fold_index, cap, seed)
                    result = replay_policy(lambda: FallbackPolicy(random), outer_req, outer_pages, cap)
                    tag = f"random_matched_deepsets_{name}"
                    outer.append(dict(nonmodel, policy=tag, seed=seed, target_override_rate=rate, **result))
                    random = random_selector(rate, fold_index, cap, seed)
                    picks = np.asarray([random(None,None,range(cap)) for _ in block["full_x"]])
                    override.append(dict(nonmodel, policy=tag, seed=seed, **diagnostics(block["full_advantage"], picks)))
        for name,(score,value,fit_cpu,fit_wall) in student.items():
            if value is None:
                result = replay_policy(CLASSICAL["lru"], outer_req, outer_pages, cap)
                picks = np.zeros(len(block["x"]), dtype=np.int64)
            else:
                select = student_selector(score, value)
                result = replay_policy(lambda: FallbackPolicy(select, counts=True), outer_req, outer_pages, cap)
                picks = np.asarray([choose_student(score(x), value) for x in block["x"]])
            outer.append(dict(nonmodel, policy=name, seed=0, threshold=value, fit_cpu_seconds=fit_cpu, **result))
            override.append(dict(nonmodel, policy=name, seed=0, **diagnostics(block["resolved"], picks)))
            if name == "student_ridge_resolved":
                rate = result["override_frequency"]
                random = random_selector(rate, fold_index, cap, SEED)
                def short_random(p, request, candidates):
                    idx = random(p, request, range(8))
                    if idx == 0:
                        return 0
                    counts = np.asarray([math.log1p(p.counts[c]) for c in candidates], dtype=np.float32)
                    order = np.lexsort((np.arange(len(candidates)), counts))
                    return int(order[order!=0][idx-1])
                factory = CLASSICAL["lru"] if rate==0 else lambda: FallbackPolicy(short_random, counts=True)
                random_result = replay_policy(factory, outer_req, outer_pages, cap)
                tag = "random_matched_resolved_ridge_k8"
                outer.append(dict(nonmodel, policy=tag, seed=SEED, target_override_rate=rate, **random_result))
                random = random_selector(rate, fold_index, cap, SEED)
                picks = np.asarray([random(None,None,range(8)) for _ in block["x"]])
                override.append(dict(nonmodel, policy=tag, seed=SEED, **diagnostics(block["resolved"], picks)))
        print(f"Outer {heldout=} {cap=}", flush=True)
    staged = WORK / f"{heldout}-{stage}-replay.tmp"
    staged.write_text(json.dumps([outer,validation,override], allow_nan=False))
    staged.replace(WORK / f"{heldout}-{stage}-replay.json")
    return heldout


def summarize(stages):
    outer, val, override = [], [], []
    for stage in stages:
        for family in PROTOCOL:
            a, b, c = json.loads((WORK / f"{family}-{stage}-replay.json").read_text())
            outer.extend(a)
            val.extend(b)
            override.extend(c)
    for policy, seed, expected in (("lru",0,565126), ("oracle_belady",0,507593),
                                   ("deepsets_h8_exact_argmin",7,565462), ("deepsets_h8_near_tie_0.05",7,565245),
                                   ("deepsets_h8_exact_argmin",19,569422)):
        if policy.startswith("deepsets") and "neural" not in stages:
            continue
        actual = sum(x["misses"] for x in outer if x["policy"]==policy and x["seed"]==seed)
        assert actual == expected, (policy, seed, actual, expected)
    write_csv("CLOSED_LOOP_RESULTS.csv", outer)
    write_csv("VALIDATION_RESULTS.csv", val)
    write_csv("OVERRIDE_DIAGNOSTICS.csv", override)
    write_csv("CHECKPOINT_FALLBACK_RESULTS.csv", [r for r in outer if
              r["policy"].startswith(("deepsets", "random_matched_deepsets")) or r["policy"] == "lru"])
    write_csv("RESOURCE_PROFILE.csv", [{k:v for k,v in r.items() if k in {
        "family","capacity","policy","seed","cpu_seconds","seconds","process_peak_rss_kib","score_cache_hits",
        "score_cache_misses","feature_cpu_seconds","score_cpu_seconds","checkpoint_bytes","fit_cpu_seconds"}}
        for r in outer])
    controls = {(x["family"], x["capacity"]): x for x in outer if x["policy"]=="lru"}
    tags = list(dict.fromkeys((x["policy"], x["seed"]) for x in outer))
    overall = []
    family_rows = []
    cap_rows = []
    for name, seed in tags:
        matching = [r for r in outer if r["policy"]==name and r["seed"]==seed]
        tag = dict(policy=name, seed=seed)
        def aggregate(rows):
            baseline = [controls[x["family"],x["capacity"]] for x in rows]
            misses = sum(x["misses"] for x in rows)
            lru = sum(x["misses"] for x in baseline)
            cpu = sum(x["cpu_seconds"] for x in rows)
            return dict(tag, misses=misses, lru_misses=lru, requests=sum(x["requests"] for x in rows),
                        relative_miss_reduction=(lru-misses)/lru, cpu_seconds=cpu,
                        cpu_vs_lru=cpu/sum(x["cpu_seconds"] for x in baseline),
                        overrides=sum(x.get("overrides",0) for x in rows),
                        evictions=sum(x.get("evictions",0) for x in rows))
        families = [dict(family=f, **aggregate([x for x in matching if x["family"]==f]))
                    for f in PROTOCOL if any(x["family"]==f for x in matching)]
        family_rows.extend(families)
        cap_rows.extend(dict(capacity=c, **aggregate([x for x in matching if x["capacity"]==c])) for c in CAPACITIES
                        if any(x["capacity"]==c for x in matching))
        loo = {f:aggregate([x for x in matching if x["family"]!=f])["relative_miss_reduction"]
               for f in PROTOCOL if any(x["family"]!=f for x in matching)}
        worst = min(families, key=lambda x:x["relative_miss_reduction"])
        overall.append(dict(**aggregate(matching),
            macro_family_relative_miss_reduction=sum(x["relative_miss_reduction"] for x in families)/len(families),
            worst_family=worst["family"], worst_family_regression=-worst["relative_miss_reduction"],
            leave_one_family_out_min=min(loo.values()), leave_one_family_out_max=max(loo.values()),
            leave_one_family_out=json.dumps(loo)))
    write_csv("FAMILY_LEVEL_RESULTS.csv", family_rows)
    write_csv("CAPACITY_LEVEL_RESULTS.csv", cap_rows)
    write_csv("POLICY_SUMMARY.csv", overall)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fold", choices=list(PROTOCOL))
    parser.add_argument("--stage", choices=("students", "neural", "summarize"), required=True)
    args = parser.parse_args()
    if args.stage == "summarize":
        summarize(("students", "neural"))
    elif args.fold:
        family_runs(args.fold, args.stage)
    else:
        remaining = [family for family in PROTOCOL if not (WORK / f"{family}-{args.stage}-replay.json").is_file()]
        with ProcessPoolExecutor(args.workers) as pool:
            for family in pool.map(family_runs, remaining, [args.stage] * len(remaining)):
                print(f"Completed fold {family}", flush=True)
        summarize(("students",) if args.stage == "students" else ("students", "neural"))
