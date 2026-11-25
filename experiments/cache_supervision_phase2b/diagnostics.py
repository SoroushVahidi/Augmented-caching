"""Verified historical joins and bounded additional development labels."""
from collections import Counter, defaultdict
import csv
from itertools import islice
import json
import math
from pathlib import Path
from time import perf_counter, process_time
import resource

import numpy as np
from verify import AUDIT, OUT, SEED, write_csv, write_json
from exact_lru import rollout
from sources import CAPACITIES, NOVEMBER, PROTOCOL, TRACES, digest, load_cell, observed_states, stratified, stream_for
from data import Capture, FEATURES, labels, matrix, trace

FIXED = (4, 8, 16, 32, 64, 128, 256, 512)
ALPHAS = (.25, .5, 1, 2, 4)
UNIFORM_SEEDS = (7, 19, 42, 101, SEED)
CELLS = [(f, c) for f in PROTOCOL for c in CAPACITIES]
WORK = OUT.parent.parent / "experiments/cache_supervision_phase2b/runs"


def archived_actions():
    groups = defaultdict(list)
    with (AUDIT / "ACTION_LEVEL_EVIDENCE.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            groups[row["family"], int(row["capacity"]), row["sample"], int(row["t"])].append(row)
    return groups


def pick_samples(t, primary, enriched, cell_index):
    eligible = np.setdiff1d(t[(t >= 10512) & (t < 49488)], np.r_[primary, enriched])
    rng = np.random.default_rng(np.random.SeedSequence([SEED, cell_index]))
    small_target, small_extra = divmod(5000, len(CELLS))
    large_target, large_extra = divmod(20000, len(CELLS))
    first = stratified(eligible, small_target + int(cell_index < small_extra) - len(primary), rng)
    second = stratified(np.setdiff1d(eligible, first),
                        large_target + int(cell_index < large_extra) - len(primary) - len(first), rng)
    return first, second


def shortlist_orders(state, cell_index, t):
    started = perf_counter()
    x = state["x"]
    n = len(x)
    rank = np.arange(n)

    def baseline_first(v):
        return np.r_[0, v[v != 0]]

    low = baseline_first(np.lexsort((rank, x[:, 1])))
    low_cost = perf_counter() - started
    start = perf_counter()
    sieve = np.lexsort((rank, state["sieve_flags"]))
    native = state["native_sieve"]
    if native >= 0:
        sieve = np.r_[native, sieve[sieve != native]]
    sieve = baseline_first(sieve)
    sieve_cost = perf_counter() - start
    start = perf_counter()
    ghost = baseline_first(np.lexsort((rank, x[:, 1], x[:, 10])))
    ghost_cost = perf_counter() - start
    start = perf_counter()
    tail = rank
    tail_cost = perf_counter() - start
    all_orders = dict(lru_tail=(tail, tail_cost), low_frequency=(low, low_cost),
                      sieve_derived=(sieve, sieve_cost), ghost_informed=(ghost, ghost_cost))
    for seed in UNIFORM_SEEDS:
        rng = np.random.default_rng(np.random.SeedSequence([seed, cell_index, t]))
        start = perf_counter()
        uniform = baseline_first(rng.permutation(n))
        all_orders[f"uniform_{seed}"] = uniform, perf_counter() - start
    return all_orders


def distinct_endpoints(future, targets):
    unique = set()
    goals = iter(sorted(set(targets)))
    target = next(goals, None)
    endpoints = {}
    for i, request in enumerate(future, 1):
        unique.add(int(request))
        if len(unique) == target:
            endpoints[target] = i
            target = next(goals, None)
            if target is None:
                break
    while target is not None:
        endpoints[target] = len(future)
        target = next(goals, None)
    return endpoints


def horizon_rows(future, resident, incoming, short, exact, output, label_times):
    cap = len(resident)
    scaled = {x: math.ceil(x * cap) for x in ALPHAS}
    distinct = distinct_endpoints(future, scaled.values())
    queries = ([("fixed_requests", h, h) for h in FIXED]
               + [("scaled_requests", a, h) for a, h in scaled.items()]
               + [("distinct_objects", a, distinct[h]) for a, h in scaled.items()])
    for kind, value, h in queries + [("resolved", 0, 0)]:
        if kind == "resolved":
            actual = exact.requests
            y = np.asarray(exact.advantage)
            elapsed = label_times
        else:
            start = perf_counter()
            snapshot = rollout(resident, incoming, islice(map(int, future), h))[1]
            elapsed = perf_counter() - start
            actual = snapshot.requests
            y = np.asarray(snapshot.advantage)
        z = np.asarray(exact.advantage)
        for name, a, full in (("all", y, z), ("low_frequency", y[short], z[short])):
            hist = Counter(a)
            n = len(a)
            stand = output[kind, value, name, n]
            stand["decisions"] += 1
            stand["all_tie"] += len(hist) == 1
            stand["all_tie_resolved_nonconstant"] += len(hist) == 1 and np.ptp(full) > 0
            stand["distinct_labels"] += len(hist)
            stand["entropy_bits"] -= sum((v/n) * math.log2(v/n) for v in hist.values())
            stand["pairwise_tie"] += sum(v*(v-1) for v in hist.values()) / (n*(n-1))
            stand["optimal_set_fraction"] += np.count_nonzero(a == max(a)) / n
            stand["lru_optimal"] += max(a) == 0
            stand["positive_decision"] += max(a) > 0
            stand["positive_actions"] += np.count_nonzero(a > 0)
            stand["value_disagreement"] += np.count_nonzero(a != full)
            stand["sign_disagreement"] += np.count_nonzero(np.sign(a) != np.sign(full))
            stand["actual_requests"] += actual
            stand["capacity_over_actual_horizon"] += cap / max(1, actual)
            stand["candidates_over_actual_horizon"] += n / max(1, actual)
            stand["actual_zero_requests"] += actual == 0
            stand["label_seconds"] += elapsed
            stand.update({f"advantage_{k}": v for k, v in hist.items()})


def prepare_cell(family, cap, all_actions, cell_index):
    began, cpu = perf_counter(), process_time()
    x, y, timestamp, manifest = load_cell(family, cap)
    assert manifest["columns"] == FEATURES and manifest["trace_end"] == 50000
    path = AUDIT / "units/state" / f"{family}-{cap}.json"
    archived = json.loads(path.read_text())
    primary = np.asarray(archived["sample"]["temporal_stratified"])
    enriched = np.asarray(archived["sample"]["learner_disagreement_enriched"])
    assert len(primary) == 56 and not set(primary) & set(enriched)
    first, second = pick_samples(timestamp, primary, enriched, cell_index)
    all_times = np.sort(np.r_[primary, enriched, first, second])
    assert len(set(all_times)) == len(all_times)
    indices = dict(zip(all_times, np.searchsorted(timestamp, all_times)))
    assert all(timestamp[i] == t for t, i in indices.items())
    stream, object_count = stream_for(family)
    _, states, _ = observed_states(stream, object_count, cap, all_times)
    req, pages = trace(TRACES, family)
    mapping = {}
    for pos, (p, r) in enumerate(zip(stream, req, strict=True)):
        assert r.t == pos
        assert mapping.setdefault(r.page_id, int(p)) == p

    def check_features(policy, request, candidates):
        t = request.t
        if t in indices:
            np.testing.assert_array_equal([mapping[p] for p in candidates], states[t]["ids"])
            np.testing.assert_array_equal(matrix(policy, request, candidates), x[indices[t]])

    capture = Capture(check_features)
    capture.reset(cap, pages)
    for request in req:
        capture.on_request(request)
    verified = []
    shortlist = defaultdict(Counter)
    horizon_data = defaultdict(Counter)
    stats = defaultdict(Counter)
    training = {key: [] for key in ("time", "short", "x", "h8", "resolved", "full_x", "full_advantage")}
    label_seconds = 0.
    for times, sample in ((primary, "temporal_stratified"), (enriched, "learner_disagreement_enriched"),
                          (first, "additional_5k"), (second, "additional_20k")):
        for t in times:
            state = states[t]
            future = stream[t+1:]
            resident = tuple(state["ids"].tolist())
            incoming = int(stream[t])
            i = indices[t]
            started = perf_counter()
            fixed, exact = rollout(resident, incoming, map(int, future), (4, 8, 16), stop_when_resolved=True)
            elapsed = perf_counter() - started
            label_seconds += elapsed
            fixed_labels = np.asarray([fixed[h].misses for h in (4, 8, 16)]).T
            np.testing.assert_array_equal(fixed_labels, y[i])
            advantage = np.asarray(exact.advantage)
            if sample in ("temporal_stratified", "learner_disagreement_enriched"):
                records = sorted(all_actions[family, cap, sample, t], key=lambda r: int(r["candidate_recency_order"]))
                assert len(records) == cap
                np.testing.assert_array_equal([int(r["candidate_recency_order"]) for r in records], np.arange(cap))
                np.testing.assert_array_equal([int(r["h16_loss"]) for r in records], fixed_labels[:, 2])
                np.testing.assert_array_equal([int(r["next_position"]) if r["next_position"] else len(stream)
                                               for r in records], state["next"])
                assert all(int(r["resolution"]) > 0 for r in records)
                np.testing.assert_array_equal([int(r["lru_paired_advantage"]) for r in records], advantage)
                np.testing.assert_array_equal(labels(np.asarray(resident), incoming, future[:16]), fixed_labels)
                verified.append(sample)
            if sample == "learner_disagreement_enriched":
                continue
            orders = shortlist_orders(state, cell_index, t) if sample == "temporal_stratified" else {}
            if orders:
                short = orders["low_frequency"][0][:8]
            else:
                rank = np.arange(cap)
                ordered = np.lexsort((rank, state["x"][:, 1]))
                short = np.r_[0, ordered[ordered != 0][:7]]
            h8 = np.asarray(fixed[8].advantage)
            for name, mask in (("all", np.arange(cap)), ("low_frequency_k8", short)):
                block = stats[sample, name]
                block.update(dict(decisions=1, resolved_positive=int(max(advantage[mask]) > 0),
                                  h8_positive=int(max(h8[mask]) > 0), candidate_rows=len(mask),
                                  trace_end_uncoalesced=int(exact.active > 0)))
            if sample != "temporal_stratified":
                continue
            for name, (order, elapsed_order) in orders.items():
                for k in (4, 8, 16, 32):
                    s = order[:k]
                    shortlist[name, k].update(dict(decisions=1, all_candidate_opportunities=int(max(advantage)>0),
                        covered_opportunities=int(max(advantage[s])>0), available_positive_actions=int(np.sum(advantage>0)),
                        shortlisted_positive_actions=int(np.sum(advantage[s]>0)),
                        sum_best_advantage=int(max(advantage[s])), all_candidate_sum_best=int(max(advantage)),
                        construction_seconds=elapsed_order))
            horizon_rows(future, resident, incoming, short, exact, horizon_data, elapsed)
            for key, value in dict(time=t,short=short,x=x[i, short],h8=h8[short],resolved=advantage[short],
                                   full_x=x[i],full_advantage=advantage).items():
                training[key].append(value)
    np.savez_compressed(WORK / f"{family}-{cap}.npz", **training)
    meta = dict(family=family, capacity=cap)
    md = dict(lru_tail="resident recency", low_frequency="Phase 2A causal resident/ghost count and recency",
              ghost_informed="Phase 2A bounded ghost reinsertion history/count and recency",
              sieve_derived="parallel SIEVE queue, visited bits, hand and resident recency")
    return dict(
        horizons=[dict(meta, kind=kind, horizon_spec=value, candidate_set=name, candidates=n, **block)
                  for (kind, value, name, n), block in horizon_data.items()],
        shortlists=[dict(meta, mechanism=name, k=k, metadata=md.get(name, "random generator and resident recency"),
                         **block) for (name,k), block in shortlist.items()],
        samples=[dict(meta, stratum=s, candidate_set=n, **block) for (s, n), block in stats.items()],
        archive=dict(meta, sha256=PROTOCOL[family]["sha256"], source_receipt_sha256=digest(path),
                     exact_matches=dict(Counter(verified)), verified_feature_groups=len(all_times),
                     warmup_primary=int((primary < 10000).sum()), scored_primary=int((primary>=10000).sum())),
        resource=dict(meta, stage="prepare_labels", seconds=perf_counter()-began, cpu_seconds=process_time()-cpu,
                      label_seconds=label_seconds, rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      stored_bytes=(WORK / f"{family}-{cap}.npz").stat().st_size),
    )


def run():
    WORK.mkdir(exist_ok=True)
    actions = archived_actions()
    tables = []
    for idx,(family, cap) in enumerate(CELLS):
        result = prepare_cell(family, cap, actions, idx)
        tables.append(result)
        print(f"Verified joins and labels for {family}/{cap}", flush=True)
    write_csv("SHORTLIST_COVERAGE.csv", [r for t in tables for r in t["shortlists"]])
    write_csv("HORIZON_SENSITIVITY.csv", [r for t in tables for r in t["horizons"]])
    samples = [r for t in tables for r in t["samples"]]
    for budget, strata in ((1176, {"temporal_stratified"}),
                           (5000, {"temporal_stratified","additional_5k"}),
                           (20000, {"temporal_stratified","additional_5k","additional_20k"})):
        for test, fold in PROTOCOL.items():
            for role, scope in (("train", fold["train"]), ("validation", [fold["validation"]]), ("test", [test])):
                for name in ("all", "low_frequency_k8"):
                    subset = [r for r in samples if r.get("family") in scope and r["stratum"] in strata
                              and r["candidate_set"] == name]
                    samples.append(dict(fold=test, role=role, budget=budget, candidate_set=name,
                                        **{k:sum(r[k] for r in subset) for k in
                                           ("decisions","resolved_positive","h8_positive","candidate_rows","trace_end_uncoalesced")}))
    write_csv("SAMPLE_ADEQUACY.csv", samples)
    write_csv("LABEL_GENERATION_RESOURCES.csv", [t["resource"] for t in tables])
    write_json("ARCHIVED_JOIN_CORRECTNESS.json", dict(
        source_action_sha256=digest(AUDIT / "ACTION_LEVEL_EVIDENCE.csv"), unexplained_mismatches=0,
        columns=FEATURES, feature_schema="evict_value_v1 / November matrix", cells=[t["archive"] for t in tables]))


if __name__ == "__main__":
    run()
