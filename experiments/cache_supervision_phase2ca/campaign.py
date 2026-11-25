"""Generate exact, grouped counterfactual supervision on the fixed development grid."""
from collections import Counter, OrderedDict
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for directory in ("cache_supervision_phase2b", "neural_eviction_20261121", "recurrent_neural_eviction"):
    sys.path.append(str(HERE.parent / directory))
sys.path.insert(0, str(ROOT / "src"))

from data import Capture, END, FEATURES, HISTORY, PROTOCOL, matrix, trace
from exact_lru import RolloutDetails, rollout
from features import CausalHistory, GLOBAL_NAMES, STATIC_NAMES
from lafc.policies.sieve import SievePolicy

BASE = "c87c230e6c1a06703be764f9e69807980b31b112"
SCHEMA = "phase2ca-grouped-v1"
SEED = 20261125
CAPS = (32, 64, 128)
DIAGNOSTIC = (256, 512, 1024)
BUDGETS = (20000, 50000, 100000)
FIXED = (4, 8, 16, 32, 64, 128, 256, 512)
ALPHAS = (.125, .25, .5, 1, 2, 4)
KS = (2, 4, 8, 16, 32)
UNIFORM = (7, 19, 42, 101, 211, 307, 421, 503, 601, SEED)
STRATUM_WIDTH = 5000
PARTITION_GROUPS = 256
MIXTURES = ("lru_tail", "low_frequency", "sieve_derived", "ghost_informed", "recency_frequency", "causal_mixture")
MECHANISMS = MIXTURES + tuple(f"uniform_{s}" for s in UNIFORM)
FEATURE_GROUPS = {
    "F0": list(FEATURES),
    "F1": list(FEATURES),  # Reduced using the supplied training-family statistics.
    "F2": ["age", "count", *(f"gap_{i}" for i in (1, 2, 3)), *(f"known_gap_{i}" for i in (1, 2, 3))],
    "F4": list(STATIC_NAMES),
    "F5": ["evictions", "reinsertions", "age_since_eviction", "known_eviction"],
    "F6": list(GLOBAL_NAMES),
}
FEATURE_GROUPS["F3"] = FEATURE_GROUPS["F2"] + ["decayed_popularity", "decayed_popularity_256"]


def dump(path, payload):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def sha(path):
    with path.open("rb") as stream:
        h = hashlib.sha256()
        while data := stream.read(1 << 20):
            h.update(data)
    return h.hexdigest()


def trace_data(root, family):
    started = perf_counter()
    req, pages = trace(root, family)
    enc = {}
    for t, r in enumerate(req):
        if r.t != t:
            raise ValueError(f"Noncanonical request position {family}/{t}")
        if r.page_id not in enc:
            enc[r.page_id] = len(enc)
    series = [enc[r.page_id] for r in req]
    return req, pages, enc, series, perf_counter() - started


def census(series, capacity):
    order = OrderedDict()
    times = []
    full = 0
    for t, item in enumerate(series):
        if item in order:
            order.move_to_end(item)
        else:
            if len(order) == capacity:
                if t >= HISTORY:
                    times.append(t)
                order.popitem(last=False)
            order[item] = None
        if t >= HISTORY:
            full += len(order) == capacity
    return np.asarray(times, dtype=np.uint32), full


def cell_quotas(counts, budgets):
    """Round-robin capacity-aware allocation, with all smaller prefixes nested."""
    quotas = [0] * len(counts)
    result = {}
    allocated = 0
    while allocated < budgets[-1]:
        advance = False
        for i, n in enumerate(counts):
            if quotas[i] == n:
                continue
            quotas[i] += 1
            advance = True
            allocated += 1
            if allocated in budgets:
                result[str(allocated)] = quotas.copy()
            if allocated == budgets[-1]:
                return result
        if not advance:
            raise ValueError(f"Only {allocated} eligible decisions for {budgets[-1]} requested")
    return result


def selection(times, quotas, family, cap):
    """Random draws within fixed chronological strata, with deterministic stratum quotas."""
    blocks = {i: times[(times >= HISTORY+i*STRATUM_WIDTH) & (times < HISTORY+(i+1)*STRATUM_WIDTH)]
              for i in range((END-HISTORY)//STRATUM_WIDTH)}
    population = {i: len(t) for i,t in blocks.items()}
    priority = sorted((((r+.5)/n if r else -1.), b, r) for b,n in population.items() for r in range(n))
    selection = {}
    drawn = {}
    rng = np.random.default_rng(np.random.SeedSequence([SEED, list(PROTOCOL).index(family), cap]))
    for i, block in blocks.items():
        drawn[i] = rng.permutation(block)
    counts = [Counter(b for _,b,_ in priority[:q]) for q in quotas]
    for _, b, rank in priority[:quotas[-1]]:
        t = int(drawn[b][rank])
        included = [rank < count[b] for count in counts]
        selection[t] = (b, included, [count[b]/population[b] for count in counts])
    summary = [dict(family=family, capacity=cap, stratum=b, start=HISTORY+b*STRATUM_WIDTH,
                    end=HISTORY+(b+1)*STRATUM_WIDTH, eligible=population[b],
                    **{f"n_{budget}": count[b] for budget,count in zip(BUDGETS, counts)},
                    **{f"p_{budget}": count[b]/population[b] if population[b] else 0.
                       for budget,count in zip(BUDGETS, counts)})
               for b in blocks]
    if len(selection) != quotas[-1]:
        raise AssertionError("Unexpected duplicate decisions")
    return selection, summary


class MultiDecay(CausalHistory):
    """Use the Phase 2A feature adapter and maintain one extra decay in its ghost scope."""

    def __init__(self, capacity):
        super().__init__(capacity)
        self.decay256 = {}

    def extra(self, residents, t):
        return np.asarray([math.log1p(self.decay256[p][0] * math.exp(-(t-self.decay256[p][1])/256))
                           for p in residents], dtype=np.float32)

    def after(self, pid, t, hit, victim):
        drop = None
        if victim is not None and len(self.ghosts) == self.ghost_limit and pid not in self.ghosts:
            drop = next(iter(self.ghosts))
        super().after(pid, t, hit, victim)
        if drop is not None:
            del self.decay256[drop]
        value, last = self.decay256.get(pid, (0., t))
        self.decay256[pid] = (1 + value * math.exp(-(t-last)/256), t)


def horizon_keys(cap):
    scaled = tuple(math.ceil(a * cap) for a in ALPHAS)
    names = [*(f"H{h}" for h in FIXED), *(f"alpha_{a:g}" for a in ALPHAS), *(f"distinct_{a:g}" for a in ALPHAS)]
    return names, FIXED + scaled, scaled


def make_labels(residents, incoming, stream, t):
    _, fixed, distinct = horizon_keys(len(residents))
    # Slicing a 50k-request trace per group would dwarf the labeler cost.
    details = RolloutDetails(distinct_targets=distinct)
    future = (stream[j] for j in range(t+1, len(stream)))
    absolute, final = rollout(residents, incoming, future, fixed, stop_when_resolved=True, details=details)
    rows = [absolute[h] for h in fixed] + [details.distinct_snapshots[h] for h in distinct]
    turnover = [details.fixed_distinct[h] for h in fixed] + [details.attained_distinct[h] for h in distinct]
    effective = [r.requests for r in rows]
    truncated = [absolute[h].requests < h for h in fixed] + [details.attained_distinct[h] < h for h in distinct]
    adv = np.asarray([row.advantage for row in rows], dtype=np.int32)
    resolved = np.asarray(final.advantage, dtype=np.int32)
    cause = np.asarray(details.additional, dtype=np.uint8)
    saves = np.asarray(details.prevented, dtype=np.uint16)
    lengths = np.asarray(details.lengths, dtype=np.uint16)
    reasons = np.asarray(details.reasons, dtype=np.uint8)
    if not np.array_equal(saves.astype(np.int32)-cause, resolved) or np.min(resolved) < -1:
        raise AssertionError("Invalid exact benefit/risk decomposition")
    maxes = adv.max(axis=1, keepdims=True)
    if np.any((adv == maxes).sum(axis=1) < np.maximum(1, len(residents)-np.asarray(effective))):
        raise AssertionError("Optimal set lower bound failed")
    endpoint = max(final.requests, int(lengths.max()))
    return dict(miss_counts=np.asarray([row.misses for row in rows], dtype=np.uint16),
                advantage=adv, resolved=resolved, prevented=saves, additional=cause,
                branch_lengths=lengths, branch_reasons=reasons, actual_requests=np.asarray(effective,dtype=np.uint16),
                distinct_objects=np.asarray(turnover,dtype=np.uint16), truncated=np.asarray(truncated,dtype=np.bool_),
                resolved_length=np.uint16(lengths.max()), resolved_through=np.uint32(t + int(lengths.max())),
                all_labels_through=np.uint32(t+endpoint), trace_end_uncoalesced=bool((reasons==2).any()))


def orders(expanded, flags, native, family, cap, position, costs):
    rank = np.arange(cap)
    makers = {
        "lru_tail": lambda: rank,
        "low_frequency": lambda: np.lexsort((rank, expanded[:, 1])),
        "sieve_derived": lambda: np.lexsort((rank, flags)),
        "ghost_informed": lambda: np.lexsort((rank, expanded[:, 1], expanded[:, 10])),
        "recency_frequency": lambda: np.argsort(rank/max(1,cap-1)+expanded[:, 7], kind="stable"),
        "causal_mixture": lambda: np.argsort(rank/max(1,cap-1)+expanded[:, 7]+expanded[:, 8]/3, kind="stable"),
    }
    family_number = list(PROTOCOL).index(family)
    for seed in UNIFORM:
        makers[f"uniform_{seed}"] = lambda s=seed: np.random.default_rng(
            np.random.SeedSequence([s,family_number,cap,position])).permutation(cap)
    values = []
    for name,maker in makers.items():
        started = perf_counter()
        order = maker()
        if name == "sieve_derived" and native > 0:
            order = np.r_[native,order[order!=native]]
        order = np.r_[0,order[order!=0][:31]] if name != "lru_tail" else order[:32]
        costs[name] += perf_counter() - started
        values.append(order)
    return np.asarray(values,dtype=np.uint16)


def write_partition(path, groups):
    arrays = {k: np.asarray([r[k] for r in groups]) for k in groups[0]}
    with path.with_suffix(".tmp").open("wb") as f:
        np.savez_compressed(f, **arrays)
    path.with_suffix(".tmp").replace(path)
    return dict(file=path.name, sha256=sha(path), bytes=path.stat().st_size, groups=len(groups),
                first=int(groups[0]["position"]), last=int(groups[-1]["position"]),
                raw_bytes=sum(v.nbytes for v in arrays.values()))


def run_cell(family, cap, data, picks, strata, target, callback=None):
    req, pages, enc, stream, decode_seconds = data
    history = MultiDecay(cap)
    sieve = SievePolicy()
    sieve.reset(cap, pages)
    feature_time = label_time = write_time = state_time = 0.
    costs = Counter()
    feature_costs = Counter()
    partitions, buffer = [], []
    wall_start, cpu_start = perf_counter(), process_time()
    if target is not None:
        target.mkdir(parents=True, exist_ok=False, mode=0o700)
    flags = None
    native_victim = None
    members = None
    trained = 0

    def flush():
        nonlocal write_time
        if target is not None and buffer:
            started = perf_counter()
            partitions.append(write_partition(target / f"groups-{len(partitions):04}.npz", buffer))
            write_time += perf_counter() - started
            buffer.clear()

    def capture(policy, request, candidates):
        nonlocal feature_time, label_time, trained
        t = request.t
        if t not in picks:
            return
        stratum, included, probability = picks[t]
        resident = tuple(enc[p] for p in candidates)
        if len(resident) != cap or enc[request.page_id] in resident:
            raise AssertionError("Invalid ordered full-cache state")
        started = perf_counter()
        legacy = matrix(policy, request, candidates)
        after_legacy = perf_counter()
        expanded = history.candidates(candidates,t)
        after_expanded = perf_counter()
        workload = history.before(request.page_id,t,False)
        after_workload = perf_counter()
        extra = history.extra(candidates,t)
        after_extra = perf_counter()
        feature_costs["legacy"] += after_legacy-started
        feature_costs["expanded"] += after_expanded-after_legacy
        feature_costs["workload"] += after_workload-after_expanded
        feature_costs["decay"] += after_extra-after_workload
        sieve_all = np.asarray([flags.get(p,True) for p in candidates],dtype=np.bool_)
        native = candidates.index(native_victim) if native_victim is not None and members == set(candidates) else -1
        feature_time += perf_counter() - started
        shortlist = orders(expanded,sieve_all,native,family,cap,t,costs)
        started = perf_counter()
        group = make_labels(resident, enc[request.page_id], stream, t)
        label_time += perf_counter() - started
        group.update(position=np.uint32(t), resident=np.asarray(resident,dtype=np.uint32),
                     baseline=np.uint32(resident[0]), incoming=np.uint32(enc[request.page_id]),
                     legacy=legacy, expanded=expanded, workload=workload, decay256=extra,
                     sieve_flags=sieve_all, sieve_native=np.int16(native), shortlist=shortlist,
                     stratum=np.uint8(stratum), budget_included=np.asarray(included,dtype=np.bool_),
                     inclusion_probability=np.asarray(probability,dtype=np.float64),
                     within_stratum_all_labels=group["all_labels_through"] < (HISTORY+(stratum+1)*STRATUM_WIDTH))
        if callback is not None:
            callback(group, resident, enc[request.page_id], t, stream)
        if target is not None:
            buffer.append(group)
            if len(buffer) >= PARTITION_GROUPS:
                flush()
        trained += 1

    policy = Capture(capture)
    policy.reset(cap, pages)
    for request in req:
        t = request.t
        pid = request.page_id
        started = perf_counter()
        if t in picks:
            flags = sieve._visited.copy()
            members = set(sieve._order)
        s = sieve.on_request(request)
        native_victim = s.evicted
        state_time += perf_counter() - started
        e = policy.on_request(request)
        started = perf_counter()
        history.after(pid,t,e.hit,e.evicted)
        state_time += perf_counter() - started
        if trained == len(picks):
            break
    assert trained == len(picks)
    flush()
    return dict(family=family, capacity=cap, trace_sha256=PROTOCOL[family]["sha256"], schema=SCHEMA,
                partitions=partitions, strata=strata, groups=trained, rows=trained*cap,
                label_seconds=label_time, feature_seconds=feature_time, state_seconds=state_time,
                write_seconds=write_time, shortlist_seconds=dict(costs), trace_seconds=decode_seconds,
                feature_stage_seconds=dict(feature_costs),
                wall_seconds=perf_counter()-wall_start,cpu_seconds=process_time()-cpu_start,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, peak_ghosts=history.peak_ghosts,
                effective_horizons=horizon_keys(cap)[0])


def scan(root):
    rows = []
    for f in PROTOCOL:
        data = trace_data(root,f)
        trace_requests = data[3]
        for c in CAPS+DIAGNOSTIC:
            times, full = census(trace_requests,c)
            rows.append(dict(family=f, capacity=c, eligible=len(times), distinct_trace_objects=len(data[2]),
                             full_occupancy_scored_requests=full, diagnostic=c in DIAGNOSTIC,
                             trace_sha256=PROTOCOL[f]["sha256"]))
    cells = [r for r in rows if not r["diagnostic"]]
    quotas = cell_quotas([r["eligible"] for r in cells], BUDGETS)
    for i,r in enumerate(cells):
        r["quotas"] = [quotas[str(b)][i] for b in BUDGETS]
    for r in rows:
        if r["diagnostic"]:
            r["quotas"] = [0,0,min(128,r["eligible"])]
    return rows


def produce(root, out, family):
    plan = json.loads((out / "PLAN.json").read_text())
    data = trace_data(root,family)
    result = []
    family_path = out / "data" / family
    family_path.mkdir(mode=0o700,parents=True,exist_ok=False)
    for cell in [r for r in plan if r["family"] == family]:
        cap = cell["capacity"]
        times, full = census(data[3],cap)
        assert len(times) == cell["eligible"] and full == cell["full_occupancy_scored_requests"]
        picks, strata = selection(times,cell["quotas"],family,cap)
        print(f"START_GROUPS {family} C={cap} N={len(picks)} eligible={len(times)}",flush=True)
        if picks:
            receipt = run_cell(family,cap,data,picks,strata,family_path / str(cap))
        else:
            receipt = dict(groups=0, partitions=[], strata=strata, schema=SCHEMA)
        result.append(dict(cell, **receipt,
                           producer_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                           producer_job_id=os.environ.get("SLURM_JOB_ID"),
                           producer_array_index=int(os.environ.get("SLURM_ARRAY_TASK_ID",list(PROTOCOL).index(family)))))
        print(f"COMPLETE_GROUPS {family} C={cap} N={len(picks)}",flush=True)
    dump(out / "receipts" / f"{family}.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("plan","produce"))
    p.add_argument("trace_root",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--family",choices=tuple(PROTOCOL))
    args = p.parse_args()
    if args.stage == "plan":
        dump(args.output / "PLAN.json", scan(args.trace_root))
    else:
        produce(args.trace_root,args.output,args.family)
