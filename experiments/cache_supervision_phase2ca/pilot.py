"""Focused authoritative parity and end-to-end profiling on the execution platform."""
from collections import OrderedDict
import argparse
import os
from pathlib import Path
import platform
import random
from time import perf_counter

import numpy as np

from campaign import CAPS, PROTOCOL, SCHEMA, SEED, census, dump, make_labels, run_cell, scan, selection, trace_data
from data import labels
from exact_lru import RolloutDetails, rollout
from lafc.evict_value_v2_rollout import simulate_rollout_misses
from lafc.types import Request


def authoritative(resident,incoming,sequence,h):
    req = [Request(j,p) for j,p in enumerate(sequence[:h])]
    return np.array([simulate_rollout_misses(
        cache_pages=(*resident[:i],*resident[i+1:],incoming),future_reqs=req,
        capacity=len(resident),reference_policy="lru") for i in range(len(resident))])


def independent_outcomes(residents, incoming, sequence, case):
    """Track exact synchronization times and paired increments without sharing branch work."""
    victims = [OrderedDict.fromkeys((*residents[:i],*residents[i+1:],incoming))
               for i in range(len(residents))]
    base = victims[0]
    costs = [0]*len(victims)
    prevented = [0]*len(victims)
    caused = [0]*len(victims)
    lengths = [0]*len(victims)
    active = set(range(1,len(victims)))
    for cursor, pid in enumerate(sequence,1):
        base_hit = pid in base
        for i,branch in enumerate(victims):
            hit = pid in branch
            if hit:
                branch.move_to_end(pid)
            else:
                branch.popitem(last=False)
                branch[pid] = None
                costs[i] += 1
            prevented[i] += int(hit and not base_hit)
            caused[i] += int(base_hit and not hit)
        for i in tuple(active):
            if tuple(victims[i]) == tuple(base):
                lengths[i] = cursor
                active.remove(i)
        if not active and cursor == len(sequence):
            break
    for i in active:
        lengths[i] = len(sequence)
    expected = ([costs[0]-a for a in costs],prevented,caused,lengths,
                [2 if i in active else 1 for i in range(len(victims))])
    seen = (list(case[0].advantage),case[1].prevented,case[1].additional,
            case[1].lengths,case[1].reasons)
    assert seen == expected, (seen, expected)


def synthetic():
    rng = random.Random(SEED)
    count = horizons = 0
    for c in (1,2,4,8,16,32):
        for i in range(14):
            residents = tuple(rng.sample(range(c),c))
            suffix = ([c]*(3*c) if i==0 else list(range(c,c*3)) if i==1 else
                      [rng.randrange(c+4) for _ in range(rng.randint(0,4*c))])
            h = sorted({0,1,4,8,16,len(suffix)+7})
            details = RolloutDetails(distinct_targets=(1,c,4*c))
            found, final = rollout(residents,c,suffix,h,stop_when_resolved=True,details=details)
            independent_outcomes(residents,c,suffix,(final,details))
            for k in h:
                np.testing.assert_array_equal(found[k].misses, authoritative(residents,c,suffix,k))
                assert found[k].requests == min(k,len(suffix))
                horizons += 1
            for target in (1,c,4*c):
                seen = set()
                end = len(suffix)
                for cursor,page in enumerate(suffix,1):
                    seen.add(page)
                    if len(seen)==target:
                        end=cursor
                        break
                snap = details.distinct_snapshots[target]
                assert snap.requests == end and details.attained_distinct[target] == len(seen)
                np.testing.assert_array_equal(snap.misses,authoritative(residents,c,suffix,end))
                if len(seen) >= c:
                    assert snap.active == 0
                horizons += 1
            count += 1
    return count,horizons


def run(root,out,alignment):
    start = perf_counter()
    count, horizons = synthetic()
    plan = scan(root)
    dump(out/"PLAN.json",plan)
    groups = 0
    resources = []
    with np.load(alignment,allow_pickle=False) as comparative:
        for f in PROTOCOL:
            data = trace_data(root,f)
            for c in CAPS:
                key = f"{f}_{c}"
                time = comparative[key+"_t"]
                lookup = {int(t): i for i,t in enumerate(time)}
                archived_picks = {t:(max(0,(t-10000)//5000),[False,False,True],[0.,0.,1.]) for t in lookup}
                def compare(group,resident,incoming,t,stream):
                    i=lookup[t]
                    np.testing.assert_array_equal(group["legacy"],comparative[key+"_x"][i])
                    np.testing.assert_array_equal(group["miss_counts"][:3].T,comparative[key+"_y"][i])
                    np.testing.assert_array_equal(group["resolved"],comparative[key+"_resolved"][i])
                    next_observed = {}
                    for j in range(t+1,len(stream)):
                        if stream[j] not in next_observed:
                            next_observed[stream[j]]=j
                    np.testing.assert_array_equal([next_observed.get(p,len(stream)) for p in resident],
                                                  comparative[key+"_next"][i])
                    expected=labels(np.asarray(resident,np.int64),incoming,np.asarray(stream[t+1:t+17],np.int64))
                    np.testing.assert_array_equal(group["miss_counts"][:3].T,expected)
                receipt=run_cell(f,c,data,archived_picks,[],None,callback=compare)
                groups+=receipt["groups"]
            print(f"ARCHIVED_PARITY {f} passed",flush=True)
            if f in ("cloudphysics","metakv","wiki2018"):
                for cap,n in ((32,128),(128,384),(1024,32)):
                    times,_=census(data[3],cap)
                    n=min(n,len(times))
                    if n:
                        picked,strata=selection(times,[0,0,n],f,cap)
                        dest=out/"pilot_data"/f/str(cap)
                        resources.append(run_cell(f,cap,data,picked,strata,dest))
                        print(f"PILOT_DATA {f}/{cap} n={n}",flush=True)
    dump(out/"PILOT_RESOURCES.json",resources)
    dump(out/"LABELER_PLATFORM_PARITY.json", dict(
        zero_unexplained_mismatches=True, synthetic_states=count,horizon_comparisons=horizons,
        archived_groups=groups,benchmarked_groups=sum(r["groups"] for r in resources),
        elapsed_seconds=perf_counter()-start, platform=platform.platform(), python=platform.python_version(),
        numpy=np.__version__, hostname=platform.node(), slurm_id=os.getenv("SLURM_JOB_ID"), feature_schema=SCHEMA))


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("trace_root",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("alignment",type=Path)
    args=p.parse_args()
    run(args.trace_root,args.out,args.alignment)
