"""Run measured CPU inference and matched control policies."""

from functools import lru_cache
from time import perf_counter_ns

import numpy as np
import torch

from legacy import CLASSICAL, PROTOCOL, TRACE_ROOT, old_policy, trace
from fit import checksum, load
from streaming import RequestPolicy


@lru_cache(maxsize=12)
def inputs(family):
    requests, pages = trace(TRACE_ROOT, family)
    identities = {}
    stream = np.asarray([identities.setdefault(r.page_id, len(identities)) for r in requests], dtype=np.int64)
    return requests, pages, stream


def replay(task):
    method, family, capacity, warm, end, checkpoint, variant, split = task
    torch.set_num_threads(1)
    requests, pages, _ = inputs(family)
    if checkpoint:
        policy = RequestPolicy(load(checkpoint), recompute=variant=="recompute")
        digest = checksum(checkpoint)
    elif method in CLASSICAL:
        policy = CLASSICAL[method]()
        digest = None
    else:
        model, h, seed, strategy = method.split("_",3)
        policy = old_policy(family,int(h[1:]),model,int(seed[1:]),.05 if strategy=="margin_lru" else 0)
        digest = None
    policy.reset(capacity,pages)
    for req in requests[:warm]:
        policy.on_request(req)
    if checkpoint:
        policy.measure = True
    hits = evictions = 0
    phase = [0]*4
    start = perf_counter_ns()
    duration = end-warm
    for i, req in enumerate(requests[warm:end]):
        event = policy.on_request(req)
        hits += int(event.hit)
        evictions += int(event.evicted is not None)
        phase[min(3,4*i//duration)] += int(not event.hit)
    seconds = (perf_counter_ns()-start) / 1e9
    misses = duration-hits
    row = dict(method=method,family=family,capacity=capacity,split=split,score_start=warm,score_end=end,
               requests=duration, hits=hits,misses=misses,evictions=evictions,miss_ratio=misses/duration,
               hit_ratio=hits/duration,seconds=seconds,us_per_request=seconds/duration*1e6,
               trace_sha256=PROTOCOL[family]["sha256"],checkpoint_sha256=digest,variant=variant,
               origin="measured_cpu", **{f"phase{i+1}_misses":v for i,v in enumerate(phase)})
    if checkpoint:
        row.update(feature_us_per_request=policy.feature_ns/duration/1e3,
                   recurrent_us_per_request=policy.update_ns/duration/1e3,
                   scoring_us_per_eviction=policy.score_ns/max(1,evictions)/1e3,
                   eviction_p50_us=float(np.percentile(policy.latencies,50))/1e3 if evictions else None,
                   eviction_p95_us=float(np.percentile(policy.latencies,95))/1e3 if evictions else None,
                   eviction_p99_us=float(np.percentile(policy.latencies,99))/1e3 if evictions else None,
                   tensor_state_peak_bytes=policy.peak_state,metadata_peak_shallow_bytes=policy.peak_metadata_estimate,
                   metadata_objects_peak=policy.peak_history_count,ghost_entries_peak=policy.causal.peak_ghosts)
    return row
