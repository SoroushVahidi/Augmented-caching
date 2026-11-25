"""State-aligned cache oracles and collision-verifiable decision representations."""
from collections import Counter
from dataclasses import dataclass
from hashlib import blake2b
import struct

import numpy as np
from numba import njit


def canonical(x, y=None):
    """Canonical rows retain any explicit recency and candidate alignment fields."""
    x = np.array(x, dtype="<f8", order="C", copy=True)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Require finite candidate feature rows")
    x[x == 0] = 0.0
    if y is not None:
        y = np.asarray(y, dtype="<f8")
        y = y.reshape(len(x), -1)
        if not np.isfinite(y).all():
            raise ValueError("Require finite targets")
        x = np.concatenate((x, y), axis=1)
        x[x == 0] = 0.0
    return struct.pack("<II", *x.shape) + b"".join(sorted(row.tobytes() for row in x))


class ExactIndex:
    """Use a digest to locate buckets and verify exact matches with canonical bytes."""

    def __init__(self):
        self.buckets = {}
        self.count = Counter()

    def add(self, data):
        key = blake2b(data, digest_size=16).digest()
        bucket = self.buckets.setdefault(key, [])
        for seen, index in bucket:
            if seen == data:
                self.count[index] += 1
                return index
        index = len(self.count)
        bucket.append((data, index))
        self.count[index] = 1
        return index


@njit(cache=True)
def next_positions(stream, objects):
    last = np.full(objects, len(stream), np.int64)
    result = np.empty(len(stream), np.int64)
    for t in range(len(stream)-1,-1,-1):
        pid = stream[t]
        result[t] = last[pid]
        last[pid] = t
    return result


@njit(cache=True)
def initial_next(stream, start, objects):
    nxt = np.full(objects,len(stream),np.int64)
    for t in range(len(stream)-1,start-1,-1):
        nxt[stream[t]] = t
    return nxt


@njit(cache=True)
def forced_state(residents, incoming, victim):
    state = np.empty_like(residents)
    state[:victim] = residents[:victim]
    state[victim:-1] = residents[victim+1:]
    state[-1] = incoming
    return state


@njit(cache=True)
def higher(left, right, nxt):
    if left < 0:
        return False
    return right < 0 or nxt[left] > nxt[right] or (nxt[left] == nxt[right] and left < right)


@njit(cache=True)
def top_two(cache, nxt):
    first = -1
    second = -1
    for x in cache:
        if higher(x, first, nxt):
            second = first
            first = x
        elif higher(x, second, nxt):
            second = x
    return first, second


@njit(cache=True)
def forced_belady(stream, next_at, start, residents, incoming, objects):
    """Exact suffix misses for every forced victim, with lockstep MIN coupling."""
    c = len(residents)
    nxt = initial_next(stream,start,objects)
    ref_victim, _ = top_two(residents,nxt)
    base_index = np.where(residents==ref_victim)[0][0]
    cache = forced_state(residents,incoming,base_index)
    in_cache = np.zeros(objects,np.bool_)
    for pid in cache:
        in_cache[pid] = True
    ref_only = residents.copy()
    alt_only = np.full(c,ref_victim,np.int64)
    ref_only[base_index] = -1
    delta = np.zeros(c,np.int64)
    base_count = 0
    active = c - 1
    for t in range(start,len(stream)):
        req = stream[t]
        hit = in_cache[req]
        nxt[req] = next_at[t]
        if hit:
            first = -1
            second = -1
            for a in ref_only:
                if a == req:
                    first, second = top_two(cache,nxt)
                    break
        else:
            base_count += 1
            first, second = top_two(cache,nxt)
        if active:
            for i in range(c):
                a, b = ref_only[i], alt_only[i]
                if a < 0:
                    continue
                if hit and req == a:
                    delta[i] += 1
                    shared_best = second if first == a else first
                    if higher(b,shared_best,nxt):
                        ref_only[i] = -1
                        active -= 1
                    else:
                        ref_only[i] = shared_best
                elif not hit and req == b:
                    delta[i] -= 1
                    if first == a:
                        ref_only[i] = -1
                        active -= 1
                    else:
                        alt_only[i] = first
                elif not hit:
                    shared_best = second if first == a else first
                    if higher(b,shared_best,nxt):
                        if first == a:
                            ref_only[i] = -1
                            active -= 1
                        else:
                            alt_only[i] = first
                    elif first == a:
                        ref_only[i] = shared_best
        if not hit:
            pos = np.where(cache==first)[0][0]
            cache[pos] = req
            in_cache[first] = False
            in_cache[req] = True
    return base_count+delta, base_index


@njit(cache=True)
def lru_step(order, req):
    i = len(order)-1
    while i >= 0 and order[i] != req:
        i -= 1
    miss = i < 0
    for j in range(max(0,i),len(order)-1):
        order[j] = order[j+1]
    order[-1] = req
    return int(miss)


@njit(cache=True)
def paired_lru(residents, incoming, future, horizons, cap):
    """Positive advantage means fewer misses than forcing the oldest victim."""
    c = len(residents)
    end = min(len(future),cap)
    baseline = forced_state(residents,incoming,0)
    reference = np.empty((end+1,c),np.int64)
    reference[0] = baseline
    misses = np.zeros(end+1,np.int64)
    for j in range(end):
        misses[j+1] = misses[j]+lru_step(baseline,future[j])
        reference[j+1] = baseline
    deltas = np.zeros((c,len(horizons)),np.int64)
    advantage = np.zeros(c,np.int64)
    length = np.zeros(c,np.int64)
    resolution = np.zeros(c,np.int64)
    for i in range(c):
        alt = forced_state(residents,incoming,i)
        alt_misses = 0
        j = 0
        h = 0
        while j < end and not np.array_equal(alt,reference[j]):
            alt_misses += lru_step(alt,future[j])
            j += 1
            while h < len(horizons) and horizons[h] <= j:
                deltas[i,h] = misses[j]-alt_misses
                h += 1
        while h < len(horizons):
            deltas[i,h] = misses[j]-alt_misses
            h += 1
        advantage[i] = misses[j]-alt_misses
        length[i] = j
        # 1 means exact ordered-state coalescence, 2 means exact end-of-trace comparison.
        resolution[i] = 1 if np.array_equal(alt,reference[j]) else (2 if j==len(future) else 0)
    realized = np.minimum(horizons,end)
    return deltas,advantage,length,resolution,misses[realized]


@njit(cache=True)
def shortlist_hindsight(stream, next_at, capacity, k, warm):
    """Closed-loop next-use oracle restricted to K oldest residents."""
    order = np.full(capacity,-1,np.int64)
    nxt = np.empty(next_at.shape[0]+1,np.int64)
    misses = 0
    count = 0
    for t, pid in enumerate(stream):
        pos = -1
        for i in range(count):
            if order[i] == pid:
                pos = i
                break
        if pos < 0:
            if t >= warm:
                misses += 1
            if count < capacity:
                order[count] = pid
                count += 1
                nxt[pid] = next_at[t]
                continue
            pos = 0
            for i in range(1,min(k,count)):
                if nxt[order[i]] > nxt[order[pos]]:
                    pos = i
        for i in range(pos,count-1):
            order[i] = order[i+1]
        order[count-1] = pid
        nxt[pid] = next_at[t]
    return misses


def censored_reuse(next_at, t, trace_end, horizon):
    """Use [t+1,t+horizon] and the explicitly observed finite suffix."""
    administrative = int(horizon)
    available = trace_end-1-t
    stop = min(administrative,available)
    delay = np.asarray(next_at)-t
    observed = (np.asarray(next_at)<trace_end) & (delay<=stop)
    valid = observed | (available>=administrative)
    return np.where(observed,delay,stop), observed, valid, stop
