"""Verified inputs and memory-mapped, group-preserving KBS supervision."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from numba import njit

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS as FEATURES
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.learned_gate.features import compute_predictor_scores
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.types import Request

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = json.loads((HERE / "protocol.json").read_text())
CAPACITIES = (32, 64, 128)
HORIZONS = (4, 8, 16)
HISTORY = 10_000
END = 50_000


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fold(family):
    spec = PROTOCOL[family]
    train, val = tuple(spec["train"]), spec["validation"]
    if len(train) != 5 or set(train) != PROTOCOL.keys() - {family, val} or family == val:
        raise ValueError(f"Invalid family split for {family}")
    return train, val


def trace(root, family):
    path = root / family / "trace.jsonl"
    if sha256(path) != PROTOCOL[family]["sha256"]:
        raise ValueError(f"Trace hash mismatch for {family}: {path}")
    requests, pages, _ = load_trace_from_any(str(path))
    if len(requests) != END:
        raise ValueError(f"Unexpected request count in {path}")
    causal = [
        Request(r.t, r.page_id, metadata={k: r.metadata[k] for k in ("bucket", "confidence") if k in r.metadata})
        for r in requests
    ]
    return causal, pages


def matrix(policy, request, candidates):
    """Compute v1's ordered features once per candidate set."""
    n = len(candidates)
    rank = np.arange(n, dtype=np.float64)
    age = rank / max(1, n - 1)
    lru = 1.0 - age
    predictor_scores = compute_predictor_scores(candidates, policy._bucket_by_page)
    predictor = np.fromiter(predictor_scores.values(), dtype=np.float64, count=n)
    pred = int(predictor.argmax())
    bucket = [float(policy._bucket_by_page.get(p, 0)) for p in candidates]
    confidence = [float(policy._confidence_by_page.get(p, 0.5)) for p in candidates]
    bucket_mean = sum(bucket) / n
    confidence_mean = sum(confidence) / n
    req_counts = Counter(policy._recent_req_hist)
    hit_counts = Counter(policy._recent_hit_hist)
    req_denom = max(1, len(policy._recent_req_hist))
    hit_denom = max(1, len(policy._recent_hit_hist))
    x = np.empty((len(FEATURES), n), dtype=np.float64)
    x[0] = int(request.metadata.get("bucket", 0))
    x[1] = float(request.metadata.get("confidence", 0.5))
    x[2] = bucket
    x[3] = confidence
    x[4] = rank
    x[5] = age
    x[6] = predictor
    x[7] = lru
    x[8] = rank == pred
    x[9] = rank == 0
    x[10] = predictor - predictor[pred]
    x[11] = lru - lru[0]
    x[12] = np.asarray(bucket) - bucket[pred]
    x[13] = np.asarray(bucket) - bucket[0]
    x[14] = np.asarray(confidence) - confidence[pred]
    x[15] = np.asarray(confidence) - confidence[0]
    x[16] = bucket_mean
    x[17] = (sum((b - bucket_mean) ** 2 for b in bucket) / n) ** 0.5
    x[18] = min(bucket)
    x[19] = max(bucket)
    x[20] = len(set(bucket))
    x[21] = confidence_mean
    x[22] = (sum((c - confidence_mean) ** 2 for c in confidence) / n) ** 0.5
    x[23] = pred != 0
    x[24] = [req_counts[p] / req_denom for p in candidates]
    x[25] = [hit_counts[p] / hit_denom for p in candidates]
    return x.T


@njit(cache=True)
def labels(candidates, incoming, future):
    """Force each victim, then replay one LRU continuation per victim for all H."""
    n = len(candidates)
    result = np.empty((n, 3), dtype=np.uint8)
    order = np.empty(n, dtype=np.int64)
    for victim in range(n):
        order[:victim] = candidates[:victim]
        order[victim:n - 1] = candidates[victim + 1:]
        order[n - 1] = incoming
        misses = 0
        for i in range(16):
            if i < len(future):
                pid = future[i]
                pos = n - 1
                while pos >= 0 and order[pos] != pid:
                    pos -= 1
                if pos == -1:
                    misses += 1
                    pos = 0
                for j in range(pos, n - 1):
                    order[j] = order[j + 1]
                order[n - 1] = pid
            if i == 3:
                result[victim, 0] = misses
            elif i == 7:
                result[victim, 1] = misses
        result[victim, 2] = misses
    return result


class Capture(EvictValueV1Policy):
    def __init__(self, callback):
        super().__init__(scorer_mode="lightweight", history_window=64)
        self.callback = callback

    def _choose_victim(self, request):
        candidates = list(self._order)
        self.callback(self, request, candidates)
        return candidates[0], {}


def build(data_root, out, family, capacity, limit, seed, end=END):
    requests, pages = trace(data_root, family)
    requests = requests[:end]
    policy = LRUPolicy()
    policy.reset(capacity, pages)
    times = np.array([r.t for r in requests if policy.on_request(r).evicted is not None], dtype=np.int64)
    eligible = len(times)
    if limit and limit < eligible:
        times = np.sort(np.random.default_rng(seed).choice(times, limit, replace=False))
    if not len(times):
        raise ValueError(f"No full-cache misses for {family}/{capacity}")
    lookup = {int(t): i for i, t in enumerate(times)}
    ids = {pid: i for i, pid in enumerate(pages)}
    stream = np.array([ids[r.page_id] for r in requests], dtype=np.int64)
    cell = out / f"{family}-{capacity}"
    cell.mkdir(parents=True, exist_ok=False)
    x = np.lib.format.open_memmap(cell / "x.npy", "w+", np.float64, (len(times), capacity, len(FEATURES)))
    y = np.lib.format.open_memmap(cell / "y.npy", "w+", np.uint8, (len(times), capacity, len(HORIZONS)))

    def capture(state, request, candidates):
        index = lookup.get(request.t)
        if index is not None:
            x[index] = matrix(state, request, candidates)
            resident = np.array([ids[p] for p in candidates], dtype=np.int64)
            y[index] = labels(resident, stream[request.t], stream[request.t + 1:request.t + 17])

    policy = Capture(capture)
    policy.reset(capacity, pages)
    for request in requests:
        policy.on_request(request)
    x.flush()
    y.flush()
    np.save(cell / "t.npy", times, allow_pickle=False)
    metadata = dict(family=family, capacity=capacity, trace_sha256=PROTOCOL[family]["sha256"],
                    trace_end=end, eligible=eligible, groups=len(times), columns=FEATURES,
                    horizons=HORIZONS, continuation="LRU", seed=seed, limit=limit)
    (cell / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_cell(data, family, capacity):
    cell = data / f"{family}-{capacity}"
    metadata = json.loads((cell / "manifest.json").read_text())
    if (metadata["family"], metadata["capacity"], metadata["trace_sha256"], metadata["columns"],
        metadata["horizons"], metadata["continuation"]) != (
        family, capacity, PROTOCOL[family]["sha256"], FEATURES, list(HORIZONS), "LRU"
    ):
        raise ValueError(f"Invalid data manifest in {cell}")
    return tuple(np.load(cell / f"{name}.npy", mmap_mode="r", allow_pickle=False) for name in ("x", "y", "t"))


def historical(reader, family, capacity):
    matches = [
        row for row in reader if row["trace_sha256"] == PROTOCOL[family]["sha256"]
        and int(row["capacity"]) == capacity and row["policy_variant"] == "primary_controlled_window"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one exact historical match for {family}/{capacity}")
    row = matches[0]
    if (row["history_start"], row["history_end"], row["score_start"], row["score_end"], row["scored_requests"],
        row["capacity_semantics"], row["object_size_semantics"], row["future_information"], row["status"]) != (
        "0", str(HISTORY), str(HISTORY), str(END), str(END - HISTORY), "object_slots", "unit", "none", "ok"
    ) or int(row["hits"]) + int(row["misses"]) != END - HISTORY:
        raise ValueError(f"Invalid historical accounting for {family}/{capacity}")
    return int(row["misses"])


def historical_misses(path, family, capacity):
    with path.open(newline="") as stream:
        return historical(csv.DictReader(stream), family, capacity)
