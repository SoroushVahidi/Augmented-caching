#!/usr/bin/env python3
"""Frozen, held-out closed-loop replay and compact campaign tooling."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lafc.evict_value_model_v1 import EvictValueV1Model
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.metrics.cost import total_misses
from lafc.policies.base import BasePolicy
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy
from lafc.policies.lfu import LFUPolicy
from lafc.policies.lru import LRUPolicy
from lafc.policies.sieve import SievePolicy
from lafc.runner.run_policy import run_policy
from lafc.types import CacheEvent, Page, PageId, Request

MODEL_SHA = "8ba5f6e17b9293615b811b1922317ec7b1fe51769d2377f9846ede579062bcd6"
RANDOM_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
POLICIES = ["learned", "lru", "mru", "random", "sieve", "lfu"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class MRUPolicy(LRUPolicy):
    name = "mru"

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        if self.in_cache(pid):
            self._order.move_to_end(pid)
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)
        cost = self._pages[pid].weight
        self._record_miss(cost)
        evicted = None
        if self._cache.is_full():
            evicted, _ = self._order.popitem(last=True)
            self._evict(evicted)
        self._add(pid)
        self._order[pid] = None
        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)


class RandomPolicy(BasePolicy):
    name = "random"

    def __init__(self, seed: int):
        self.seed = seed

    def reset(self, capacity: int, pages: dict[PageId, Page]) -> None:
        super().reset(capacity, pages)
        self._residents: list[PageId] = []
        self._rng = random.Random(self.seed)

    def on_request(self, request: Request) -> CacheEvent:
        pid = request.page_id
        if self.in_cache(pid):
            self._record_hit()
            return CacheEvent(t=request.t, page_id=pid, hit=True, cost=0.0)
        cost = self._pages[pid].weight
        self._record_miss(cost)
        evicted = None
        if self._cache.is_full():
            evicted = self._rng.choice(self._residents)
            self._residents.remove(evicted)
            self._evict(evicted)
        self._add(pid)
        self._residents.append(pid)
        return CacheEvent(t=request.t, page_id=pid, hit=False, cost=cost, evicted=evicted)


def policy(name: str, model_path: Path, seed: int) -> BasePolicy:
    if name == "learned":
        return EvictValueV1Policy(model_path=str(model_path), scorer_mode="artifact")
    if name == "lru":
        return LRUPolicy()
    if name == "mru":
        return MRUPolicy()
    if name == "random":
        return RandomPolicy(seed)
    if name == "sieve":
        return SievePolicy()
    if name == "lfu":
        return LFUPolicy()
    raise ValueError(f"unknown policy {name}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def model_path(args: argparse.Namespace) -> Path:
    return Path(args.model)


def task_manifest(args: argparse.Namespace) -> dict[str, Any]:
    return load_json(Path(args.manifest))


def run_task(args: argparse.Namespace) -> None:
    manifest = task_manifest(args)
    task = manifest["tasks"][int(args.task_id)]
    out = Path(args.out_root) / task["output_key"] / "result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        try:
            old = load_json(out)
            if old.get("status") == "COMPLETED" and old.get("protocol_sha256") == manifest["protocol_sha256"]:
                print(json.dumps({"status": "SKIPPED_EXISTING", "path": str(out)}))
                return
        except (OSError, ValueError, KeyError):
            pass
    spec = manifest["cells_by_key"][task["cell_key"]]
    trace_path = Path(spec["trace_path"])
    reqs, pages, _ = load_trace_from_any(str(trace_path))
    lo, hi = [int(x) for x in spec["test_request_range"].split("..")]
    scored = [r for r in reqs if lo <= int(r.t) <= hi]
    if not scored or int(scored[0].t) != lo or int(scored[-1].t) != hi:
        raise RuntimeError(f"test range did not resolve exactly for {trace_path}: {lo}..{hi}")
    start = time.perf_counter()
    result = run_policy(policy(task["policy"], model_path(args), int(task["seed"])), scored, pages, int(spec["capacity"]))
    wall = time.perf_counter() - start
    misses = int(total_misses(result.events))
    payload = {
        "status": "COMPLETED",
        "protocol_sha256": manifest["protocol_sha256"],
        "model_sha256": MODEL_SHA,
        "runner_commit": manifest["runner_commit"],
        "family": spec["family"], "capacity": spec["capacity"],
        "policy": task["policy"], "seed": task["seed"],
        "trace_path": str(trace_path), "test_request_range": spec["test_request_range"],
        "scored_requests": len(scored), "hits": len(scored) - misses, "misses": misses,
        "miss_ratio": misses / len(scored), "evictions": sum(e.evicted is not None for e in result.events),
        "wallclock_seconds": wall, "model_inference_time_seconds": None,
        "warmup": "none; common empty cache and empty bounded-history state at test boundary",
    }
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(json.dumps(payload, sort_keys=True))


def validate(args: argparse.Namespace) -> None:
    m = task_manifest(args); root = Path(args.out_root); errors = []
    expected = {(t["output_key"], t["policy"], t["seed"]) for t in m["tasks"]}
    seen = set()
    for key, pol, seed in expected:
        p = root / key / "result.json"
        if not p.exists(): errors.append(f"missing:{p}"); continue
        x = load_json(p); seen.add((key, pol, seed))
        for field, expected_value in [("status", "COMPLETED"), ("protocol_sha256", m["protocol_sha256"]), ("model_sha256", MODEL_SHA)]:
            if x.get(field) != expected_value: errors.append(f"{p}:{field}")
        if not (0 <= float(x.get("miss_ratio", -1)) <= 1): errors.append(f"range:{p}")
    report = {"status": "PASS" if not errors and seen == expected else "FAIL", "expected_tasks": len(expected), "completed_tasks": len(seen), "errors": errors}
    Path(args.report).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if errors or seen != expected: raise SystemExit(1)


def aggregate(args: argparse.Namespace) -> None:
    m = task_manifest(args); root = Path(args.out_root); rows=[]
    for t in m["tasks"]:
        x=load_json(root/t["output_key"]/"result.json"); rows.append(x)
    out=Path(args.project_root); out.mkdir(parents=True, exist_ok=True)
    (out/"per_cell_metrics.csv").write_text("family,capacity,policy,seed,scored_requests,misses,miss_ratio,evictions,wallclock_seconds\n"+"\n".join(",".join(str(x[k]) for k in ["family","capacity","policy","seed","scored_requests","misses","miss_ratio","evictions","wallclock_seconds"]) for x in rows)+"\n",encoding="utf-8")
    by_policy={}
    for x in rows: by_policy.setdefault(x["policy"],[]).append(x)
    summary={"campaign_status":"COMPLETE","protocol_sha256":m["protocol_sha256"],"model_sha256":MODEL_SHA,"cells":len(m["cells_by_key"]),"tasks":len(rows),"by_policy":{p:{"mean_miss_ratio":sum(x["miss_ratio"] for x in xs)/len(xs),"tasks":len(xs)} for p,xs in by_policy.items()}}
    (out/"campaign_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"random_seed_metrics.csv").write_text("\n".join(",".join(str(x[k]) for k in ["family","capacity","seed","miss_ratio"]) for x in rows if x["policy"]=="random")+"\n",encoding="utf-8")
    (out/"provenance.json").write_text(json.dumps({"protocol_sha256":m["protocol_sha256"],"model_sha256":MODEL_SHA,"runner_commit":m["runner_commit"],"status":"COMPACT_AGGREGATION"},indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(summary,sort_keys=True))


def main() -> None:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    for name in ("run-task","validate","aggregate"):
        p=sub.add_parser(name); p.add_argument("--manifest",required=True); p.add_argument("--out-root",required=True)
        if name=="run-task": p.add_argument("--task-id",required=True,type=int); p.add_argument("--model",required=True)
        if name=="validate": p.add_argument("--report",required=True)
        if name=="aggregate": p.add_argument("--project-root",required=True)
    args=ap.parse_args(); {"run-task":run_task,"validate":validate,"aggregate":aggregate}[args.cmd](args)

if __name__ == "__main__": main()
