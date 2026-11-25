"""Read the verified November evidence and reuse its simulator and checkpoint loader."""

import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
PROJECT = WORKTREE.parents[1]
NOVEMBER = PROJECT / "worktrees/neural-eviction-20261121/experiments/neural_eviction_20261121"
TRACE_ROOT = PROJECT / "repo/data/processed"
REFERENCE_MODELS = PROJECT / "worktrees/fairness/models/cross_family_v1_staging"
OLD_RUN = NOVEMBER / "runs/campaign"
BELADY = PROJECT / ("archive/Augmented-caching-preserve-20260808-104203/artifacts/"
                    "reviewer_fairness/policy_comparison_offline_belady.csv")

sys.path.insert(0, str(WORKTREE / "src"))
sys.path.append(str(HERE.parent / NOVEMBER.name))

from data import PROTOCOL, historical_misses, sha256, trace  # noqa: E402
from metrics import oldest_within  # noqa: E402
from models import load as load_old, predict  # noqa: E402
from replay import BatchPolicy, CLASSICAL, FAIRNESS, SELECTED, run_policy  # noqa: E402
from lafc.policies.offline_belady import OfflineBeladyPolicy  # noqa: E402


def old_policy(family, horizon, method, seed, margin=0):
    root = OLD_RUN / f"h{horizon}" / family
    info = next(m for m in json.loads((root/"offline.json").read_text())["models"]
                if m["method"] == method and m["seed"] == seed)
    path = root / info["file"]
    if sha256(path) != info["sha256"]:
        raise ValueError(path)
    model, _ = load_old(path, family, horizon)
    return BatchPolicy(lambda x: predict(model, x)[1][0], margin)


def verify_archive():
    traces = {}
    checkpoints = {}
    for family, info in PROTOCOL.items():
        key = TRACE_ROOT / family / "trace.jsonl"
        traces[family] = sha256(key)
        if traces[family] != info["sha256"]:
            raise ValueError(key)
        for kind, expected in info["models"].items():
            key = REFERENCE_MODELS / family / f"evict_value_wulver_v1_{kind}.pkl"
            if sha256(key) != expected:
                raise ValueError(key)
    for run in ("smoke", "pilot", "campaign"):
        for fold in (NOVEMBER / f"runs/{run}").glob("h*/*/offline.json"):
            for file, meta in json.loads(fold.read_text())["fits"].items():
                if file.endswith(".pt"):
                    path = fold.parent / file
                    actual = sha256(path)
                    if actual != meta["checkpoint_sha256"]:
                        raise ValueError(path)
                    checkpoints[str(path.relative_to(NOVEMBER))] = actual
    if len(checkpoints) != 137:
        raise ValueError("Unexpected November checkpoint inventory")
    rows = list(csv.DictReader((OLD_RUN/"replay.csv").open()))
    if len(rows) != 882 or any(int(r["hits"]) + int(r["misses"]) != int(r["requests"]) for r in rows):
        raise ValueError("November accounting error")
    referenced = {"lru", "sieve", "fifo_reinsertion", "lrb", "three_l_cache", "cacheus", "halp", "selected_kbs"}
    for method in referenced:
        file = SELECTED if method == "selected_kbs" else FAIRNESS / f"policy_comparison_{method}.csv"
        for r in (row for row in rows if row["method"] == method):
            if int(r["misses"]) != historical_misses(file, r["family"], int(r["capacity"])):
                raise ValueError((method, r["family"], r["capacity"]))
    return dict(traces=traces, prior_checkpoints=checkpoints, prior_replay_rows=len(rows))


def archived_rows():
    rows = list(csv.DictReader((OLD_RUN/"replay.csv").open()))
    for row in rows:
        method = row["method"]
        if (method in {"lru","sieve","fifo_reinsertion","lrb","three_l_cache","cacheus","halp","selected_kbs","mlp","deepsets"}
                and row["horizon"] == "4"
                or method == "deepsets" and row["horizon"] == "8" and row["seed"] == "7"):
            if method in {"mlp","deepsets"}:
                method = "{}_h{}_s{}_{}".format(method,row["horizon"],row["seed"],row["strategy"])
            yield dict(method=method, family=row["family"], capacity=int(row["capacity"]),
                       misses=int(row["misses"]), requests=int(row["requests"]), split="test",
                       origin="archived_matched", trace_sha256=row["trace_sha256"])
    with BELADY.open() as stream:
        oracle = {(r["trace_sha256"], int(r["capacity"])): r for r in csv.DictReader(stream)
                  if r["policy_variant"] == "primary_controlled_window"}
    for family, info in PROTOCOL.items():
        for cap in (32,64,128):
            row = oracle[(info["sha256"], cap)]
            assert (row["score_start"],row["score_end"],row["scored_requests"],row["status"]) == ("10000","50000","40000","ok")
            yield dict(method="belady", family=family, capacity=cap,
                       misses=int(row["misses"]), requests=40000, split="test",
                       origin="archived_oracle", trace_sha256=info["sha256"])
