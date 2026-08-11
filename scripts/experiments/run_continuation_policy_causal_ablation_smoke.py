"""Tiny local smoke for the continuation-policy causal ablation.

This is intentionally not a campaign runner. It samples a small bounded
number of C1/C2 decision-aligned labels, trains one pi2 model from C2
labels, and evaluates frozen pi1 versus pi2 on a short held-out prefix.
Full seven-fold execution belongs on Wulver under the frozen protocol
config.
"""

from __future__ import annotations

import argparse
import csv
import json
import tempfile
import time
from pathlib import Path
from typing import Dict, List

from lafc.continuation_policy_ablation import (
    ContinuationAblationConfig,
    build_decision_aligned_continuation_rows,
    label_agreement_metrics,
    load_frozen_pi1_from_registry,
    sha256_of_file,
    train_pi2_from_c2_labels,
)
from lafc.distribution_shift_ablation import DistributionShiftEvalPolicy
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.policies.lru import LRUPolicy
from lafc.policies.base import BasePolicy
from lafc.runner.run_policy import run_policy


FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
DEFAULT_REGISTRY = Path("analysis/supervision_objective_ablation_v1/model_registry.json")


def _load_fold(family: str) -> Dict[str, object]:
    return json.loads((FOLDS_DIR / f"{family}.json").read_text(encoding="utf-8"))


def _load_split_map(family: str) -> Dict[str, str]:
    return json.loads((FOLDS_DIR / f"{family}_family_split_map.json").read_text(encoding="utf-8"))


def _load_train_manifest(family: str) -> List[Dict[str, str]]:
    fold = _load_fold(family)
    with open(fold["train_manifest"], newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _collect_rows_for_split(
    *,
    held_out_family: str,
    split_name: str,
    capacity: int,
    max_decisions: int,
    max_requests_per_trace: int,
    cfg: ContinuationAblationConfig,
    pi1_model,
    pi1_provenance,
    data_read_root: Path,
) -> List[Dict[str, object]]:
    split_map = _load_split_map(held_out_family)
    rows: List[Dict[str, object]] = []
    for rec in _load_train_manifest(held_out_family):
        if split_map[rec["trace_family"]] != split_name:
            continue
        reqs, _pages, _src = load_trace_from_any(str(data_read_root / rec["path"]))
        if max_requests_per_trace:
            reqs = reqs[:max_requests_per_trace]
        need = max_decisions - len({r["decision_id"] for r in rows})
        if need <= 0:
            break
        new_rows = build_decision_aligned_continuation_rows(
            requests=reqs,
            capacity=capacity,
            trace_name=rec["trace_name"],
            trace_family=rec["trace_family"],
            cfg=cfg,
            pi1_model=pi1_model,
            pi1_provenance=pi1_provenance,
            max_decisions=need,
        )
        for row in new_rows:
            row["split"] = split_name
        rows.extend(new_rows)
    return rows


def _misses_for_policy(policy: BasePolicy, reqs, pages, capacity: int) -> Dict[str, float]:
    t0 = time.time()
    result = run_policy(policy, reqs, pages, capacity)
    wall_s = time.time() - t0
    return {
        "misses": float(result.total_misses),
        "requests": float(len(result.events)),
        "miss_ratio": float(result.total_misses / max(len(result.events), 1)),
        "runtime_seconds": wall_s,
    }


def _misses_for_model(model_path: Path, reqs, pages, capacity: int) -> Dict[str, float]:
    return _misses_for_policy(
        DistributionShiftEvalPolicy(model_path=str(model_path)),
        reqs,
        pages,
        capacity,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--held-out-family", default="brightkite")
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--prefix", type=int, default=1200)
    ap.add_argument("--max-requests-per-train-trace", type=int, default=2500)
    ap.add_argument("--max-train-decisions", type=int, default=20)
    ap.add_argument("--max-val-decisions", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--data-read-root", type=Path, default=Path("."))
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    started = time.time()
    cfg = ContinuationAblationConfig(horizon=args.horizon)
    pi1_model, pi1_prov = load_frozen_pi1_from_registry(
        registry_path=args.registry,
        held_out_family=args.held_out_family,
    )

    train_rows = _collect_rows_for_split(
        held_out_family=args.held_out_family,
        split_name="train",
        capacity=args.capacity,
        max_decisions=args.max_train_decisions,
        max_requests_per_trace=args.max_requests_per_train_trace,
        cfg=cfg,
        pi1_model=pi1_model,
        pi1_provenance=pi1_prov,
        data_read_root=args.data_read_root,
    )
    val_rows = _collect_rows_for_split(
        held_out_family=args.held_out_family,
        split_name="val",
        capacity=args.capacity,
        max_decisions=args.max_val_decisions,
        max_requests_per_trace=args.max_requests_per_train_trace,
        cfg=cfg,
        pi1_model=pi1_model,
        pi1_provenance=pi1_prov,
        data_read_root=args.data_read_root,
    )
    if not train_rows or not val_rows:
        raise ValueError(f"insufficient smoke rows: train={len(train_rows)} val={len(val_rows)}")

    pi2 = train_pi2_from_c2_labels(
        train_rows=train_rows,
        val_rows=val_rows,
        seed=args.seed,
        pi1_provenance=pi1_prov,
    )

    fold = _load_fold(args.held_out_family)
    held_reqs, held_pages, _src = load_trace_from_any(str(args.data_read_root / fold["test_trace_path"]))
    held_reqs = held_reqs[:args.prefix]
    with tempfile.TemporaryDirectory(prefix="lafc_continuation_smoke_") as td:
        pi2_path = Path(td) / "pi2.pkl"
        pi2.save(pi2_path)
        c0_result = _misses_for_policy(LRUPolicy(), held_reqs, held_pages, args.capacity)
        pi1_result = _misses_for_model(Path(pi1_prov.model_path), held_reqs, held_pages, args.capacity)
        pi2_result = _misses_for_model(pi2_path, held_reqs, held_pages, args.capacity)
        pi2_hash = sha256_of_file(pi2_path)

    label_metrics = label_agreement_metrics(train_rows + val_rows)
    summary = {
        "protocol_id": "continuation_policy_causal_ablation_v1_smoke",
        "held_out_family": args.held_out_family,
        "capacity": args.capacity,
        "prefix": args.prefix,
        "horizon": args.horizon,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_decisions": len({r["decision_id"] for r in train_rows}),
        "val_decisions": len({r["decision_id"] for r in val_rows}),
        "pi1_hash": pi1_prov.model_sha256,
        "pi2_hash": pi2_hash,
        "c0_lru_result": c0_result,
        "pi1_result": pi1_result,
        "pi2_result": pi2_result,
        "delta_pi1_minus_c0_misses": pi1_result["misses"] - c0_result["misses"],
        "delta_pi2_minus_c0_misses": pi2_result["misses"] - c0_result["misses"],
        "delta_pi2_minus_pi1_misses": pi2_result["misses"] - pi1_result["misses"],
        "label_metrics": label_metrics,
        "runtime_seconds": time.time() - started,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
