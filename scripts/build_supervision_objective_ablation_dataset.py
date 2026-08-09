"""Build the shared multi-objective candidate dataset for ONE cross-family
fold of the supervision-objective ablation (docs/supervision_objective_ablation_protocol.md,
configs/supervision_objective_ablation_v1.json).

Reuses the frozen `reviewer_fair_cross_family_v1` fold definitions
(configs/fair_cross_family_v1/folds/<family>.json,
<family>_family_split_map.json) UNCHANGED -- no new family rotation or
split logic is invented here. Streams candidate rows to disk in bounded
shards (never materializes a whole fold's dataset in memory at once): the
canonical single-objective pipeline OOM-killed at Stage 2 training
(57GB RSS, see journalctl evidence, PID 253445, 2026-08-06 23:51:09) when
it loaded an entire multi-family manifest into memory with no row cap.
This module's rows carry 5 label columns instead of 1, so the same
in-memory-list pattern would be worse; every stage here streams instead.

For pairwise rows, C(k,2) grows quadratically in the candidate-set size k
(up to capacity=128 -> 8128 pairs/decision) -- this build processes ONE
decision's candidate rows at a time and immediately derives that
decision's (capped) pairwise rows before moving on, so peak memory is
O(capacity), never O(decisions x capacity^2).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_wulver_v1 import load_trace_from_any
from lafc.experiments.external_baseline_common import base_provenance, sha256_of_file
from lafc.supervision_objective_ablation import (
    ObjectiveAblationConfig,
    build_pairwise_rows,
    iter_multi_label_candidate_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
FOLDS_DIR = Path("configs/fair_cross_family_v1/folds")
FAIRNESS_WORKTREE_FOLDS_DIR = REPO_ROOT.parent / "Augmented-caching-fairness" / "configs/fair_cross_family_v1/folds"

SCALAR_LABEL_COLUMNS = [
    "eviction_loss_label",
    "next_arrival_label_raw",
    "next_arrival_label_censored",
    "next_arrival_censored_flag",
    "reuse_distance_label_raw",
    "reuse_distance_label_censored",
    "reuse_distance_censored_flag",
]
SCALAR_FIELDNAMES = (
    ["example_id", "trace_name", "trace_family", "split", "capacity", "horizon", "decision_id", "decision_t", "candidate_page_id"]
    + SCALAR_LABEL_COLUMNS
    + list(EVICT_VALUE_V1_FEATURE_COLUMNS)
)
PAIRWISE_FIELDNAMES = (
    ["pair_id", "trace_name", "trace_family", "split", "capacity", "horizon", "decision_id",
     "pairwise_label_source", "candidate_i_page_id", "candidate_j_page_id", "value_i", "value_j",
     "label_i_preferred", "is_tie"]
    + [f"i_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
    + [f"j_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
    + [f"delta_{c}" for c in EVICT_VALUE_V1_FEATURE_COLUMNS]
)


class _ShardWriter:
    """Bounded-memory CSV shard writer: holds at most max_rows_per_shard
    rows in memory before flushing to disk, mirroring
    scripts/build_evict_value_dataset_wulver_v1.py's shard pattern."""

    def __init__(self, out_dir: Path, prefix: str, fieldnames: List[str], max_rows_per_shard: int):
        self.out_dir = out_dir
        self.prefix = prefix
        self.fieldnames = fieldnames
        self.max_rows_per_shard = max_rows_per_shard
        self._buf: List[Dict[str, object]] = []
        self._shard_index = 0
        self.shards: List[Dict[str, object]] = []
        self.total_rows = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def add(self, row: Dict[str, object]) -> None:
        self._buf.append(row)
        self.total_rows += 1
        if len(self._buf) >= self.max_rows_per_shard:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        path = self.out_dir / f"{self.prefix}.part{self._shard_index:04d}.csv"
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=self.fieldnames)
            w.writeheader()
            w.writerows(self._buf)
        self.shards.append({"path": str(path), "row_count": len(self._buf)})
        self._shard_index += 1
        self._buf = []

    def close(self) -> None:
        self._flush()


def _verify_fold_identical_to_fairness_worktree(family: str) -> Dict[str, object]:
    local_path = FOLDS_DIR / f"{family}.json"
    if not local_path.exists():
        raise FileNotFoundError(f"Fold manifest not found: {local_path}")
    local_bytes = local_path.read_bytes()
    if FAIRNESS_WORKTREE_FOLDS_DIR.exists():
        ref_path = FAIRNESS_WORKTREE_FOLDS_DIR / f"{family}.json"
        if ref_path.exists():
            ref_bytes = ref_path.read_bytes()
            if hashlib.sha256(local_bytes).hexdigest() != hashlib.sha256(ref_bytes).hexdigest():
                raise ValueError(
                    f"Fold {family}'s manifest differs from the frozen copy in "
                    f"{FAIRNESS_WORKTREE_FOLDS_DIR} -- refusing to build a dataset "
                    "against a protocol that may have drifted. Aborting."
                )
    return json.loads(local_bytes)


def _example_id(decision_id: str, candidate: str) -> str:
    return f"{decision_id}|{candidate}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--held-out-family", required=True)
    ap.add_argument("--data-read-root", type=Path, default=Path("."),
                     help="Root to resolve fold trace paths against (this worktree's "
                     "data/processed/ is empty; point at the primary checkout).")
    ap.add_argument("--capacities", default="32,64,128")
    ap.add_argument("--horizon", type=int, default=4, help="MUST match the frozen protocol's H=4. Not free to change.")
    ap.add_argument("--max-rows-per-shard", type=int, default=200000)
    ap.add_argument("--max-pairs-per-decision", type=int, default=6,
                     help="Resource-safety cap on pairwise TRAINING rows per decision "
                     "(does not alter pairwise semantics or evaluation). C(4,2)=6.")
    ap.add_argument("--pairwise-sample-seed", type=int, default=0)
    ap.add_argument("--max-requests-per-trace", type=int, default=None,
                     help="Optional prefix cap for smoke testing only.")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    if args.horizon != 4:
        raise ValueError(
            f"--horizon={args.horizon} != 4: the frozen protocol "
            "(configs/supervision_objective_ablation_v1.json) fixes H=4. Aborting."
        )

    capacities = [int(x.strip()) for x in args.capacities.split(",") if x.strip()]
    fold = _verify_fold_identical_to_fairness_worktree(args.held_out_family)
    family_split_map: Dict[str, str] = json.loads(
        (FOLDS_DIR / f"{args.held_out_family}_family_split_map.json").read_text(encoding="utf-8")
    )
    held_out = fold["test_family"]
    if held_out in family_split_map:
        raise ValueError(f"Held-out family {held_out} appears in its own family_split_map -- fold isolation violated.")
    if set(family_split_map.values()) - {"train", "val"}:
        raise ValueError(f"family_split_map for {held_out} contains a split other than train/val: {family_split_map}")
    expected_families = set(fold["training_families"]) | {fold["validation_family"]}
    if set(family_split_map.keys()) != expected_families:
        raise ValueError(
            f"family_split_map keys {set(family_split_map.keys())} != fold's declared "
            f"training+validation families {expected_families}"
        )

    train_manifest_path = Path(fold["train_manifest"])
    rows_manifest = list(csv.DictReader(train_manifest_path.open(encoding="utf-8")))
    families_in_manifest = {r["trace_family"] for r in rows_manifest}
    if held_out in families_in_manifest:
        raise ValueError(f"Held-out family {held_out} appears in its own train manifest -- aborting.")
    if families_in_manifest != expected_families:
        raise ValueError(
            f"Train manifest families {families_in_manifest} != expected {expected_families} for fold {held_out}"
        )

    out_root = args.out_dir or Path(f"data/derived/supervision_objective_ablation_v1/{held_out}")
    out_root.mkdir(parents=True, exist_ok=True)
    scalar_dir = out_root / "scalar" / "shards"
    pairwise_dir = out_root / "pairwise" / "shards"

    cfg = ObjectiveAblationConfig(horizon=args.horizon)
    trace_stats: List[Dict[str, object]] = []
    scalar_manifest_items: List[Dict[str, object]] = []
    pairwise_manifest_items: List[Dict[str, object]] = []
    label_stats: Dict[str, object] = {"per_capacity": {}}

    for rec in rows_manifest:
        family = rec["trace_family"]
        split = family_split_map[family]
        trace_path = args.data_read_root / rec["path"]
        if not trace_path.exists():
            raise FileNotFoundError(f"Training trace not found: {trace_path}")
        trace_hash = sha256_of_file(trace_path)
        reqs, _pages, _src = load_trace_from_any(str(trace_path))
        if args.max_requests_per_trace is not None:
            reqs = reqs[: args.max_requests_per_trace]
        trace_name = rec["trace_name"]
        trace_stats.append({
            "trace_name": trace_name, "trace_family": family, "split": split,
            "path": str(trace_path), "trace_sha256": trace_hash, "request_count": len(reqs),
        })

        for cap in capacities:
            safe_trace = trace_name.replace("/", "__").replace(":", "_")
            prefix = f"{safe_trace}__cap{cap}"
            scalar_writer = _ShardWriter(scalar_dir, prefix, SCALAR_FIELDNAMES, args.max_rows_per_shard)
            pairwise_writer = _ShardWriter(pairwise_dir, prefix, PAIRWISE_FIELDNAMES, args.max_rows_per_shard)

            key = f"cap{cap}"
            stat = label_stats["per_capacity"].setdefault(
                key, {"scalar_rows": 0, "pairwise_rows": 0, "decisions": 0,
                      "next_arrival_censored_count": 0, "reuse_distance_censored_count": 0}
            )

            decision_buf: List[Dict[str, object]] = []
            current_decision_id = None

            def _flush_decision(buf: List[Dict[str, object]]) -> None:
                if not buf:
                    return
                stat["decisions"] += 1
                for row in buf:
                    scalar_row = dict(row)
                    scalar_row["example_id"] = _example_id(str(row["decision_id"]), str(row["candidate_page_id"]))
                    scalar_row["split"] = split
                    na_censored = row["next_arrival_label_raw"] != row["next_arrival_label_censored"]
                    rd_censored = row["reuse_distance_label_raw"] != row["reuse_distance_label_censored"]
                    scalar_row["next_arrival_censored_flag"] = int(na_censored)
                    scalar_row["reuse_distance_censored_flag"] = int(rd_censored)
                    if na_censored:
                        stat["next_arrival_censored_count"] += 1
                    if rd_censored:
                        stat["reuse_distance_censored_count"] += 1
                    scalar_writer.add({k: scalar_row[k] for k in SCALAR_FIELDNAMES})
                    stat["scalar_rows"] += 1

                pairs = build_pairwise_rows(
                    buf, source="next_arrival",
                    max_pairs_per_decision=args.max_pairs_per_decision,
                    sample_seed=args.pairwise_sample_seed,
                )
                for p in pairs:
                    prow = dict(p)
                    prow["pair_id"] = f"{p['decision_id']}|{p['candidate_i_page_id']}|{p['candidate_j_page_id']}"
                    prow["trace_name"] = trace_name
                    prow["split"] = split
                    pairwise_writer.add({k: prow[k] for k in PAIRWISE_FIELDNAMES})
                    stat["pairwise_rows"] += 1

            for row in iter_multi_label_candidate_rows(reqs, cap, trace_name, family, cfg):
                if row["decision_id"] != current_decision_id:
                    _flush_decision(decision_buf)
                    decision_buf = []
                    current_decision_id = row["decision_id"]
                decision_buf.append(row)
            _flush_decision(decision_buf)

            scalar_writer.close()
            pairwise_writer.close()
            scalar_manifest_items.extend(scalar_writer.shards)
            pairwise_manifest_items.extend(pairwise_writer.shards)
            print(
                f"[done] {prefix} split={split} scalar_shards={len(scalar_writer.shards)} "
                f"scalar_rows={scalar_writer.total_rows} pairwise_shards={len(pairwise_writer.shards)} "
                f"pairwise_rows={pairwise_writer.total_rows}"
            )

    manifest = {
        "format": "supervision_objective_ablation_v1_candidate_csv_shards",
        "protocol_id": "supervision_objective_ablation_v1",
        "held_out_family": held_out,
        "fold_id": fold["fold_id"],
        "training_families": fold["training_families"],
        "validation_family": fold["validation_family"],
        "horizon": args.horizon,
        "capacities": capacities,
        "max_pairs_per_decision": args.max_pairs_per_decision,
        "pairwise_sample_seed": args.pairwise_sample_seed,
        "max_requests_per_trace": args.max_requests_per_trace,
        "trace_stats": trace_stats,
        "scalar_shards": scalar_manifest_items,
        "pairwise_shards": pairwise_manifest_items,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_root / "label_statistics.json").write_text(json.dumps(label_stats, indent=2), encoding="utf-8")
    provenance = {
        **base_provenance(),
        "held_out_family": held_out,
        "fold_id": fold["fold_id"],
        "scalar_row_total": sum(s["row_count"] for s in scalar_manifest_items),
        "pairwise_row_total": sum(s["row_count"] for s in pairwise_manifest_items),
    }
    (out_root / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote manifest={out_root / 'manifest.json'}")


if __name__ == "__main__":
    main()
