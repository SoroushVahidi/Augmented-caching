from __future__ import annotations

import collections
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from lafc.evict_value_dataset_v1 import _simulate_lru_misses
from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS, compute_candidate_features_v1
from lafc.simulator.request_trace import build_requests_from_lists, load_trace
from lafc.types import Page, PageId, Request


@dataclass(frozen=True)
class WulverDatasetConfig:
    horizons: Tuple[int, ...]
    history_window: int = 64
    split_mode: str = "trace_chunk"
    chunk_size: int = 4096
    split_train_pct: int = 70
    split_val_pct: int = 15
    split_seed: int = 0


@dataclass(frozen=True)
class TraceSpec:
    path: str
    trace_name: str
    dataset_source: str
    trace_family: str


def _stable_bucket(key: str, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100


def assign_split(
    *,
    split_mode: str,
    trace_name: str,
    dataset_source: str,
    trace_family: str,
    t: int,
    chunk_size: int,
    train_pct: int,
    val_pct: int,
    seed: int,
) -> str:
    if split_mode not in {"trace_chunk", "source_family"}:
        raise ValueError(f"Unsupported split_mode={split_mode}")
    if train_pct <= 0 or val_pct <= 0 or train_pct + val_pct >= 100:
        raise ValueError("train_pct and val_pct must be positive and sum to < 100")

    if split_mode == "trace_chunk":
        chunk_id = max(t, 0) // max(chunk_size, 1)
        key = f"trace={trace_name}|chunk={chunk_id}"
    else:
        key = f"source={dataset_source}|family={trace_family}"

    bucket = _stable_bucket(key, seed)
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + val_pct:
        return "val"
    return "test"


def _parse_jsonl_trace(path: Path) -> Tuple[List[Request], Dict[PageId, Page], str]:
    page_ids: List[PageId] = []
    records: List[Dict[str, object]] = []
    dataset_source = "unknown"
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            page_ids.append(str(row["item_id"]))
            records.append(dict(row))
            dataset_source = str(row.get("source_dataset", dataset_source))

    prediction_records: List[Dict[str, object]] = []
    for rec in records:
        md = rec.get("metadata", {})
        bucket = None
        conf = None
        if isinstance(md, Mapping):
            bucket = md.get("bucket")
            conf = md.get("confidence")
        prediction_records.append({"bucket": bucket, "confidence": conf})
    reqs, pages = build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)
    return reqs, pages, dataset_source


def load_trace_from_any(path: str) -> Tuple[List[Request], Dict[PageId, Page], str]:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        return _parse_jsonl_trace(p)
    reqs, pages = load_trace(path)
    return reqs, pages, "legacy"


def infer_trace_family(dataset_source: str, trace_name: str) -> str:
    source = dataset_source.lower().strip()
    if source and source != "unknown":
        return source
    tn = trace_name.lower()
    if "stress::" in tn:
        return "stress"
    if "example" in tn:
        return "example"
    return "unknown"


def discover_trace_specs(trace_globs: Sequence[str]) -> List[TraceSpec]:
    specs: List[TraceSpec] = []
    for patt in trace_globs:
        for path in sorted(Path(".").glob(patt)):
            if path.is_dir():
                continue
            trace_name = str(path)
            dataset_source = path.parent.name if path.name == "trace.jsonl" else "legacy"
            trace_family = infer_trace_family(dataset_source=dataset_source, trace_name=trace_name)
            specs.append(
                TraceSpec(
                    path=str(path),
                    trace_name=trace_name,
                    dataset_source=dataset_source,
                    trace_family=trace_family,
                )
            )
    unique: Dict[str, TraceSpec] = {}
    for s in specs:
        unique[s.path] = s
    return list(unique.values())


def parse_trace_manifest(manifest_path: Optional[str], fallback_globs: Sequence[str]) -> List[TraceSpec]:
    if not manifest_path:
        return discover_trace_specs(fallback_globs)
    specs: List[TraceSpec] = []
    with Path(manifest_path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            path = str(row["path"]).strip()
            if not path:
                continue
            trace_name = str(row.get("trace_name", path)).strip() or path
            dataset_source = str(row.get("dataset_source", "unknown")).strip() or "unknown"
            trace_family = str(row.get("trace_family", "")).strip() or infer_trace_family(dataset_source, trace_name)
            specs.append(
                TraceSpec(
                    path=path,
                    trace_name=trace_name,
                    dataset_source=dataset_source,
                    trace_family=trace_family,
                )
            )
    return specs


def iter_candidate_rows(
    requests: Sequence[Request],
    capacity: int,
    trace_name: str,
    dataset_source: str,
    trace_family: str,
    cfg: WulverDatasetConfig,
) -> Iterator[Dict[str, object]]:
    order: collections.OrderedDict[PageId, None] = collections.OrderedDict()
    bucket_by_page: Dict[PageId, int] = {}
    conf_by_page: Dict[PageId, float] = {}
    recent_req_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)
    recent_hit_hist: Deque[PageId] = collections.deque(maxlen=cfg.history_window)

    for t, req in enumerate(requests):
        pid = req.page_id
        if req.metadata.get("bucket") is not None:
            bucket_by_page[pid] = int(req.metadata["bucket"])
        if req.metadata.get("confidence") is not None:
            conf_by_page[pid] = max(0.0, min(1.0, float(req.metadata["confidence"])))

        hit = pid in order
        if hit:
            order.move_to_end(pid)
            recent_req_hist.append(pid)
            recent_hit_hist.append(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            recent_req_hist.append(pid)
            continue

        candidates = list(order.keys())
        req_bucket = int(req.metadata.get("bucket", 0))
        req_conf = float(req.metadata.get("confidence", 0.5))
        decision_id = f"{trace_name}|cap={capacity}|t={t}"
        split = assign_split(
            split_mode=cfg.split_mode,
            trace_name=trace_name,
            dataset_source=dataset_source,
            trace_family=trace_family,
            t=t,
            chunk_size=cfg.chunk_size,
            train_pct=cfg.split_train_pct,
            val_pct=cfg.split_val_pct,
            seed=cfg.split_seed,
        )

        future = requests[t + 1 :]
        for candidate in candidates:
            req_rate = (sum(1 for x in recent_req_hist if x == candidate) / len(recent_req_hist)) if recent_req_hist else 0.0
            hit_rate = (sum(1 for x in recent_hit_hist if x == candidate) / len(recent_hit_hist)) if recent_hit_hist else 0.0
            feats = compute_candidate_features_v1(
                request_bucket=req_bucket,
                request_confidence=req_conf,
                candidates=candidates,
                candidate=candidate,
                bucket_by_page=bucket_by_page,
                confidence_by_page=conf_by_page,
                recent_request_rate=req_rate,
                recent_hit_rate=hit_rate,
            ).as_dict()
            for h in cfg.horizons:
                fut_h = future[:h]
                after = [p for p in candidates if p != candidate] + [pid]
                y_loss = float(_simulate_lru_misses(after, fut_h, capacity=capacity))
                row: Dict[str, object] = {
                    "trace_name": trace_name,
                    "trace_family": trace_family,
                    "dataset_source": dataset_source,
                    "capacity": int(capacity),
                    "horizon": int(h),
                    "decision_id": decision_id,
                    "decision_t": int(t),
                    "decision_chunk_id": int(t // max(cfg.chunk_size, 1)),
                    "candidate_page_id": candidate,
                    "split": split,
                    "y_loss": y_loss,
                    "y_value": -y_loss,
                }
                row.update(feats)
                yield row

        lru_victim = candidates[0]
        order.pop(lru_victim)
        order[pid] = None
        recent_req_hist.append(pid)


def dataset_columns() -> List[str]:
    return [
        "trace_name",
        "trace_family",
        "dataset_source",
        "capacity",
        "horizon",
        "decision_id",
        "decision_t",
        "decision_chunk_id",
        "candidate_page_id",
        "split",
        "y_loss",
        "y_value",
        *EVICT_VALUE_V1_FEATURE_COLUMNS,
    ]


def update_summary_maps(
    row: Mapping[str, object],
    *,
    rows_by_key: MutableMapping[str, int],
    decisions_by_key: MutableMapping[str, set[str]],
) -> None:
    split = str(row["split"])
    family = str(row["trace_family"])
    cap = str(row["capacity"])
    hor = str(row["horizon"])
    did = str(row["decision_id"])
    key = f"split={split}|family={family}|capacity={cap}|horizon={hor}"
    rows_by_key[key] = int(rows_by_key.get(key, 0)) + 1
    decisions_by_key.setdefault(key, set()).add(did)


def materialize_summary(rows_by_key: Mapping[str, int], decisions_by_key: Mapping[str, set[str]]) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for key in sorted(rows_by_key.keys()):
        tokens = dict(tok.split("=", 1) for tok in key.split("|"))
        out.append(
            {
                "split": tokens["split"],
                "trace_family": tokens["family"],
                "capacity": int(tokens["capacity"]),
                "horizon": int(tokens["horizon"]),
                "row_count": int(rows_by_key[key]),
                "decision_count": int(len(decisions_by_key.get(key, set()))),
            }
        )
    return out
