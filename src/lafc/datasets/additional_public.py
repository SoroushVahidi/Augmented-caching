from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from lafc.datasets.base import CanonicalTraceRecord, write_request_sequence

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetDescriptor:
    name: str
    title: str
    source_urls: Sequence[str]
    supports_auto_download: bool
    raw_expectation: str
    notes: str


DATASET_DESCRIPTORS: Dict[str, DatasetDescriptor] = {
    "twemcache": DatasetDescriptor(
        name="twemcache",
        title="Twitter cache-trace (Twemcache/Pelikan)",
        source_urls=(
            "https://github.com/twitter/cache-trace",
            "https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source",
        ),
        supports_auto_download=False,
        raw_expectation="CSV/JSONL request logs provided locally via manifest.json",
        notes="Large raw mirrors are manifest-ingested by design.",
    ),
    "metakv": DatasetDescriptor(
        name="metakv",
        title="MetaKV",
        source_urls=(
            "https://github.com/cacheMon/cache_dataset",
            "https://ftp.pdl.cmu.edu/pub/datasets/cacheDatasets/original/metaKV/",
            "https://cachelib.org/docs/Cache_Library_User_Guides/Cachebench_FB_HW_eval/",
        ),
        supports_auto_download=False,
        raw_expectation="oracleGeneral-like CSV/TSV/JSONL or local exports via manifest.json",
        notes="Supports direct oracle-style ingestion.",
    ),
    "metacdn": DatasetDescriptor(
        name="metacdn",
        title="MetaCDN",
        source_urls=(
            "https://github.com/cacheMon/cache_dataset",
            "https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/metaCDN/",
            "https://cachelib.org/docs/Cache_Library_User_Guides/Cachebench_FB_HW_eval/",
        ),
        supports_auto_download=False,
        raw_expectation="object-cache request logs or oracleGeneral-like files via manifest.json",
        notes="Supports optional range aggregation to object-level requests.",
    ),
    "cloudphysics": DatasetDescriptor(
        name="cloudphysics",
        title="CloudPhysics block I/O traces",
        source_urls=(
            "https://github.com/cacheMon/cache_dataset",
        ),
        supports_auto_download=False,
        raw_expectation="block I/O CSV/TSV/JSONL via manifest.json",
        notes="Supports LBN->page_id conversion with configurable page size.",
    ),
}


def _read_manifest(raw_dir: Path) -> Dict[str, Any]:
    manifest_path = raw_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing manifest.json in {raw_dir}. Provide local raw files and a manifest."
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _iter_rows(path: Path) -> Iterable[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        with path.open("r", encoding="utf-8") as fh:
            if suffix == ".json":
                data = json.load(fh)
                if isinstance(data, list):
                    for row in data:
                        yield dict(row)
                    return
                raise ValueError(f"Expected list JSON in {path}")
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
        return

    delimiter = "\t" if suffix in {".tsv", ".txt"} else ","
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        for row in reader:
            yield dict(row)


def _first_of(row: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _to_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any, default: float = 1.0) -> float:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_op(op: Any) -> str:
    text = str(op or "").strip().lower()
    if not text:
        return "get"
    return text


def _is_read_like(op: str) -> bool:
    return op in {"get", "read", "hit", "lookup", "fetch", "r"}


def parse_twemcache(
    raw_dir: Path,
    *,
    limit: int | None = None,
    cluster: str | None = None,
    sample_only: bool = False,
    paging_view: bool = False,
) -> List[CanonicalTraceRecord]:
    manifest = _read_manifest(raw_dir)
    files = manifest.get("files", [])
    if not files:
        raise ValueError("twemcache manifest requires non-empty 'files'")

    records: List[CanonicalTraceRecord] = []
    idx = 0
    for f in files:
        path = raw_dir / str(f)
        if not path.exists():
            raise FileNotFoundError(f"twemcache raw file not found: {path}")
        for row in _iter_rows(path):
            cluster_id = str(_first_of(row, ["cluster", "cluster_id", "pool", "clusterName"], ""))
            if cluster and cluster_id and cluster_id != cluster:
                continue
            key = str(_first_of(row, ["key", "cacheKey", "obj_id", "item_id", "page_id"], "")).strip()
            if not key:
                continue
            key_size = _to_int(_first_of(row, ["key_size", "keySize"]))
            value_size = _to_int(_first_of(row, ["value_size", "valueSize", "obj_size", "size"]))
            ttl = _to_int(_first_of(row, ["ttl", "TTL"]))
            op = _normalize_op(_first_of(row, ["op", "op_type", "operation", "cmd"]))
            client_id = _first_of(row, ["client_id", "client", "host"], None)
            ts = _first_of(row, ["timestamp", "ts", "time"], None)
            total_size = (key_size or 0) + (value_size or 0)
            rec = CanonicalTraceRecord(
                request_index=idx,
                item_id=key,
                source_dataset="twemcache",
                timestamp=str(ts) if ts is not None else None,
                size=total_size if total_size > 0 else None,
                metadata={
                    "obj_id": key,
                    "obj_size": value_size,
                    "key_size": key_size,
                    "value_size": value_size,
                    "ttl": ttl,
                    "op_type": op,
                    "client_id": client_id,
                    "cluster": cluster_id or None,
                    "source_dataset": "twemcache",
                },
            )
            records.append(rec)
            idx += 1
            if limit is not None and len(records) >= limit:
                return records
            if sample_only and len(records) >= 100:
                return records

    if paging_view:
        write_request_sequence(records, raw_dir / ".." / "processed" / "twemcache" / "requests_only.txt")
    return records


def parse_meta_oracle_dataset(
    raw_dir: Path,
    *,
    dataset_name: str,
    limit: int | None = None,
    usecase: str | None = None,
    read_only: bool = False,
    keep_usecase_fields: bool = True,
    aggregate_ranges: bool = False,
    sample_only: bool = False,
) -> List[CanonicalTraceRecord]:
    manifest = _read_manifest(raw_dir)
    files = manifest.get("files", [])
    if not files:
        raise ValueError(f"{dataset_name} manifest requires non-empty 'files'")

    records: List[CanonicalTraceRecord] = []
    idx = 0
    for f in files:
        path = raw_dir / str(f)
        if not path.exists():
            raise FileNotFoundError(f"{dataset_name} raw file not found: {path}")
        for row in _iter_rows(path):
            op = _normalize_op(_first_of(row, ["op_type", "op", "operation", "cmd"]))
            if read_only and not _is_read_like(op):
                continue
            ucase = _first_of(row, ["usecase", "tenant", "workload"], None)
            if usecase and str(ucase) != usecase:
                continue

            obj = str(
                _first_of(
                    row,
                    [
                        "obj_id",
                        "cacheKey",
                        "key",
                        "object_id",
                        "objectId",
                        "item_id",
                        "page_id",
                        "lbn",
                    ],
                    "",
                )
            ).strip()
            if not obj:
                continue

            if aggregate_ranges:
                obj = obj.split(":")[0]

            obj_size = _to_int(_first_of(row, ["obj_size", "objectSize", "value_size", "size", "length"]))
            ttl = _to_int(_first_of(row, ["ttl", "TTL"]))
            next_access_vtime = _to_int(_first_of(row, ["next_access_vtime", "next_access", "next_vtime"]))
            metadata = {
                "obj_id": obj,
                "obj_size": obj_size,
                "ttl": ttl,
                "op_type": op,
                "next_access_vtime": next_access_vtime,
                "cache_hit": _first_of(row, ["cache_hit", "cache_hits", "hit"], None),
                "sub_usecase": _first_of(row, ["sub_usecase", "subUsecase"], None),
                "response_size": _to_int(_first_of(row, ["responseSize", "response_size"])),
                "range_start": _to_int(_first_of(row, ["rangeStart", "range_start"])),
                "range_end": _to_int(_first_of(row, ["rangeEnd", "range_end"])),
                "content_type_id": _first_of(row, ["contentTypeId", "content_type_id"], None),
                "sampling_rate": _to_float(_first_of(row, ["sampling_rate", "sampleRate"], None), default=0.0) or None,
                "source_dataset": dataset_name,
            }
            if keep_usecase_fields:
                metadata["usecase"] = ucase

            rec = CanonicalTraceRecord(
                request_index=idx,
                item_id=obj,
                source_dataset=dataset_name,
                timestamp=str(_first_of(row, ["timestamp", "ts", "time"], "")) or None,
                size=obj_size,
                metadata=metadata,
            )
            records.append(rec)
            idx += 1
            if limit is not None and len(records) >= limit:
                return records
            if sample_only and len(records) >= 100:
                return records

    return records


def parse_cloudphysics(
    raw_dir: Path,
    *,
    limit: int | None = None,
    page_size: int = 4096,
    read_only: bool = False,
    sample_only: bool = False,
) -> List[CanonicalTraceRecord]:
    if page_size <= 0:
        raise ValueError("page_size must be positive")

    manifest = _read_manifest(raw_dir)
    files = manifest.get("files", [])
    if not files:
        raise ValueError("cloudphysics manifest requires non-empty 'files'")

    records: List[CanonicalTraceRecord] = []
    idx = 0
    for f in files:
        path = raw_dir / str(f)
        if not path.exists():
            raise FileNotFoundError(f"cloudphysics raw file not found: {path}")
        for row in _iter_rows(path):
            cmd = _normalize_op(_first_of(row, ["command", "cmd", "op_type", "op"]))
            if read_only and cmd not in {"r", "read"}:
                continue
            lbn_val = _to_int(_first_of(row, ["lbn", "block_id", "block", "obj_id"]))
            if lbn_val is None:
                continue
            length = _to_int(_first_of(row, ["length", "len", "size"])) or page_size
            page_id = str(int((lbn_val * 512) // page_size))
            rec = CanonicalTraceRecord(
                request_index=idx,
                item_id=page_id,
                source_dataset="cloudphysics",
                timestamp=str(_first_of(row, ["timestamp", "ts", "time"], "")) or None,
                size=length,
                metadata={
                    "page_id": page_id,
                    "lbn": lbn_val,
                    "length": length,
                    "op_type": cmd,
                    "version": _first_of(row, ["version", "trace_version"], None),
                    "next_access_vtime": _to_int(_first_of(row, ["next_access_vtime", "next_vtime"])),
                    "page_size": page_size,
                    "source_dataset": "cloudphysics",
                },
            )
            records.append(rec)
            idx += 1
            if limit is not None and len(records) >= limit:
                return records
            if sample_only and len(records) >= 100:
                return records

    return records
