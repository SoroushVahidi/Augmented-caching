from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List

from lafc.datasets.base import CanonicalTraceRecord, write_records, write_request_sequence
from lafc.datasets.additional_public import (
    DATASET_DESCRIPTORS,
    parse_cloudphysics,
    parse_meta_oracle_dataset,
    parse_twemcache,
)
from lafc.datasets.brightkite import BRIGHTKITE_FILENAME, parse_brightkite
from lafc.datasets.citibike import parse_citibike
from lafc.datasets.spec_cpu2006 import parse_spec_from_manifest
from lafc.datasets.wiki2018 import parse_wiki2018


def _prepare_brightkite(raw_dir: Path, limit: int | None) -> List[CanonicalTraceRecord]:
    return parse_brightkite(raw_dir / BRIGHTKITE_FILENAME, limit=limit)


def _prepare_citibike(raw_dir: Path, limit: int | None) -> List[CanonicalTraceRecord]:
    candidates = sorted(raw_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CitiBike CSV files found in {raw_dir}")
    return parse_citibike(candidates[0], limit=limit)


def _prepare_spec(raw_dir: Path, limit: int | None) -> List[CanonicalTraceRecord]:
    return parse_spec_from_manifest(raw_dir / "manifest.json", limit=limit)


def _prepare_wiki(raw_dir: Path, limit: int | None) -> List[CanonicalTraceRecord]:
    candidates = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.tsv")) + list(raw_dir.glob("*.txt")))
    if not candidates:
        raise FileNotFoundError(f"No wiki2018 raw file found in {raw_dir}")
    return parse_wiki2018(candidates[0], limit=limit)


PREPARERS: Dict[str, Callable[[Path, int | None], List[CanonicalTraceRecord]]] = {
    "brightkite": _prepare_brightkite,
    "citibike": _prepare_citibike,
    "spec_cpu2006": _prepare_spec,
    "wiki2018": _prepare_wiki,
}


def run_dataset(dataset: str, raw_root: Path, out_root: Path, fmt: str, limit: int | None, **kwargs: object) -> Path:
    raw_dir = raw_root / dataset
    out_dir = out_root / dataset
    if dataset in PREPARERS:
        records = PREPARERS[dataset](raw_dir, limit)
    elif dataset in {"twemcache", "metakv", "metacdn", "cloudphysics"}:
        sample_only = bool(kwargs.get("sample_only", False))
        try:
            if dataset == "twemcache":
                records = parse_twemcache(
                    raw_dir,
                    limit=limit,
                    cluster=kwargs.get("cluster"),  # type: ignore[arg-type]
                    sample_only=sample_only,
                    paging_view=bool(kwargs.get("paging_view", False)),
                )
            elif dataset == "metakv":
                records = parse_meta_oracle_dataset(
                    raw_dir,
                    dataset_name="metakv",
                    limit=limit,
                    usecase=kwargs.get("usecase"),  # type: ignore[arg-type]
                    read_only=bool(kwargs.get("read_only", False)),
                    keep_usecase_fields=bool(kwargs.get("keep_usecase_fields", True)),
                    aggregate_ranges=False,
                    sample_only=sample_only,
                )
            elif dataset == "metacdn":
                records = parse_meta_oracle_dataset(
                    raw_dir,
                    dataset_name="metacdn",
                    limit=limit,
                    usecase=kwargs.get("usecase"),  # type: ignore[arg-type]
                    read_only=bool(kwargs.get("read_only", False)),
                    keep_usecase_fields=bool(kwargs.get("keep_usecase_fields", True)),
                    aggregate_ranges=bool(kwargs.get("aggregate_ranges", False)),
                    sample_only=sample_only,
                )
            else:
                records = parse_cloudphysics(
                    raw_dir,
                    limit=limit,
                    page_size=int(kwargs.get("page_size", 4096)),
                    read_only=bool(kwargs.get("read_only", False)),
                    sample_only=sample_only,
                )
        except FileNotFoundError:
            if not sample_only:
                raise
            sample_path = Path("data/examples") / f"{dataset}_sample.jsonl"
            if not sample_path.exists():
                raise
            records = _load_sample_records(sample_path, dataset, limit=limit)
    else:
        raise ValueError(f"Unsupported dataset {dataset}")

    ext = "jsonl" if fmt == "jsonl" else "csv"
    trace_path = out_dir / f"trace.{ext}"
    write_records(records, trace_path, fmt=fmt)

    if dataset in {"wiki2018", "twemcache", "metakv", "metacdn", "cloudphysics"} or bool(kwargs.get("paging_view", False)):
        write_request_sequence(records, out_dir / "requests_only.txt")
    return trace_path


def _load_sample_records(path: Path, dataset: str, limit: int | None) -> List[CanonicalTraceRecord]:
    records: List[CanonicalTraceRecord] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            rec = CanonicalTraceRecord(
                request_index=i,
                item_id=str(row["item_id"]),
                source_dataset=dataset,
                split=str(row.get("split", "sample")),
                timestamp=row.get("timestamp"),
                size=row.get("size"),
                cost=float(row.get("cost", 1.0)),
                metadata=row.get("metadata"),
            )
            records.append(rec)
            if limit is not None and len(records) >= limit:
                break
    return records


def _write_dataset_descriptor(dataset: str, out_dir: Path) -> None:
    if dataset not in DATASET_DESCRIPTORS:
        return
    descriptor = DATASET_DESCRIPTORS[dataset]
    payload = {
        "name": descriptor.name,
        "title": descriptor.title,
        "source_urls": list(descriptor.source_urls),
        "supports_auto_download": descriptor.supports_auto_download,
        "raw_expectation": descriptor.raw_expectation,
        "notes": descriptor.notes,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset_descriptor.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets into canonical trace format")
    parser.add_argument(
        "--dataset",
        default="all",
        choices=[
            "brightkite",
            "citibike",
            "spec_cpu2006",
            "wiki2018",
            "twemcache",
            "metakv",
            "metacdn",
            "cloudphysics",
            "all",
        ],
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-only", action="store_true", help="Write at most 100 records")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    parser.add_argument("--cluster", default=None, help="Optional cluster filter (twemcache).")
    parser.add_argument("--usecase", default=None, help="Optional usecase filter (metakv/metacdn).")
    parser.add_argument("--read-only", action="store_true", help="Keep read-like operations only.")
    parser.add_argument("--drop-usecase-fields", action="store_true", help="Drop usecase/sub_usecase metadata fields.")
    parser.add_argument("--aggregate-ranges", action="store_true", help="Aggregate range requests to object-level (metacdn).")
    parser.add_argument("--page-size", type=int, default=4096, help="Page size for CloudPhysics LBN->page mapping.")
    parser.add_argument("--paging-view", action="store_true", help="Always write requests_only.txt.")
    args = parser.parse_args()

    limit = 100 if args.sample_only and args.limit is None else args.limit
    datasets = (list(PREPARERS.keys()) + ["twemcache", "metakv", "metacdn", "cloudphysics"]) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        path = run_dataset(
            ds,
            args.raw_dir,
            args.output_dir,
            args.format,
            limit,
            cluster=args.cluster,
            usecase=args.usecase,
            read_only=args.read_only,
            keep_usecase_fields=not args.drop_usecase_fields,
            aggregate_ranges=args.aggregate_ranges,
            page_size=args.page_size,
            sample_only=args.sample_only,
            paging_view=args.paging_view,
        )
        _write_dataset_descriptor(ds, args.output_dir / ds)
        print(f"Prepared {ds}: {path}")

    return 0
