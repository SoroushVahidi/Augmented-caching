from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict, List

from lafc.datasets.base import CanonicalTraceRecord, write_records, write_request_sequence
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


def run_dataset(dataset: str, raw_root: Path, out_root: Path, fmt: str, limit: int | None) -> Path:
    raw_dir = raw_root / dataset
    out_dir = out_root / dataset
    records = PREPARERS[dataset](raw_dir, limit)
    ext = "jsonl" if fmt == "jsonl" else "csv"
    trace_path = out_dir / f"trace.{ext}"
    write_records(records, trace_path, fmt=fmt)

    if dataset == "wiki2018":
        write_request_sequence(records, out_dir / "requests_only.txt")
    return trace_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare benchmark datasets into canonical trace format")
    parser.add_argument("--dataset", default="all", choices=["brightkite", "citibike", "spec_cpu2006", "wiki2018", "all"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--sample-only", action="store_true", help="Write at most 100 records")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--format", choices=["jsonl", "csv"], default="jsonl")
    args = parser.parse_args()

    limit = 100 if args.sample_only and args.limit is None else args.limit
    datasets = list(PREPARERS.keys()) if args.dataset == "all" else [args.dataset]

    for ds in datasets:
        path = run_dataset(ds, args.raw_dir, args.output_dir, args.format, limit)
        print(f"Prepared {ds}: {path}")

    return 0
