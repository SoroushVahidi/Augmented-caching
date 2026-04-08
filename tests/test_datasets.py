from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lafc.datasets.base import CanonicalTraceRecord, validate_records
from lafc.datasets.brightkite import parse_brightkite
from lafc.datasets.citibike import parse_citibike
from lafc.datasets.spec_cpu2006 import parse_spec_from_manifest
from lafc.datasets.wiki2018 import parse_wiki2018


def test_brightkite_parser(tmp_path: Path):
    raw = tmp_path / "bk.txt"
    raw.write_text(
        "1\t2008-01-01T00:00:00Z\t40.0\t-70.0\tvenueA\n"
        "2\t2008-01-01T00:01:00Z\t41.0\t-71.0\tvenueB\n",
        encoding="utf-8",
    )
    recs = parse_brightkite(raw)
    assert [r.item_id for r in recs] == ["venueA", "venueB"]
    assert recs[0].request_index == 0


def test_citibike_parser(tmp_path: Path):
    raw = tmp_path / "cb.csv"
    raw.write_text(
        "start_station_id,started_at,end_station_id,bikeid,tripduration\n"
        "72,2024-01-01 00:00:00,79,B100,600\n",
        encoding="utf-8",
    )
    recs = parse_citibike(raw)
    assert len(recs) == 1
    assert recs[0].item_id == "72"


def test_spec_parser_manifest(tmp_path: Path):
    trace = tmp_path / "401.trace"
    trace.write_text("0x100 R 64\n0x104,W,64\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"traces": [{"name": "401", "path": "401.trace"}]}), encoding="utf-8")

    recs = parse_spec_from_manifest(manifest)
    assert [r.item_id for r in recs] == ["0x100", "0x104"]


def test_wiki_parser(tmp_path: Path):
    raw = tmp_path / "wiki.csv"
    raw.write_text("object_id,timestamp,size,cost,url\na,2018-01-01T00:00:00Z,10,2.0,/wiki/A\n", encoding="utf-8")
    recs = parse_wiki2018(raw)
    assert recs[0].size == 10
    assert recs[0].cost == 2.0


def test_canonical_validation_deterministic():
    recs = [
        CanonicalTraceRecord(request_index=0, item_id="a", source_dataset="x", cost=1.0),
        CanonicalTraceRecord(request_index=1, item_id="a", source_dataset="x", cost=1.0),
    ]
    validate_records(recs)


def test_missing_raw_data_fails_gracefully(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_brightkite(tmp_path / "missing.txt")


def test_prepare_cli_brightkite(tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    (raw_root / "brightkite").mkdir(parents=True)
    with gzip.open(raw_root / "brightkite" / "loc-brightkite_totalCheckins.txt.gz", "wt", encoding="utf-8") as fh:
        fh.write("1\t2008-01-01T00:00:00Z\t40.0\t-70.0\tvenueA\n")
    cmd = [
        sys.executable,
        "scripts/datasets/prepare_all.py",
        "--dataset",
        "brightkite",
        "--raw-dir",
        str(raw_root),
        "--output-dir",
        str(out_root),
        "--format",
        "jsonl",
    ]
    subprocess.run(cmd, check=True)
    assert (out_root / "brightkite" / "trace.jsonl").exists()

def test_deterministic_spec_preprocessing(tmp_path: Path):
    trace = tmp_path / "401.trace"
    trace.write_text("0x100 R 64\n0x100 R 64\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"traces": [{"name": "401", "path": "401.trace"}]}), encoding="utf-8")
    recs1 = parse_spec_from_manifest(manifest)
    recs2 = parse_spec_from_manifest(manifest)
    assert [r.item_id for r in recs1] == [r.item_id for r in recs2]


def test_prepare_cli_spec_missing_manifest_fails(tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    (raw_root / "spec_cpu2006").mkdir(parents=True)
    cmd = [
        sys.executable,
        "scripts/datasets/prepare_all.py",
        "--dataset",
        "spec_cpu2006",
        "--raw-dir",
        str(raw_root),
        "--output-dir",
        str(out_root),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode != 0
    assert "manifest" in (result.stderr + result.stdout).lower()
