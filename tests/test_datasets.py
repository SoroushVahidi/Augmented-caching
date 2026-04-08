from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path

import pytest

from lafc.datasets.base import CanonicalTraceRecord, validate_records
from lafc.datasets.additional_public import (
    parse_cloudphysics,
    parse_meta_oracle_dataset,
    parse_twemcache,
)
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


def test_twemcache_parser(tmp_path: Path):
    raw_dir = tmp_path / "twemcache"
    raw_dir.mkdir(parents=True)
    manifest = raw_dir / "manifest.json"
    data = raw_dir / "sample.csv"
    data.write_text(
        "timestamp,key,key_size,value_size,client_id,operation,ttl,cluster\n"
        "1,k1,10,100,c1,get,300,cacheA\n"
        "2,k2,11,101,c2,set,120,cacheA\n",
        encoding="utf-8",
    )
    manifest.write_text(json.dumps({"files": ["sample.csv"]}), encoding="utf-8")
    recs = parse_twemcache(raw_dir, cluster="cacheA")
    assert [r.item_id for r in recs] == ["k1", "k2"]
    assert recs[0].metadata["ttl"] == 300


def test_metakv_oracle_parser_read_filter(tmp_path: Path):
    raw_dir = tmp_path / "metakv"
    raw_dir.mkdir(parents=True)
    (raw_dir / "manifest.json").write_text(json.dumps({"files": ["mk.csv"]}), encoding="utf-8")
    (raw_dir / "mk.csv").write_text(
        "timestamp,obj_id,obj_size,op_type,ttl,usecase,sub_usecase,next_access_vtime,cache_hit\n"
        "1,o1,100,get,30,timeline,feed,9,1\n"
        "2,o2,200,set,40,timeline,feed,10,0\n",
        encoding="utf-8",
    )
    recs = parse_meta_oracle_dataset(raw_dir, dataset_name="metakv", read_only=True)
    assert len(recs) == 1
    assert recs[0].item_id == "o1"


def test_metacdn_parser_aggregate_ranges(tmp_path: Path):
    raw_dir = tmp_path / "metacdn"
    raw_dir.mkdir(parents=True)
    (raw_dir / "manifest.json").write_text(json.dumps({"files": ["cdn.csv"]}), encoding="utf-8")
    (raw_dir / "cdn.csv").write_text(
        "timestamp,cacheKey,objectSize,rangeStart,rangeEnd,op_type\n"
        "1,objA:0-100,1000,0,100,get\n",
        encoding="utf-8",
    )
    recs = parse_meta_oracle_dataset(raw_dir, dataset_name="metacdn", aggregate_ranges=True)
    assert recs[0].item_id == "objA"


def test_cloudphysics_page_size_conversion(tmp_path: Path):
    raw_dir = tmp_path / "cloudphysics"
    raw_dir.mkdir(parents=True)
    (raw_dir / "manifest.json").write_text(json.dumps({"files": ["cp.csv"]}), encoding="utf-8")
    (raw_dir / "cp.csv").write_text(
        "timestamp,lbn,length,command,version\n"
        "1,8,4096,read,v1\n",
        encoding="utf-8",
    )
    recs = parse_cloudphysics(raw_dir, page_size=4096)
    assert recs[0].item_id == "1"


def test_additional_dataset_missing_manifest_fails(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        parse_twemcache(tmp_path / "twemcache_missing")


def test_prepare_cli_sample_only_for_new_dataset(tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    ds_dir = raw_root / "twemcache"
    ds_dir.mkdir(parents=True)
    (ds_dir / "manifest.json").write_text(json.dumps({"files": ["sample.csv"]}), encoding="utf-8")
    (ds_dir / "sample.csv").write_text(
        "timestamp,key,key_size,value_size,operation,cluster\n"
        + "".join([f"{i},k{i},10,100,get,cacheA\n" for i in range(300)]),
        encoding="utf-8",
    )
    cmd = [
        sys.executable,
        "scripts/datasets/prepare_all.py",
        "--dataset",
        "twemcache",
        "--raw-dir",
        str(raw_root),
        "--output-dir",
        str(out_root),
        "--sample-only",
    ]
    subprocess.run(cmd, check=True)
    lines = (out_root / "twemcache" / "trace.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 100


def test_prepare_cli_sample_only_without_raw_uses_examples(tmp_path: Path):
    raw_root = tmp_path / "raw"
    out_root = tmp_path / "out"
    cmd = [
        sys.executable,
        "scripts/datasets/prepare_all.py",
        "--dataset",
        "metakv",
        "--raw-dir",
        str(raw_root),
        "--output-dir",
        str(out_root),
        "--sample-only",
    ]
    subprocess.run(cmd, check=True)
    assert (out_root / "metakv" / "trace.jsonl").exists()
