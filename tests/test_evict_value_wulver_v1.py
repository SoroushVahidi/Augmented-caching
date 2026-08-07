from __future__ import annotations

import pytest

from lafc.evict_value_wulver_v1 import WulverDatasetConfig, assign_split, iter_candidate_rows, materialize_summary, update_summary_maps
from lafc.simulator.request_trace import build_requests_from_lists


def test_assign_split_trace_chunk_stable():
    s1 = assign_split(
        split_mode="trace_chunk",
        trace_name="traceA",
        dataset_source="metakv",
        trace_family="metakv",
        t=17,
        chunk_size=8,
        train_pct=70,
        val_pct=15,
        seed=9,
    )
    s2 = assign_split(
        split_mode="trace_chunk",
        trace_name="traceA",
        dataset_source="metakv",
        trace_family="metakv",
        t=18,
        chunk_size=8,
        train_pct=70,
        val_pct=15,
        seed=9,
    )
    assert s1 == s2


def test_assign_split_family_map_deterministic_and_explicit():
    fmap = {"brightkite": "train", "citibike": "val"}
    for t in (0, 100, 9999):
        assert assign_split(
            split_mode="family_map", trace_name="t", dataset_source="brightkite",
            trace_family="brightkite", t=t, chunk_size=4096, train_pct=70,
            val_pct=15, seed=0, family_split_map=fmap,
        ) == "train"
        assert assign_split(
            split_mode="family_map", trace_name="t", dataset_source="citibike",
            trace_family="citibike", t=t, chunk_size=4096, train_pct=70,
            val_pct=15, seed=0, family_split_map=fmap,
        ) == "val"


def test_assign_split_family_map_rejects_unmapped_family():
    # A held-out test family must never silently receive a split -- it
    # must be excluded from the input manifest entirely, and if it
    # somehow appears anyway, this must raise, not default to "train".
    with pytest.raises(ValueError, match="wiki2018"):
        assign_split(
            split_mode="family_map", trace_name="t", dataset_source="wiki2018",
            trace_family="wiki2018", t=0, chunk_size=4096, train_pct=70,
            val_pct=15, seed=0, family_split_map={"brightkite": "train"},
        )


def test_assign_split_family_map_rejects_test_value():
    with pytest.raises(ValueError, match="'test'|only 'train' or 'val'"):
        assign_split(
            split_mode="family_map", trace_name="t", dataset_source="brightkite",
            trace_family="brightkite", t=0, chunk_size=4096, train_pct=70,
            val_pct=15, seed=0, family_split_map={"brightkite": "test"},
        )


def test_assign_split_family_map_requires_map():
    with pytest.raises(ValueError):
        assign_split(
            split_mode="family_map", trace_name="t", dataset_source="brightkite",
            trace_family="brightkite", t=0, chunk_size=4096, train_pct=70,
            val_pct=15, seed=0, family_split_map=None,
        )


def test_wulver_rows_include_required_metadata():
    reqs, _ = build_requests_from_lists(
        page_ids=["A", "B", "C", "A", "D", "A", "B", "C", "D"],
        prediction_records=[{"bucket": 1, "confidence": 0.8}] * 9,
    )
    cfg = WulverDatasetConfig(horizons=(4,))
    rows = list(
        iter_candidate_rows(
            requests=reqs,
            capacity=2,
            trace_name="toy_trace",
            dataset_source="toy_source",
            trace_family="toy_family",
            cfg=cfg,
        )
    )
    assert rows
    sample = rows[0]
    assert sample["trace_name"] == "toy_trace"
    assert sample["trace_family"] == "toy_family"
    assert sample["dataset_source"] == "toy_source"
    assert "split" in sample


def test_summary_counts_rows_and_decisions():
    rows = [
        {
            "split": "train",
            "trace_family": "metakv",
            "capacity": 64,
            "horizon": 8,
            "decision_id": "d1",
        },
        {
            "split": "train",
            "trace_family": "metakv",
            "capacity": 64,
            "horizon": 8,
            "decision_id": "d1",
        },
        {
            "split": "val",
            "trace_family": "metakv",
            "capacity": 64,
            "horizon": 8,
            "decision_id": "d2",
        },
    ]
    rows_by_key = {}
    decisions_by_key = {}
    for row in rows:
        update_summary_maps(row, rows_by_key=rows_by_key, decisions_by_key=decisions_by_key)
    summary = materialize_summary(rows_by_key=rows_by_key, decisions_by_key=decisions_by_key)
    train_row = next(r for r in summary if r["split"] == "train")
    assert train_row["row_count"] == 2
    assert train_row["decision_count"] == 1
