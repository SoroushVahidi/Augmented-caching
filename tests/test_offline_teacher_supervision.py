from __future__ import annotations

import json
from pathlib import Path

from lafc.offline_teacher_supervision import (
    OfflineTeacherLabelConfig,
    build_offline_teacher_candidate_rows,
    build_offline_teacher_pairwise_rows,
)
from lafc.simulator.request_trace import build_requests_from_lists


def test_exact_teacher_type_and_labels_generated():
    requests, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "C"])
    sizes = {pid: 1.0 for pid in pages}

    rows = build_offline_teacher_candidate_rows(
        requests=requests,
        pages=pages,
        page_sizes=sizes,
        capacity=2.0,
        trace_name="toy_uniform",
        trace_family="toy",
        cfg=OfflineTeacherLabelConfig(horizon=8),
    )

    assert rows
    assert all(r["teacher_type"] == "exact_teacher_belady" for r in rows)
    assert any(float(r["teacher_best"]) == 1.0 for r in rows)


def test_approx_teacher_used_for_non_uniform_case():
    requests, pages = build_requests_from_lists(
        ["A", "B", "A", "C", "A"],
        {"A": 10.0, "B": 1.0, "C": 2.0},
    )
    sizes = {"A": 3.0, "B": 2.0, "C": 2.0}

    rows = build_offline_teacher_candidate_rows(
        requests=requests,
        pages=pages,
        page_sizes=sizes,
        capacity=4.0,
        trace_name="toy_general",
        trace_family="toy",
        cfg=OfflineTeacherLabelConfig(horizon=8),
    )

    assert rows
    assert all(r["teacher_type"] == "approx_teacher_lp" for r in rows)


def test_pairwise_rows_created():
    requests, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "C"])
    sizes = {pid: 1.0 for pid in pages}
    rows = build_offline_teacher_candidate_rows(
        requests=requests,
        pages=pages,
        page_sizes=sizes,
        capacity=2.0,
        trace_name="toy_uniform",
        trace_family="toy",
        cfg=OfflineTeacherLabelConfig(horizon=8),
    )

    pairwise = build_offline_teacher_pairwise_rows(rows)
    assert pairwise
    assert {"candidate_i_page_id", "candidate_j_page_id", "label_i_better"}.issubset(pairwise[0].keys())


def test_first_check_style_outputs(tmp_path):
    # lightweight smoke that mirrors expected output workflow using plain file writes
    requests, pages = build_requests_from_lists(["A", "B", "C", "A", "B", "C"])
    sizes = {pid: 1.0 for pid in pages}
    rows = build_offline_teacher_candidate_rows(
        requests=requests,
        pages=pages,
        page_sizes=sizes,
        capacity=2.0,
        trace_name="toy_uniform",
        trace_family="toy",
        cfg=OfflineTeacherLabelConfig(horizon=8),
    )

    out = Path(tmp_path)
    (out / "summary.json").write_text(json.dumps({"rows": len(rows)}), encoding="utf-8")
    assert (out / "summary.json").exists()
