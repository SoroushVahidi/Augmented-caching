from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List

from lafc.offline.trace_inputs import load_trace_with_sizes
from lafc.offline_teacher_supervision import OfflineTeacherLabelConfig, build_offline_teacher_candidate_rows
from lafc.simulator.request_trace import load_trace


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _group_by_decision(rows: Iterable[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["decision_id"]), []).append(row)
    return grouped


def _compute_diff_stats(rows: List[Dict[str, object]]) -> Dict[str, float]:
    grouped = _group_by_decision(rows)
    disagree = 0
    total = 0
    regrets: List[float] = []
    for _, items in grouped.items():
        total += 1
        teacher_best = {str(r["candidate_page_id"]) for r in items if float(r["teacher_best"]) == 1.0}
        heur_best = {str(r["candidate_page_id"]) for r in items if float(r["heuristic_proxy_best"]) == 1.0}
        if teacher_best != heur_best:
            disagree += 1
        regrets.extend(float(r["teacher_regret"]) for r in items)

    return {
        "decisions_total": float(total),
        "decision_disagreement_count": float(disagree),
        "decision_disagreement_rate": (float(disagree) / float(total)) if total else 0.0,
        "mean_teacher_regret": mean(regrets) if regrets else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="First-check for offline-teacher supervision labels")
    ap.add_argument("--traces", default="data/example_unweighted.json,data/example_general_caching.json")
    ap.add_argument("--capacity", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=24)
    ap.add_argument("--output-dir", default="analysis/offline_teacher_first_check")
    args = ap.parse_args()

    trace_paths = [x.strip() for x in args.traces.split(",") if x.strip()]
    cfg = OfflineTeacherLabelConfig(horizon=args.horizon)

    rows: List[Dict[str, object]] = []
    for trace_path in trace_paths:
        try:
            requests, pages, sizes = load_trace_with_sizes(trace_path)
        except Exception:
            requests, pages = load_trace(trace_path)
            sizes = {pid: 1.0 for pid in pages}

        rows.extend(
            build_offline_teacher_candidate_rows(
                requests=requests,
                pages=pages,
                page_sizes=sizes,
                capacity=float(args.capacity),
                trace_name=trace_path,
                trace_family=Path(trace_path).parent.name or "unknown",
                cfg=cfg,
            )
        )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "candidate_rows.csv", rows)

    stats = _compute_diff_stats(rows)
    by_teacher: Dict[str, int] = {}
    for r in rows:
        k = str(r["teacher_type"])
        by_teacher[k] = by_teacher.get(k, 0) + 1

    summary = {
        "traces": trace_paths,
        "capacity": args.capacity,
        "horizon": args.horizon,
        "rows_total": len(rows),
        "teacher_row_counts": by_teacher,
        **stats,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = (
        "# Offline Teacher Label First Check\n\n"
        f"- Rows: {len(rows)}\n"
        f"- Decisions: {int(stats['decisions_total'])}\n"
        f"- Teacher/heuristic disagreement rate: {stats['decision_disagreement_rate']:.4f}\n"
        f"- Mean teacher regret: {stats['mean_teacher_regret']:.4f}\n"
        f"- Teacher row counts: {json.dumps(by_teacher)}\n"
    )
    (out / "report.md").write_text(report, encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
