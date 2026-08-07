from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate chain-witness theorem campaign artifacts.")
    ap.add_argument("--jobs-root", type=Path, default=Path("analysis/pairwise_chain_witness_campaign/jobs"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/pairwise_chain_witness_campaign"))
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    all_instances: List[Dict[str, object]] = []
    all_violations: List[Dict[str, object]] = []
    all_min: List[Dict[str, object]] = []
    summaries: List[Dict[str, object]] = []

    for job_dir in sorted([p for p in args.jobs_root.iterdir() if p.is_dir()]):
        summary_path = job_dir / "summary.json"
        if summary_path.exists():
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
        all_instances.extend(_read_csv(job_dir / "instances.csv"))
        all_violations.extend(_read_csv(job_dir / "violations.csv"))
        all_min.extend(_read_csv(job_dir / "minimized_counterexamples.csv"))

    _write_csv(out / "all_instances.csv", all_instances)
    _write_csv(out / "all_violations.csv", all_violations)
    _write_csv(out / "all_minimized_counterexamples.csv", all_min)
    (out / "job_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    lemma_fail = Counter(v.get("lemma", "unknown") for v in all_violations)
    mode_ctr = Counter(i.get("mode", "unknown") for i in all_instances)
    pass_all = sum(1 for i in all_instances if str(i.get("all_pass", "")).lower() == "true")
    total = len(all_instances)
    fail = total - pass_all

    smallest = None
    if all_min:
        smallest = sorted(all_min, key=lambda r: (int(r.get("min_trace_len", 10**9)), int(r.get("orig_trace_len", 10**9))))[0]

    summary = {
        "jobs": len(summaries),
        "instances_checked": total,
        "instances_all_pass": pass_all,
        "instances_with_any_failure": fail,
        "violations_total": len(all_violations),
        "violations_by_lemma": dict(lemma_fail),
        "instances_by_mode": dict(mode_ctr),
        "counterexample_found": len(all_violations) > 0,
        "smallest_minimized_counterexample": smallest,
    }
    (out / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Pairwise chain-witness theorem campaign report")
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- Jobs aggregated: {summary['jobs']}")
    lines.append(f"- Instances checked: {summary['instances_checked']}")
    lines.append(f"- Instances passing all lemmas: {summary['instances_all_pass']}")
    lines.append(f"- Instances with at least one failure: {summary['instances_with_any_failure']}")
    lines.append(f"- Total lemma violations: {summary['violations_total']}")
    lines.append("")
    lines.append("## Violations by lemma")
    if lemma_fail:
        for lemma, n in sorted(lemma_fail.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {lemma}: {n}")
    else:
        lines.append("- No violations observed.")
    lines.append("")
    lines.append("## Coverage by workload mode")
    for mode, n in sorted(mode_ctr.items()):
        lines.append(f"- {mode}: {n}")
    lines.append("")
    lines.append("## Bottleneck diagnosis")
    if lemma_fail:
        top = sorted(lemma_fail.items(), key=lambda x: (-x[1], x[0]))[0]
        lines.append(f"- Most frequent failing lemma: `{top[0]}` ({top[1]} failures).")
    else:
        lines.append("- No empirical bottleneck observed in checked workload envelope.")
    lines.append("")
    lines.append("## Smallest minimized counterexample")
    if smallest:
        lines.append(f"- Lemma: `{smallest.get('lemma', '')}`")
        lines.append(f"- k={smallest.get('k', '')}, ranking_mode={smallest.get('ranking_mode', '')}")
        lines.append(f"- trace length: {smallest.get('orig_trace_len', '')} -> {smallest.get('min_trace_len', '')}")
        lines.append(f"- min trace: `{smallest.get('min_trace', '')}`")
    else:
        lines.append("- None found.")
    lines.append("")
    lines.append("## Credibility readout")
    if all_violations:
        lines.append("- Counterexamples exist in this campaign; theorem direction needs structural restrictions before claiming generality.")
    else:
        lines.append("- No counterexamples found in this campaign; this is strong empirical support (not a proof).")
    lines.append("")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
