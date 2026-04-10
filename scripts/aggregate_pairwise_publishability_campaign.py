from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
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
    ap = argparse.ArgumentParser(description="Aggregate pairwise publishability campaign results.")
    ap.add_argument("--jobs-root", type=Path, default=Path("analysis/pairwise_publishability_campaign/jobs"))
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/pairwise_publishability_campaign"))
    args = ap.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, object]] = []
    offline: List[Dict[str, object]] = []
    online: List[Dict[str, object]] = []
    wtl: List[Dict[str, object]] = []
    fam: List[Dict[str, object]] = []
    hor: List[Dict[str, object]] = []

    for job_dir in sorted([p for p in args.jobs_root.iterdir() if p.is_dir()]):
        sp = job_dir / "summary.json"
        if sp.exists():
            summaries.append(json.loads(sp.read_text(encoding="utf-8")))
        offline.extend(_read_csv(job_dir / "offline_metrics.csv"))
        online.extend(_read_csv(job_dir / "online_metrics.csv"))
        wtl.extend(_read_csv(job_dir / "wtl_vs_baselines.csv"))
        fam.extend(_read_csv(job_dir / "offline_per_family.csv"))
        hor.extend(_read_csv(job_dir / "offline_per_horizon.csv"))

    _write_csv(out / "offline_metrics_all.csv", offline)
    _write_csv(out / "online_metrics_all.csv", online)
    _write_csv(out / "wtl_vs_baselines_all.csv", wtl)
    _write_csv(out / "offline_per_family_all.csv", fam)
    _write_csv(out / "offline_per_horizon_all.csv", hor)
    (out / "job_summaries.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    rec_ctr = Counter(str(s.get("recommendation", "unknown")) for s in summaries)

    # aggregate offline test metrics by (model_family,label_variant,label_noise)
    grouped_off: Dict[str, List[float]] = defaultdict(list)
    grouped_top1: Dict[str, List[float]] = defaultdict(list)
    for r in offline:
        if str(r.get("split", "")) != "test":
            continue
        key = f"{r.get('model_family','')}|{r.get('label_variant','')}|noise={r.get('label_noise','')}"
        grouped_off[key].append(float(r.get("mean_regret_vs_best", 0.0)))
        grouped_top1[key].append(float(r.get("top1_reconstruction", 0.0)))
    off_rows = []
    for k in sorted(grouped_off):
        off_rows.append(
            {
                "config": k,
                "mean_regret_vs_best": float(mean(grouped_off[k])),
                "top1_reconstruction": float(mean(grouped_top1[k])) if grouped_top1[k] else 0.0,
                "n": len(grouped_off[k]),
            }
        )
    _write_csv(out / "offline_ablation_summary.csv", off_rows)

    # online means and W/T/L collapsed.
    by_pol: Dict[str, List[float]] = defaultdict(list)
    for r in online:
        by_pol[str(r.get("policy", ""))].append(float(r.get("misses", 0.0)))
    online_means = {k: float(mean(v)) for k, v in sorted(by_pol.items()) if v}

    wtl_by_base: Dict[str, Dict[str, int]] = defaultdict(lambda: {"W": 0, "T": 0, "L": 0})
    for r in wtl:
        b = str(r.get("baseline", ""))
        wtl_by_base[b]["W"] += int(float(r.get("W", 0)))
        wtl_by_base[b]["T"] += int(float(r.get("T", 0)))
        wtl_by_base[b]["L"] += int(float(r.get("L", 0)))
    _write_csv(out / "wtl_collapsed.csv", [{"baseline": b, **v} for b, v in sorted(wtl_by_base.items())])

    final_rec = "not promising enough"
    if rec_ctr.get("promising_mainline", 0) > 0:
        final_rec = "promising enough for manuscript mainline"
    elif rec_ctr.get("promising_appendix_ablation", 0) > 0:
        final_rec = "promising only as appendix/ablation"

    summary = {
        "jobs": len(summaries),
        "offline_rows": len(offline),
        "online_rows": len(online),
        "recommendation_counts": dict(rec_ctr),
        "online_mean_misses": online_means,
        "final_recommendation": final_rec,
    }
    (out / "campaign_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# Pairwise publishability campaign report")
    lines.append("")
    lines.append("## Scale")
    lines.append(f"- Jobs aggregated: {summary['jobs']}")
    lines.append(f"- Offline rows: {summary['offline_rows']}")
    lines.append(f"- Online rows: {summary['online_rows']}")
    lines.append("")
    lines.append("## Offline ablation highlights")
    if off_rows:
        best = sorted(off_rows, key=lambda r: (float(r["mean_regret_vs_best"]), -float(r["top1_reconstruction"])))[0]
        lines.append(f"- Best config by mean_regret_vs_best: `{best['config']}` ({best['mean_regret_vs_best']:.4f})")
    else:
        lines.append("- No offline test rows found.")
    lines.append("")
    lines.append("## Online means (lower misses is better)")
    for p, v in sorted(online_means.items(), key=lambda x: x[1]):
        lines.append(f"- {p}: {v:.4f}")
    lines.append("")
    lines.append("## W/T/L vs baselines (pairwise policy)")
    for b, v in sorted(wtl_by_base.items()):
        lines.append(f"- {b}: W={v['W']} T={v['T']} L={v['L']}")
    lines.append("")
    lines.append("## Final recommendation")
    lines.append(f"- {final_rec}")
    lines.append("")
    lines.append("## Caution")
    lines.append("- This is empirical evidence only; mixed results should be reported conservatively.")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
