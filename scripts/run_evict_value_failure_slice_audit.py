from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from lafc.evict_value_wulver_v1 import (
    TraceSpec,
    infer_trace_family,
    load_trace_from_any,
    parse_trace_manifest,
)
from lafc.policies.evict_value_v1 import EvictValueV1Policy
from lafc.policies.lru import LRUPolicy
from lafc.policies.predictive_marker import PredictiveMarkerPolicy
from lafc.policies.rest_v1 import RestV1Policy
from lafc.policies.trust_and_doubt import TrustAndDoubtPolicy
from lafc.predictors.offline_from_trace import attach_predicted_caches
from lafc.runner.run_policy import run_policy

OUT_CSV = Path("analysis/evict_value_failure_slice_audit.csv")
OUT_MD = Path("analysis/evict_value_failure_slice_summary.md")

POLICY_ORDER = [
    "evict_value_v1",
    "predictive_marker",
    "trust_and_doubt",
    "rest_v1",
    "lru",
]


def _build_policies(evict_value_model_path: str):
    return {
        "evict_value_v1": lambda: EvictValueV1Policy(model_path=evict_value_model_path),
        "predictive_marker": lambda: PredictiveMarkerPolicy(),
        "trust_and_doubt": lambda: TrustAndDoubtPolicy(seed=7),
        "rest_v1": lambda: RestV1Policy(),
        "lru": lambda: LRUPolicy(),
    }


def _parse_caps(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("--capacities must include at least one integer")
    return vals


def _position_bucket(idx: int, n: int) -> str:
    if n <= 1:
        return "early"
    frac = idx / (n - 1)
    if frac < 1.0 / 3.0:
        return "early"
    if frac < 2.0 / 3.0:
        return "mid"
    return "late"


def _trace_specs_from_args(args: argparse.Namespace) -> List[TraceSpec]:
    globs = [x.strip() for x in args.trace_glob.split(",") if x.strip()]
    specs = parse_trace_manifest(None, fallback_globs=globs)

    if args.max_traces is not None:
        specs = specs[: args.max_traces]
    return specs


def _extract_margin(diag: Dict[str, object]) -> Optional[float]:
    for key in ("score_margin", "margin", "predicted_margin", "top2_margin", "loss_margin"):
        if key in diag and diag[key] is not None:
            try:
                return float(diag[key])
            except (TypeError, ValueError):
                return None
    return None


def _extract_spreads(diag: Dict[str, object]) -> Tuple[Optional[float], Optional[float]]:
    bucket = diag.get("bucket_spread")
    conf = diag.get("confidence_spread")
    out_b = None
    out_c = None
    try:
        if bucket is not None:
            out_b = float(bucket)
    except (TypeError, ValueError):
        out_b = None
    try:
        if conf is not None:
            out_c = float(conf)
    except (TypeError, ValueError):
        out_c = None
    return out_b, out_c


def _as_int_bool(v: bool) -> int:
    return 1 if bool(v) else 0


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row.keys()}) if rows else [
        "trace_name",
        "trace_family",
        "capacity",
        "t",
        "request_page",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pct(n: int, d: int) -> float:
    return (100.0 * n / d) if d else 0.0


def _summary_markdown(
    rows: List[Dict[str, object]],
    skipped: Dict[str, str],
    caps: Sequence[int],
    trace_count: int,
) -> str:
    lines: List[str] = []
    lines.append("# evict_value_v1 failure-slice audit")
    lines.append("")
    lines.append("## Run scope")
    lines.append(f"- Traces processed: {trace_count}")
    lines.append(f"- Capacities: {', '.join(str(c) for c in caps)}")
    lines.append(f"- Eviction-decision rows (anchored on evict_value_v1): {len(rows)}")
    if skipped:
        lines.append("- Skipped policies:")
        for p, reason in sorted(skipped.items()):
            lines.append(f"  - {p}: {reason}")
    else:
        lines.append("- Skipped policies: none")
    lines.append("")

    competitors = [p for p in POLICY_ORDER if p != "evict_value_v1"]
    lines.append("## Overall comparison counts vs each competitor")
    for comp in competitors:
        comp_hit = sum(int(r.get(f"{comp}_hit", 0)) for r in rows)
        ev_hit = sum(int(r.get("evict_value_v1_hit", 0)) for r in rows)
        ev_lose = sum(
            1
            for r in rows
            if int(r.get("evict_value_v1_hit", 0)) < int(r.get(f"{comp}_hit", 0))
        )
        ev_win = sum(
            1
            for r in rows
            if int(r.get("evict_value_v1_hit", 0)) > int(r.get(f"{comp}_hit", 0))
        )
        disagree = sum(int(r.get(f"evict_value_v1_diff_vs_{comp}", 0)) for r in rows)
        lines.append(
            f"- {comp}: disagree={disagree}/{len(rows)} ({_pct(disagree, len(rows)):.1f}%), "
            f"evict_value_v1 loses={ev_lose}, wins={ev_win}, tie={len(rows)-ev_lose-ev_win}, "
            f"evict_hits={ev_hit}, {comp}_hits={comp_hit}"
        )
    lines.append("")

    def _breakdown(dim: str) -> List[Tuple[str, int, Dict[str, int]]]:
        grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for r in rows:
            grouped[str(r.get(dim, "unknown"))].append(r)
        out: List[Tuple[str, int, Dict[str, int]]] = []
        for key, subset in sorted(grouped.items()):
            losses = {
                comp: sum(
                    1
                    for r in subset
                    if int(r.get("evict_value_v1_hit", 0)) < int(r.get(f"{comp}_hit", 0))
                )
                for comp in competitors
            }
            out.append((key, len(subset), losses))
        return out

    lines.append("## Per-family breakdown")
    for fam, cnt, losses in _breakdown("trace_family"):
        loss_txt = ", ".join(f"{k}:{v}" for k, v in losses.items())
        lines.append(f"- {fam} (rows={cnt}): {loss_txt}")
    lines.append("")

    lines.append("## Per-capacity breakdown")
    for cap, cnt, losses in _breakdown("capacity"):
        loss_txt = ", ".join(f"{k}:{v}" for k, v in losses.items())
        lines.append(f"- cap={cap} (rows={cnt}): {loss_txt}")
    lines.append("")

    lines.append("## Disagreement-slice breakdown")
    disagree_counts = Counter(int(r.get("disagreement_count", 0)) for r in rows)
    for k in sorted(disagree_counts):
        c = disagree_counts[k]
        lines.append(f"- disagree_with_{k}_competitors: {c} ({_pct(c, len(rows)):.1f}%)")
    lines.append("")

    margins = [r for r in rows if r.get("evict_value_v1_score_margin") not in (None, "")]
    lines.append("## Low-margin vs high-margin breakdown")
    if margins:
        vals = sorted(float(r["evict_value_v1_score_margin"]) for r in margins)
        med = vals[len(vals) // 2]
        low = [r for r in margins if float(r["evict_value_v1_score_margin"]) <= med]
        high = [r for r in margins if float(r["evict_value_v1_score_margin"]) > med]

        def _any_loss(rs: Iterable[Dict[str, object]]) -> int:
            return sum(
                1
                for r in rs
                if any(int(r.get("evict_value_v1_hit", 0)) < int(r.get(f"{c}_hit", 0)) for c in competitors)
            )

        lines.append(f"- margin available rows: {len(margins)}")
        lines.append(f"- median margin split: {med:.6f}")
        lines.append(f"- low-margin rows: {len(low)}, rows with any competitor better: {_any_loss(low)}")
        lines.append(f"- high-margin rows: {len(high)}, rows with any competitor better: {_any_loss(high)}")
    else:
        lines.append("- Score margin was not available in diagnostics for this run; all rows have null margin.")
    lines.append("")

    lines.append("## Top recurring failure patterns")
    pattern_counts: Counter[str] = Counter()
    for r in rows:
        losers = [
            comp
            for comp in competitors
            if int(r.get("evict_value_v1_hit", 0)) < int(r.get(f"{comp}_hit", 0))
        ]
        if not losers:
            continue
        req_bucket = r.get("request_bucket")
        position = str(r.get("position_bucket", "unknown"))
        pattern = f"loses_to={'+'.join(sorted(losers))}|pos={position}|req_bucket={req_bucket}"
        pattern_counts[pattern] += 1

    if not pattern_counts:
        lines.append("- No rows where a competitor hits and evict_value_v1 misses inside the audited eviction slices.")
    else:
        for pattern, count in pattern_counts.most_common(10):
            lines.append(f"- {pattern}: {count}")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Audit rows are anchored on steps where evict_value_v1 performs an eviction decision (cache full + miss).")
    lines.append("- Candidate spread summaries are populated only when diagnostics expose them; otherwise null by design.")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit where evict_value_v1 loses against strong baselines on aligned eviction slices.")
    ap.add_argument(
        "--trace-manifest",
        default="auto",
        help="Path to manifest CSV, 'auto' to use analysis/wulver_trace_manifest.csv when present, or empty string to disable.",
    )
    ap.add_argument(
        "--trace-glob",
        default="data/example_unweighted.json,data/example_atlas_v1.json",
        help="Used only when --trace-manifest is missing/unset.",
    )
    ap.add_argument("--max-traces", type=int, default=8)
    ap.add_argument("--capacities", default="2,3,4")
    ap.add_argument("--max-requests-per-trace", type=int, default=3000)
    ap.add_argument("--evict-value-model", default="models/evict_value_v1_hist_gb.pkl")
    ap.add_argument("--out-csv", type=Path, default=OUT_CSV)
    ap.add_argument("--out-md", type=Path, default=OUT_MD)
    args = ap.parse_args()

    capacities = _parse_caps(args.capacities)
    trace_manifest: Optional[Path]
    raw_manifest = str(args.trace_manifest).strip()
    if raw_manifest.lower() == "auto":
        default_manifest = Path("analysis/wulver_trace_manifest.csv")
        trace_manifest = default_manifest if default_manifest.exists() else None
    elif raw_manifest == "":
        trace_manifest = None
    else:
        trace_manifest = Path(raw_manifest)

    specs: List[TraceSpec]
    if trace_manifest and trace_manifest.exists():
        specs = parse_trace_manifest(str(trace_manifest), fallback_globs=[])
    else:
        specs = _trace_specs_from_args(args)

    if args.max_traces is not None:
        specs = specs[: args.max_traces]
    if not specs:
        raise SystemExit("No traces found from manifest/glob inputs.")

    policies = _build_policies(args.evict_value_model)
    rows: List[Dict[str, object]] = []
    skipped: Dict[str, str] = {}

    for spec in specs:
        requests, pages, dataset_source = load_trace_from_any(spec.path)
        if args.max_requests_per_trace is not None:
            requests = requests[: args.max_requests_per_trace]
        if not requests:
            continue
        trace_family = spec.trace_family or infer_trace_family(dataset_source, spec.trace_name)

        for capacity in capacities:
            td_requests = attach_predicted_caches(requests, capacity=capacity)
            results = {}
            for pname in POLICY_ORDER:
                try:
                    policy = policies[pname]()
                    in_reqs = td_requests if pname == "trust_and_doubt" else requests
                    results[pname] = run_policy(policy, in_reqs, pages, capacity)
                except Exception as exc:  # robust skipping is intentional for lightweight auditing
                    skipped[pname] = f"{type(exc).__name__}: {exc}"

            if "evict_value_v1" not in results:
                continue

            evict_events = results["evict_value_v1"].events
            for i, ev in enumerate(evict_events):
                if ev.evicted is None:
                    continue

                req = requests[i]
                ev_diag = ev.diagnostics or {}
                margin = _extract_margin(ev_diag)
                bucket_spread, conf_spread = _extract_spreads(ev_diag)

                row: Dict[str, object] = {
                    "trace_name": spec.trace_name,
                    "trace_family": trace_family,
                    "capacity": capacity,
                    "t": req.t,
                    "request_page": req.page_id,
                    "candidate_count": ev_diag.get("candidate_count"),
                    "request_bucket": req.metadata.get("bucket"),
                    "request_confidence": req.metadata.get("confidence"),
                    "evict_value_v1_victim": ev.evicted,
                    "evict_value_v1_hit": _as_int_bool(ev.hit),
                    "evict_value_v1_miss": _as_int_bool(not ev.hit),
                    "evict_value_v1_score_margin": margin,
                    "bucket_spread_summary": bucket_spread,
                    "confidence_spread_summary": conf_spread,
                    "position_bucket": _position_bucket(i, len(evict_events)),
                }
                disagree_count = 0
                for comp in [p for p in POLICY_ORDER if p != "evict_value_v1"]:
                    res = results.get(comp)
                    if res is None or i >= len(res.events):
                        row[f"{comp}_victim"] = None
                        row[f"{comp}_hit"] = None
                        row[f"{comp}_miss"] = None
                        row[f"evict_value_v1_diff_vs_{comp}"] = None
                        continue

                    cev = res.events[i]
                    row[f"{comp}_victim"] = cev.evicted
                    row[f"{comp}_hit"] = _as_int_bool(cev.hit)
                    row[f"{comp}_miss"] = _as_int_bool(not cev.hit)
                    diff = (ev.evicted != cev.evicted) if (cev.evicted is not None) else 1
                    row[f"evict_value_v1_diff_vs_{comp}"] = int(diff)
                    disagree_count += int(diff)
                row["disagreement_count"] = disagree_count
                rows.append(row)

    _write_csv(args.out_csv, rows)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_summary_markdown(rows, skipped, capacities, len(specs)), encoding="utf-8")
    print(f"Wrote {args.out_csv} and {args.out_md} (rows={len(rows)})")


if __name__ == "__main__":
    main()
