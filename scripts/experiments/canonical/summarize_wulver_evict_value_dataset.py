from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def main() -> None:
    ap = argparse.ArgumentParser(description="Extended summaries for Wulver evict_value_v1 dataset.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split-summary", type=Path, default=None, help="split_summary.csv next to manifest if omitted")
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--out-md", type=Path, default=None)
    args = ap.parse_args()

    split_path = args.split_summary or (args.manifest.parent / "split_summary.csv")
    rows = list(csv.DictReader(split_path.open(encoding="utf-8")))
    for r in rows:
        r["row_count"] = int(r["row_count"])
        r["decision_count"] = int(r["decision_count"])
        r["capacity"] = int(r["capacity"])
        r["horizon"] = int(r["horizon"])

    total_rows = sum(r["row_count"] for r in rows)
    total_decisions = sum(r["decision_count"] for r in rows)

    by_split: Dict[str, int] = defaultdict(int)
    by_family: Dict[str, int] = defaultdict(int)
    by_cap: Dict[int, int] = defaultdict(int)
    by_horizon: Dict[int, int] = defaultdict(int)
    fam_split: Dict[Tuple[str, str], int] = defaultdict(int)

    for r in rows:
        by_split[r["split"]] += r["row_count"]
        by_family[r["trace_family"]] += r["row_count"]
        by_cap[r["capacity"]] += r["row_count"]
        by_horizon[r["horizon"]] += r["row_count"]
        fam_split[(r["trace_family"], r["split"])] += r["row_count"]

    payload = {
        "total_row_count": total_rows,
        "total_decision_count": total_decisions,
        "rows_by_split": dict(by_split),
        "rows_by_family": dict(by_family),
        "rows_by_capacity": {str(k): v for k, v in sorted(by_cap.items())},
        "rows_by_horizon": {str(k): v for k, v in sorted(by_horizon.items())},
        "family_x_split_rows": {f"{a}|{b}": c for (a, b), c in sorted(fam_split.items())},
    }

    manifest_meta = {}
    if args.manifest.exists():
        m = json.loads(args.manifest.read_text(encoding="utf-8"))
        manifest_meta = {
            "split_mode": m.get("split_mode"),
            "chunk_size": m.get("chunk_size"),
            "capacities": m.get("capacities"),
            "horizons": m.get("horizons"),
            "trace_count": m.get("trace_count"),
            "shard_count": m.get("shard_count"),
        }
    payload["manifest"] = manifest_meta

    out_json = args.out_json or (args.manifest.parent / "dataset_summary_extended.json")
    out_md = args.out_md or (Path("analysis") / "evict_value_v1_wulver_dataset_summary.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("# evict_value_v1 Wulver dataset summary")
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- **Rows:** {total_rows}")
    lines.append(f"- **Decisions (unique decision_id):** {total_decisions}")
    lines.append("")
    lines.append("## Rows by split")
    for k in sorted(by_split.keys()):
        lines.append(f"- {k}: {by_split[k]}")
    lines.append("")
    lines.append("## Rows by trace_family")
    for k in sorted(by_family.keys()):
        lines.append(f"- {k}: {by_family[k]}")
    lines.append("")
    lines.append("## Rows by capacity")
    for k in sorted(by_cap.keys()):
        lines.append(f"- {k}: {by_cap[k]}")
    lines.append("")
    lines.append("## Rows by horizon")
    for k in sorted(by_horizon.keys()):
        lines.append(f"- {k}: {by_horizon[k]}")
    lines.append("")
    lines.append("## Family × split (rows)")
    for (fam, sp), c in sorted(fam_split.items()):
        lines.append(f"- {fam} / {sp}: {c}")
    lines.append("")
    lines.append("## Manifest meta")
    lines.append(f"- {manifest_meta}")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out_json} and {out_md}")


if __name__ == "__main__":
    main()
