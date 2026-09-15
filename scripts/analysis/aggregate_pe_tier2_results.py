from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate PE Tier-2 result CSVs.")
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(args.input.open(encoding="utf-8")))
    required = {"policy", "family", "capacity", "misses", "requests"}
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise SystemExit(f"missing columns: {sorted(missing)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["policy", "family", "capacity", "misses", "requests", "miss_ratio"]
    with args.output.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            requests = int(row["requests"])
            misses = int(row["misses"])
            out = {
                "policy": row["policy"],
                "family": row["family"],
                "capacity": int(row["capacity"]),
                "misses": misses,
                "requests": requests,
                "miss_ratio": misses / requests if requests else "",
            }
            writer.writerow(out)


if __name__ == "__main__":
    main()
