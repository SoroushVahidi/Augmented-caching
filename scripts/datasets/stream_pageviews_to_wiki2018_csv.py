#!/usr/bin/env python3
"""Convert Wikimedia pageviews (hourly .gz text) lines to wiki2018-style CSV.

Input format (space-separated, one row per line):
  <project> <page_title...> <views> <agents>

Example:
  en Main_Page 12345 0

We emit CSV with columns: object_id,timestamp,size,cost
object_id := "<project>:<page_title>" (title may contain spaces; we join middle fields).
"""
from __future__ import annotations

import argparse
import csv
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-rows", type=int, default=100_000, help="Max output rows (excluding header).")
    ap.add_argument(
        "--projects",
        default="en,en.mw,en.m",
        help="Comma-separated project prefixes to keep (e.g. en,en.mw).",
    )
    args = ap.parse_args()
    allowed = {p.strip() for p in args.projects.split(",") if p.strip()}

    writer = csv.DictWriter(sys.stdout, fieldnames=["object_id", "timestamp", "size", "cost"])
    writer.writeheader()

    n = 0
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split(" ")
        if len(parts) < 4:
            continue
        project = parts[0]
        if project not in allowed:
            continue
        try:
            agents = int(parts[-1])
            views = int(parts[-2])
        except ValueError:
            continue
        title = " ".join(parts[1:-2])
        if not title:
            continue
        object_id = f"{project}:{title}"
        writer.writerow(
            {
                "object_id": object_id,
                "timestamp": "",
                "size": "",
                "cost": 1.0,
            }
        )
        n += 1
        if n >= args.max_rows:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
