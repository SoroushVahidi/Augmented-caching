#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a manifest.json template for dataset ingestion")
    parser.add_argument("--dataset", required=True, choices=["twemcache", "metakv", "metacdn", "cloudphysics"])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--files", nargs="*", default=["sample.csv"])
    args = parser.parse_args()

    ds_dir = args.raw_dir / args.dataset
    ds_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = ds_dir / "manifest.json"
    payload = {
        "dataset": args.dataset,
        "files": args.files,
        "notes": "List local relative raw files for ingestion.",
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
