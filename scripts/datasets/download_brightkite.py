#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = "https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download BrightKite check-ins from SNAP")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/brightkite"))
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    out = args.raw_dir / "loc-brightkite_totalCheckins.txt.gz"
    if out.exists():
        print(f"BrightKite already exists: {out}")
        return 0

    print(f"Downloading {URL} -> {out}")
    urllib.request.urlretrieve(URL, out)
    print("Done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
