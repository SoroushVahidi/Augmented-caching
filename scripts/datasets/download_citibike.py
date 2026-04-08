#!/usr/bin/env python3
from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


URL_PATTERNS = [
    "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.csv.zip",
    "https://s3.amazonaws.com/tripdata/{month}-citibike-tripdata.zip",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CitiBike monthly trip data")
    parser.add_argument("--month", required=True, help="YYYYMM, e.g. 202401")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/citibike"))
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.raw_dir / f"{args.month}-citibike.zip"

    downloaded = False
    for pattern in URL_PATTERNS:
        url = pattern.format(month=args.month)
        try:
            print(f"Attempting download: {url}")
            urllib.request.urlretrieve(url, zip_path)
            downloaded = True
            break
        except Exception as exc:
            print(f"Failed URL {url}: {exc}")

    if not downloaded:
        raise RuntimeError(
            f"Could not download CitiBike month {args.month}. "
            "Check month availability or download manually per docs/datasets.md."
        )

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(args.raw_dir)
        print(f"Extracted files: {zf.namelist()}")
    print(f"Downloaded and extracted to {args.raw_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
