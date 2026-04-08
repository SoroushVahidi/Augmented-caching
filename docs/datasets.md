# Dataset ingestion and preprocessing

This repository supports four benchmark families and converts them into a canonical trace format for caching experiments.

## Canonical processed format

Output path pattern:

- `data/processed/<dataset>/trace.jsonl` (default)
- or `data/processed/<dataset>/trace.csv` when `--format csv` is used.

Canonical fields:

- `request_index` (strictly increasing integer)
- `item_id` (cacheable object identifier)
- `source_dataset` (`brightkite`, `citibike`, `spec_cpu2006`, `wiki2018`)
- `split` (default: `full`)
- `timestamp` (if available)
- `size` (if available)
- `cost` (default 1.0)
- `metadata` (dataset-specific safe extras)

## Unified CLI

```bash
python scripts/datasets/prepare_all.py --dataset brightkite --raw-dir data/raw --output-dir data/processed
```

Supported datasets for `--dataset`: `brightkite`, `citibike`, `spec_cpu2006`, `wiki2018`, `all`.

Useful flags:

- `--raw-dir`
- `--output-dir`
- `--sample-only` (caps output to 100 rows)
- `--limit N`
- `--format jsonl|csv`

Module entry-point also works:

```bash
python -m lafc.datasets.prepare --dataset all
```

## BrightKite

- Represents location check-ins.
- Auto-download: **yes** (`scripts/datasets/download_brightkite.py` from SNAP).
- Expected raw file: `data/raw/brightkite/loc-brightkite_totalCheckins.txt.gz`.
- Default cache mapping: `item_id := venue_id`.
  - Alternative mappings (documented but not default): user-level (`user_id`) or geo-cell IDs.
- Prepare command:

```bash
python scripts/datasets/download_brightkite.py
python scripts/datasets/prepare_all.py --dataset brightkite
```

## CitiBike NYC

- Represents bike trip starts/ends in NYC.
- Auto-download: **yes** for monthly public archives (`scripts/datasets/download_citibike.py --month YYYYMM`).
- Expected raw file: a monthly CSV under `data/raw/citibike/`.
- Default cache mapping: `item_id := start_station_id` (station-level demand).
  - Alternative mappings: bike id, end station id, or route pair.
- Prepare command:

```bash
python scripts/datasets/download_citibike.py --month 202401
python scripts/datasets/prepare_all.py --dataset citibike
```

## SPEC CPU2006 traces

- Represents memory-access traces derived from licensed SPEC CPU2006 workloads.
- Auto-download: **no** (license restricted).
- Required local files:
  - `data/raw/spec_cpu2006/manifest.json`
  - trace files referenced by manifest.
- Supported trace line formats:
  - whitespace: `0xADDR OP SIZE`
  - comma: `0xADDR,OP,SIZE`
- Default cache mapping: `item_id := memory address`.
- Prepare command:

```bash
python scripts/datasets/prepare_all.py --dataset spec_cpu2006
```

## wiki2018 CDN trace

- Represents CDN-style object requests.
- Auto-download: **manual by default** (depends on source/terms).
- Required local file in `data/raw/wiki2018/`: CSV/TSV with `object_id` and optionally `timestamp,size,cost`.
- Default cache mapping: `item_id := object_id`.
- Preserve `size` for byte-aware caching.
- Produces both:
  - canonical `trace.jsonl`/`trace.csv`
  - `requests_only.txt` (paging-style sequence)
- Prepare command:

```bash
python scripts/datasets/prepare_all.py --dataset wiki2018
```

## Caveats

- Raw data is intentionally not versioned in git.
- Manual-ingestion datasets fail with explicit errors and instructions.
- Use `data/examples/*.jsonl` for smoke tests and parser validation.
