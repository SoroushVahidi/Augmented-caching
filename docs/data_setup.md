# Dataset setup guide

This document explains how to obtain and prepare the trace datasets used in this project.

## Trace families

| Family | Source | Characteristics | Access |
|--------|--------|-----------------|--------|
| **BrightKite** | SNAP Check-ins | Social network check-ins; unweighted | Public |
| **CitiBike** | Citi Bike System Data | Bicycle trip records; unweighted | Public |
| **SPEC CPU2006** | SPEC traces | Memory access patterns | Manual |
| **wiki2018** | Wikimedia Pageviews | Web object requests; unweighted | Public |
| **Twemcache** | Twitter Cache-trace | Multi-cluster K-V cache traces | Public |
| **MetaKV** | Meta Key-Value | Facebook production workloads | Public |
| **MetaCDN** | Meta CDN | Content delivery network traces | Public |
| **CloudPhysics** | Block I/O traces | Virtualized storage workloads | Public |

## Preparation workflow

We use a two-stage pipeline: **Download** (if supported) → **Prepare** (standardization).

### 1. Unified preparation script

The primary entry point is `scripts/setup/prepare_all.py`:

```bash
python scripts/setup/prepare_all.py --dataset <name>
```

Supported names: `brightkite`, `citibike`, `spec_cpu2006`, `wiki2018`, `twemcache`, `metakv`, `metacdn`, `cloudphysics`, `all`.

### 2. Family-specific instructions

#### BrightKite
```bash
python scripts/setup/download_brightkite.py
python scripts/setup/prepare_all.py --dataset brightkite
```

#### CitiBike
```bash
python scripts/setup/download_citibike.py --month 202401
python scripts/setup/prepare_all.py --dataset citibike
```

#### wiki2018
We use a specific snapshot of Wikimedia pageviews. For reproduction of the canonical `wiki2018` trace:
```bash
python scripts/setup/stream_pageviews_to_wiki2018_csv.py --output data/raw/wiki2018/pageviews.csv
python scripts/setup/prepare_all.py --dataset wiki2018
```

#### Heavier production traces (Twitter/Meta/CloudPhysics)
These traces are often distributed as large collections.
1. Download the raw files from the sources listed in `docs/datasets.md`.
2. Place them in `data/raw/<dataset>/`.
3. Create a `manifest.json` in the raw directory if the prepare script expects one.
4. Run:
```bash
python scripts/setup/prepare_all.py --dataset <dataset>
```

## Directory structure

- `data/raw/<dataset>/` — original trace files (excluded from Git).
- `data/processed/<dataset>/` — standardized traces in JSONL/CSV format.
- `data/processed/<dataset>/requests_only.txt` — simple item ID sequence for unweighted paging simulators.

## Standardization details

The preparation scripts normalize diverse trace formats into a common schema:
- `request_index`: strictly increasing integer.
- `item_id`: unique identifier for the object.
- `size`: object size (if applicable).
- `cost`: miss cost (if applicable).
- `metadata`: preserved original fields (ttl, op_type, cluster, etc.).

For detailed format specifications, see `docs/datasets.md`.
