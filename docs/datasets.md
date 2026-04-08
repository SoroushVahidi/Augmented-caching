# Dataset ingestion and preprocessing

This repository supports canonical preprocessing into `data/processed/<dataset>/trace.(jsonl|csv)`.

## Canonical format

Canonical processed records include:
- `request_index`
- `item_id` (object/page key used by simulators)
- `source_dataset`
- `timestamp` (if present)
- `size` (if present)
- `cost`
- `metadata` (dataset-specific fields such as `obj_id`, `obj_size`, `ttl`, `op_type`, `next_access_vtime`)

A paging-view export is also written as:
- `data/processed/<dataset>/requests_only.txt`

## Unified CLI

```bash
python -m lafc.datasets.prepare --dataset <name> --raw-dir data/raw --output-dir data/processed
# or
python scripts/datasets/prepare_all.py --dataset <name>
```

Supported `--dataset` values:
- `brightkite`, `citibike`, `spec_cpu2006`, `wiki2018`
- `twemcache`, `metakv`, `metacdn`, `cloudphysics`
- `all`

Important flags:
- `--sample-only` / `--limit`
- `--format {jsonl,csv}`
- `--cluster` (twemcache)
- `--usecase` (metakv/metacdn)
- `--read-only`
- `--drop-usecase-fields`
- `--aggregate-ranges` (metacdn)
- `--page-size` (cloudphysics)
- `--paging-view`

---

## Existing datasets (baseline support)

### BrightKite
- Auto-download: **yes** (`scripts/datasets/download_brightkite.py`).
- Raw expectation: SNAP check-ins gz file.
- Prepare:
```bash
python scripts/datasets/download_brightkite.py
python scripts/datasets/prepare_all.py --dataset brightkite
```

### CitiBike
- Auto-download: **yes** (`scripts/datasets/download_citibike.py`).
- Raw expectation: monthly CSV.
- Prepare:
```bash
python scripts/datasets/download_citibike.py --month 202401
python scripts/datasets/prepare_all.py --dataset citibike
```

### SPEC CPU2006
- Auto-download: **no** (license/manual ingestion).
- Raw expectation: `data/raw/spec_cpu2006/manifest.json` + referenced traces.
- Prepare:
```bash
python scripts/datasets/prepare_all.py --dataset spec_cpu2006
```

### wiki2018
- Auto-download: **manual by default**.
- Raw expectation: CSV/TSV/TXT with `object_id` and optional `timestamp,size,cost`.
- Prepare:
```bash
python scripts/datasets/prepare_all.py --dataset wiki2018
```

For a reproducible **public** trace without manual licensing steps, see `docs/datasets_wulver_trace_acquisition.md` (Wikimedia pageviews → `object_id` CSV).

---

## New public production-grade datasets

## 1) Twitter cache-trace (Twemcache / Pelikan)

Sources:
- https://github.com/twitter/cache-trace
- https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/open_source

Acquisition policy:
- Auto-download: **no** (large mirrors; manifest-based ingestion only).
- Put local raw files under `data/raw/twemcache/` and create `manifest.json`.

Expected raw fields (any superset/subset supported):
- `timestamp`, `key`, `key_size`, `value_size`, `client_id`, `operation`, `ttl`, `cluster`

Preserved canonical metadata:
- `obj_id`, `obj_size`, `key_size`, `value_size`, `ttl`, `op_type`, `client_id`, `cluster`

Prepare examples:
```bash
python -m lafc.datasets.prepare --dataset twemcache --raw-dir data/raw --output-dir data/processed
python -m lafc.datasets.prepare --dataset twemcache --cluster cacheA --sample-only
```

## 2) MetaKV

Sources:
- https://github.com/cacheMon/cache_dataset
- https://ftp.pdl.cmu.edu/pub/datasets/cacheDatasets/original/metaKV/
- https://cachelib.org/docs/Cache_Library_User_Guides/Cachebench_FB_HW_eval/

Acquisition policy:
- Auto-download: **no** (manifest/local ingestion).
- Supports oracleGeneral-like exports via CSV/TSV/JSONL through manifest.

Preserved fields where available:
- `timestamp`, `obj_id`, `obj_size`, `next_access_vtime`, `op_type`, `ttl`, `usecase`, `sub_usecase`, `cache_hit`

Prepare examples:
```bash
python -m lafc.datasets.prepare --dataset metakv --raw-dir data/raw
python -m lafc.datasets.prepare --dataset metakv --read-only --usecase timeline
```

## 3) MetaCDN

Sources:
- https://github.com/cacheMon/cache_dataset
- https://ftp.pdl.cmu.edu/pub/datasets/twemcacheWorkload/cacheDatasets/metaCDN/
- https://cachelib.org/docs/Cache_Library_User_Guides/Cachebench_FB_HW_eval/

Acquisition policy:
- Auto-download: **no** (manifest/local ingestion).
- Supports oracleGeneral-style ingestion.

Preserved fields where available:
- `timestamp`, `cacheKey/obj_id`, `objectSize`, `responseSize`, `rangeStart/rangeEnd`, `ttl`, `contentTypeId`, `cache_hit`, `sampling_rate`, `next_access_vtime`

Prepare examples:
```bash
python -m lafc.datasets.prepare --dataset metacdn --raw-dir data/raw
python -m lafc.datasets.prepare --dataset metacdn --aggregate-ranges --sample-only
```

## 4) CloudPhysics block I/O traces

Source family:
- cache_dataset README / associated public mirrors

Acquisition policy:
- Auto-download: **no** (manifest/local ingestion).

Preserved fields where available:
- `timestamp`, `lbn`, `length`, `command`, `version`, `next_access_vtime`

Additional conversion:
- LBN to logical page id with `--page-size`.
- Read-only filtering with `--read-only`.

Prepare examples:
```bash
python -m lafc.datasets.prepare --dataset cloudphysics --raw-dir data/raw --page-size 4096
python -m lafc.datasets.prepare --dataset cloudphysics --read-only --sample-only
```

---

## Output locations

Processed traces:
- `data/processed/<dataset>/trace.jsonl` (default)
- `data/processed/<dataset>/trace.csv` (with `--format csv`)
- `data/processed/<dataset>/requests_only.txt`
- `data/processed/<dataset>/dataset_descriptor.json`

Sample traces in repo:
- `data/examples/twemcache_sample.jsonl`
- `data/examples/metakv_sample.jsonl`
- `data/examples/metacdn_sample.jsonl`
- `data/examples/cloudphysics_sample.jsonl`

## Legal/access caveats

- Do not commit raw production datasets into git.
- Some datasets are large and mirror-hosted; use manifest-based local ingestion.
- Respect source licenses/terms for each dataset provider.

## Modeling relevance notes

- Unweighted paging: use `requests_only.txt` or `item_id` sequences.
- Weighted/file/object caching: use `size` and `obj_size` metadata.
- Confidence-aware/coarse prediction experiments: use preserved context (`op_type`, `ttl`, `usecase`, `cluster`, ranges).
