# Acquiring real processed traces for Wulver (`evict_value_v1`)

This note documents how the **non-sample** `data/processed/<dataset>/trace.jsonl` files used for Wulver-scale runs were produced in a reproducible way.

All large raw/processed artifacts live under `data/raw/**` and `data/processed/**` (gitignored). Only commands and small helper scripts are tracked in git.

## 1) citibike — auto-download + prepare

```bash
PYTHONPATH=src python scripts/datasets/download_citibike.py --month 202401
PYTHONPATH=src python -m lafc.datasets.prepare --dataset citibike --format jsonl --limit 50000 --output-dir data/processed --raw-dir data/raw
```

## 2) wiki2018 — public pageviews → repo CSV schema

The repo’s `wiki2018` preparer expects CSV with an `object_id` column (`src/lafc/datasets/wiki2018.py`).

We derive a real trace from **Wikimedia hourly pageviews** (public dumps), keeping project `en`, and mapping each row to `object_id = "<project>:<page_title>"`.

```bash
mkdir -p data/raw/wiki2018
curl -sL 'https://dumps.wikimedia.org/other/pageviews/2024/2024-01/pageviews-20240101-000000.gz' \
  | gzip -dc \
  | awk '$1=="en"{print}' \
  | python scripts/datasets/stream_pageviews_to_wiki2018_csv.py --max-rows 100000 --projects en \
  > data/raw/wiki2018/pageviews_en_head.csv

PYTHONPATH=src python -m lafc.datasets.prepare --dataset wiki2018 --format jsonl --limit 50000 --output-dir data/processed --raw-dir data/raw
```

**Caveat:** pageviews are aggregated counts per `(project, title)` for an hour, not byte-for-byte CDN logs. They are still a legitimate **real public request-mixture** over Wikipedia page titles for paging experiments.

## 3) twemcache — CMU sample + headered CSV

Raw Twemcache open traces are distributed as **headerless** CSV inside `.zst`. The repo’s CSV path expects a header row for `csv.DictReader`.

Example used here: `sample100/cluster26.sort.sample100.zst` from CMU (`~14MiB` compressed), decompressed, first 100k lines, with a synthetic header matching Twemcache field names.

Then:

```bash
# After creating data/raw/twemcache/manifest.json and cluster26_sample100_head.csv
PYTHONPATH=src python -m lafc.datasets.prepare --dataset twemcache --format jsonl --limit 50000 --output-dir data/processed --raw-dir data/raw
```

## 4) metakv — CMU kvcache shard (streamed head)

Stream the first ~100k lines of `kvcache_202206/kvcache_traces_1.csv.zst` into `data/raw/metakv/kvcache_traces_1_head.csv`, list it in `manifest.json`, then prepare.

## 5) metacdn — CMU CDN shard (streamed head)

Same pattern using `cdn_202303/reag0c01_20230315_20230322_0.2000.csv.zst`.

## 6) cloudphysics — Alibaba block trace (partial zstd + head)

`alibabaBlock2020.csv.zst` is extremely large. A **prefix of the compressed blob** (range download) still decompresses cleanly for early blocks; we take the first 100k logical rows and add a CSV header (`idx,command,lbn,length,timestamp`) so `lbn`/`length`/`command` match `parse_cloudphysics`.

## Full-manifest Wulver dataset generation

Point the evict-value Wulver builder at:

- `analysis/wulver_trace_manifest_full.csv`
