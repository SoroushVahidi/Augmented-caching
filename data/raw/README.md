# Raw dataset staging area

Place raw datasets here before preprocessing.

- `brightkite/`: auto-downloadable via `scripts/datasets/download_brightkite.py`
- `citibike/`: auto-downloadable by month via `scripts/datasets/download_citibike.py`
- `spec_cpu2006/`: **manual ingestion only** (licensed data). Provide `manifest.json` + local traces.
- `wiki2018/`: generally manual ingestion unless you already have a permitted local copy.
- `twemcache/`: manifest-based local ingestion (`manifest.json` + listed files).
- `metakv/`: manifest-based local/oracle-style ingestion.
- `metacdn/`: manifest-based local/oracle-style ingestion.
- `cloudphysics/`: manifest-based local ingestion for block I/O traces.

The preprocessing CLI reads from this directory by default.
