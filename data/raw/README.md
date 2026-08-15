# Raw dataset staging area

Place raw datasets here before preprocessing.

These families are **third-party** sources (or licensed data). This project
does not claim ownership of upstream traces. See
[`docs/datasets.md`](../../docs/datasets.md) for ingestion notes.

A related public **derived** Hugging Face release (wiki2018-only candidate
rows, not the full manuscript experiment payload) is
[LAFC-Evict](https://huggingface.co/datasets/SoroushVahidi/lafc-evict).

- `brightkite/`: auto-downloadable via `scripts/datasets/download_brightkite.py`
- `citibike/`: auto-downloadable by month via `scripts/datasets/download_citibike.py`
- `spec_cpu2006/`: **manual ingestion only** (licensed data). Provide `manifest.json` + local traces.
- `wiki2018/`: generally manual ingestion unless you already have a permitted local copy.
- `twemcache/`: manifest-based local ingestion (`manifest.json` + listed files).
- `metakv/`: manifest-based local/oracle-style ingestion.
- `metacdn/`: manifest-based local/oracle-style ingestion.
- `cloudphysics/`: manifest-based local ingestion for block I/O traces.

The preprocessing CLI reads from this directory by default.
