# evict_value_v1 Wulver split design (multi-family phase)

## Split logic used

- **Mode:** `trace_chunk` (default in `scripts/build_evict_value_dataset_wulver_v1.py`).
- **Assignment key:** `(trace_name, decision_chunk_id)` with fixed `chunk_size` (default 4096 request indices per chunk).
- **Partitioning:** deterministic seeded hash into train / val / test using `--split-train-pct`, `--split-val-pct`, `--split-seed`.

This keeps **all horizons and capacities** for a given decision in the **same** split bucket (because `split` is attached at row generation and keyed by chunk, not by horizon).

## Leakage caveats (still relevant)

- **Cross-capacity correlation:** the same underlying request stream is replayed at multiple capacities; train/val/test are **not** independent across capacity for the same trace.
- **Chunk adjacency:** decisions inside a chunk are correlated; chunk-level split reduces but does not remove within-chunk dependence.
- **wiki2018:** pageview-derived proxy (not a canonical CDN object trace); interpret family-level results cautiously.

## Strength for “first serious” validation

- **Adequate for:** screening whether `evict_value_v1` remains competitive vs strong baselines across **diverse families** under the same labeling pipeline.
- **Not yet:** a publication-grade leakage-free benchmark; treat as **first multi-family Wulver validation**, not a final word.
