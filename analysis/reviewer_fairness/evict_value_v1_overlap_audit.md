# evict_value_v1 Train/Test Overlap Audit

**Finding: CRITICAL — confirmed train/test overlap.** See
`docs/reviewer_fairness_protocol.md` section 6 for the narrative; this file
is the machine-verifiable evidence record.

## Evidence

- Contaminated training manifest:
  `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json`
  (sha256 `7e06f18d99313eb7f8ff1158719243f85230880f27dd395a237d6aed0648d6bf`)
- Contaminated model artifact:
  `models/evict_value_wulver_v1_best.pkl`
  (sha256 `0c9e8a48066f8bb80bfab31b023c9785ca5b955f408d50c18008dbcc314ea61b`)
- `split_mode: "trace_chunk"`, `chunk_size: 4096`, `max_requests_per_trace: 50000`,
  `trace_count: 7`.
- `preflight.trace_stats[*].path` in that manifest lists the exact same 7
  files as `analysis/wulver_trace_manifest_full.csv`
  (`data/processed/{brightkite,citibike,wiki2018,twemcache,metakv,metacdn,
  cloudphysics}/trace.jsonl`), same `[0, 50000)` request range.
- Split assignment (`src/lafc/evict_value_wulver_v1.py:assign_split`):
  ```python
  chunk_id = t // chunk_size
  key = f"trace={trace_name}|chunk={chunk_id}"
  bucket = _stable_bucket(key, seed)
  # bucket < 70 -> train, < 85 -> val, else -> test
  ```
  This is a **hash-bucketed assignment per chunk**, not a contiguous
  prefix/suffix split. With `chunk_size=4096` over 50,000 requests, each
  trace has ~13 chunks, independently and pseudo-randomly distributed
  across train/val/test — **scattered across the entire `[0, 50000)`
  range**, not confined to an early portion.

## Consequence

The model that produced the manuscript's headline end-to-end table was fit
on candidate examples drawn from ~70% of each canonical evaluation trace's
own request positions, then scored end-to-end over the same, unrestricted
`[0, 50000)` range of those same 7 traces. **This model is not eligible
for the primary fairness comparison.**

## Corrective action taken this session

See `docs/evict_value_v1_fair_training_protocol.md` and
`configs/evict_value_v1_fair_training_protocol.json` for the frozen,
strictly-disjoint replacement protocol (`evict_value_v1_fair_v1`), and
`data/processed_fair_v1/PROVENANCE.json` for the machine-verified
non-overlap evidence of its training corpus.

**The contaminated `heavy_r1` artifact is preserved unchanged** —
`data/derived/evict_value_v1_wulver_heavy_r1/`,
`models/evict_value_wulver_v1_best.pkl`, and its replay results are not
deleted, modified, or relabeled. They remain valid evidence of the old
model's behavior (deployment-only classification, per
`docs/reviewer_fairness_protocol.md` section 13) but are explicitly marked
`eligible_for_primary_comparison: false` in `evict_value_v1_overlap_audit.json`.
