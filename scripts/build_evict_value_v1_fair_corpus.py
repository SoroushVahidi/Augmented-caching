"""Build a strictly non-overlapping training corpus for evict_value_v1_fair_v1.

Corrects the CRITICAL finding in docs/reviewer_fairness_protocol.md section
6: the canonical evict_value_v1 model was trained on chunk-level splits
drawn from the SAME 7 traces and SAME [0, 50000) request range later used
for the manuscript's headline end-to-end evaluation.

Strategy chosen (Section 3 of the fairness task, option B: "later
non-overlapping time range from the same source dataset"), verified before
choosing it: every one of the 7 raw source files
(data/raw/{family}/...) has strictly more than 50,000 usable records --
confirmed by direct inspection (twemcache/metacdn/cloudphysics/metakv raw
files are named/sized ~100k rows; citibike ~1.9M rows; brightkite ~4.7M
rows; wiki2018 ~100k rows). The canonical dataset-prep parsers
(src/lafc/datasets/*.py) take the first `limit` parsed records in file
order (confirmed: `if limit is not None and len(records) >= limit: break`
in every parser) -- so positions [50000, 100000) of the same raw file are
guaranteed disjoint from the canonical [0, 50000) evaluation range, using
the SAME source dataset, SAME family, SAME parser, SAME default kwargs.

This script:
1. Parses each raw source with limit=100000 using the EXISTING,
   UNMODIFIED parser functions (no algorithmic changes).
2. VALIDATES non-overlap empirically: the first 50,000 parsed records'
   item_ids must exactly match the existing canonical
   data/processed/{family}/trace.jsonl -- proving this script's parsing
   methodology reproduces the exact same extraction that built the
   canonical traces, so positions [50000:] are provably the same
   underlying source continuing on, not a different extraction.
3. Slices records[50000:100000] (or fewer if the raw source is shorter),
   re-indexes request_index starting at 0, and writes to a NEW output
   root: data/processed_fair_v1/{family}/trace.jsonl -- never touching
   data/processed/ or any *_heavy_r1 artifact.
4. Writes a machine-verifiable provenance manifest recording source file
   hashes, extracted index ranges, and per-family record counts.

Usage:
    python scripts/build_evict_value_v1_fair_corpus.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import List

from lafc.datasets.additional_public import (
    parse_cloudphysics,
    parse_meta_oracle_dataset,
    parse_twemcache,
)
from lafc.datasets.base import CanonicalTraceRecord, write_records
from lafc.datasets.brightkite import BRIGHTKITE_FILENAME, parse_brightkite
from lafc.datasets.citibike import parse_citibike
from lafc.datasets.wiki2018 import parse_wiki2018

# This worktree's data/raw and data/processed are gitignored and empty by
# design (see .gitignore); the actual raw/processed files live only in
# whichever checkout originally built them. Overridable via env vars so
# this script can read them read-only from the primary checkout without
# copying or modifying anything there.
RAW_ROOT = Path(os.environ.get("LAFC_RAW_ROOT", "data/raw"))
CANONICAL_PROCESSED_ROOT = Path(os.environ.get("LAFC_PROCESSED_ROOT", "data/processed"))
FAIR_OUTPUT_ROOT = Path("data/processed_fair_v1")
PARSE_LIMIT = 100_000
CANONICAL_TEST_SIZE = 50_000

# (family, trace_name matching analysis/wulver_trace_manifest_full.csv)
FAMILIES = [
    "brightkite", "citibike", "wiki2018", "twemcache", "metakv", "metacdn", "cloudphysics",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_family(family: str) -> List[CanonicalTraceRecord]:
    raw_dir = RAW_ROOT / family
    if family == "brightkite":
        return parse_brightkite(raw_dir / BRIGHTKITE_FILENAME, limit=PARSE_LIMIT)
    if family == "citibike":
        candidates = sorted(raw_dir.glob("*.csv"))
        return parse_citibike(candidates[0], limit=PARSE_LIMIT)
    if family == "wiki2018":
        candidates = sorted(list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.tsv")) + list(raw_dir.glob("*.txt")))
        return parse_wiki2018(candidates[0], limit=PARSE_LIMIT)
    if family == "twemcache":
        return parse_twemcache(raw_dir, limit=PARSE_LIMIT, cluster=None, sample_only=False, paging_view=False)
    if family in ("metakv", "metacdn"):
        return parse_meta_oracle_dataset(
            raw_dir, dataset_name=family, limit=PARSE_LIMIT, usecase=None, read_only=False,
            keep_usecase_fields=True, aggregate_ranges=False, sample_only=False,
        )
    if family == "cloudphysics":
        return parse_cloudphysics(raw_dir, limit=PARSE_LIMIT, page_size=4096, read_only=False, sample_only=False)
    raise ValueError(f"Unsupported family {family!r}")


def main() -> None:
    provenance = {"parse_limit": PARSE_LIMIT, "canonical_test_size": CANONICAL_TEST_SIZE, "families": {}}
    FAIR_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for family in FAMILIES:
        print(f"=== {family} ===", flush=True)
        records = _parse_family(family)
        n_parsed = len(records)

        canonical_path = CANONICAL_PROCESSED_ROOT / family / "trace.jsonl"
        if canonical_path.exists() and n_parsed >= CANONICAL_TEST_SIZE:
            canonical_item_ids = []
            with canonical_path.open() as fh:
                for i, line in enumerate(fh):
                    if i >= CANONICAL_TEST_SIZE:
                        break
                    canonical_item_ids.append(json.loads(line)["item_id"])
            reparsed_item_ids = [r.item_id for r in records[:CANONICAL_TEST_SIZE]]
            if canonical_item_ids != reparsed_item_ids:
                first_mismatch = next(
                    (i for i, (a, b) in enumerate(zip(canonical_item_ids, reparsed_item_ids)) if a != b),
                    None,
                )
                raise AssertionError(
                    f"{family}: reparsed first {CANONICAL_TEST_SIZE} records do NOT match the "
                    f"canonical trace.jsonl (first mismatch at index {first_mismatch}). "
                    "Refusing to build the 'fair' corpus from an extraction that cannot be "
                    "proven disjoint from the canonical evaluation range."
                )
            print(f"  Verified: first {CANONICAL_TEST_SIZE} reparsed records == canonical trace.jsonl", flush=True)
        else:
            print(f"  WARNING: could not verify against canonical trace.jsonl "
                  f"(canonical_path exists={canonical_path.exists()}, n_parsed={n_parsed})", file=__import__("sys").stderr)

        disjoint = records[CANONICAL_TEST_SIZE:]
        n_disjoint = len(disjoint)
        reindexed = [
            dataclasses.replace(r, request_index=i, split="fair_v1_train_source")
            for i, r in enumerate(disjoint)
        ]

        out_path = FAIR_OUTPUT_ROOT / family / "trace.jsonl"
        write_records(reindexed, out_path, fmt="jsonl")
        out_hash = _sha256(out_path)

        provenance["families"][family] = {
            "n_parsed_from_raw": n_parsed,
            "n_disjoint_records_written": n_disjoint,
            "source_index_range": [CANONICAL_TEST_SIZE, CANONICAL_TEST_SIZE + n_disjoint],
            "canonical_test_range": [0, CANONICAL_TEST_SIZE],
            "overlap_with_canonical_test_range": False,
            "output_path": str(out_path),
            "output_trace_sha256": out_hash,
        }
        print(f"  Wrote {n_disjoint} disjoint records to {out_path}", flush=True)

    (FAIR_OUTPUT_ROOT / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"\nProvenance written to {FAIR_OUTPUT_ROOT / 'PROVENANCE.json'}")


if __name__ == "__main__":
    main()
