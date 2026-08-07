"""Generate the 7 leave-one-family-out fold manifests for
evict_value_v1_cross_family_v1 (configs/reviewer_fairness_cross_family_v1.json).

For each held-out test family, writes:
    configs/fair_cross_family_v1/folds/<family>.json
        -- full fold description (test/val/train assignment, hashes, paths)
    configs/fair_cross_family_v1/folds/<family>_train_manifest.csv
        -- trace-manifest CSV of the 5 training families only
    configs/fair_cross_family_v1/folds/<family>_family_split_map.json
        -- {family: "train"|"val"} for the 6 non-held-out families, fed to
           build_evict_value_dataset_wulver_v1.py --split-mode family_map

The held-out test family never appears in either output file for its own
fold -- verified by an explicit assertion, not just by construction.

Usage:
    python scripts/build_evict_value_v1_cross_family_folds.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

CANONICAL_MANIFEST = Path("analysis/wulver_trace_manifest_full.csv")
OUT_DIR = Path("configs/fair_cross_family_v1/folds")
# This worktree's data/processed/ is gitignored and empty by design; read
# canonical traces read-only from wherever they were actually built (e.g.
# the primary checkout) for hash computation only -- never written to.
DATA_READ_ROOT = Path(os.environ.get("LAFC_DATA_READ_ROOT", "."))

SORTED_FAMILIES = ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_rows = list(csv.DictReader(CANONICAL_MANIFEST.open()))
    by_family = {r["trace_family"]: r for r in manifest_rows}
    missing = set(SORTED_FAMILIES) - set(by_family)
    if missing:
        raise SystemExit(f"Canonical manifest missing families: {missing}")

    for i, test_family in enumerate(SORTED_FAMILIES):
        val_family = SORTED_FAMILIES[(i + 1) % len(SORTED_FAMILIES)]
        train_families = [f for f in SORTED_FAMILIES if f not in (test_family, val_family)]

        assert test_family not in train_families
        assert test_family != val_family
        assert val_family not in train_families
        assert len(train_families) == 5

        test_row = by_family[test_family]
        test_path = DATA_READ_ROOT / test_row["path"]
        test_hash = _sha256(test_path) if test_path.exists() else None

        family_split_map = {val_family: "val"}
        for tf in train_families:
            family_split_map[tf] = "train"
        assert test_family not in family_split_map

        train_manifest_path = OUT_DIR / f"{test_family}_train_manifest.csv"
        with train_manifest_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "trace_name", "trace_family"])
            w.writeheader()
            for tf in train_families + [val_family]:
                r = by_family[tf]
                w.writerow({"path": r["path"], "trace_name": r["trace_name"], "trace_family": tf})

        # Verify the held-out family's path string appears nowhere in the
        # manifest we just wrote -- a direct, mechanical check, not just
        # "we didn't add it on purpose".
        written = train_manifest_path.read_text()
        assert test_row["path"] not in written, (
            f"held-out family {test_family}'s path leaked into its own fold's training manifest"
        )
        assert test_row["trace_name"] not in written

        family_split_map_path = OUT_DIR / f"{test_family}_family_split_map.json"
        family_split_map_path.write_text(json.dumps(family_split_map, indent=2) + "\n")

        fold = {
            "fold_id": f"cross_family_v1_{test_family}",
            "test_family": test_family,
            "test_trace_name": test_row["trace_name"],
            "test_trace_path": test_row["path"],
            "test_trace_sha256": test_hash,
            "history": [0, 10000],
            "score": [10000, 50000],
            "validation_family": val_family,
            "training_families": train_families,
            "train_manifest": str(train_manifest_path),
            "family_split_map": str(family_split_map_path),
            "held_out_family_rows_in_train_manifest": 0,
            "model_output_path": f"models/evict_value_v1_cross_family_v1_{test_family}.pkl",
            "dataset_output_root": f"data/derived/evict_value_v1_cross_family_v1/{test_family}/",
        }
        (OUT_DIR / f"{test_family}.json").write_text(json.dumps(fold, indent=2) + "\n")
        print(f"[fold] test={test_family} val={val_family} train={train_families}")

    print(f"\nWrote 7 fold manifests to {OUT_DIR}/")


if __name__ == "__main__":
    main()
