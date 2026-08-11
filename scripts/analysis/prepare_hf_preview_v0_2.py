#!/usr/bin/env python3
"""Build a small local Hugging Face v0.2 preview package.

This script is intentionally conservative:
- it reads only selected real derived CSV shards;
- it writes a small staging package under analysis/;
- it pseudonymizes object identifiers before writing Parquet;
- it writes relative provenance references, never machine-local source paths.

It does not upload anything.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("analysis/huggingface_dataset_preview_v0_2")
SEED = "lafc-evict-sample-v0.2-preview-seed-20260811"
PSEUDONYM_SALT = "lafc-evict-v0.2-public-preview-object-id-v1"
SAFE_FAMILY = "wiki2018"
CAPACITIES = (32, 64, 128)

REPO_ROOT = Path(__file__).resolve().parents[2]
OBJECTIVE_ROOT = Path("/home/soroush/Augmented-caching-objective-ablation/data/derived/supervision_objective_ablation_v1")
CROSS_FAMILY_ROOT = Path("/home/soroush/Augmented-caching-fairness/data/derived/evict_value_v1_cross_family_v1")

OBJECTIVE_FOLD = "brightkite"
CROSS_FAMILY_FOLD = "brightkite"

OBJECTIVE_ROWS_PER_CAPACITY = 700_000
CROSS_FAMILY_ROWS_PER_CAPACITY = 900_000
SHARDS_PER_CAPACITY = 4

FORBIDDEN_LITERAL_PATTERNS = (
    "/home/soroush",
    "sv96",
    "wulver",
    "/scratch",
    "/tmp/",
    "HF_TOKEN",
    "GITHUB_TOKEN",
)
TOKEN_PATTERNS = (
    re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
)


@dataclass(frozen=True)
class SourceSelection:
    dataset_key: str
    source_root: Path
    root_label: str
    fold: str
    rel_paths: tuple[str, ...]
    rows_per_capacity: int


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_score(*parts: object) -> int:
    text = "|".join(str(part) for part in parts)
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def pseudonymize(value: object) -> str:
    raw = str(value)
    digest = hashlib.sha256(f"{PSEUDONYM_SALT}|{raw}".encode("utf-8")).hexdigest()
    return "obj_" + digest[:24]


def selected_rel_paths(root: Path, fold: str, dataset_key: str, capacity: int) -> tuple[str, ...]:
    if dataset_key == "cross_family_evict_value_v1":
        shard_dir = root / fold / "shards"
        matches = sorted(shard_dir.glob(f"wiki2018_pageviews_en_50k__cap{capacity}.part*.csv"))
    elif dataset_key == "objective_ablation_scalar":
        shard_dir = root / fold / "scalar" / "shards"
        matches = sorted(shard_dir.glob(f"wiki2018_pageviews_en_50k__cap{capacity}.part*.csv"))
    else:
        raise ValueError(f"Unknown dataset_key: {dataset_key}")
    if not matches:
        raise FileNotFoundError(f"No wiki2018 cap={capacity} shards found for {dataset_key}")
    if len(matches) <= SHARDS_PER_CAPACITY:
        chosen = matches
    else:
        indexes = sorted(
            {
                round(i * (len(matches) - 1) / (SHARDS_PER_CAPACITY - 1))
                for i in range(SHARDS_PER_CAPACITY)
            }
        )
        chosen = [matches[i] for i in indexes]
    return tuple(str(path.relative_to(root)) for path in chosen)


def build_source_selections() -> list[SourceSelection]:
    objective_paths: list[str] = []
    cross_paths: list[str] = []
    for cap in CAPACITIES:
        objective_paths.extend(selected_rel_paths(OBJECTIVE_ROOT, OBJECTIVE_FOLD, "objective_ablation_scalar", cap))
        cross_paths.extend(selected_rel_paths(CROSS_FAMILY_ROOT, CROSS_FAMILY_FOLD, "cross_family_evict_value_v1", cap))
    return [
        SourceSelection(
            dataset_key="cross_family_evict_value_v1",
            source_root=CROSS_FAMILY_ROOT,
            root_label="evict_value_v1_cross_family_v1",
            fold=CROSS_FAMILY_FOLD,
            rel_paths=tuple(cross_paths),
            rows_per_capacity=CROSS_FAMILY_ROWS_PER_CAPACITY,
        ),
        SourceSelection(
            dataset_key="objective_ablation_scalar",
            source_root=OBJECTIVE_ROOT,
            root_label="supervision_objective_ablation_v1",
            fold=OBJECTIVE_FOLD,
            rel_paths=tuple(objective_paths),
            rows_per_capacity=OBJECTIVE_ROWS_PER_CAPACITY,
        ),
    ]


def read_selection(selection: SourceSelection) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    frames: list[pd.DataFrame] = []
    source_entries: list[dict[str, object]] = []
    for rel_path in selection.rel_paths:
        path = selection.source_root / rel_path
        source_entries.append(
            {
                "dataset_key": selection.dataset_key,
                "source_root_label": selection.root_label,
                "source_fold": selection.fold,
                "source_relpath": rel_path,
                "source_csv_bytes": path.stat().st_size,
                "source_sha256": sha256_file(path),
            }
        )
        frame = pd.read_csv(path)
        frame["source_root_label"] = selection.root_label
        frame["source_fold"] = selection.fold
        frame["source_relpath"] = rel_path
        frame["source_row_number"] = range(len(frame))
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    return df, source_entries


def deterministic_sample(df: pd.DataFrame, *, dataset_key: str, rows_per_capacity: int) -> pd.DataFrame:
    samples: list[pd.DataFrame] = []
    for capacity, group in df.groupby("capacity", sort=True):
        group = group.copy()
        scores = [
            stable_score(SEED, dataset_key, row.source_relpath, row.source_row_number)
            for row in group[["source_relpath", "source_row_number"]].itertuples(index=False)
        ]
        group["_sample_score"] = scores
        take = min(rows_per_capacity, len(group))
        samples.append(group.nsmallest(take, "_sample_score"))
    result = pd.concat(samples, ignore_index=True)
    result = result.sort_values(["capacity", "_sample_score"]).reset_index(drop=True)
    return result


def normalize_preview(df: pd.DataFrame, *, dataset_key: str) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "preview_dataset", dataset_key)
    out.insert(1, "preview_version", "v0.2")
    out.insert(2, "sampling_seed", SEED)
    out["object_id_public"] = [pseudonymize(value) for value in out["candidate_page_id"]]
    out["candidate_page_id_original_present"] = False
    out["candidate_page_id"] = out["object_id_public"]
    if "example_id" in out.columns:
        out["example_id"] = out["decision_id"].astype(str) + "|" + out["candidate_page_id"].astype(str)
    if "trace_family" in out.columns:
        out = out[out["trace_family"].astype(str) == SAFE_FAMILY].copy()
    return out.drop(columns=["_sample_score"], errors="ignore")


def csv_equivalent_bytes(df: pd.DataFrame) -> int:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return len(buffer.getvalue().encode("utf-8"))


def write_schema(out_dir: Path, data_files: Iterable[Path]) -> dict[str, object]:
    schema: dict[str, object] = {}
    for path in data_files:
        df = pd.read_parquet(path)
        schema[path.name] = [
            {"name": str(name), "dtype": str(dtype)}
            for name, dtype in zip(df.columns, df.dtypes)
        ]
    (out_dir / "schema.json").write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return schema


def write_checksums(out_dir: Path) -> None:
    lines = []
    checksum_path = out_dir / "checksums.sha256"
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file() and p != checksum_path):
        lines.append(f"{sha256_file(path)}  {path.relative_to(out_dir).as_posix()}")
    checksum_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def security_scan(out_dir: Path) -> dict[str, object]:
    findings: list[dict[str, str]] = []
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if path.suffix.lower() in {".md", ".json", ".csv", ".sha256"}:
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in FORBIDDEN_LITERAL_PATTERNS:
                if pattern.lower() in text.lower():
                    findings.append({"path": str(path.relative_to(out_dir)), "pattern": pattern})
            for regex in TOKEN_PATTERNS:
                if regex.search(text):
                    findings.append({"path": str(path.relative_to(out_dir)), "pattern": regex.pattern})
        if path.suffix.lower() == ".parquet":
            df = pd.read_parquet(path)
            object_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col])]
            for col in object_cols:
                values = df[col].dropna().astype(str)
                joined_sample = "\n".join(values.head(50_000).tolist())
                for pattern in FORBIDDEN_LITERAL_PATTERNS:
                    if pattern.lower() in joined_sample.lower():
                        findings.append({"path": str(path.relative_to(out_dir)), "column": col, "pattern": pattern})
                for regex in TOKEN_PATTERNS:
                    if regex.search(joined_sample):
                        findings.append({"path": str(path.relative_to(out_dir)), "column": col, "pattern": regex.pattern})
    result = {"passed": not findings, "findings": findings}
    (out_dir / "security_scan.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_text_artifacts(out_dir: Path, stats: dict[str, object], provenance_rows: list[dict[str, object]]) -> None:
    readme = """# LAFC-Evict Sample v0.2 Preview

This is a local staging package for a proposed Hugging Face dataset update. It has not been uploaded.

Version history:

- v0.1: synthetic publication-workflow dry run built from `examples/tiny_candidate_rows.csv`.
- v0.2: small real derived-data preview using Wikimedia pageview-derived rows only, pending final release review.
- Full release: future work, pending storage, licensing, and provenance review.

The preview contains derived candidate-row features and counterfactual labels. It does not include raw trace rows, raw Wikimedia page titles, machine-local paths, model files, or experiment logs. Object identifiers are deterministic release pseudonyms.

Read `dataset_card.md`, `manifest.json`, `sampling_manifest.json`, and `provenance_summary.csv` before upload consideration.
"""
    card = """---
license: other
tags:
- tabular
- caching
- cache-eviction
- learning-augmented-algorithms
- counterfactual-supervision
- parquet
- preview
pretty_name: LAFC-Evict Sample v0.2 Real-Data Preview
task_categories:
- tabular-regression
- tabular-classification
---

# LAFC-Evict Sample v0.2 Real-Data Preview

This is a proposed small real-data preview for `SoroushVahidi/lafc-evict-sample`. It is prepared locally and is not yet ready for public upload until final license and attribution review is complete.

The preview includes only Wikimedia pageview-derived rows. Upstream pageviews are public Wikimedia dumps; the rows here are derived cache-eviction supervision examples, not raw pageview logs. Object identifiers are pseudonymized.

The existing v0.1 release remains a synthetic workflow dry run and is not suitable for scientific benchmarking. This v0.2 preview is intended to demonstrate the real derived schema at small scale. It is not the full benchmark release.

## Configs

- `cross_family_evict_value_v1`: finite-horizon eviction-loss candidate rows from the corrected cross-family dataset.
- `objective_ablation_scalar`: scalar multi-target objective-ablation candidate rows.

## Limitations

- Only `wiki2018` is included.
- Other families are excluded pending redistribution review.
- This package is a preview, not the full LAFC-Evict release.
- License metadata should remain conservative (`other`) until final attribution/license text is approved.
"""
    notes = """# v0.2 Preview Release Notes

This proposed update transitions `lafc-evict-sample` from a synthetic-only v0.1 workflow smoke test to a small real derived-data preview.

Changes:

- Adds real Wikimedia pageview-derived candidate-row examples.
- Preserves v0.1 as synthetic-only history.
- Uses Parquet payloads.
- Pseudonymizes object identifiers.
- Excludes Brightkite, CitiBike, Twemcache, MetaKV, MetaCDN, and CloudPhysics until final redistribution review.

Do not upload until `docs/DATASET_PUBLICATION_PROVENANCE_AUDIT.md` is reviewed and approved.
"""
    out_dir.joinpath("README.md").write_text(readme, encoding="utf-8")
    out_dir.joinpath("dataset_card.md").write_text(card, encoding="utf-8")
    out_dir.joinpath("RELEASE_NOTES_v0_2.md").write_text(notes, encoding="utf-8")

    with out_dir.joinpath("provenance_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "family",
            "source_name",
            "source_url",
            "redistribution_status",
            "included",
            "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(provenance_rows)

    out_dir.joinpath("statistics.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the local LAFC-Evict v0.2 HF preview staging package.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir.resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {out_dir}")
        for child in out_dir.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    data_dir.mkdir(exist_ok=True)

    source_entries: list[dict[str, object]] = []
    data_files: list[Path] = []
    stats: dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "preview_version": "v0.2",
        "sampling_seed": SEED,
        "included_families": [SAFE_FAMILY],
        "excluded_families": ["brightkite", "citibike", "cloudphysics", "metacdn", "metakv", "twemcache"],
        "datasets": {},
    }

    for selection in build_source_selections():
        df, entries = read_selection(selection)
        source_entries.extend(entries)
        sample = deterministic_sample(df, dataset_key=selection.dataset_key, rows_per_capacity=selection.rows_per_capacity)
        preview = normalize_preview(sample, dataset_key=selection.dataset_key)
        parquet_path = data_dir / f"{selection.dataset_key}.parquet"
        csv_bytes = csv_equivalent_bytes(preview)
        preview.to_parquet(parquet_path, index=False, compression="zstd")
        parquet_bytes = parquet_path.stat().st_size
        data_files.append(parquet_path)
        stats["datasets"][selection.dataset_key] = {
            "row_count": int(len(preview)),
            "decision_count": int(preview["decision_id"].nunique()),
            "capacities": sorted(int(v) for v in preview["capacity"].unique()),
            "splits": sorted(str(v) for v in preview["split"].unique()),
            "source_csv_equivalent_bytes": int(csv_bytes),
            "parquet_bytes": int(parquet_bytes),
            "csv_to_parquet_ratio": float(csv_bytes / parquet_bytes) if parquet_bytes else math.nan,
            "columns": list(preview.columns),
        }

    total_csv = sum(int(d["source_csv_equivalent_bytes"]) for d in stats["datasets"].values())
    total_parquet = sum(path.stat().st_size for path in data_files)
    stats["totals"] = {
        "row_count": int(sum(int(d["row_count"]) for d in stats["datasets"].values())),
        "source_csv_equivalent_bytes": total_csv,
        "parquet_bytes": total_parquet,
        "csv_to_parquet_ratio": float(total_csv / total_parquet) if total_parquet else math.nan,
        "source_files_read": len(source_entries),
        "source_files_read_bytes": int(sum(int(entry["source_csv_bytes"]) for entry in source_entries)),
        "staging_total_bytes_before_checksums": int(sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())),
    }
    # Estimates use conservative full-tree CSV sizes measured locally and the preview's CSV->Parquet ratio.
    ratio = float(stats["totals"]["csv_to_parquet_ratio"])
    stats["full_release_size_estimates"] = {
        "basis": "local du sizes divided by sampled CSV-to-Parquet ratio; rough upper-level estimate",
        "objective_ablation_csv_tree_bytes": 121 * 1024**3,
        "cross_family_csv_tree_bytes": 93 * 1024**3,
        "objective_ablation_parquet_estimate_bytes": int((121 * 1024**3) / ratio),
        "cross_family_parquet_estimate_bytes": int((93 * 1024**3) / ratio),
        "combined_parquet_estimate_bytes": int(((121 + 93) * 1024**3) / ratio),
    }

    sampling_manifest = {
        "method": "deterministic SHA-256 row sampling by source shard relative path and source row number",
        "seed": SEED,
        "object_id_pseudonymization": "candidate_page_id replaced with obj_<first24_sha256(salt|original)>",
        "pseudonym_salt_id": "v0.2-public-preview-object-id-v1",
        "source_entries": source_entries,
    }
    out_dir.joinpath("sampling_manifest.json").write_text(json.dumps(sampling_manifest, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "dataset_repo": "SoroushVahidi/lafc-evict-sample",
        "version": "v0.2-preview",
        "release_title": "LAFC-Evict Sample v0.2 Real-Data Preview",
        "upload_status": "NOT_UPLOADED",
        "readiness": "READY_AFTER_LICENSE_REVIEW",
        "included_families": [SAFE_FAMILY],
        "excluded_families": stats["excluded_families"],
        "data_files": [str(path.relative_to(out_dir)) for path in data_files],
        "no_raw_trace_rows": True,
        "object_ids_pseudonymized": True,
        "machine_paths_in_metadata": False,
    }
    out_dir.joinpath("manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    provenance_rows = [
        {
            "family": "wiki2018",
            "source_name": "Wikimedia public pageviews derived proxy trace",
            "source_url": "https://dumps.wikimedia.org/other/pageviews/",
            "redistribution_status": "ALLOWED_WITH_ATTRIBUTION",
            "included": "yes",
            "reason": "local docs record public pageview source; object IDs are pseudonymized in preview",
        },
        {
            "family": "twemcache",
            "source_name": "Twitter cache trace / Twemcache open trace collection",
            "source_url": "https://github.com/twitter/cache-trace",
            "redistribution_status": "UNCLEAR",
            "included": "no",
            "reason": "local registry says eligible_pending_final_review; final license/attribution not recorded",
        },
        {
            "family": "metakv",
            "source_name": "MetaKV trace family via open cache trace collection",
            "source_url": "https://github.com/cacheMon/cache_dataset",
            "redistribution_status": "UNCLEAR",
            "included": "no",
            "reason": "local registry says eligible_pending_final_review; final license/attribution not recorded",
        },
        {
            "family": "metacdn",
            "source_name": "MetaCDN trace family via open cache trace collection",
            "source_url": "https://github.com/cacheMon/cache_dataset",
            "redistribution_status": "UNCLEAR",
            "included": "no",
            "reason": "local registry says eligible_pending_final_review; final license/attribution not recorded",
        },
        {
            "family": "cloudphysics",
            "source_name": "CloudPhysics / open cache trace collection block I/O family",
            "source_url": "https://github.com/cacheMon/cache_dataset",
            "redistribution_status": "UNCLEAR",
            "included": "no",
            "reason": "specific open trace collection provenance and terms require confirmation",
        },
        {
            "family": "citibike",
            "source_name": "Citi Bike trip-data derived trace family",
            "source_url": "https://citibikenyc.com/system-data",
            "redistribution_status": "UNCLEAR",
            "included": "no",
            "reason": "local governance marks blocked_pending_review with privacy review required",
        },
        {
            "family": "brightkite",
            "source_name": "Brightkite / SNAP-derived trace family",
            "source_url": "https://snap.stanford.edu/data/loc-brightkite.html",
            "redistribution_status": "UNCLEAR",
            "included": "no",
            "reason": "local governance marks blocked_pending_review with license/privacy review required",
        },
    ]
    write_text_artifacts(out_dir, stats, provenance_rows)
    write_schema(out_dir, data_files)
    scan = security_scan(out_dir)
    write_checksums(out_dir)
    print(json.dumps({"output_dir": str(out_dir), "statistics": stats["totals"], "security_scan": scan}, indent=2))


if __name__ == "__main__":
    main()
