from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS as FEATURES

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "train_evict_value_wulver_v1.py"

HEADER = (
    ["trace_name", "trace_family", "dataset_source", "capacity", "horizon",
     "decision_id", "decision_t", "decision_chunk_id", "candidate_page_id", "split",
     "y_loss", "y_value"]
    + list(FEATURES)
)


def _load_module():
    spec = importlib.util.spec_from_file_location("train_evict_value_wulver_v1", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    # The module defines a @dataclass (SplitMeta); dataclasses' string-annotation
    # resolution looks the module up via sys.modules[cls.__module__], which
    # requires the module to be registered there before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_shard(path: Path, family: str, split: str, n_rows: int, horizon: int, seed: int) -> None:
    rng = np.random.RandomState(seed)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(HEADER)
        n_decisions = max(1, n_rows // 3)
        for i in range(n_rows):
            decision_id = f"{family}-d{i % n_decisions}"
            row = [
                f"{family}_trace", family, family, 32, horizon,
                decision_id, i, i // 8, f"page{i}", split,
                float(rng.uniform(0, 10)), float(rng.uniform(0, 10)),
            ] + [float(rng.uniform(-1, 1)) for _ in FEATURES]
            w.writerow(row)


def _build_manifest(tmp_path: Path, horizon: int = 4) -> Path:
    shards_dir = tmp_path / "shards"
    shards_dir.mkdir()
    entries = []
    specs = [
        ("famA", "train", 240, 1),
        ("famB", "train", 180, 2),
        ("famC", "val", 90, 3),
    ]
    for family, split, n_rows, seed in specs:
        p = shards_dir / f"{family}.csv"
        _write_shard(p, family, split, n_rows, horizon, seed)
        entries.append({"path": str(p), "row_count": n_rows})
    manifest = {"format": "csv", "horizons": [horizon], "shards": entries}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_load_split_compact_matches_dict_loader_no_subsampling(tmp_path):
    module = _load_module()
    manifest_path = _build_manifest(tmp_path)

    old_rows = module._load_rows_from_manifest(manifest_path, 4, "train", None, seed=0)
    x_old, y_old = module._xy(old_rows)

    x_new, y_new, meta = module._load_split_compact(manifest_path, 4, "train", None, seed=0, need_metadata=False)

    assert meta is None
    assert x_new.shape == x_old.shape
    assert np.array_equal(x_new, x_old)
    assert np.array_equal(y_new, y_old)


def test_load_split_compact_metadata_matches_dict_loader(tmp_path):
    module = _load_module()
    manifest_path = _build_manifest(tmp_path)

    old_rows = module._load_rows_from_manifest(manifest_path, 4, "val", None, seed=1)
    x_old, y_old = module._xy(old_rows)

    x_new, y_new, meta = module._load_split_compact(manifest_path, 4, "val", None, seed=1, need_metadata=True)

    assert np.array_equal(x_new, x_old)
    assert np.array_equal(y_new, y_old)
    assert meta.decision_id == [str(r["decision_id"]) for r in old_rows]
    assert meta.candidate_page_id == [str(r["candidate_page_id"]) for r in old_rows]
    assert meta.trace_family == [str(r["trace_family"]) for r in old_rows]


def test_load_split_compact_subsampling_selects_same_positions(tmp_path):
    module = _load_module()
    manifest_path = _build_manifest(tmp_path)
    max_rows = 50

    old_rows = module._load_rows_from_manifest(manifest_path, 4, "train", max_rows, seed=7)
    x_old, y_old = module._xy(old_rows)

    x_new, y_new, _ = module._load_split_compact(manifest_path, 4, "train", max_rows, seed=7, need_metadata=False)

    assert x_new.shape[0] == max_rows
    # random.sample over a sequence of length n depends only on n, not content,
    # so sampling row-indices selects the same positions as sampling row-dicts
    # of the same length -- resulting y arrays (as sets, since dict iteration
    # order == file order == array order in both paths) must match exactly.
    assert np.array_equal(np.sort(y_new), np.sort(y_old))
    assert np.array_equal(y_new, y_old)


def test_ranking_and_family_metrics_arr_match_dict_versions(tmp_path):
    module = _load_module()
    manifest_path = _build_manifest(tmp_path)

    rows = module._load_rows_from_manifest(manifest_path, 4, "train", None, seed=0)
    x, y = module._xy(rows)
    rng = np.random.RandomState(0)
    preds = y + rng.normal(scale=0.5, size=y.shape)

    old_ranking = module._ranking_metrics(rows, preds)
    old_family = module._family_metrics(rows, preds)

    _, _, meta = module._load_split_compact(manifest_path, 4, "train", None, seed=0, need_metadata=True)
    new_ranking = module._ranking_metrics_arr(meta.decision_id, meta.candidate_page_id, y, preds)
    new_family = module._family_metrics_arr(meta, y, preds)

    assert old_ranking == new_ranking
    assert old_family == new_family


def test_memory_guard_raises_before_and_never_after_threshold():
    module = _load_module()
    with pytest.raises(module.MemoryBudgetExceeded):
        module._check_memory_guard(0.0, stage="test")
    module._check_memory_guard(None, stage="test")  # no-op, must not raise
    module._check_memory_guard(1e6, stage="test")  # far above any real usage, must not raise


def test_cli_memory_bounded_matches_default_end_to_end(tmp_path):
    """The equivalence contract, exercised through the real CLI: identical
    comparison rows and metrics with and without --memory-bounded, including
    through the --max-*-rows subsampling path."""
    manifest_path = _build_manifest(tmp_path)

    def run(memory_bounded: bool, out_tag: str) -> Path:
        out_dir = tmp_path / out_tag
        out_dir.mkdir()
        cmd = [
            sys.executable, str(SCRIPT_PATH),
            "--manifest", str(manifest_path), "--horizons", "4", "--seed", "0",
            "--max-train-rows", "100", "--max-val-rows", "30",
            "--models-dir", str(out_dir / "models"),
            "--metrics-json", str(out_dir / "metrics.json"),
            "--comparison-csv", str(out_dir / "comparison.csv"),
            "--best-config-json", str(out_dir / "best.json"),
        ]
        if memory_bounded:
            cmd += ["--memory-bounded", "--memory-guard-gb", "40"]
        subprocess.run(cmd, check=True, cwd=REPO_ROOT, capture_output=True, text=True)
        return out_dir

    old_dir = run(False, "old")
    new_dir = run(True, "new")

    old_csv = (old_dir / "comparison.csv").read_text()
    new_csv = (new_dir / "comparison.csv").read_text()
    assert old_csv == new_csv

    old_metrics = json.loads((old_dir / "metrics.json").read_text())
    new_metrics = json.loads((new_dir / "metrics.json").read_text())
    old_metrics.pop("execution_mode", None)
    new_metrics.pop("execution_mode", None)
    assert old_metrics == new_metrics
