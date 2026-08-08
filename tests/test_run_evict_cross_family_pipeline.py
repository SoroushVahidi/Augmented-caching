from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "experiments" / "run_evict_cross_family_pipeline.py"


def _load_module(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("run_evict_cross_family_pipeline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Redirect every path constant into an isolated tmp_path so tests never
    # touch the real repo's data/models/analysis directories.
    monkeypatch.setattr(module, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(module, "STAGING_MODELS_DIR", tmp_path / "staging")
    monkeypatch.setattr(module, "FINAL_MODELS_DIR", tmp_path / "models")
    monkeypatch.setattr(module, "ANALYSIS_DIR", tmp_path / "analysis")
    return module


def test_stage1_valid_false_when_manifest_missing(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    assert module.stage1_valid("brightkite") is False


def test_stage1_valid_true_for_complete_manifest(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fold_dir = module.DATA_DIR / "brightkite"
    shards_dir = fold_dir / "shards"
    shards_dir.mkdir(parents=True)
    shard_path = shards_dir / "citibike_x.csv"
    shard_path.write_text("a,b\n1,2\n")
    manifest = {"shards": [{"path": str(shard_path), "row_count": 1}]}
    (fold_dir / "manifest.json").write_text(json.dumps(manifest))
    assert module.stage1_valid("brightkite") is True


def test_stage1_valid_false_when_shard_file_missing(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fold_dir = module.DATA_DIR / "brightkite"
    fold_dir.mkdir(parents=True)
    manifest = {"shards": [{"path": str(fold_dir / "shards" / "missing.csv"), "row_count": 1}]}
    (fold_dir / "manifest.json").write_text(json.dumps(manifest))
    assert module.stage1_valid("brightkite") is False


def test_stage1_valid_false_when_held_out_family_contaminates_shards(monkeypatch, tmp_path):
    """Train/test separation invariant: a shard whose filename prefix is the
    held-out family itself must never be treated as a valid training shard."""
    module = _load_module(monkeypatch, tmp_path)
    fold_dir = module.DATA_DIR / "brightkite"
    shards_dir = fold_dir / "shards"
    shards_dir.mkdir(parents=True)
    contaminated = shards_dir / "brightkite_should_not_be_here.csv"
    contaminated.write_text("a,b\n1,2\n")
    manifest = {"shards": [{"path": str(contaminated), "row_count": 1}]}
    (fold_dir / "manifest.json").write_text(json.dumps(manifest))
    assert module.stage1_valid("brightkite") is False


def test_stage2_valid_requires_all_four_artifacts(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fam = "brightkite"
    assert module.stage2_valid(fam) is False

    staging = module.STAGING_MODELS_DIR / fam
    staging.mkdir(parents=True)
    (staging / "evict_value_wulver_v1_best.pkl").write_bytes(b"x")
    assert module.stage2_valid(fam) is False  # still missing analysis artifacts

    analysis_fold = module.ANALYSIS_DIR / fam
    analysis_fold.mkdir(parents=True)
    (analysis_fold / "best_config.json").write_text("{}")
    (analysis_fold / "model_comparison.csv").write_text("a,b\n1,2\n")
    (analysis_fold / "train_metrics.json").write_text("{}")
    assert module.stage2_valid(fam) is True


def test_stage3_valid_checks_final_pkl(monkeypatch, tmp_path):
    module = _load_module(monkeypatch, tmp_path)
    fam = "brightkite"
    assert module.stage3_valid(fam) is False
    module.FINAL_MODELS_DIR.mkdir(parents=True)
    (module.FINAL_MODELS_DIR / f"evict_value_v1_cross_family_v1_{fam}.pkl").write_bytes(b"model-bytes")
    assert module.stage3_valid(fam) is True


def test_dry_run_never_writes_state(monkeypatch, tmp_path, capsys):
    """Regression test: an earlier version of this runner wrote 'complete'
    into the checkpoint file even under --dry-run, which would have made a
    subsequent real run silently skip folds that were never actually built."""
    module = _load_module(monkeypatch, tmp_path)
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(sys, "argv", [
        "run_evict_cross_family_pipeline.py", "--dry-run",
        "--families", "brightkite",
        "--state-file", str(state_file),
    ])
    module.main()
    assert not state_file.exists()
    out = capsys.readouterr().out
    assert "no state was written" in out
