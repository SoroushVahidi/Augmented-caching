from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
NAMESPACE = "pe_long_horizon_canonical_regen_20260916"
CONFIG_DIR = REPO / "configs" / NAMESPACE
PROVENANCE_DIR = CONFIG_DIR / "provenance"
ANALYSIS_DIR = REPO / "analysis" / NAMESPACE
CANONICAL_MANIFEST = REPO / "analysis" / "wulver_trace_manifest_full.csv"
RELEASE_DECISION_VIEW = Path(
    "/home/soroush/projects/lafc-evict-dataset/repo/release/"
    "lafc-evict-v0.1-open-current-contract-preserved/data/decision_view/decision_view.parquet"
)
TRACE_ROOT_CANDIDATES = [
    Path(os.environ["TRACE_SOURCE_ROOT"]) if os.environ.get("TRACE_SOURCE_ROOT") else None,
    REPO,
    Path("/mmfs1/scratch/ikoutis/sv96/lafc-evict/pe_long_horizon_canonical_regen_20260916/source_traces"),
    Path("/mmfs1/home/sv96/lafc-work/Augmented-caching"),
    Path("/home/soroush/projects/augmented-caching/repo"),
]
FAMILIES = ["cloudphysics", "metacdn", "metakv", "twemcache", "wiki2018"]
READER_NAMES = {"cloudphysics": "alibaba-block"}
CAPACITIES = [32, 64, 128, 256]
HORIZONS = [32, 64, 128]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def load_trace_manifest() -> dict[str, dict[str, str]]:
    rows = {}
    with CANONICAL_MANIFEST.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["trace_family"] in FAMILIES:
                rows[row["trace_family"]] = dict(row)
    missing = sorted(set(FAMILIES) - set(rows))
    if missing:
        raise SystemExit(f"missing canonical manifest rows: {missing}")
    return rows


def resolve_trace_path(rel_path: str) -> Path | None:
    p = Path(rel_path)
    if p.is_absolute() and p.exists():
        return p
    for root in TRACE_ROOT_CANDIDATES:
        if root is None:
            continue
        p = root / rel_path
        if p.exists():
            return p
    return None


def canonical_h16_key_summaries() -> dict[str, dict[str, object]]:
    import duckdb

    con = duckdb.connect()
    summaries: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        for cap in CAPACITIES:
            rows = con.execute(
                """
                select trace_name, trace_family, capacity, decision_t, split
                from read_parquet(?)
                where horizon = 16 and trace_family = ? and capacity = ?
                order by trace_name, trace_family, capacity, decision_t, split
                """,
                [str(RELEASE_DECISION_VIEW), family, cap],
            ).fetchall()
            if not rows:
                raise SystemExit(f"no canonical H16 rows for {family} cap={cap}")
            keys = [f"{r[0]}|{r[1]}|cap={int(r[2])}|t={int(r[3])}|split={r[4]}" for r in rows]
            if len(keys) != len(set(keys)):
                raise SystemExit(f"duplicate canonical H16 physical keys for {family} cap={cap}")
            h = hashlib.sha256()
            for key in keys:
                h.update(key.encode("utf-8"))
                h.update(b"\n")
            summaries[f"{family}__cap{cap}"] = {
                "horizon": 16,
                "decision_count": len(keys),
                "key_sha256": h.hexdigest(),
                "key_format": "trace_name|trace_family|cap=<capacity>|t=<decision_t>|split=<split>",
                "min_decision_t": min(int(r[3]) for r in rows),
                "max_decision_t": max(int(r[3]) for r in rows),
                "split_counts": dict(
                    con.execute(
                        """
                        select split, count(*)
                        from read_parquet(?)
                        where horizon = 16 and trace_family = ? and capacity = ?
                        group by split
                        order by split
                        """,
                        [str(RELEASE_DECISION_VIEW), family, cap],
                    ).fetchall()
                ),
            }
    return summaries


def build_manifest() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    trace_rows = load_trace_manifest()
    key_summaries = canonical_h16_key_summaries()
    tasks = []
    cells = []
    index = 0
    for family in FAMILIES:
        trace = trace_rows[family]
        resolved_trace_path = resolve_trace_path(trace["path"])
        if resolved_trace_path is None:
            raise SystemExit(f"canonical trace source is not present for {family}: {trace['path']}")
        for cap in CAPACITIES:
            task_id = f"{family}__cap{cap}"
            task = {
                "array_index": index,
                "task_id": task_id,
                "family": family,
                "reader_facing_family": READER_NAMES.get(family, family),
                "capacity": cap,
                "horizons": HORIZONS,
                "trace_path": trace["path"],
                "canonical_manifest_trace_path": trace["path"],
                "preparation_resolved_trace_path": str(resolved_trace_path),
                "trace_line_count": sum(1 for _ in resolved_trace_path.open("rb")),
                "trace_sha256": sha256_file(resolved_trace_path),
                "trace_name": trace["trace_name"],
                "dataset_source": trace["dataset_source"],
                "trace_family": trace["trace_family"],
                "out_dir": f"/mmfs1/scratch/ikoutis/sv96/lafc-evict/{NAMESPACE}/raw/{task_id}",
                "canonical_h16": key_summaries[task_id],
                "status": "NOT_RUN",
            }
            tasks.append(task)
            for h in HORIZONS:
                cells.append({"family": family, "capacity": cap, "horizon": h, "task_id": task_id})
            index += 1
    manifest = OrderedDict(
        [
            ("format", f"{NAMESPACE}_manifest_v1"),
            ("namespace", NAMESPACE),
            ("run_root", f"/mmfs1/scratch/ikoutis/sv96/lafc-evict/{NAMESPACE}"),
            ("supersedes", "analysis/pe_long_horizon_production_v1"),
            ("supersession_reason", "stale/undercovered twemcache provenance and non-canonical trace-name split hashing"),
            ("canonical_h16_release_decision_view", str(RELEASE_DECISION_VIEW)),
            ("canonical_trace_manifest", str(CANONICAL_MANIFEST.relative_to(REPO))),
            ("canonical_trace_manifest_sha256", sha256_file(CANONICAL_MANIFEST)),
            (
                "trace_source_resolution",
                "Resolve task.trace_path relative to TRACE_SOURCE_ROOT, or to the execution worktree/source-trace root.",
            ),
            ("families", FAMILIES),
            ("excluded_families", ["brightkite", "citibike"]),
            ("capacities", CAPACITIES),
            ("production_horizons", HORIZONS),
            ("h16_regenerated", False),
            ("split_mode", "trace_chunk"),
            ("chunk_size", 4096),
            ("split_seed", 7),
            ("split_train_pct", 70),
            ("split_val_pct", 15),
            ("history_window", 64),
            ("max_requests_per_trace", 50000),
            ("max_rows_per_shard", 500000),
            ("expected_source_head_at_preparation", git_value("rev-parse", "HEAD")),
            ("expected_source_branch_at_preparation", git_value("branch", "--show-current")),
            ("generator_lib_sha256", sha256_file(REPO / "src/lafc/evict_value_wulver_v1.py")),
            (
                "generator_driver_sha256",
                sha256_file(REPO / "scripts/experiments/canonical/build_evict_value_dataset_wulver_v1.py"),
            ),
            ("num_production_tasks", len(tasks)),
            ("num_logical_cells", len(cells)),
            ("tasks", tasks),
            ("cells", cells),
        ]
    )
    (CONFIG_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (CONFIG_DIR / "resource_params.env").write_text(
        "\n".join(
            [
                "CPUS_PER_TASK=2",
                "MEMORY_PER_TASK=4G",
                "WALLTIME_PER_TASK=10:00:00",
                "ARRAY_SPEC=0-19%10",
                "NUM_PRODUCTION_TASKS=20",
                "NUM_LOGICAL_CELLS=60",
                f"EXPECTED_GENERATOR_LIB_SHA256={manifest['generator_lib_sha256']}",
                f"EXPECTED_GENERATOR_DRIVER_SHA256={manifest['generator_driver_sha256']}",
                f"EXPECTED_MANIFEST_SHA256={sha256_file(CONFIG_DIR / 'manifest.json')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(CONFIG_DIR / "manifest.json")


def source_audit() -> dict[str, object]:
    trace_rows = load_trace_manifest()
    traces = {}
    for family in FAMILIES:
        rel = trace_rows[family]["path"]
        path = resolve_trace_path(rel)
        traces[family] = {
            "path": rel,
            "trace_name": trace_rows[family]["trace_name"],
            "resolved_local_path": str(path) if path else None,
            "exists_locally": bool(path),
            "line_count": sum(1 for _ in path.open("rb")) if path else None,
            "sha256": sha256_file(path) if path else None,
        }
    old_table = REPO / "analysis/pe_long_horizon_production_v1/summaries/table_A_family_capacity_horizon.csv"
    old_twem_rows = 0
    if old_table.exists():
        with old_table.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row["family"] == "twemcache":
                    old_twem_rows += int(row["candidate_rows"])
    import duckdb

    con = duckdb.connect()
    canonical_twem_h16_rows = con.execute(
        "select sum(candidate_count) from read_parquet(?) where horizon = 16 and trace_family = 'twemcache'",
        [str(RELEASE_DECISION_VIEW)],
    ).fetchone()[0]
    return {
        "canonical_manifest": str(CANONICAL_MANIFEST.relative_to(REPO)),
        "canonical_manifest_sha256": sha256_file(CANONICAL_MANIFEST),
        "trace_sources": traces,
        "twemcache_full_source_proof": {
            "canonical_h16_candidate_rows_h16_all_caps": int(canonical_twem_h16_rows),
            "canonical_h4_h8_h16_candidate_rows_all_caps": int(canonical_twem_h16_rows) * 3,
            "superseded_long_horizon_twemcache_candidate_rows_all_horizons": old_twem_rows,
            "canonical_expected_total": 44330208,
            "superseded_expected_total": 10591392,
        },
    }


def run_pilot() -> dict[str, object]:
    from lafc.evict_value_wulver_v1 import WulverDatasetConfig, assign_split, load_trace_from_any

    import duckdb

    cfg = WulverDatasetConfig(horizons=(32,), chunk_size=4096, split_seed=7)
    trace_rows = load_trace_manifest()
    con = duckdb.connect()
    results = []
    for family in ["twemcache", "metakv"]:
        trace = trace_rows[family]
        path = resolve_trace_path(trace["path"])
        if not path:
            raise SystemExit(f"pilot source missing for {family}: {trace['path']}")
        reqs, _pages, _source = load_trace_from_any(str(path))
        reqs = reqs[:50000]
        for cap in [32, 128]:
            order: OrderedDict[str, None] = OrderedDict()
            generated = []
            for t, req in enumerate(reqs):
                pid = req.page_id
                if pid in order:
                    order.move_to_end(pid)
                    continue
                if len(order) < cap:
                    order[pid] = None
                    continue
                split = assign_split(
                    split_mode=cfg.split_mode,
                    trace_name=trace["trace_name"],
                    dataset_source=trace["dataset_source"],
                    trace_family=trace["trace_family"],
                    t=t,
                    chunk_size=cfg.chunk_size,
                    train_pct=cfg.split_train_pct,
                    val_pct=cfg.split_val_pct,
                    seed=cfg.split_seed,
                )
                generated.append(f"{trace['trace_name']}|{family}|cap={cap}|t={t}|split={split}")
                order.popitem(last=False)
                order[pid] = None
            canonical = [
                f"{r[0]}|{r[1]}|cap={int(r[2])}|t={int(r[3])}|split={r[4]}"
                for r in con.execute(
                    """
                    select trace_name, trace_family, capacity, decision_t, split
                    from read_parquet(?)
                    where horizon = 16 and trace_family = ? and capacity = ?
                    order by trace_name, trace_family, capacity, decision_t, split
                    """,
                    [str(RELEASE_DECISION_VIEW), family, cap],
                ).fetchall()
            ]
            results.append(
                {
                    "family": family,
                    "capacity": cap,
                    "generated_count": len(generated),
                    "canonical_h16_count": len(canonical),
                    "unexpected_generated_keys": len(set(generated) - set(canonical)),
                    "missing_canonical_keys": len(set(canonical) - set(generated)),
                    "split_hash_identity": generated == canonical,
                    "min_generated_t": min(int(k.split("|t=")[1].split("|")[0]) for k in generated),
                    "max_generated_t": max(int(k.split("|t=")[1].split("|")[0]) for k in generated),
                }
            )
    return {"scope": "twemcache/metakv capacities 32 and 128, H32 decision enumeration", "results": results}


def preflight() -> int:
    errors: list[str] = []
    manifest = json.loads((CONFIG_DIR / "manifest.json").read_text(encoding="utf-8"))
    if len(manifest["tasks"]) != 20 or len(manifest["cells"]) != 60:
        errors.append("manifest cardinality is not 20 tasks / 60 logical cells")
    if any(16 in task["horizons"] for task in manifest["tasks"]):
        errors.append("H16 appears in production horizons")
    if {task["family"] for task in manifest["tasks"]} != set(FAMILIES):
        errors.append("family set mismatch")
    if {task["capacity"] for task in manifest["tasks"]} != set(CAPACITIES):
        errors.append("capacity set mismatch")
    if shutil.which("sbatch") is None:
        errors.append("sbatch is not available in this environment")
    if not Path("/mmfs1/scratch/ikoutis/sv96").exists():
        errors.append("/mmfs1/scratch/ikoutis/sv96 is not mounted; quota/storage gate cannot be validated here")
    audit = source_audit()
    pilot = run_pilot()
    for result in pilot["results"]:
        if result["unexpected_generated_keys"] or result["missing_canonical_keys"] or not result["split_hash_identity"]:
            errors.append(f"pilot mismatch: {result}")
    report = {
        "preflight_status": "PASS" if not errors else "FAIL_STOP_BEFORE_SUBMISSION",
        "errors": errors,
        "source_audit": audit,
        "pilot": pilot,
        "environment": {
            "hostname": os.uname().nodename,
            "sbatch": shutil.which("sbatch"),
            "mmfs1_scratch_mounted": Path("/mmfs1/scratch/ikoutis/sv96").exists(),
        },
    }
    out = PROVENANCE_DIR / "preflight_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if not errors else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build-manifest", "preflight"])
    args = ap.parse_args()
    if args.command == "build-manifest":
        build_manifest()
        return 0
    return preflight()


if __name__ == "__main__":
    sys.exit(main())
