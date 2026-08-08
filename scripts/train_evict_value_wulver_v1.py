from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model

FEATURES = list(EVICT_VALUE_V1_FEATURE_COLUMNS)


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


class MemoryBudgetExceeded(RuntimeError):
    pass


def _check_memory_guard(threshold_gb: Optional[float], stage: str) -> None:
    """Soft guard: abort cleanly (before the kernel OOM-killer would) once peak
    RSS crosses ``threshold_gb``. Checked between coarse-grained units of work
    (per shard while loading, before each model fit) -- never mid-write of a
    model artifact, so a triggered guard never leaves a corrupt/partial file."""
    if threshold_gb is None:
        return
    rss = _rss_gb()
    if rss >= threshold_gb:
        raise MemoryBudgetExceeded(
            f"peak RSS {rss:.1f} GiB >= memory guard threshold {threshold_gb:.1f} GiB at stage={stage!r}"
        )


@dataclass
class SplitMeta:
    decision_id: List[str]
    candidate_page_id: List[str]
    trace_family: List[str]


def _ranking_metrics(rows: List[Dict[str, object]], preds: np.ndarray) -> Dict[str, float]:
    grouped: Dict[str, List[Tuple[Dict[str, object], float]]] = {}
    for row, pred in zip(rows, preds):
        grouped.setdefault(str(row["decision_id"]), []).append((row, float(pred)))
    top1 = 0
    regrets: List[float] = []
    for items in grouped.values():
        chosen = min(items, key=lambda x: (x[1], str(x[0]["candidate_page_id"])))
        best = min(items, key=lambda x: (float(x[0]["y_loss"]), str(x[0]["candidate_page_id"])))
        top1 += int(chosen[0]["candidate_page_id"] == best[0]["candidate_page_id"])
        regrets.append(float(chosen[0]["y_loss"]) - float(best[0]["y_loss"]))
    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_eviction_match": float(top1 / denom),
        "mean_regret_vs_oracle": float(np.mean(regrets) if regrets else 0.0),
    }


def _metrics(y: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
    }


def _family_metrics(rows: List[Dict[str, object]], preds: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    fams = sorted({str(r["trace_family"]) for r in rows})
    for fam in fams:
        idx = [i for i, r in enumerate(rows) if str(r["trace_family"]) == fam]
        if not idx:
            continue
        sub_rows = [rows[i] for i in idx]
        sub_p = preds[idx]
        sub_y = np.asarray([float(r["y_loss"]) for r in sub_rows], dtype=float)
        rm = _ranking_metrics(sub_rows, sub_p)
        out[fam] = {**_metrics(sub_y, sub_p), **rm}
    return out


def _load_rows_from_manifest(
    manifest_path: Path,
    horizon: int,
    split: str,
    max_rows: int | None,
    seed: int,
) -> List[Dict[str, object]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = [Path(s["path"]) for s in payload.get("shards", [])]
    rng_order = random.Random(seed + 31)
    paths = list(paths)
    rng_order.shuffle(paths)
    rows: List[Dict[str, object]] = []
    for sp in paths:
        if not sp.exists():
            continue
        with sp.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                if int(r["horizon"]) != horizon:
                    continue
                if str(r["split"]) != split:
                    continue
                row = dict(r)
                for c in FEATURES + ["y_loss", "y_value"]:
                    row[c] = float(row[c])
                row["horizon"] = int(row["horizon"])
                rows.append(row)
                if max_rows is not None and len(rows) >= max_rows * 2:
                    # Enough raw rows to subsample; avoids reading multi-GB shards fully.
                    break
        if max_rows is not None and len(rows) >= max_rows * 2:
            break
    if max_rows is not None and len(rows) > max_rows:
        rng = random.Random(seed)
        rows = rng.sample(rows, max_rows)
    return rows


def _xy(rows: List[Dict[str, object]]) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(r[c]) for c in FEATURES] for r in rows], dtype=float)
    y = np.asarray([float(r["y_loss"]) for r in rows], dtype=float)
    return x, y


def _load_split_compact(
    manifest_path: Path,
    horizon: int,
    split: str,
    max_rows: int | None,
    seed: int,
    need_metadata: bool,
    memory_guard_gb: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, Optional[SplitMeta]]:
    """Memory-bounded equivalent of ``_xy(_load_rows_from_manifest(...))``.

    Streams matching rows directly into a preallocated float64 array instead
    of materializing a Python dict per row (the OOM root cause -- see
    docs/evict_cross_family_oom_diagnosis.md, measured at ~2.3KB/row for the
    dict path vs ~0.2KB/row here, same dtype, same values). Shard visitation
    order, the (horizon, split) filter, the raw-row accumulation cap
    (``max_rows * 2``), and the final ``random.Random(seed).sample`` selection
    are all bit-for-bit identical to ``_load_rows_from_manifest`` -- sampling
    over row-indices selects the same positions as sampling over row-dicts of
    the same length, since ``random.sample`` depends only on population size.
    """
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_entries = payload.get("shards", [])
    entries = [(Path(s["path"]), int(s.get("row_count", 0))) for s in shard_entries]
    rng_order = random.Random(seed + 31)
    rng_order.shuffle(entries)
    entries = [(p, rc) for p, rc in entries if p.exists()]

    raw_cap = None if max_rows is None else max_rows * 2
    prealloc = sum(rc for _, rc in entries)
    if raw_cap is not None:
        prealloc = min(prealloc, raw_cap)

    n_features = len(FEATURES)
    x = np.empty((prealloc, n_features), dtype=np.float64)
    y = np.empty(prealloc, dtype=np.float64)
    decision_id: List[str] = [] if need_metadata else None  # type: ignore[assignment]
    candidate_page_id: List[str] = [] if need_metadata else None  # type: ignore[assignment]
    trace_family: List[str] = [] if need_metadata else None  # type: ignore[assignment]

    n = 0
    col_idx = None
    for shard_i, (sp, _rc) in enumerate(entries):
        with sp.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader)
            if col_idx is None:
                col_idx = {name: i for i, name in enumerate(header)}
                feat_idx = [col_idx[c] for c in FEATURES]
                h_idx, s_idx, y_idx = col_idx["horizon"], col_idx["split"], col_idx["y_loss"]
                if need_metadata:
                    did_idx = col_idx["decision_id"]
                    cpid_idx = col_idx["candidate_page_id"]
                    fam_idx = col_idx["trace_family"]
            for r in reader:
                if r[h_idx] != str(horizon) or r[s_idx] != split:
                    continue
                if n >= prealloc:
                    # A later shard contributed more matches than the
                    # row_count-derived preallocation assumed (row_count
                    # counts ALL rows in the shard, matches are a subset, so
                    # this only grows the buffer, never truncates data).
                    grow_to = max(n + 1, int(prealloc * 1.5) + 1)
                    x = np.resize(x, (grow_to, n_features))
                    y = np.resize(y, (grow_to,))
                    prealloc = grow_to
                for j, fi in enumerate(feat_idx):
                    x[n, j] = float(r[fi])
                y[n] = float(r[y_idx])
                if need_metadata:
                    decision_id.append(r[did_idx])
                    candidate_page_id.append(r[cpid_idx])
                    trace_family.append(r[fam_idx])
                n += 1
                if raw_cap is not None and n >= raw_cap:
                    break
        if raw_cap is not None and n >= raw_cap:
            break
        if memory_guard_gb is not None and shard_i % 4 == 0:
            _check_memory_guard(memory_guard_gb, stage=f"loading split={split} shard={shard_i}")

    x = x[:n]
    y = y[:n]

    if max_rows is not None and n > max_rows:
        rng = random.Random(seed)
        idx = rng.sample(range(n), max_rows)
        x = x[idx]
        y = y[idx]
        if need_metadata:
            decision_id = [decision_id[i] for i in idx]
            candidate_page_id = [candidate_page_id[i] for i in idx]
            trace_family = [trace_family[i] for i in idx]

    meta = SplitMeta(decision_id, candidate_page_id, trace_family) if need_metadata else None
    return x, y, meta


def _ranking_metrics_arr(
    decision_id: List[str], candidate_page_id: List[str], y_loss: np.ndarray, preds: np.ndarray
) -> Dict[str, float]:
    grouped: Dict[str, List[int]] = {}
    for i, d in enumerate(decision_id):
        grouped.setdefault(d, []).append(i)
    top1 = 0
    regrets: List[float] = []
    for idxs in grouped.values():
        chosen_i = min(idxs, key=lambda i: (float(preds[i]), candidate_page_id[i]))
        best_i = min(idxs, key=lambda i: (float(y_loss[i]), candidate_page_id[i]))
        top1 += int(candidate_page_id[chosen_i] == candidate_page_id[best_i])
        regrets.append(float(y_loss[chosen_i]) - float(y_loss[best_i]))
    denom = max(len(grouped), 1)
    return {
        "decision_count": float(len(grouped)),
        "top1_eviction_match": float(top1 / denom),
        "mean_regret_vs_oracle": float(np.mean(regrets) if regrets else 0.0),
    }


def _family_metrics_arr(meta: SplitMeta, y_loss: np.ndarray, preds: np.ndarray) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    tf = np.asarray(meta.trace_family)
    fams = sorted(set(meta.trace_family))
    for fam in fams:
        idx = np.where(tf == fam)[0]
        if idx.size == 0:
            continue
        sub_p = preds[idx]
        sub_y = y_loss[idx]
        sub_did = [meta.decision_id[i] for i in idx]
        sub_cpid = [meta.candidate_page_id[i] for i in idx]
        rm = _ranking_metrics_arr(sub_did, sub_cpid, sub_y, sub_p)
        out[fam] = {**_metrics(sub_y, sub_p), **rm}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Train evict_value_v1 on Wulver shard manifest (ridge / RF / HistGB).")
    ap.add_argument("--manifest", type=Path, default=Path("data/derived/evict_value_v1_wulver_multi/manifest.json"))
    ap.add_argument("--horizons", default="4,8,16")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-train-rows", type=int, default=None, help="Subsample training rows per horizon (optional).")
    ap.add_argument("--max-val-rows", type=int, default=None)
    ap.add_argument("--max-test-rows", type=int, default=None)
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    ap.add_argument("--metrics-json", type=Path, default=Path("analysis/evict_value_wulver_v1_train_metrics.json"))
    ap.add_argument("--comparison-csv", type=Path, default=Path("analysis/evict_value_wulver_v1_model_comparison.csv"))
    ap.add_argument("--best-config-json", type=Path, default=Path("analysis/evict_value_wulver_v1_best_config.json"))
    ap.add_argument(
        "--memory-bounded", action="store_true",
        help="Stream rows directly into preallocated NumPy arrays instead of materializing a "
             "Python dict per row (see docs/evict_cross_family_oom_diagnosis.md). Same rows, same "
             "dtype (float64), same subsampling, same model/selection behavior as the default path "
             "-- implementation-only change, measured ~10x peak-RSS reduction on the loading stage.",
    )
    ap.add_argument(
        "--memory-guard-gb", type=float, default=None,
        help="With --memory-bounded, abort cleanly (raising, never mid-artifact-write) if peak RSS "
             "crosses this threshold. Recommended: comfortably below total machine RAM.",
    )
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {
        "manifest": str(args.manifest), "horizons": {},
        "execution_mode": {
            "memory_bounded": bool(args.memory_bounded),
            "memory_guard_gb": args.memory_guard_gb,
        },
    }
    comparison_rows: List[Dict[str, object]] = []

    for h in horizons:
        if args.memory_bounded:
            _check_memory_guard(args.memory_guard_gb, stage=f"before load h={h}")
            x_train, y_train, _ = _load_split_compact(
                args.manifest, h, "train", args.max_train_rows, args.seed,
                need_metadata=False, memory_guard_gb=args.memory_guard_gb,
            )
            x_val, y_val, meta_val = _load_split_compact(
                args.manifest, h, "val", args.max_val_rows, args.seed + 1,
                need_metadata=True, memory_guard_gb=args.memory_guard_gb,
            )
            n_train, n_val = x_train.shape[0], x_val.shape[0]
            if n_train == 0 or n_val == 0:
                print(f"[skip] horizon={h}: insufficient rows train={n_train} val={n_val}")
                continue
            x_test, y_test, meta_test = _load_split_compact(
                args.manifest, h, "test", args.max_test_rows, args.seed + 2,
                need_metadata=True, memory_guard_gb=args.memory_guard_gb,
            )
            if x_test.shape[0] == 0:
                print(f"[warn] horizon={h}: empty test split; metrics will duplicate val for test")
                x_test, y_test, meta_test = x_val, y_val, meta_val

            def ranking_val(preds: np.ndarray, _m=meta_val, _y=y_val) -> Dict[str, float]:
                return _ranking_metrics_arr(_m.decision_id, _m.candidate_page_id, _y, preds)

            def ranking_test(preds: np.ndarray, _m=meta_test, _y=y_test) -> Dict[str, float]:
                return _ranking_metrics_arr(_m.decision_id, _m.candidate_page_id, _y, preds)

            def family_val(preds: np.ndarray, _m=meta_val, _y=y_val) -> Dict[str, Dict[str, float]]:
                return _family_metrics_arr(_m, _y, preds)

        else:
            train = _load_rows_from_manifest(args.manifest, h, "train", args.max_train_rows, args.seed)
            val = _load_rows_from_manifest(args.manifest, h, "val", args.max_val_rows, args.seed + 1)
            test = _load_rows_from_manifest(args.manifest, h, "test", args.max_test_rows, args.seed + 2)
            if not train or not val:
                print(f"[skip] horizon={h}: insufficient rows train={len(train)} val={len(val)}")
                continue
            if not test:
                print(f"[warn] horizon={h}: empty test split; metrics will duplicate val for test")
                test = list(val)

            x_train, y_train = _xy(train)
            x_val, y_val = _xy(val)
            x_test, y_test = _xy(test)
            del train  # not referenced again -- only x_train/y_train are used below
            gc.collect()

            def ranking_val(preds: np.ndarray, _rows=val) -> Dict[str, float]:
                return _ranking_metrics(_rows, preds)

            def ranking_test(preds: np.ndarray, _rows=test) -> Dict[str, float]:
                return _ranking_metrics(_rows, preds)

            def family_val(preds: np.ndarray, _rows=val) -> Dict[str, Dict[str, float]]:
                return _family_metrics(_rows, preds)

        models: Dict[str, object] = {
            "ridge": Pipeline([("scale", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
            "random_forest": RandomForestRegressor(
                n_estimators=80, max_depth=12, min_samples_leaf=4, random_state=args.seed, n_jobs=1
            ),
            "hist_gb": HistGradientBoostingRegressor(
                max_depth=6, learning_rate=0.05, max_iter=150, random_state=args.seed
            ),
        }

        horizon_payload: Dict[str, object] = {}
        horizon_rows: List[Dict[str, object]] = []

        for name, est in models.items():
            if args.memory_bounded:
                _check_memory_guard(args.memory_guard_gb, stage=f"before fit h={h} model={name}")
            est.fit(x_train, y_train)
            p_val = est.predict(x_val)
            p_test = est.predict(x_test)
            m_val = _metrics(y_val, p_val)
            m_test = _metrics(y_test, p_test)
            r_val = ranking_val(p_val)
            r_test = ranking_test(p_test)
            fam_val = family_val(p_val)

            horizon_payload[name] = {
                "val": {**m_val, **r_val, "per_family_val": fam_val},
                "test": {**m_test, **r_test},
            }

            row_cmp = {
                "horizon": h,
                "model": name,
                "val_mae": m_val["mae"],
                "val_rmse": m_val["rmse"],
                "test_mae": m_test["mae"],
                "test_rmse": m_test["rmse"],
                "val_top1": r_val["top1_eviction_match"],
                "test_top1": r_test["top1_eviction_match"],
                "val_mean_regret": r_val["mean_regret_vs_oracle"],
                "test_mean_regret": r_test["mean_regret_vs_oracle"],
            }
            comparison_rows.append(row_cmp)
            horizon_rows.append(row_cmp)

            EvictValueV1Model(model_name=f"wulver_h{h}_{name}", estimator=est, feature_columns=list(FEATURES)).save(
                args.models_dir / f"evict_value_wulver_v1_h{h}_{name}.pkl"
            )

        best_name = None
        best_key = None
        for row in horizon_rows:
            key = (float(row["val_mean_regret"]), float(row["val_mae"]), float(row["val_rmse"]))
            if best_key is None or key < best_key:
                best_key = key
                best_name = str(row["model"])
        horizon_payload["winner_by_val_mean_regret"] = best_name
        all_results["horizons"][str(h)] = horizon_payload

    # Global best: min val regret; tie-break with lower val MAE, then RMSE
    best_row = None
    best_key = None
    for row in comparison_rows:
        key = (float(row["val_mean_regret"]), float(row["val_mae"]), float(row["val_rmse"]))
        if best_key is None or key < best_key:
            best_key = key
            best_row = row
    best_h = int(best_row["horizon"]) if best_row else None
    best_m = str(best_row["model"]) if best_row else None
    best_r = float(best_row["val_mean_regret"]) if best_row else float("inf")
    all_results["best_overall"] = {"horizon": best_h, "model": best_m, "val_mean_regret": best_r}

    if best_h is not None and best_m is not None:
        src = args.models_dir / f"evict_value_wulver_v1_h{best_h}_{best_m}.pkl"
        dst = args.models_dir / "evict_value_wulver_v1_best.pkl"
        if src.exists():
            dst.write_bytes(src.read_bytes())
        args.best_config_json.write_text(
            json.dumps(
                {
                    "horizon": best_h,
                    "model": best_m,
                    "model_path": str(dst),
                    "selection_rule": "minimize validation mean_regret_vs_oracle across (horizon, model)",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    args.metrics_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    with args.comparison_csv.open("w", newline="", encoding="utf-8") as fh:
        if comparison_rows:
            w = csv.DictWriter(fh, fieldnames=list(comparison_rows[0].keys()))
            w.writeheader()
            w.writerows(comparison_rows)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
