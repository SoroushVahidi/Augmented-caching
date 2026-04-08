from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lafc.evict_value_features_v1 import EVICT_VALUE_V1_FEATURE_COLUMNS
from lafc.evict_value_model_v1 import EvictValueV1Model

FEATURES = list(EVICT_VALUE_V1_FEATURE_COLUMNS)


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
    args = ap.parse_args()

    horizons = [int(x.strip()) for x in args.horizons.split(",") if x.strip()]
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, object] = {"manifest": str(args.manifest), "horizons": {}}
    comparison_rows: List[Dict[str, object]] = []

    for h in horizons:
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
            est.fit(x_train, y_train)
            p_val = est.predict(x_val)
            p_test = est.predict(x_test)
            m_val = _metrics(y_val, p_val)
            m_test = _metrics(y_test, p_test)
            r_val = _ranking_metrics(val, p_val)
            r_test = _ranking_metrics(test, p_test)
            fam_val = _family_metrics(val, p_val)

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
