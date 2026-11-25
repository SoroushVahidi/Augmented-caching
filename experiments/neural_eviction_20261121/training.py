"""Fold-local preprocessing, validation search, fit and offline measurement."""
from dataclasses import asdict, dataclass
import json
from itertools import product
from pathlib import Path
import resource
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch

from data import FEATURES, HORIZONS, PROTOCOL, fold, load_cell, sha256
from lafc.evict_value_model_v1 import EvictValueV1Model
from metrics import STRATEGIES, add_stats, decision_stats, rates
from models import EvictionNetwork, NetworkConfig, load, objective, predict, save, seed_everything


@dataclass
class Cell:
    family: str
    capacity: int
    x: np.ndarray
    y: np.ndarray
    indices: np.ndarray


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def cells(data, families, capacities, limit, sample_seed, start=0, end=None):
    result = []
    for family, capacity in product(families, capacities):
        x, y, times = load_cell(data, family, capacity)
        stop = len(times) if end is None else int(np.searchsorted(times, end))
        indices = np.arange(int(np.searchsorted(times, start)), stop)
        if limit and len(indices) > limit:
            seed = np.random.SeedSequence([sample_seed, list(PROTOCOL).index(family), capacity])
            indices = np.sort(np.random.default_rng(seed).choice(indices, limit, replace=False))
        if len(indices) == 0:
            raise ValueError(f"No usable decisions in {family}/{capacity}")
        result.append(Cell(family, capacity, x, y, indices))
    return result


def batches(items, batch_size, horizon, rng=None):
    order = [rng.permutation(cell.indices) if rng is not None else cell.indices for cell in items]
    units = [(i, start) for i, ix in enumerate(order) for start in range(0, len(ix), batch_size)]
    if rng is not None:
        rng.shuffle(units)
    h = HORIZONS.index(horizon)
    for i, start in units:
        cell = items[i]
        ix = order[i][start:start + batch_size]
        yield cell.x[ix], cell.y[ix, :, h]


def fit_scaler(items, batch_size):
    scaler = StandardScaler()
    for x, _ in batches(items, batch_size, HORIZONS[0]):
        scaler.partial_fit(x.reshape(-1, len(FEATURES)))
    return scaler.mean_, scaler.scale_


def measure(items, predictor, batch_size, horizon, strategies=STRATEGIES, margin=0.05):
    report = []
    for cell in items:
        totals = {strategy: {} for strategy in strategies}
        for x, y in batches([cell], batch_size, horizon):
            costs, scores = predictor(x) if predictor else (None, None)
            for strategy in strategies:
                subtotal = decision_stats(y, costs, scores, strategy, margin)
                totals[strategy] = add_stats(totals[strategy], subtotal)
        for strategy, stats in totals.items():
            report.append(dict(family=cell.family, capacity=cell.capacity, strategy=strategy,
                               margin=margin if strategy == "margin_lru" else 0.0,
                               horizon=horizon, **stats, **rates(stats)))
    return report


def selection_key(rows):
    return (sum(r["regret"] for r in rows) / sum(r["decisions"] for r in rows),
            sum(r["absolute_error"] for r in rows) / sum(r["regression_candidates"] for r in rows),
            (sum(r["squared_error"] for r in rows) / sum(r["regression_candidates"] for r in rows)) ** 0.5)


def fit_one(config, horizon, mean, scale, train, val, settings, device, seed, checkpoint, provenance):
    seed_everything(seed)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = EvictionNetwork(config, horizon, mean, scale).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    use_amp = device == "cuda" and settings["mixed_precision"] and torch.cuda.is_bf16_supported()
    rng = np.random.default_rng(seed)
    best_key, best_epoch, curve = None, 0, []
    for epoch in range(1, settings["epochs"] + 1):
        model.train()
        for x, y in batches(train, settings["batch_size"], horizon, rng):
            features = torch.as_tensor(x, dtype=torch.float32, device=device)
            costs = torch.as_tensor(y, dtype=torch.float32, device=device)
            mask = torch.ones_like(costs, dtype=torch.bool)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_amp):
                loss = objective(model, features, costs, mask)
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        val_rows = measure(val, lambda x: predict(model, x, device),
                           settings["batch_size"], horizon, ("argmin",))
        key = selection_key(val_rows)
        curve.append(dict(epoch=epoch, key=key, cells=val_rows))
        if best_key is None or key < best_key:
            best_key, best_epoch = key, epoch
            save(checkpoint, model, dict(provenance, seed=seed, epoch=epoch))
        if epoch - best_epoch >= settings["patience"]:
            break
    return dict(key=best_key, selected_epoch=best_epoch, curve=curve, seed=seed, config=asdict(config),
                seconds=time.perf_counter() - started, batch_size=settings["batch_size"],
                learning_rate=settings["learning_rate"], weight_decay=settings["weight_decay"],
                mixed_precision=use_amp, peak_cuda_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                checkpoint_sha256=sha256(checkpoint), checkpoint_bytes=checkpoint.stat().st_size)


def reference_estimator(root, family, kind):
    key = "best" if kind == "selected_kbs" else f"h4_{kind}"
    path = root / family / f"evict_value_wulver_v1_{key}.pkl"
    if sha256(path) != PROTOCOL[family]["models"][key]:
        raise ValueError(f"Checkpoint hash mismatch: {path}")
    model = EvictValueV1Model.load(path)
    if model.feature_columns != FEATURES:
        raise ValueError(f"Feature order mismatch: {path}")
    return model.estimator, path


def sklearn_predict(estimator, x):
    values = estimator.predict(x.reshape(-1, len(FEATURES))).reshape(x.shape[:2])
    return values, values


def fit_reference(kind, train, horizon, path, n_jobs):
    candidates = {
        "hist_gb": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=150,
                                                 random_state=0),
        "random_forest": RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=4,
                                                random_state=0, n_jobs=n_jobs),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }
    count = sum(len(cell.indices) * cell.capacity for cell in train)
    x = np.empty((count, len(FEATURES)), dtype=np.float32)
    y = np.empty(count, dtype=np.float32)
    cursor = 0
    for features, costs in batches(train, 512, horizon):
        size = costs.size
        x[cursor:cursor + size] = features.reshape(-1, len(FEATURES))
        y[cursor:cursor + size] = costs.ravel()
        cursor += size
    estimator = candidates[kind]
    started = time.perf_counter()
    estimator.fit(x, y)
    seconds = time.perf_counter() - started
    if kind == "random_forest":
        estimator.n_jobs = 1
    joblib.dump(estimator, path)
    return estimator, dict(seconds=seconds, train_candidates=count, seed=0,
                           peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def fold_output(out, horizon, family):
    return out / f"h{horizon}" / family


def run_fold(out, family, horizon, settings, model_root, device):
    dest = fold_output(out, horizon, family)
    if (dest / "offline.json").exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    train_families, val_family = fold(family)
    capacities, seed = settings["capacities"], settings["sample_seed"]
    train = cells(out / "data", train_families, capacities, settings["train_groups_per_cell"], seed)
    val = cells(out / "data", [val_family], capacities, settings["val_groups_per_cell"], seed)
    mean, scale = fit_scaler(train, settings["batch_size"])
    provenance = dict(test=family, train=list(train_families), validation=val_family,
                      capacity=list(capacities), sample_seed=seed,
                      train_groups={f"{c.family}/{c.capacity}": len(c.indices) for c in train},
                      validation_groups={f"{c.family}/{c.capacity}": len(c.indices) for c in val},
                      history=settings["history"], end=settings["end"])
    fits, selected = {}, []
    for architecture in settings["architectures"]:
        if architecture == "set_regression" and horizon != 4:
            continue
        candidates = settings["network_grid"][:1] if architecture == "set_regression" else settings["network_grid"]
        primary_seed = settings["seeds"][0]
        for index, hyperparameters in enumerate(candidates):
            config = NetworkConfig(architecture, **hyperparameters)
            checkpoint = dest / f"{architecture}_grid{index}.pt"
            fits[checkpoint.name] = fit_one(config, horizon, mean, scale, train, val, settings, device,
                                           primary_seed, checkpoint, provenance)
        champion = min((dest / f"{architecture}_grid{i}.pt" for i in range(len(candidates))),
                       key=lambda path: fits[path.name]["key"])
        selected.append((architecture, primary_seed, champion))
        for other_seed in settings["seeds"][1:] if architecture != "set_regression" else []:
            checkpoint = dest / f"{architecture}_seed{other_seed}.pt"
            config = NetworkConfig(**fits[champion.name]["config"])
            fits[checkpoint.name] = fit_one(config, horizon, mean, scale, train, val, settings, device,
                                           other_seed, checkpoint, provenance)
            selected.append((architecture, other_seed, checkpoint))

    # All neural model selection is frozen before reading outer-test features or labels.
    test = cells(out / "data", [family], capacities, settings["test_groups_per_cell"], seed,
                 settings["history"], settings["end"])
    rows, models = [], []
    for name, run_seed, path in selected:
        model, _ = load(path, family, horizon)
        model.to(device)
        predictor = lambda x: predict(model, x, device)
        models.append(dict(method=name, seed=run_seed, file=path.name, format="torch",
                           size_bytes=path.stat().st_size, sha256=sha256(path),
                           config=fits[path.name]["config"], epoch=fits[path.name]["selected_epoch"]))
        for split, items in [("validation", val), ("test", test)]:
            rows.extend(dict(split=split, method=name, seed=run_seed, **r) for r in
                        measure(items, predictor, settings["batch_size"], horizon, ("argmin", "margin_lru")))

    for name in ("hist_gb", "random_forest", "ridge", "selected_kbs"):
        versions = []
        if horizon == 4 and settings["historical_models"]:
            estimator, path = reference_estimator(model_root, family, name)
            versions.append((name, estimator, path, "original"))
        if name != "selected_kbs" and horizon in settings["refit_horizons"]:
            path = dest / f"{name}_refit.joblib"
            estimator, fits[path.name] = fit_reference(name, train, horizon, path, settings["fit_tree_jobs"])
            versions.append((f"{name}_refit", estimator, path, "refit"))
        for method, estimator, path, fmt in versions:
            models.append(dict(method=method, seed=0, file=path.name, format=fmt,
                               size_bytes=path.stat().st_size, sha256=sha256(path)))
            predictor = lambda x: sklearn_predict(estimator, x)
            for split, items in [("validation", val), ("test", test)]:
                rows.extend(dict(split=split, method=method, seed=0, **r) for r in
                            measure(items, predictor, settings["batch_size"], horizon, ("argmin", "margin_lru")))
    for split, items in [("validation", val), ("test", test)]:
        rows.extend(dict(split=split, method=r["strategy"], seed=0, **r) for r in
                    measure(items, None, settings["batch_size"], horizon, ("lru", "random", "exact")))
    write_json(dest / "offline.json", dict(fold=provenance, fits=fits, models=models, rows=rows))
    print(f"Trained and evaluated offline: {family} H={horizon}, {len(models)} artifacts", flush=True)
