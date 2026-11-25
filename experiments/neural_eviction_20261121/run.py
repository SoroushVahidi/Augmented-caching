#!/usr/bin/env python3
"""Prepare verified cells, fit frozen outer folds, and run matched replays."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import product
import json
import multiprocessing
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
import torch

from data import CAPACITIES, END, FEATURES, HORIZONS, PROTOCOL, build, load_cell
from replay import run_replay
from training import run_fold, write_json


def audit_cell(out, family, capacity):
    x, y, t = load_cell(out / "data", family, capacity)
    feature_flat = x.reshape(-1, len(FEATURES))
    minima = np.full(len(FEATURES), np.inf)
    maxima = np.full(len(FEATURES), -np.inf)
    nonfinite = np.zeros(len(FEATURES), dtype=np.int64)
    sums = np.zeros(len(FEATURES))
    for start in range(0, len(feature_flat), 32_768):
        block = feature_flat[start:start + 32_768]
        minima = np.minimum(minima, block.min(0))
        maxima = np.maximum(maxima, block.max(0))
        nonfinite += (~np.isfinite(block)).sum(0)
        sums += block.sum(0)
    if nonfinite.any():
        raise ValueError(f"Nonfinite feature for {family}/{capacity}: {nonfinite}")
    mean = sums / len(feature_flat)
    covariance_sum = np.zeros((len(FEATURES), len(FEATURES)))
    varying = minima != maxima
    for start in range(0, len(feature_flat), 32_768):
        centered = feature_flat[start:start + 32_768, varying] - mean[varying]
        covariance_sum[np.ix_(varying, varying)] += centered.T @ centered
    moments = covariance_sum / len(feature_flat)
    std = np.diag(moments) ** 0.5
    corr = moments[np.ix_(varying, varying)] / np.outer(std[varying], std[varying])
    eigenvalues = np.linalg.eigvalsh(corr)
    rank = int((eigenvalues > 1e-10).sum())
    def audit_labels(ys, h):
        size = (ys == ys.min(1, keepdims=True)).sum(1)
        return dict(horizon=h, decisions=len(size), all_tied=int((size == capacity).sum()),
                    unique=int((size == 1).sum()), optimal_fraction=float(size.mean() / capacity))

    report = dict(family=family, capacity=capacity, groups=len(t), candidate_rows=len(feature_flat),
                  first_t=int(t[0]), last_t=int(t[-1]), centered_linear_rank=rank,
                  variation_eigenvalues=eigenvalues.tolist(),
                  features=[dict(feature=k, min=float(lo), max=float(hi), mean=float(mu),
                                 std=float(sd), nonfinite=int(nf))
                            for k, lo, hi, mu, sd, nf in zip(FEATURES, minima, maxima, mean, std, nonfinite)],
                  labels=[audit_labels(y[:, :, i], h) for i, h in enumerate(HORIZONS)])
    write_json(out / "data" / f"{family}-{capacity}" / "audit.json", report)
    return report


def prepare_cell(data_root, out, family, capacity, settings):
    cell = out / "data" / f"{family}-{capacity}"
    if not (cell / "manifest.json").exists():
        build(data_root, out / "data", family, capacity, settings["build_groups_per_cell"],
              settings["sample_seed"], end=settings["end"])
    report = audit_cell(out, family, capacity)
    print(f"Prepared {family}/{capacity}, {report['groups']} whole decisions", flush=True)


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, list(dict.fromkeys(k for row in rows for k in row)), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


def reduce_run(out):
    for name, pattern in (("offline", "h*/*/offline.json"), ("replay", "h*/*/replay-*.json")):
        rows = []
        for path in sorted(out.glob(pattern)):
            payload = json.loads(path.read_text())
            if name == "offline":
                rows.extend(dict(fold=payload["fold"]["test"], **r) for r in payload["rows"])
            else:
                rows.extend(payload)
        write_csv(out / f"{name}.csv", rows)
    audits = [json.loads(p.read_text()) for p in sorted(out.glob("data/*/audit.json"))]
    write_json(out / "data_audit.json", audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "fold", "replay", "all", "reduce"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--original-model-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--families", nargs="+", choices=tuple(PROTOCOL), default=list(PROTOCOL))
    parser.add_argument("--horizons", nargs="+", type=int, choices=HORIZONS)
    parser.add_argument("--serial-replay", action="store_true")
    args = parser.parse_args()
    settings = json.loads(args.config.read_text())
    if any(c not in CAPACITIES for c in settings["capacities"]) or not 0 <= settings["history"] < settings["end"] <= END:
        parser.error("Invalid matched capacities or replay window")
    if settings["historical_models"] and args.original_model_root is None and args.stage != "prepare":
        parser.error("--original-model-root is required with original KBS checkpoints")
    if settings["historical_models"] and sklearn.__version__ != "1.9.0":
        parser.error("Original KBS checkpoints require scikit-learn 1.9.0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out / "config.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != settings:
        parser.error("Output directory belongs to a different config")
    if not snapshot.exists():
        write_json(snapshot, settings)
    metadata = dict(python=sys.version, platform=platform.platform(), numpy=np.__version__,
                    sklearn=sklearn.__version__, torch=str(torch.__version__),
                    cuda=torch.version.cuda, requested_device=args.device,
                    gpu=torch.cuda.get_device_name() if args.device == "cuda" else None)
    if args.stage != "reduce":
        write_json(args.out / f"environment-{args.stage}-{'-'.join(args.families)}.json", metadata)
    horizons = args.horizons or settings["horizons"]
    cpu_context = multiprocessing.get_context("spawn")

    if args.stage in {"prepare", "all"}:
        # All seven input caches are required even when a single outer fold is requested.
        units = product(PROTOCOL, settings["capacities"])
        if args.workers == 1:
            for family, cap in units:
                prepare_cell(args.trace_root, args.out, family, cap, settings)
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=cpu_context) as pool:
                futures = [pool.submit(prepare_cell, args.trace_root, args.out, family, cap, settings)
                           for family, cap in units]
                for future in futures:
                    future.result()
        reduce_run(args.out)
    torch.set_num_threads(1)
    for h in horizons:
        if args.stage in {"fold", "all"}:
            for family in args.families:
                run_fold(args.out, family, h, settings, args.original_model_root, args.device)
        if args.stage in {"replay", "all"}:
            units = product(args.families, settings["capacities"])
            def work(family, cap):
                return args.out, family, h, cap, settings, args.trace_root, args.original_model_root

            if args.workers == 1 or args.serial_replay:
                for family, cap in units:
                    run_replay(*work(family, cap))
            else:
                with ProcessPoolExecutor(max_workers=args.workers, mp_context=cpu_context) as pool:
                    futures = [pool.submit(run_replay, *work(family, cap)) for family, cap in units]
                    for future in futures:
                        future.result()
        if args.stage in {"all", "reduce"}:
            reduce_run(args.out)


if __name__ == "__main__":
    main()
