"""Check regenerated feature moments against the actual H=4 training estimators."""
import argparse
import json
from pathlib import Path

import numpy as np

from data import CAPACITIES, FEATURES, END, HISTORY, PROTOCOL, fold, historical_misses, trace
from replay import CLASSICAL, FAIRNESS, run_policy
from run import write_csv
from training import reference_estimator, write_json


def compare_moments(audits, model_root):
    rows = []
    for test in PROTOCOL:
        train, validation = fold(test)
        estimator, _ = reference_estimator(model_root, test, "ridge")
        scaler = estimator.named_steps["scale"]
        training = [r for r in audits if r["family"] in train]
        if {(r["family"], r["capacity"]) for r in training} != {(f, c) for f in train for c in CAPACITIES}:
            raise ValueError("Moment check requires all original training cells")
        counts = [r["candidate_rows"] for r in training]
        means = np.array([[v["mean"] for v in row["features"]] for row in training])
        stds = np.array([[v["std"] for v in row["features"]] for row in training])
        mean = np.average(means, weights=counts, axis=0)
        variance = np.average(stds**2 + (means - mean)**2, weights=counts, axis=0)
        np.testing.assert_allclose(mean, scaler.mean_, rtol=0, atol=1e-6)
        np.testing.assert_allclose(variance, scaler.var_, rtol=0, atol=1e-6)
        assert sum(counts) == int(scaler.n_samples_seen_)
        rows.append(dict(held_out=test, training=list(train), validation=validation, rows_seen=sum(counts),
                         ridge_sha256=PROTOCOL[test]["models"]["h4_ridge"],
                         max_absolute_mean_difference=float(np.max(np.abs(mean - scaler.mean_))),
                         max_absolute_variance_difference=float(np.max(np.abs(variance - scaler.var_))),
                         features={k: dict(mean=float(m), variance=float(v))
                                   for k, m, v in zip(FEATURES, scaler.mean_, scaler.var_)}))
    return rows


def validate_classical(data_root):
    rows = []
    for family in PROTOCOL:
        requests, pages = trace(data_root, family)
        for capacity in CAPACITIES:
            for method, factory in CLASSICAL.items():
                m = run_policy(factory(), requests, pages, capacity, HISTORY, END)
                expected = historical_misses(FAIRNESS / f"policy_comparison_{method}.csv", family, capacity)
                if m["misses"] != expected:
                    raise ValueError(f"{method} disagreement for {family}/{capacity}")
                rows.append(dict(family=family, capacity=capacity, method=method, expected_misses=expected,
                                 observed_misses=m["misses"], hits=m["hits"], requests=m["requests"]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", required=True, type=Path)
    p.add_argument("--model-root", required=True, type=Path)
    p.add_argument("--trace-root", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    audits = json.loads(args.audit.read_text())
    write_json(args.out / "original_ridge_feature_moments.json", compare_moments(audits, args.model_root))
    coverage = []
    for a in audits:
        for label in a["labels"]:
            coverage.append(dict(family=a["family"], capacity=a["capacity"], decisions=a["groups"],
                                 candidate_rows=a["candidate_rows"], centered_linear_rank=a["centered_linear_rank"],
                                 horizon=label["horizon"], all_tied=label["all_tied"], unique_best=label["unique"],
                                 mean_optimal_fraction=label["optimal_fraction"]))
    write_csv(args.out / "data_coverage.csv", coverage)
    write_csv(args.out / "native_baseline_parity.csv", validate_classical(args.trace_root))
    print("Verified seven Ridge scalers and all 63 native baseline cells.")


if __name__ == "__main__":
    main()
