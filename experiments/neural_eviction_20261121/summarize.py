"""Reduce completed primary/diagnostic cells and plot matched evidence."""
import argparse
from collections import defaultdict
import csv
from math import comb
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import add_stats, rates
from run import write_csv


def sign_test(values):
    positive, negative = sum(v > 0 for v in values), sum(v < 0 for v in values)
    n = positive + negative
    return min(1.0, 2 * sum(comb(n, k) for k in range(min(positive, negative) + 1)) / 2**n)


def summarize_offline(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["split"], row["horizon"], row["method"], row["strategy"], row["seed"])].append(row)
    summary = []
    for (split, h, method, strategy, seed), cells in groups.items():
        totals = {}
        for row in cells:
            counts = {k: float(row[k]) for k in (
                "decisions", "candidates", "absolute_error", "squared_error", "regression_candidates",
                "all_tied", "discriminative", "unique", "optimal_fraction", "agreements", "regret",
                "positive_regret", "agreements_discriminative", "regret_discriminative",
                "positive_discriminative", "agreements_unique")}
            totals = add_stats(totals, counts)
        summary.append(dict(split=split, horizon=h, method=method, strategy=strategy, seed=seed,
                            cells=len(cells), families=len({c["family"] for c in cells}),
                            **totals, **rates(totals)))
    return summary


def replay_aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["horizon"], row["method"], row["strategy"], row["seed"], row["origin"])].append(row)
    output = []
    for (h, method, strategy, seed, origin), cells in groups.items():
        requests = sum(int(r["requests"]) for r in cells)
        misses = sum(int(r["misses"]) for r in cells)
        timed = [r for r in cells if r.get("replay_seconds")]
        duration = sum(float(r["replay_seconds"]) for r in timed)
        def per_eviction(key):
            avail = [r for r in timed if r.get(key)]
            return (sum(float(r[key]) * int(r["evictions"]) for r in avail)
                    / sum(int(r["evictions"]) for r in avail)) if avail else None

        output.append(dict(horizon=h, method=method, strategy=strategy, seed=seed, origin=origin,
                           cells=len(cells), families=len({r["family"] for r in cells}),
                           requests=requests, misses=misses, miss_ratio=misses / requests,
                           mean_us_per_request=duration * 1e6 / requests if len(timed) == len(cells) else None,
                           mean_us_per_scoring_call=per_eviction("mean_us_per_scoring_call"),
                           mean_us_per_victim_selection=per_eviction("mean_us_per_victim_selection"),
                           max_process_peak_rss_kib=max((int(r["process_peak_rss_kib"]) for r in timed), default=None),
                           max_model_bytes=max((int(r["checkpoint_bytes"]) for r in cells if r.get("checkpoint_bytes")),
                                               default=None)))
    return output


def pairing(rows):
    key = lambda row: (row["family"], row["capacity"], row["trace_sha256"], row["score_start"], row["score_end"])
    groups = defaultdict(dict)
    for row in rows:
        label = (row["horizon"], row["method"], row["strategy"], row["seed"])
        if key(row) in groups[label]:
            raise ValueError("Duplicate replay cell")
        groups[label][key(row)] = row
    report = []
    for (horizon, model, strategy, seed), cells in groups.items():
        for ref in ["lru", "hist_gb", "hist_gb_refit", "selected_kbs", "random_forest", "random_forest_refit"]:
            controls = groups.get((horizon, ref, "argmin", "0"), {})
            pairs = []
            for k in sorted(cells.keys() & controls.keys()):
                a, b = cells[k], controls[k]
                if int(a["requests"]) != int(b["requests"]):
                    raise ValueError("Replay exposure mismatch")
                pairs.append((k[0], int(a["misses"]), int(b["misses"]), int(a["requests"])))
            if not pairs or model == ref and strategy == "argmin" and seed == "0":
                continue
            totals = defaultdict(lambda: [0, 0])
            for family, a, b, _ in pairs:
                totals[family][0] += a
                totals[family][1] += b
            total_model, total_ref = sum(p[1] for p in pairs), sum(p[2] for p in pairs)
            sensitivity = [(total_model - a) / (total_ref - b) - 1
                           for a, b in totals.values() if total_ref > b]
            diffs = [a - b for _, a, b, _ in pairs]
            sign = sign_test([a - b for a, b in totals.values()]) if len(totals) == 7 else None
            report.append(dict(horizon=horizon, method=model, strategy=strategy, seed=seed, baseline=ref,
                               cells=len(pairs), families=len(totals), requests=sum(p[3] for p in pairs),
                               model_misses=total_model, baseline_misses=total_ref,
                               relative_delta=total_model / total_ref - 1 if total_ref else None,
                               wins=sum(d < 0 for d in diffs), ties=sum(d == 0 for d in diffs),
                               losses=sum(d > 0 for d in diffs), family_sign_p=sign,
                               drop_family_min_relative_delta=min(sensitivity, default=None),
                               drop_family_max_relative_delta=max(sensitivity, default=None)))
    return report


def figure(out, offline, paired):
    if not offline:
        return
    primary_methods = {"mlp", "deepsets", "hist_gb", "random", "selected_kbs"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), layout="constrained")
    records = [r for r in offline if r["split"] == "test" and r["horizon"] == "4"
               and r["strategy"] in {"argmin", "random"} and r["method"] in primary_methods
               and r["seed"] in {"0", "7"}]
    if records:
        names = [r["method"].replace("_", " ") for r in records]
        x = np.arange(len(names))
        axes[0].bar(x - 0.18, [r["optimal_set_rate"] for r in records], 0.36, label="All decisions")
        axes[0].bar(x + 0.18, [r["discriminative_accuracy"] or 0 for r in records], 0.36,
                    label="Discriminative")
        axes[0].set_xticks(x, names, rotation=25, ha="right")
        axes[0].set(ylabel="Optimal-set selection rate", ylim=(0, 1.03),
                    title=f"H=4 offline · {records[0]['cells']} cells")
        axes[0].legend(fontsize=8)
    records = [r for r in paired if r["horizon"] == "4" and r["strategy"] == "argmin"
               and r["baseline"] == "lru" and r["method"] in primary_methods and r["seed"] in {"0", "7"}
               and r["relative_delta"] is not None]
    if records:
        axes[1].bar([r["method"].replace("_", " ") for r in records],
                    [100 * r["relative_delta"] for r in records], color="#335c81")
        axes[1].tick_params(axis="x", labelrotation=25)
        axes[1].axhline(0, lw=0.8, color="black")
        comparison = f"{records[0]['cells']} cells" if len({r["cells"] for r in records}) == 1 else "varying coverage"
        axes[1].set(ylabel="Misses relative to LRU (%)", title=f"H=4 replay · {comparison}")
    else:
        axes[1].set_axis_off()
    fig.savefig(out / "comparison.png", dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    def read(name):
        path = args.out / f"{name}.csv"
        if not path.exists():
            return []
        with path.open(newline="") as stream:
            return list(csv.DictReader(stream))

    offline = summarize_offline(read("offline"))
    replay = read("replay")
    paired = pairing(replay)
    write_csv(args.out / "offline_summary.csv", offline)
    write_csv(args.out / "replay_summary.csv", replay_aggregate(replay))
    write_csv(args.out / "paired.csv", paired)
    figure(args.out, offline, paired)


if __name__ == "__main__":
    main()
