"""Reduce measured records into audit tables and scientific plots."""

from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics

from fit import atomic_json


def write_csv(path, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    if not columns:
        columns = ["status"]
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def comparisons(rows):
    cells = defaultdict(dict)
    for row in rows:
        key = (row["split"], row["method"])
        cell = (row["family"],row["capacity"],row.get("score_start",10000),row.get("score_end",50000))
        cells[key][cell] = row
    output = []
    for (split, method), values in cells.items():
        for (base_split, base), other in cells.items():
            if split != base_split or base == method or base.startswith("R"):
                continue
            keys = sorted(values.keys() & other.keys())
            if not keys:
                continue
            deltas = [values[k]["misses"]-other[k]["misses"] for k in keys]
            by_family = defaultdict(lambda:[0,0])
            for k,d in zip(keys,deltas):
                by_family[k[0]][0] += d
                by_family[k[0]][1] += other[k]["misses"]
            denom = sum(other[k]["misses"] for k in keys)
            runtime = [values[k]["seconds"]-other[k]["seconds"] for k in keys
                       if values[k].get("seconds") is not None and other[k].get("seconds") is not None]
            drops = [(sum(deltas)-a)/(denom-b) for a,b in by_family.values() if denom>b]
            output.append(dict(split=split,method=method,baseline=base,cells=len(keys),requests=sum(values[k]["requests"] for k in keys),
                               misses=sum(values[k]["misses"] for k in keys),base_misses=denom,
                               relative_reduction=-sum(deltas)/denom if denom else None,
                               wins=sum(d<0 for d in deltas),ties=deltas.count(0),losses=sum(d>0 for d in deltas),
                               worst_family_delta=max((a/b for a,b in by_family.values() if b), default=None),
                               drop_family_min_delta=min(drops,default=None),drop_family_max_delta=max(drops,default=None),
                               break_even_miss_penalty_seconds=max(0,sum(runtime))/-sum(deltas)
                               if sum(deltas)<0 and len(runtime)==len(keys) else None))
    return output


def assemble(root):
    records = [json.loads(path.read_text()) for path in sorted((root/"fits").glob("*/*/result.json"))]
    replay = [json.loads(path.read_text()) for path in sorted((root/"replays").glob("*.json"))]
    if (root/"archived_baselines.json").exists():
        archive = json.loads((root/"archived_baselines.json").read_text())
        measured = {(r["method"],r["family"],r["capacity"],r["split"]) for r in replay}
        replay.extend(row for row in archive if (row["method"],row["family"],row["capacity"],row["split"]) not in measured)
    offline = [dict(run=r["run"],fold=r["fold"],split=split,cell=cell,
                    objective=r["objective"], horizon=r["horizon"], **metrics)
               for r in records for split in ("validation","test") for cell,metrics in r.get(split,{}).items()]
    curves = []
    for r in records:
        for e in r["fit"]["curve"]:
            base = {k:v for k,v in e.items() if k!="validation"}
            for cell, metrics in e["validation"].items():
                curves.append(dict(run=r["run"],fold=r["fold"],cell=cell,**base,
                                   **{"val_"+k:v for k,v in metrics.items()}))
    checkpoint_manifest = {f"{r['fold']}/{r['run']}":dict(
        path=r["checkpoint"],selected_sha256=r["fit"]["selected_sha256"],latest_sha256=r["fit"]["latest_sha256"],
        selected_epoch=r["fit"]["selected_epoch"],completed_epochs=r["fit"]["epochs"]) for r in records}
    paired = comparisons(replay)
    ablation = [r for r in paired if r["baseline"] == "lru" and r["method"].startswith("R")]
    runtime = [dict(run=r["run"],fold=r["fold"],device="training",**{k:r["fit"][k] for k in
                     ("training_seconds","parameters","peak_gpu_bytes")}) for r in records]
    runtime.extend(dict(device="cpu_replay",**r) for r in replay if r["origin"]=="measured_cpu")
    for name,rows in (("offline_results",offline),("closed_loop_results",replay),("baseline_comparisons",paired),
                      ("ablation_results",ablation),("runtime_results",runtime),("training_curves",curves)):
        write_csv(root/f"{name}.csv", rows)
    atomic_json(root/"checkpoint_manifest.json",checkpoint_manifest)
    if ablation:
        plots(root,ablation,replay,curves,offline)


def plots(root,paired,replay,curves,offline):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dest=root/"plots"
    dest.mkdir(exist_ok=True)
    items=sorted(paired,key=lambda r:(r["split"],r["relative_reduction"] if r["relative_reduction"] is not None else 0))
    labels=[r["method"] for r in items]
    for name,values,xlabel in (
        ("closed_loop_miss_ratio",[r["misses"]/r["requests"]*100 for r in items],"Miss ratio (%)"),
        ("improvement_vs_lru",[100*(r["relative_reduction"] or 0) for r in items],"Reduction in misses against LRU (%)"),
    ):
        fig,ax=plt.subplots(figsize=(8,max(3,.26*len(labels))))
        ax.barh(labels,values,color=["#2b708b" if "GRU" in s or s[1:3] in {"03","04","05","06","07"}
                                     else "#8b6c41" for s in labels])
        ax.set_xlabel(xlabel)
        fig.tight_layout()
        fig.savefig(dest/f"{name}.png",dpi=135)
        plt.close(fig)
    selected=[r for r in replay if r["method"] in {"lru","fifo_reinsertion"} or r["method"].startswith(("R02","R05","R06","R08"))]
    by=defaultdict(list)
    for r in selected:
        by[(r["family"],r["method"])].append(r["misses"]/r["requests"])
    if by:
        families=sorted({k[0] for k in by})
        fig,ax=plt.subplots(figsize=(9,4))
        for method in sorted({k[1] for k in by}):
            ax.plot(families,[statistics.mean(by[(f,method)]) if by[(f,method)] else float("nan") for f in families], "o-",label=method)
        ax.set_ylabel("Miss ratio")
        ax.legend(fontsize=7,ncols=2)
        fig.tight_layout()
        fig.savefig(dest/"by_family.png",dpi=135)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    names=sorted({r["run"] for r in curves if r["run"].startswith(("R02","R05","R06","R08"))})
    for name in names:
        group=defaultdict(list)
        for r in curves:
            if r["run"]==name:
                group[r["epoch"]].append(r["train_loss"])
        ax.plot(sorted(group),[statistics.mean(group[i]) for i in sorted(group)],label=name)
    ax.set(xlabel="Epoch",ylabel="Training objective")
    if names:
        ax.legend(fontsize=7,ncols=2)
    fig.tight_layout()
    fig.savefig(dest/"training_curves.png",dpi=135)
    plt.close(fig)
    acc=defaultdict(list)
    cost=defaultdict(list)
    for r in offline:
        if r["discriminative_accuracy"] is not None:
            acc[r["run"]].append(r["discriminative_accuracy"])
    for r in replay:
        if r.get("us_per_request") is not None:
            cost[r["method"]].append(r["us_per_request"])
    fig,ax=plt.subplots(figsize=(9,5))
    for name in sorted(acc.keys() & cost.keys()):
        x,y=statistics.mean(cost[name]),statistics.mean(acc[name])
        ax.scatter(x,y)
        ax.annotate(name,(x,y),fontsize=6)
    ax.set(xlabel="CPU replay µs/request",ylabel="Discriminative selection accuracy")
    fig.tight_layout()
    fig.savefig(dest/"accuracy_vs_compute.png",dpi=135)
    plt.close(fig)
