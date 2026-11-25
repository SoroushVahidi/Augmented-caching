"""Report predeclared contrasts from finished pilot records only."""

import argparse
import csv
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize import write_csv

CONTRASTS = (
    ("expanded_vs_legacy", "R02_T01_h16", "deepsets_h4_s7_argmin"),
    ("global_vs_static", "R03_T03_h16", "R02_T03_h16"),
    ("object_vs_static", "R04_T03_h16", "R02_T03_h16"),
    ("global_vs_object", "R03_T03_h16", "R04_T03_h16"),
    ("dual_vs_static", "R05_T03_h16", "R02_T03_h16"),
    ("dual_vs_flattened", "R05_T03_h16", "R08_T03_h16"),
    ("gru_vs_lstm", "R05_T03_h16", "R06_T03_h16"),
    ("attention_vs_lstm", "R07_T03_h16", "R06_T03_h16"),
    ("multitask_vs_imitation", "R05_T03_h16", "R05_T02_h16"),
    ("counterfactual_vs_imitation", "R05_T01_h16", "R05_T02_h16"),
    ("reset_vs_stateful", "R05_T03_h16_reset", "R05_T03_h16"),
    ("permuted_vs_delayed", "R05_T03_h16_permuted", "R05_T03_h16_delayed"),
    ("global32_vs_64", "R05_T03_h16_global32", "R05_T03_h16"),
    ("object4_vs_8", "R05_T03_h16_object4", "R05_T03_h16"),
    ("attention_history32_vs_128", "R07_T03_h16_history32", "R07_T03_h16"),
    ("recomputed_vs_stateful", "R05_T03_h16_recompute", "R05_T03_h16"),
)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("root",type=Path)
    args=parser.parse_args()
    root=args.root
    if not (root/"PILOT_COMPLETE").is_file():
        raise SystemExit("A completed pilot is required")
    with (root/"closed_loop_results.csv").open() as stream:
        evidence=list(csv.DictReader(stream))
    rows={}
    for r in evidence:
        if r["split"]=="development":
            rows.setdefault(r["method"],{})[(r["family"],r["capacity"],r["score_start"],r["score_end"])]=r
    comparisons=list(CONTRASTS)
    for m in sorted(rows):
        if m.endswith("_aggregate_targeted"):
            parent=m.removesuffix("_aggregate_targeted")
            comparisons.extend([("targeted_vs_parent",m,parent),
                                ("targeted_vs_uniform",m,parent+"_aggregate_uniform")])
    output=[]
    for label,method,reference in comparisons:
        if method not in rows or reference not in rows:
            continue
        a,b=rows[method],rows[reference]
        cells=sorted(a.keys() & b.keys())
        if not cells:
            continue
        delta=[int(a[k]["misses"])-int(b[k]["misses"]) for k in cells]
        total=sum(int(b[k]["misses"]) for k in cells)
        costs=[float(a[k]["us_per_request"])-float(b[k]["us_per_request"]) for k in cells]
        output.append(dict(contrast=label,method=method,reference=reference,cells=len(cells),
                           miss_delta=sum(delta),reduction=-sum(delta)/total if total else 0,
                           wins=sum(x<0 for x in delta),ties=delta.count(0),losses=sum(x>0 for x in delta),
                           median_extra_us_per_request=statistics.median(costs)))
    write_csv(root/"pilot_ablation_contrasts.csv",output)
    for group, labels in (
        ("recurrent_vs_nonrecurrent",{"global_vs_static","object_vs_static","dual_vs_static","dual_vs_flattened"}),
        ("gru_lstm_attention",{"gru_vs_lstm","attention_vs_lstm","global_vs_object"}),
        ("hidden_state_history_sensitivity",{"reset_vs_stateful","permuted_vs_delayed","global32_vs_64",
                                             "object4_vs_8","attention_history32_vs_128"}),
    ):
        chosen=[r for r in output if r["contrast"] in labels]
        if not chosen:
            continue
        fig,ax=plt.subplots(figsize=(8,max(2.3,.48*len(chosen)+1)))
        ax.barh([r["contrast"].replace("_"," ") for r in chosen],[100*r["reduction"] for r in chosen],
                color=["#22614f" if r["reduction"]>0 else "#95483e" for r in chosen])
        ax.axvline(0,color="#333333",lw=.8)
        ax.set_xlabel("Reduction in misses against contrast reference (%)")
        fig.tight_layout()
        fig.savefig(root/"plots"/f"{group}.png",dpi=150)
        plt.close(fig)


if __name__=="__main__":
    main()
