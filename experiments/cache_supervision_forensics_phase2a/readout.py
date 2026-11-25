"""Create compact scientific summaries and publication plots from recorded audit units."""
import csv
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from run import dump_csv,atomic

FAMILIES=("brightkite","citibike","cloudphysics","metacdn","metakv","twemcache","wiki2018")
LABELS=("BrightKite","Citi Bike","Alibaba Block","MetaCDN","MetaKV","Twemcache","Wikimedia")
COLORS=("#275d78","#b14d42","#417954","#80612e","#77609a","#339394","#666666")


def table(root,name):
    with (root/(name+".csv")).open() as f:
        return list(csv.DictReader(f))


def value(r,key):
    return float(r[key])


def sumcol(rows, key):
    return sum(float(r[key]) for r in rows)


def ratio(rows,num,den):
    n=sumcol(rows,den)
    return sumcol(rows,num)/n if n else None


def boolean(r,key):
    return r[key]=="True"


def aggregate(r):
    alternatives=sumcol(r,"alternatives")
    return dict(groups=len(r),alternatives=int(alternatives),coalesced=ratio(r,"coalesced","alternatives"),
             exact_at_end=ratio(r,"exact_at_trace_end","alternatives"),unresolved=ratio(r,"unresolved","alternatives"),
             profitable=ratio(r,"profitable_resolved","alternatives"),
             mean_length=sum(value(a,"alternatives")*value(a,"mean_effective_length") for a in r)/alternatives,
             maximum_length=max(value(a,"max_effective_length") for a in r),
             oracle_cpu_wall_seconds=sumcol(r,"paired_seconds"))


def opt_summary(r):
    sieve = [value(x,"sieve_regret")>0 for x in r if x["sieve_regret"]]
    return dict(groups=len(r),all_tied=float(np.mean([boolean(x,"full_tie") for x in r])),
                unique=float(np.mean([boolean(x,"unique_optimal") for x in r])),
                mean_action_optimal_fraction=float(np.mean([value(x,"optimal_count")/value(x,"candidates") for x in r])),
                mean_farthest_fraction=float(np.mean([value(x,"farthest_next_count")/value(x,"candidates") for x in r])),
                groups_with_additional_optimum=int(sum(value(x,"opt_outside_farthest_next")>0 for x in r)),
                lru_positive=float(np.mean([value(x,"lru_regret")>0 for x in r])),
                november_positive=float(np.mean([value(x,"november_regret")>0 for x in r])),
                phase1_positive=float(np.mean([value(x,"phase1_regret")>0 for x in r])),
                sieve_shared_groups=len(sieve),
                sieve_positive_shared=float(np.mean(sieve)) if sieve else None,
                median_oracle_seconds=float(np.median([value(x,"oracle_seconds") for x in r])))


def outcome(root):
    red=table(root,"DECISION_GROUP_REDUNDANCY")
    opt=table(root,"OPTIMAL_CONTINUATION_REGRET")
    roll=table(root,"PAIRED_ROLLOUT_RESULTS")
    h=table(root,"SUPERVISION_HORIZON_ANALYSIS")
    sh=table(root,"SHORTLIST_ORACLE_RESULTS")
    ae=table(root,"FEATURE_ALIASING_ANALYSIS")
    near=table(root,"NEAR_DUPLICATE_ANALYSIS")
    ce=table(root,"CENSORING_AUDIT")
    comp=json.loads((root/"COMPRESSION_VALIDATION.json").read_text())
    split=json.loads((root/"LAFC_SPLIT_CENSUS.json").read_text())
    nov=[r for r in red if r["surface"]=="November exact census" and r["role"]=="all" and r["horizon"]=="16"]
    pub=[r for r in red if r["surface"]=="LAFC v1.0 row-group stratified sample"]
    def uniqueness(rs):
        n=sumcol(rs,"groups")
        return dict(groups=int(n),candidate_rows=int(sumcol(rs,"candidate_rows")),
                    exact_input_duplicate_fraction=1-ratio(rs,"unique_input_groups","groups"),
                    exact_input_target_duplicate_fraction=1-ratio(rs,"unique_input_target_groups","groups"),
                    conflicting_groups_fraction=ratio(rs,"groups_in_conflicting_buckets","groups"))
    exp={}
    for form in ("november","expanded","expanded_with_request"):
        sub=[r for r in ae if r["surface"]=="KBS matched LRU decision sample" and r["representation"]==form
                         and r["signature"]=="exact" and r["horizon"]=="16"]
        nn=[r for r in near if r["representation"]==form and r["threshold"]=="0.15"]
        exp[form]=dict(groups=int(sumcol(sub,"signatures")+sumcol(sub,"rows_in_repeated_signatures")-
                           sumcol(sub,"repeated_signatures")),signature_hits=int(sumcol(sub,"signature_hits")),
                       repeated_rows=int(sumcol(sub,"rows_in_repeated_signatures")),
                       heldout_groups=int(sumcol(sub,"heldout_groups")),near_hits=int(sumcol(nn,"accepted")),
                       near_heldout=int(sumcol(nn,"heldout")))
        for k in ("conditional_label_variance","baseline_advantage_variance","in_sample_min_regret",
                  "incompatible_optimal_set_fraction","heldout_mean_regret","heldout_lru_regret"):
            usable=[r for r in sub if r[k]]
            weight="heldout_groups" if k.startswith("heldout") else "rows_in_repeated_signatures"
            denom=sumcol(usable,weight)
            exp[form][k]=sum(value(r,k)*value(r,weight) for r in usable)/denom if denom else None
    compare=[]
    for kind in ("lru_tail","sieve_derived","low_frequency","recency_frequency","ghost_informed","uniform","causal_mixture"):
        for k in (2,4,8,12,16,32):
            rr=[r for r in sh if r["sample"]=="temporal_stratified" and r["kind"]==kind and int(r["k"])==k]
            compare.append(dict(method=kind,k=k,groups=len(rr),
                  belady_next_use_coverage=float(np.mean([boolean(r,"cover_next_use") for r in rr])),
                  exact_min_coverage=float(np.mean([boolean(r,"cover_exact_optimal") for r in rr])),
                  mean_best_exact_regret=sumcol(rr,"best_regret")/len(rr),
                  alternative_count_within_shortlist=ratio(rr,"profitable_in_shortlist","profitable_available"),
                  states_with_profitable_override=sum(boolean(r,"beneficial_override_found") for r in rr),
                  states_with_any_profitable=sum(value(r,"profitable_available")>0 for r in rr)))
    h_summary=[]
    for hi in (4,8,16,32,64,128,256,512,1024,2048,4096):
        rr=[r for r in h if r["sample"]=="temporal_stratified" and int(r["horizon"])==hi]
        if not rr:
            continue
        h_summary.append(dict(horizon=hi,states=len(rr),capacities=sorted(set(int(x["capacity"]) for x in rr)),
                     all_tied=sumcol(rr,"all_tied")/len(rr),optimal_fraction=sumcol(rr,"random_optimal_accuracy")/len(rr),
                     lru_optimal=sumcol(rr,"lru_optimal_accuracy")/len(rr),mean_spread=sumcol(rr,"mean_spread")/len(rr),
                     value_disagreement=ratio(rr,"fixed_vs_exact_value_disagreement","compared_to_exact_advantage"),
                     sign_disagreement=ratio(rr,"fixed_vs_exact_sign_disagreement","compared_to_exact_advantage")))
    censor=[]
    for scale in (.25,.5,1,2,4,8,16,32):
        rr=[r for r in ce if r["sample"]=="temporal_stratified" and  float(r["horizon"])/float(r["capacity"])==scale]
        censor.append(dict(h_over_c=scale,states=len(rr),observed_event=ratio(rr,"events","candidates"),
                     trace_end_unknown=ratio(rr,"unknown_due_to_trace_end","candidates"),
                     event_rate_among_resolved_profitable=ratio(rr,"resolved_profitable_event_count","resolved_profitable_candidates")))
    loop={}
    for k in (1,4,8):
        rr=[r for r in sh if r["sample"]=="actual closed-loop hindsight" and int(r["k"])==k]
        loop[str(k)]=int(sumcol(rr,"misses"))
    loop["C"]=int(sumcol([r for r in sh if r["sample"]=="actual closed-loop hindsight" and r["k"]==r["capacity"]],"misses"))
    phase=json.loads((root/"PHASE1_TERMINAL.json").read_text())
    if phase["status"]=="COMPLETED_SUCCESS":
        lru=next(r["misses"] for r in phase["results"] if r["method"]=="lru")
        if loop["1"]!=lru:
            raise ValueError("Hindsight K1 baseline differs from validated LRU")
        belady=next(r["misses"] for r in phase["results"] if r["method"]=="belady")
        if loop["C"]!=belady:
            raise ValueError("Hindsight unrestricted oracle differs from validated MIN")
    details=dict(november=uniqueness(nov),november_by_family={f:uniqueness([r for r in nov if r["family"]==f]) for f in FAMILIES},
         public_sample=uniqueness(pub),public_physical_split_overlap=split["cross_split_physical_keys"],
         public_horizon_agnostic=split["cross_split_horizon_agnostic_timestamps"],alias=exp,horizon=h_summary,
         optimal={arm:opt_summary([r for r in opt if r["sample"]==arm])
                  for arm in ("temporal_stratified","learner_disagreement_enriched")},
         paired={arm:aggregate([r for r in roll if r["sample"]==arm])
                  for arm in ("temporal_stratified","learner_disagreement_enriched")},
         optimal_by_family={f:opt_summary([r for r in opt if r["sample"]=="temporal_stratified" and r["family"]==f])
                  for f in FAMILIES},
         shortlist=compare,closed_loop_hindsight=loop,censor=censor)
    records=comp["cell_pilots"]
    details["compression"]=dict(groups=sum(r["rows"] for r in records),unique=sum(r["unique"] for r in records),
      raw_bytes=sum(r["raw_bytes"] for r in records),compressed_bytes=sum(r["compressed_payload_bytes"] for r in records),
      construction_seconds=sum(r["construction_seconds"] for r in records),
      raw_scan_seconds=sum(r["uncompressed_memory_scan_seconds"] for r in records),
      compressed_scan_seconds=sum(r["compressed_weighted_scan_seconds"] for r in records),
      max_loss_delta=max(r["loss_delta"] for r in comp["tests"].values()),
      max_grad_delta=max(r["gradient_max_abs_delta"] for r in comp["tests"].values()))
    atomic(root/"SUMMARY.json",details)
    dump_csv(root/"SHORTLIST_SUMMARY.csv",compare)
    dump_csv(root/"HORIZON_SUMMARY.csv",h_summary)
    dump_csv(root/"CENSORING_SUMMARY.csv",censor)
    return details,nov,exp,comp


def plots(root,a,nov,alias,comp):
    d=root/"plots"
    d.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":140,"axes.grid":True,"grid.alpha":.18})
    def finish(fig,name):
        fig.tight_layout()
        fig.savefig(d/(name+".pdf"))
        fig.savefig(d/(name+".png"),dpi=180)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for f,l,col in zip(FAMILIES,LABELS,COLORS):
        merged=Counter()
        for r in nov:
            if r["family"]==f:
                merged.update({int(k):v for k,v in json.loads(r["multiplicities"]).items()})
        if merged:
            vals=np.array(sorted(merged))
            mass=np.array([n*merged[n] for n in vals],float); mass/=mass.sum()
            ax.step(vals,np.cumsum(mass[::-1])[::-1],where="post",label=l,color=col)
    ax.set(xscale="log",yscale="log",xlabel="Exact input-group multiplicity",ylabel="Fraction of decisions at least this frequent")
    ax.legend(ncol=2,fontsize=8); finish(fig,"duplicate_multiplicity")
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    for r in table(root,"FEATURE_RANK_AUDIT"):
        if r["scope"]=="pooled":
            s=np.array(json.loads(r["singular_values"]))
            ax.plot(np.arange(1,len(s)+1),np.maximum(s/s[0],1e-10),label=r["representation"])
    ax.set(yscale="log",xlabel="Singular-value index",ylabel="Fraction of largest singular value")
    ax.legend(); finish(fig,"feature_singular_spectra")
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    labels=["November","Expanded","Expanded + request"]
    rr=[alias[n] for n in ("november","expanded","expanded_with_request")]
    ax.bar(labels,[v["repeated_rows"]/v["groups"] for v in rr],color=COLORS[:3])
    ax.set(ylabel="Fraction in exact repeated signatures",ylim=(0,1))
    finish(fig,"conditional_label_uncertainty")
    h=table(root,"SUPERVISION_HORIZON_ANALYSIS")
    fig,ax=plt.subplots(figsize=(7,4))
    for cap,col in zip((32,64,128),COLORS):
        rows=[r for r in h if r["sample"]=="temporal_stratified" and int(r["capacity"])==cap]
        keys=sorted({int(r["horizon"]) for r in rows})
        ax.plot(keys,[np.mean([value(x,"random_optimal_accuracy") for x in rows if int(x["horizon"])==q]) for q in keys],
                "o-",label=f"C={cap}",color=col)
    ax.set(xscale="log",xlabel="Look-ahead horizon (requests)",ylabel="Mean optimal-set fraction")
    ax.legend(); finish(fig,"optimal_sets_by_horizon")
    fig,ax=plt.subplots(figsize=(7,4))
    hs=[4,8,16]
    for hi,col in zip(hs,COLORS):
        rows=[r for r in table(root,"LABEL_DISTRIBUTIONS") if r["sample"]=="November all LRU decisions"
              and int(r["horizon"])==hi and r["metric"]=="lru_regret"]
        keys=sorted({int(r["value"]) for r in rows})
        count=sumcol(rows,"count")
        ax.plot(keys,[sumcol([r for r in rows if int(r["value"])==k],"count")/count for k in keys],"o-",label=f"H={hi}",color=col)
    ax.set(yscale="log",xlabel="LRU finite-horizon regret (misses)",ylabel="Fraction of decisions")
    ax.legend(); finish(fig,"baseline_regret_distribution")
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9,3.8))
    r=[x for x in table(root,"PAIRED_ROLLOUT_RESULTS") if x["sample"]=="temporal_stratified"]
    for cap,col in zip((32,64,128),COLORS):
        z=[float(x["mean_effective_length"]) for x in r if int(x["capacity"])==cap]
        xx=np.sort(z); ax.step(xx,np.arange(1,len(xx)+1)/len(xx),color=col,label=f"C={cap}")
    ax.set(xlabel="Mean rollout length per state",ylabel="Empirical CDF"); ax.legend()
    keys=["coalesced","exact_at_end","unresolved"]
    left=0.
    for k,col in zip(keys,COLORS):
        x=a["paired"]["temporal_stratified"][k]
        bx.barh(["Stratified"],[x],left=left,color=col,label=k.replace("_"," ")); left+=x
    bx.set(xlabel="Fraction of alternative actions",xlim=(0,1)); bx.legend(fontsize=8)
    finish(fig,"rollout_length_and_coalescence")
    fig,ax=plt.subplots(figsize=(7,4))
    for m,col in zip(dict.fromkeys(r["method"] for r in a["shortlist"]),COLORS):
        rr=[r for r in a["shortlist"] if r["method"]==m]
        ax.plot([r["k"] for r in rr],[r["alternative_count_within_shortlist"] for r in rr],"o-",
                label=m.replace("_"," "),color=col)
    ax.set(xlabel="Shortlist size including LRU victim",ylabel="Fraction of resolved profitable alternatives retained")
    ax.legend(ncol=2,fontsize=8); finish(fig,"shortlist_coverage")
    fig,ax=plt.subplots(figsize=(6.8,3.8))
    records=a["censor"]
    ax.plot([r["h_over_c"] for r in records],[r["observed_event"] for r in records],"o-",label="Observed reuse")
    ax.plot([r["h_over_c"] for r in records],[r["trace_end_unknown"] for r in records],"o-",label="Unknown from trace end")
    ax.set(xscale="log",xlabel="Horizon / capacity",ylabel="Fraction of candidate actions")
    ax.legend(); finish(fig,"reuse_and_censoring")
    fig,ax=plt.subplots(figsize=(7.2,3.8))
    dsum=[]
    for f in FAMILIES:
        r=[r for r in comp["cell_pilots"] if r["family"]==f]
        dsum.append(1-sum(x["compressed_payload_bytes"] for x in r)/sum(x["raw_bytes"] for x in r))
    ax.bar(LABELS,dsum,color=COLORS)
    ax.tick_params(axis="x",labelrotation=22)
    ax.set(ylabel="Fraction of payload bytes saved")
    finish(fig,"compression_by_workload")


if __name__=="__main__":
    root=Path(sys.argv[1])
    if not (root/"COMPLETED_SUCCESS").is_file():
        raise SystemExit("Wait for terminal audit units and reduction")
    v,nov,alias,comp=outcome(root)
    plots(root,v,nov,alias,comp)
