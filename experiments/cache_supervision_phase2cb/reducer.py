"""Validate complete stage receipts, report every configuration, and package aggregates."""
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
from shutil import copyfile
import sys
from statistics import mean,stdev
from zipfile import ZipFile,ZIP_DEFLATED

from assets import CAPS, PROTOCOL, dump, rows, sha
from run import ARCHS, RANDOM_SEEDS, STAGE_ORDER

EXPECTED={"A":16,"B":8,"C":9,"D":44}


def table(root,name,items):
    if not items:
        items=[{"status":"NO_ROWS"}]
    fields=list(dict.fromkeys(k for r in items for k in r))
    with (root/name).open("w",newline="") as f:
        w=csv.DictWriter(f,fields)
        w.writeheader()
        for r in items:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})


def portfolio(row):
    name=row["policy"]
    if name.startswith("B_"):
        return "B_"+name.split("_",2)[2]
    if name.startswith("C_") and name.endswith("_k16"):
        return "C_validation_selected_k16"
    if name.startswith("C_"):
        return "C_"+name.split("_",2)[2]
    if name.endswith("_two_part_risk"):
        return "D_two_part_risk"
    return name


def inventory(out):
    run=json.loads((out/"evidence/WULVER_JOB_RECEIPTS.json").read_text())
    fit, replay, validation, parity, chron, resource=[],[],[],[],[],[]
    selection={}
    for stage in STAGE_ORDER:
        for family,spec in PROTOCOL.items():
            r=json.loads((out/"receipts"/f"{stage}-{family}.json").read_text())
            s=json.loads((out/"selections"/f"{stage}-{family}.json").read_text())
            assert r["status"]=="COMPLETED" and (r["stage"],r["fold"])==(stage,family)
            assert r["slurm_array_job_id"]==run["stages"][stage]["id"]
            assert (s["train"],s["validation_family"],s["outer_family"])==(spec["train"],spec["validation"],family)
            assert len(s["configs"])==EXPECTED[stage] and len({x["policy"] for x in s["configs"]})==EXPECTED[stage]
            selection[stage,family]=s
            current_val=[x for x in r["replays"] if x["role"]=="validation"]
            assert len(current_val)==EXPECTED[stage]*len(CAPS)
            assert hashlib.sha256(json.dumps(current_val,sort_keys=True).encode()).hexdigest()==s["validation_replay_sha_pre_outer"]
            combinations={(x["policy"],x["role"],x["capacity"]) for x in r["replays"] if not x["policy"].endswith("_reference")}
            assert combinations=={(c["policy"],role,k) for c in s["configs"] for role in ("validation","outer") for k in CAPS}
            for x in r["replays"]:
                relevant=spec["validation"] if x["role"]=="validation" else family
                assert x["family"]==relevant and x["trace_sha256"]==PROTOCOL[relevant]["sha256"] and x["requests"]==40000
                x["portfolio"]=portfolio(x)
                x["origin"]="phase2cb_measured"
            for model in r["fit"]:
                assert sha(out/model["model_file"])==model["checkpoint_sha256"]
            fit.extend(r["fit"])
            replay.extend(r["replays"])
            validation.extend(r["validation"])
            parity.extend(r["parity"])
            chron.extend(r["replay_parity"])
            resource.append(r["resource"])
    assert sum(x["fit_components"] for x in fit)==231
    assert len(chron)==14
    assert len([x for x in replay if not x["policy"].endswith("_reference")])==3234
    # Baselines used to isolate target/feature/shortlist changes have the exact
    # architecture chosen for that fold by Stage A validation.
    comparisons=[]
    by_name={(x["fold"],x["policy"],x["role"],x["capacity"]):x for x in replay}
    for family in PROTOCOL:
        arch=selection["C",family]["selection_scope"]["full_model"].split("_")[1]
        for cap in CAPS:
            for role in ("validation","outer"):
                for target in ("h8","resolved"):
                    x=by_name[family,f"A_{arch}_100000_{target}",role,cap]
                    comparisons.append(dict(x,portfolio=f"selected_arch_F0_{target}",origin="matched_reuse"))
    return fit,replay+comparisons,validation,parity,chron,resource,run


def historical(out):
    archive=out/"inputs/CLOSED_LOOP_RESULTS.csv"
    override={(r["family"],r["capacity"],r["policy"],r["seed"]):r
              for r in rows(out/"inputs/OVERRIDE_DIAGNOSTICS.csv")}
    included={"lru","fifo_reinsertion","sieve","deepsets_h8_exact_argmin","deepsets_h8_near_tie_0.05",
              "deepsets_h8_calibrated_baseline_gate","student_hist_gradient_boosting_h8",
              "student_hist_gradient_boosting_resolved","student_ridge_h8","student_ridge_resolved",
              "random_matched_resolved_ridge_k8","random_matched_deepsets_calibrated_approximate_tie",
              "random_matched_deepsets_near_tie_0.05","oracle_belady","oracle_lru_tail_k8_next_use"}
    typed=[]
    for r in rows(archive):
        if r["policy"] not in included:
            continue
        f,c,p,s=r["family"],int(r["capacity"]),r["policy"],int(r["seed"])
        assert r["trace_sha256"]==PROTOCOL[f]["sha256"] and c in CAPS and int(r["requests"])==40000
        d=override.get((f,str(c),p,str(s)),{})
        typed.append(dict(family=f,capacity=c,policy=p,portfolio=f"{p}_s{s}" if s else p,
                          trace_sha256=r["trace_sha256"],requests=40000,misses=int(r["misses"]),
                          cpu_seconds=float(r["cpu_seconds"]),overrides=int(r["overrides"] or 0),
                          evictions=int(r["evictions"] or 0),peak_rss_kib=int(r["process_peak_rss_kib"] or 0),
                          sampled_overrides=int(d.get("overrides",0)),sampled_beneficial=int(d.get("beneficial",0)),
                          sampled_harmful=int(d.get("harmful",0)),sampled_neutral=int(d.get("neutral",0)),
                          role="outer",rule="historical",stage="baseline",seed=s,
                          origin="phase2b_retrieved",audit_surface="Phase 2B common LRU states"))
    return typed


def summarize(outer):
    historical_lru={(r["family"],r["capacity"]):r for r in outer if r["portfolio"]=="lru"}
    lru={(r["family"],r["capacity"]):r for r in outer if r["portfolio"]=="D_lru_profiled"}
    assert set(lru)==set(historical_lru)
    assert sum(r["misses"] for r in lru.values())==565126
    assert all(lru[key]["misses"]==historical_lru[key]["misses"] for key in lru)
    groups=defaultdict(list)
    for r in outer:
        groups[r["portfolio"]].append(r)
    family_rows, cap_rows, total, oracle = [],[],[],[]
    for p,records in groups.items():
        index={(r["family"],r["capacity"]):r for r in records}
        assert len(index)==len(records)
        def measures(block):
            baseline=[lru[r["family"],r["capacity"]] for r in block]
            scope_cpu=[historical_lru[r["family"],r["capacity"]] if r["origin"]=="phase2b_retrieved"
                       else lru[r["family"],r["capacity"]] for r in block]
            misses=sum(r["misses"] for r in block)
            bm=sum(r["misses"] for r in baseline)
            n=sum(r["requests"] for r in block)
            cpu=sum(r["cpu_seconds"] for r in block)
            evictions=sum(r["evictions"] for r in block)
            delta=[r["misses"]-b["misses"] for r,b in zip(block,baseline)]
            return dict(portfolio=p,misses=misses,lru_misses=bm,requests=n,
                        relative_miss_reduction=(bm-misses)/bm,
                        absolute_miss_ratio_difference=(misses-bm)/n,
                        wins=sum(x<0 for x in delta),ties=sum(x==0 for x in delta),losses=sum(x>0 for x in delta),
                        cpu_us_per_request=cpu*1e6/n,cpu_ratio=cpu/sum(r["cpu_seconds"] for r in scope_cpu),
                        requests_per_cpu_second=n/cpu,peak_rss_kib=max(r["peak_rss_kib"] for r in block),
                        override_rate=sum(r["overrides"] for r in block)/max(1,evictions),
                        sampled_beneficial=sum(r["sampled_beneficial"] for r in block),
                        sampled_neutral=sum(r["sampled_neutral"] for r in block),
                        sampled_harmful=sum(r["sampled_harmful"] for r in block))
        family_data=[dict(family=f,**measures([r for r in records if r["family"]==f]))
                     for f in PROTOCOL if any(r["family"]==f for r in records)]
        family_rows.extend(family_data)
        cap_rows.extend(dict(capacity=c,**measures([r for r in records if r["capacity"]==c]))
                        for c in CAPS if any(r["capacity"]==c for r in records))
        loo={f:measures([r for r in records if r["family"]!=f])["relative_miss_reduction"]
             for f in PROTOCOL if any(r["family"]==f for r in records) and any(r["family"]!=f for r in records)}
        worst=min(family_data,key=lambda r:r["relative_miss_reduction"])
        result=dict(**measures(records),complete_grid=len(index)==21,
                    origin=records[0]["origin"],worst_family=worst["family"],
                    worst_family_regression=-worst["relative_miss_reduction"],
                    macro_family_improvement=sum(x["relative_miss_reduction"] for x in family_data)/len(family_data),
                    leave_one_family_out=loo,loo_min=min(loo.values()) if loo else None)
        (oracle if p.startswith("oracle_") else total).append(result)
    return total,family_rows,cap_rows,oracle


def screening(total,outer,chron):
    index={r["portfolio"]:r for r in total}
    report=[]
    for r in total:
        name=r["portfolio"]
        if r["origin"]!="phase2cb_measured" or name.startswith("D_lru") or ("random" in name or "low_frequency_budget" in name):
            continue
        matched=None
        if name.startswith("A_") and name.endswith("_resolved"):
            matched=name.removesuffix("_resolved")+"_h8"
        elif name in ("C_two_part","D_two_part_risk"):
            matched="selected_arch_F0_h8"
        elif name.endswith("_finalist"):
            rows_=[x for x in outer if x["portfolio"]==name]
            if all(x["spec"]["target"] in ("resolved","two_part") and x["spec"]["features"]=="F0" and
                   x["spec"]["shortlist"]=="low_frequency" and x["spec"]["k"]==8 for x in rows_):
                arch=name.split("_")[1]
                matched=f"A_{arch}_100000_h8"
        random_comparisons={}
        if name.endswith("_finalist"):
            prefix="D_"+name.split("_")[1]+"_"
            controls=[x for p,x in index.items() if p.startswith(prefix) and (
                "random" in p or "low_frequency_budget" in p)]
            assert len(controls)==16
            random_comparisons={x["portfolio"]:x["misses"]-r["misses"] for x in controls}
        timing_parity=[c["parity"] for c in chron if portfolio(c)==name]
        eligible=dict(at_least_half_percent=r["relative_miss_reduction"]>=.005,
                      matched_h8_improvement=r["misses"]<index[matched]["misses"] if matched else None,
                      macro_positive=r["macro_family_improvement"]>0,
                      leave_one_out_positive=(r["loo_min"] is not None and r["loo_min"]>0),
                      worst_family_within_two_percent=r["worst_family_regression"]<=.02,
                      beats_all_matched_controls=min(random_comparisons.values())>0 if random_comparisons else None,
                      cpu_within_five_x=r["cpu_ratio"]<=5.)
        report.append(dict(portfolio=name,complete_grid=r["complete_grid"],matched_h8=matched,
                           matched_control_miss_differences=random_comparisons,
                           optimization_decision_parity=all(timing_parity) if timing_parity else None,
                           passes=bool(r["complete_grid"] and name.endswith("_finalist") and all(eligible.values()) and all(timing_parity)),
                           **eligible))
    return report


def breakdown(all_rows,chron):
    source=[]
    samples=[r for r in all_rows if r["role"]=="outer" and r.get("profile_samples")]
    for r in samples:
        source.append(dict(family=r["family"],capacity=r["capacity"],policy=r["portfolio"],path="optimized",
                           samples=r["profile_samples"],full_replay_cpu_seconds=r["cpu_seconds"],
                           trace_decode_seconds=r["trace_decode_seconds"],**r["profile_ns"]))
    for r in chron:
        source.append(dict(family=r["fold"],capacity=r["capacity"],policy=r["policy"],path="reference",
                           samples=r["reference_profile_samples"],full_replay_cpu_seconds=r["reference_cpu_seconds"],
                           **r["reference_profile"]))
    time_keys=("membership","request_metadata","candidate_list_and_sieve","shortlist","feature_vector_total",
               "feature_buffer_gather","estimator_api_or_packed_score","normalization","model_api",
               "linear_dot","tree_traversal_with_normalization","gate","decision_total","cache_state",
               "feature_state_maintenance","counters_and_diagnostics")
    result=[]
    for r in source:
        begin={k:r[k] for k in ("family","capacity","policy","path","samples","full_replay_cpu_seconds")}
        total=r.get("sampled_total",1)
        for k in time_keys:
            ns=r.get(k,0)
            result.append(dict(begin,component=k,sampled_ns=ns,us_per_profiled_request=ns/r["samples"]/1000,
                              percentage_of_sampled_time=100*ns/max(1,total)))
        if "trace_decode_seconds" in r:
            result.append(dict(begin,component="trace_decode_separate",whole_trace_wall_seconds=r["trace_decode_seconds"]))
    return result


def execute(out):
    e=out/"evidence"
    fit, replays, valid, parity, chron, resources, jobs=inventory(out)
    archived=historical(out)
    fitting={(m["fold"],m["spec"]["name"]):m for m in fit}
    validated={(v["fold"],v["policy"]):v for v in valid if v.get("selected")}
    for r in replays:
        if r["spec"] is None:
            continue
        model=fitting[r["fold"],r["spec"]["name"]]
        r.update({name:model[name] for name in ("train_groups","candidate_rows","train_resolved_positive",
                                                   "train_target_positive","training_cpu_seconds","checkpoint_bytes")})
        measure=validated.get((r["fold"],r["policy"]),{})
        r["validation_positive_groups"]=measure.get("positive_groups")
    outer=[r for r in replays if r["role"]=="outer"]+archived
    total,families,capacities,oracle=summarize(outer)
    criteria=screening(total,outer,chron)
    table(e,"FINAL_POLICY_COMPARISON.csv",total)
    table(e,"FUTURE_AWARE_ORACLES.csv",oracle)
    table(e,"VALIDATION_RESULTS.csv",valid+[r for r in replays if r["role"]=="validation"])
    table(e,"CLOSED_LOOP_RESULTS.csv",outer)
    table(e,"FAMILY_LEVEL_RESULTS.csv",families)
    table(e,"CAPACITY_LEVEL_RESULTS.csv",capacities)
    table(e,"MODEL_FIT_RESULTS.csv",fit)
    table(e,"RESOURCE_PROFILE.csv",resources)
    table(e,"ADVANCEMENT_CRITERIA_RESULTS.csv",criteria)
    table(e,"CPU_OVERHEAD_BREAKDOWN.csv",breakdown(replays,chron))
    def subset(tags,predicate=lambda _:True):
        return [r for r in replays if r["stage"] in tags and predicate(r)]
    table(e,"TRAINING_DATA_SCALING.csv",subset("A"))
    table(e,"SUPERVISION_COMPARISON.csv",subset("AB",lambda r:r["spec"]["features"]=="F0" and r["spec"]["budget"]==100000))
    table(e,"FEATURE_ABLATION_RESULTS.csv",subset("B",lambda r:r["spec"]["target"]=="resolved")+
          [r for r in replays if r["portfolio"]=="selected_arch_F0_resolved"])
    table(e,"SHORTLIST_COMPARISON.csv",subset("C")+[r for r in replays if r["portfolio"]=="selected_arch_F0_resolved"])
    table(e,"BENEFIT_RISK_DECOMPOSITION_RESULTS.csv",subset("CD",lambda r:"two_part" in r["policy"])+
          [r for r in replays if r["portfolio"]=="selected_arch_F0_resolved"])
    table(e,"GATE_CALIBRATION_RESULTS.csv",valid)
    random=[r for r in subset("D") if "random" in r["rule"] or r["rule"]=="low_frequency_budget"]
    computed={(r["fold"],r["capacity"],r["role"],r["policy"]):r for r in subset("D") if r["policy"].endswith("_finalist")}
    for r in random:
        arch=r["policy"].split("_")[1]
        reference=computed[r["fold"],r["capacity"],r["role"],f"D_{arch}_finalist"]
        r["misses_minus_matched_learned"]=r["misses"]-reference["misses"]
    table(e,"RANDOM_OVERRIDE_CONTROL_RESULTS.csv",random)
    seed_blocks=defaultdict(lambda:defaultdict(list))
    for r in random:
        if r["role"]=="outer" and r["rule"]!="low_frequency_budget":
            model=r["policy"].split("_")[1]
            seed=r["settings"]["seed"]
            for f,c in (("all","all"),(r["family"],"all"),("all",r["capacity"]),(r["family"],r["capacity"])):
                seed_blocks[model,r["rule"],f,c][seed].append(r["misses_minus_matched_learned"])
    variability=[]
    for (model,rule,f,c),seeds in seed_blocks.items():
        assert set(seeds)==set(RANDOM_SEEDS)
        totals={s:sum(v) for s,v in seeds.items()}
        variability.append(dict(model=model,rule=rule,family=f,capacity=c,miss_differences_by_seed=totals,
                                mean_difference=mean(totals.values()),sd=stdev(totals.values()),
                                learned_wins=sum(x>0 for x in totals.values()),
                                ties=sum(x==0 for x in totals.values())))
    table(e,"RANDOM_CONTROL_SEED_VARIABILITY.csv",variability)
    parity_data=json.loads((out/"smoke/evidence/OPTIMIZED_INFERENCE_VALIDATION.json").read_text())
    dump(e/"OPTIMIZED_INFERENCE_VALIDATION.json",
         dict(preflight=parity_data,offline_prediction_parity=parity,finalist_chronological_parity=chron))
    winners=sorted((x for x in total if x["complete_grid"]),key=lambda x:x["misses"])[:8]
    nfit=sum(r["fit_components"] for r in fit)
    d=sum(x["passes"] for x in criteria)
    text=f"""# Phase 2C-B results

All four stage arrays and seven folds passed receipt verification. The campaign fit {nfit} deterministic model components and measured 3,234 validation and outer replay configurations, plus 14 full chronological reference-path checks. Phase 2B controls were retrieved. The seven families are retrospective development workloads.

{d} finalists passed all recorded advancement screens. See the criterion table for each component. Candidate-level intervention classifications are bounded samples from the policy's own evolving cache. Historical Phase 2B override diagnostics come from common LRU states. The CPU measurements use the same replay harness for Phase 2C-B models and LRU.

| Complete-grid configuration | Misses | Relative reduction | CPU vs LRU |
|---|---:|---:|---:|
"""
    text+="".join(f"| {w['portfolio']} | {w['misses']:,} | {w['relative_miss_reduction']:+.3%} | {w['cpu_ratio']:.2f}× |\n" for w in winners)
    (e/"EXPERIMENT_CONCLUSIONS.md").write_text(text)
    permitted=sorted(file for file in e.iterdir() if file.is_file() and file.name!="MANIFEST.sha256")
    # Group-level positions, checkpoints and raw traces live outside evidence.
    (e/"MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in permitted))
    archive=out/"phase2cb-private-review.zip"
    with ZipFile(archive,"w",compression=ZIP_DEFLATED) as z:
        for p in [*permitted,e/"MANIFEST.sha256"]:
            z.write(p,p.name)
    with ZipFile(archive) as z:
        assert z.testzip() is None
    dump(out/"receipts/REDUCER_SUCCESS.json",dict(status="COMPLETED",fits=nfit,replays=3234,
            sha256=sha(archive),manifest_sha256=sha(e/"MANIFEST.sha256"),phase2cb_source=jobs["source"]))


if __name__=="__main__":
    execute(Path(sys.argv[1]))
