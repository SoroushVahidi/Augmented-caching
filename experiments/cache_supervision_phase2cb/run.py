"""Phase 2C-B staged runner. All selection receipts precede outer replay."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import resource
from time import perf_counter, process_time

import joblib
import numpy as np

from assets import CAPS,BUDGETS, PROTOCOL, GroupData, dump, rows, sha, verify
from campaign import trace_data
from deploy import replay
from learning import ARCHS,Spec,fit,validate

ALL_BUDGETS=(1176,*BUDGETS)
RANDOM_SEEDS=(7,19,42,101,211)
UNIFORM_SEEDS=(7,19,42)
STAGE_ORDER=("A","B","C","D")
ARCHITECTURES={"ridge":"deterministic Cholesky Ridge alpha 10",
               "hgb":"deterministic 30-iteration depth-3 histogram boosting"}


def stage_a():
    return [Spec("A",f"A_{arch}_{n}_{target}",arch,n,target)
            for n in ALL_BUDGETS for arch in ARCHS for target in ("h8","resolved")]


def pick(rows_,arch=None,stage=None):
    grouped={}
    for row in rows_:
        if row["role"]!="validation" or row["rule"]!="gate" or row["settings"].get("shortlist")=="full":
            continue
        spec=row["spec"]
        if spec["budget"]!=100000 or (arch is not None and spec["arch"]!=arch) or (stage and spec["stage"]!=stage):
            continue
        if spec["name"] not in grouped:
            grouped[spec["name"]]=[]
        grouped[spec["name"]].append(row)
    for candidates in grouped.values():
        assert len(candidates)==3
    if not grouped:
        raise ValueError("No eligible validation results")
    return min(grouped.values(),key=lambda group:(sum(v["misses"] for v in group),
               sum(v["overrides"] for v in group),group[0]["policy"]))[0]


def choose_arch(rows_):
    vals={}
    for arch in ARCHS:
        a=[r for r in rows_ if r["role"]=="validation" and r["spec"]["stage"]=="A"
           and r["spec"]["arch"]==arch and r["spec"]["budget"]==100000]
        assert len(a)==6
        vals[arch]=(sum(r["misses"] for r in a),sum(r["overrides"] for r in a))
    return min(ARCHS,key=lambda a:vals[a])


def choose_k16(root,nonouter):
    eligible={"low_frequency","sieve_derived","ghost_informed","uniform_7"}
    sums=Counter()
    for r in rows(root/"evidence"/"SHORTLIST_COVERAGE.csv"):
        if (r["family"] in nonouter and int(r["capacity"]) in CAPS and int(r["k"])==16
                and r["mechanism"] in eligible):
            sums[r["mechanism"]]+=float(r["estimated_captured_gain"])
    return min(eligible,key=lambda x:(-sums[x],x))


def prior(out,stage,family):
    path=out/"receipts"/f"{stage}-{family}.json"
    result=json.loads(path.read_text())
    assert result["status"]=="COMPLETED"
    return result


def load_checkpoint(out,spec):
    path=out/"models"/f"{spec['fold']}-{spec['name']}.joblib"
    return joblib.load(path)


def checkpoint(out,family,artifact):
    name=artifact["spec"]["name"]
    p=out/"models"/f"{family}-{name}.joblib"
    t=p.with_suffix(".tmp")
    joblib.dump(artifact,t,compress=3)
    t.replace(p)
    return dict(model_file=str(p.relative_to(out)),checkpoint_sha256=sha(p),checkpoint_bytes=p.stat().st_size)


def build_plans(stage,out,dataset,family):
    split=PROTOCOL[family]
    train=tuple(split["train"])
    val=split["validation"]
    assert val!=family and family not in train and val not in train
    feats=[]
    if stage=="A":
        return stage_a(),{},train,val
    prev=prior(out,"A",family)["replays"]
    arch=choose_arch(prev)
    if stage=="B":
        feats=[Spec("B",f"B_{arch}_{target}",arch,100000,target) for target in ("h32","hc")]
        feats += [Spec("B",f"B_{arch}_{name}",arch,100000,"resolved",name) for name in ("F1","F2","F3","F4","F5","F6")]
    if stage=="C":
        k16=choose_k16(dataset,set(train)|{val})
        feats=[Spec("C",f"C_{arch}_two_part",arch,100000,"two_part")]
        feats += [Spec("C",f"C_{arch}_{s}_k8",arch,100000,"resolved","F0",s,8)
                  for s in ("lru_tail","sieve_derived","ghost_informed",*(f"uniform_{x}" for x in UNIFORM_SEEDS))]
        feats.append(Spec("C",f"C_{arch}_{k16}_k16",arch,100000,"resolved","F0",k16,16))
        return feats,{"full_model":f"A_{arch}_100000_resolved"},train,val
    if stage=="D":
        previous=prev+prior(out,"B",family)["replays"]+prior(out,"C",family)["replays"]
        return feats,{x:pick(previous,arch=x) for x in ARCHS}|{"two_part_arch":arch},train,val
    return feats,{},train,val


def run_stage(stage,out,dataset,old,trace_root,family,smoke=False):
    whole_cpu,whole_wall=process_time(),perf_counter()
    if not smoke:
        gate=json.loads((out/"evidence/DATASET_VERIFICATION.json").read_text())
        assert gate["status"]=="VERIFIED" and gate["phase2ca_manifest_sha256"]==sha(dataset/"evidence/MANIFEST.sha256")
    specs,selection_scope,train,val=build_plans(stage,out,dataset,family)
    if smoke:
        specs=[Spec("A",f"A_{arch}_100000_{target}",arch,100000,target)
               for arch in ARCHS for target in ("h8","resolved")]
    data=GroupData(dataset,old,set(train)|{val})
    vtrace=trace_data(trace_root,val)
    train_limit=None
    val_limit=1024 if smoke else None
    fit_notes=[]
    validation=[]
    parity=[]
    configs=[]
    end=12000 if smoke else 50000
    staged_data=None
    staged_key=None
    for spec in specs:
        key=(spec.budget,spec.features,spec.shortlist,spec.k)
        if stage=="A":
            if key!=staged_key:
                staged_data=(data.read(train,spec.budget,spec.features,"h8",spec.shortlist,spec.k,train_limit),
                             data.read([val],spec.budget,spec.features,"h8",spec.shortlist,spec.k,val_limit))
                staged_key=key
            tr,va=staged_data
            if spec.target=="resolved":
                tr,va=dict(tr,y=tr["resolved"]),dict(va,y=va["resolved"])
        else:
            tr=data.read(train,spec.budget,spec.features,spec.target,spec.shortlist,spec.k,train_limit)
            va=data.read([val],spec.budget,spec.features,spec.target,spec.shortlist,spec.k,val_limit)
        model=fit(spec,tr)
        ck=checkpoint(out,family,model)
        gates,stats,check=validate(model,va)
        parity.append(dict(fold=family,policy=spec.name,**check))
        validation.extend(dict(fold=family,validation_family=val,policy=spec.name,stage=stage,**r) for r in stats)
        fit_notes.append({k:v for k,v in model.items() if k not in ("columns","scaler","models","packed")}|
                         dict(fold=family,**ck,features_kept=model["columns"].tolist(),fit_components=len(model["models"])))
        configs.append(dict(policy=spec.name,spec=model["spec"],model=spec.name,settings=gates["gate"]))
        print(f"FIT_VALIDATED {stage} {family} {spec.name} groups={model['train_groups']} cpu={model['training_cpu_seconds']:.3f}",flush=True)
    if stage=="C":
        m=selection_scope["full_model"]
        artifact=load_checkpoint(out,dict(fold=family,name=m))
        # Full-candidate expansion is an inference-only diagnostic. It reuses the
        # matched K=8 training fit and the previously frozen K=8 threshold.
        a=next(r for r in prior(out,"A",family)["replays"] if r["policy"]==m and r["role"]=="validation")
        configs.append(dict(policy=f"C_{artifact['spec']['arch']}_full_scoring",spec=artifact["spec"],
                            model=m,settings=dict(a["settings"],shortlist="full")))
    if stage=="D":
        for arch in ARCHS:
            win=selection_scope[arch]
            spec=win["spec"]
            name=spec["name"]
            original=load_checkpoint(out,dict(fold=family,name=name))
            info=Spec(**spec)
            va=data.read([val],info.budget,info.features,info.target,info.shortlist,info.k)
            rules={rule:validate(original,va,rule) for rule in ("raw","oldest_tie","approx","preliminary")}
            short={k:v for k,v in win["settings"].items() if k in ("shortlist","k")}
            for rule,(choice,v,check) in rules.items():
                p=f"D_{arch}_{rule}"
                validation.extend(dict(fold=family,validation_family=val,policy=p,stage=stage,**r) for r in v)
                parity.append(dict(fold=family,policy=p,**check))
                configs.append(dict(policy=p,spec=spec,model=name,settings=dict(choice[rule],**short)))
            historical=prior(out,win["stage"],family)["replays"]
            rates={r["capacity"]:r["override_rate"] for r in historical
                   if r["policy"]==win["policy"] and r["role"]=="validation"}
            assert set(rates)==set(CAPS)
            configs.append(dict(policy=f"D_{arch}_finalist",spec=spec,model=name,settings=win["settings"]))
            for policy in ("random_full","random_shortlist","random_gate","low_frequency_budget"):
                for seed in RANDOM_SEEDS if policy!="low_frequency_budget" else (0,):
                    p=f"D_{arch}_{policy}_{seed}"
                    config=dict(win["settings"],rule=policy,seed=seed)
                    configs.append(dict(policy=p,spec=spec,model=name,settings=config,rates=rates))
        arch=selection_scope["two_part_arch"]
        name=f"C_{arch}_two_part"
        artifact=load_checkpoint(out,dict(fold=family,name=name))
        va=data.read([val],100000,"F0","two_part","low_frequency",8)
        choice,stats,check=validate(artifact,va,"risk")
        p=f"D_{arch}_two_part_risk"
        validation.extend(dict(fold=family,validation_family=val,policy=p,stage=stage,**r) for r in stats)
        parity.append(dict(fold=family,policy=p,**check))
        configs.append(dict(policy=p,spec=artifact["spec"],model=name,settings=choice["risk"]))
        configs.append(dict(policy="D_lru_profiled",spec=None,model=None,settings={"rule":"lru"}))
    cpu,wall=process_time(),perf_counter()
    result=[]
    replay_parity=[]
    # Close-loop validation evaluates every fixed configuration. Finalists use
    # only earlier stages' validation rows. The outer trace opens afterwards.
    for c in configs:
        artifact=load_checkpoint(out,dict(fold=family,name=c["model"])) if c["model"] else None
        for cap in CAPS:
            cfg=dict(c["settings"])
            if "rates" in c:
                cfg["rate"]=c["rates"][cap]
            row=replay(vtrace,val,cap,artifact,cfg,end=end)
            result.append(dict(row,policy=c["policy"],rule=cfg["rule"],fold=family,role="validation",
                               stage=stage,spec=c["spec"],settings=cfg))
    select={"fold":family,"train":list(train),"validation_family":val,"outer_family":family,
            "configs":configs,"validation_replay_sha_pre_outer":__import__("hashlib").sha256(json.dumps(result,sort_keys=True).encode()).hexdigest(),
            "selection_scope":{k:(v["policy"] if isinstance(v,dict) else v) for k,v in selection_scope.items()}}
    dump(out/"selections"/f"{stage}-{family}.json",select)
    vals_frozen=perf_counter()
    otrace=trace_data(trace_root,family)
    for c in configs:
        artifact=load_checkpoint(out,dict(fold=family,name=c["model"])) if c["model"] else None
        for cap in CAPS:
            cfg=dict(c["settings"])
            if "rates" in c:
                cfg["rate"]=c["rates"][cap]
            parity_scope=(stage=="D" and c["policy"].endswith("_finalist") and cap==64)
            row=replay(otrace,family,cap,artifact,cfg,end=end,profile=stage=="D",audit=True,signature=parity_scope)
            result.append(dict(row,policy=c["policy"],rule=cfg["rule"],fold=family,role="outer",stage=stage,spec=c["spec"],settings=cfg))
            if parity_scope:
                ref=replay(otrace,family,cap,artifact,cfg,end=end,profile=True,reference=True,signature=True)
                agrees=all(row[k]==ref[k] for k in ("misses","overrides","eviction_sha256"))
                replay_parity.append(dict(fold=family,policy=c["policy"],capacity=cap,parity=agrees,
                                         reference_cpu_seconds=ref["cpu_seconds"],optimized_cpu_seconds=row["cpu_seconds"],
                                         reference_profile=ref["profile_ns"],reference_profile_samples=ref["profile_samples"],
                                         reference_misses=ref["misses"],optimized_misses=row["misses"]))
                if not agrees:
                    result.append(dict(ref,policy=c["policy"]+"_reference",rule=cfg["rule"],fold=family,role="outer",
                                       stage=stage,spec=c["spec"],settings=cfg))
        print(f"OUTER_REPLAY_COMPLETE {stage} {family} {c['policy']}",flush=True)
    payload=dict(status="COMPLETED",stage=stage,fold=family,slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                 validation=validation,replays=result,fit=fit_notes,parity=parity,replay_parity=replay_parity,
                 slurm_array_job_id=os.environ.get("SLURM_ARRAY_JOB_ID"),
                 resource=dict(stage=stage,fold=family,cpu_seconds=process_time()-whole_cpu,wall_seconds=perf_counter()-whole_wall,
                               replay_cpu_seconds=process_time()-cpu,replay_wall_seconds=perf_counter()-wall,
                               latest_validation_freeze_elapsed=vals_frozen-wall,
                               rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    dump(out/"receipts"/f"{stage}-{family}.json",payload)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("stage",choices=(*STAGE_ORDER,"verify","smoke"))
    p.add_argument("out",type=Path)
    p.add_argument("dataset",type=Path)
    p.add_argument("old",type=Path)
    p.add_argument("trace",type=Path)
    p.add_argument("--family",choices=tuple(PROTOCOL))
    args=p.parse_args()
    if args.stage=="verify":
        result=verify(args.dataset,args.old,args.out/"evidence/DATASET_VERIFICATION.json",
                      verification_job=os.environ.get("SLURM_JOB_ID"))
        print(result,flush=True)
    else:
        run_stage("A" if args.stage=="smoke" else args.stage,args.out,args.dataset,args.old,args.trace,
                  args.family,smoke=args.stage=="smoke")
