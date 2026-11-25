"""Run the frozen pilot and retrospective held-out campaign."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
from time import perf_counter
from datetime import datetime, timezone
import traceback

import numpy as np
import torch

from legacy import OLD_RUN, PROTOCOL, archived_rows, verify_archive
from features import GLOBAL_NAMES, STATIC_NAMES
from fit import atomic_json, evaluate, fit, load
from network import Architecture
from sequences import build_blocks
from simulation import inputs, replay
from streaming import Stream
from summarize import assemble

DEVELOPMENT = ("brightkite", "cloudphysics", "twemcache")
PILOT_WARM = 4096
PILOT_END = 8192
FULL_WARM = 10000
FULL_END = 50000
ARCHITECTURES = tuple(f"R{i:02}" for i in range(1, 9))
HERE = Path(__file__).resolve().parent


def now():
    return datetime.now(timezone.utc).isoformat()


def state(root, status, stage, **extra):
    atomic_json(root/"RUN_STATUS.json",dict(status=status,stage=stage,updated_at=now(),pid=os.getpid(),**extra))
    print(f"{now()} {status} {stage}",flush=True)


def key(family, cap):
    return f"{family}/{cap}"


def make_blocks(families,caps,split,start,end,bins,decisions,legacy=False,policy=None,selection="uniform"):
    result = {}
    for f in families:
        _,_,stream = inputs(f)
        for cap in caps:
            result[key(f,cap)] = build_blocks(stream,cap,family=f,split=split,score_start=start,end=end,
                bins=bins,decisions=decisions,legacy_root=OLD_RUN/"data" if legacy else None,
                policy=policy,selection=selection)
    return result


def compute_feature_audit(root, data):
    audit = {"request":{},"candidate":{}}
    cells = {"request":{},"candidate":{}}
    for key_, blocks in data.items():
        if not blocks:
            continue
        cells["request"][key_] = np.concatenate([b.requests for b in blocks])
        cells["candidate"][key_] = np.concatenate([x for b in blocks for x in b.features])
    for kind, names in (("request",GLOBAL_NAMES),("candidate",STATIC_NAMES)):
        arrays = cells[kind]
        matrix=np.concatenate(list(arrays.values()))
        std = matrix.std(0, dtype=np.float64)
        sample = matrix[np.linspace(0,len(matrix)-1,min(30000,len(matrix)),dtype=int)].astype(np.float64)
        rank = np.linalg.matrix_rank(sample - sample.mean(0),tol=.005)
        correlations = []
        nonconstant = np.flatnonzero(std>1e-8)
        if len(nonconstant)>1:
            corr=np.corrcoef(matrix[:,nonconstant].T)
            correlations=[(names[nonconstant[i]], names[nonconstant[j]], float(corr[i,j]))
                          for i,j in zip(*np.where(np.triu(np.abs(corr)>.999,1)))]
        audit[kind] = dict(observations=len(matrix),constants=[names[i] for i in np.flatnonzero(std<1e-8)],
                          names=names,means=matrix.mean(0,dtype=np.float64).tolist(),std=std.tolist(),
                          nonfinite=int((~np.isfinite(matrix)).sum()),estimated_centered_rank=int(rank),
                          high_correlation=correlations, cells={cell:dict(
                              mean=x.mean(0,dtype=np.float64).tolist(),std=x.std(0,dtype=np.float64).tolist(),
                              count=len(x)) for cell,x in arrays.items()})
        if kind=="candidate":
            audit[kind]["missing_gap_fraction"] = (1-matrix[:,34:50].mean(0,dtype=np.float64)).tolist()
    atomic_json(root/"feature_audit.json",audit)


def fit_one(root, fold, run, train, val, conf, objective, args, provenance, initial=None, seed=7, epochs=None, horizon=16, lr=.001):
    dest=root/"fits"/fold/run
    summary=fit(dest,[b for cell in train.values() for b in cell],val,conf,provenance,
                epochs=epochs or args.epochs,device=args.device,seed=seed,objective=objective,horizon=horizon,
                initial=initial,lr=lr)
    selected=dest/"selected.pt"
    path=dest/"result.json"
    result=json.loads(path.read_text()) if path.exists() else {}
    result.update(run=run,fold=fold,objective=objective,horizon=horizon,checkpoint=str(selected),
                  checkpoint_config=vars(conf),fit=summary,
                  validation=next(r["validation"] for r in summary["curve"] if r["epoch"]==summary["selected_epoch"]))
    atomic_json(path,result)
    print(f"{now()} FIT {fold} {run} epochs={summary['epochs']} selected={summary['selected_epoch']} sha256={summary['selected_sha256']}",flush=True)
    return run,selected,conf,objective,horizon


def replay_path(root,task):
    label="-".join(str(t) for t in (*task[:5],*task[-2:]))
    return root/"replays"/f"{label}.json"


def measure(root,pool,tasks):
    rows=[None]*len(tasks)
    pending=[(i,t) for i,t in enumerate(tasks) if not replay_path(root,t).exists()]
    for (i,task),row in zip(pending,pool.map(replay,[t for _,t in pending])):
        atomic_json(replay_path(root,task),row)
        rows[i]=row
        print(f"{now()} REPLAY {row['split']} {row['method']} {row['family']}/{row['capacity']} {row['misses']}/{row['requests']}",flush=True)
    for i,t in enumerate(tasks):
        if rows[i] is None:
            rows[i]=json.loads(replay_path(root,t).read_text())
    return rows


def baseline_tasks(families,caps,warm,end,split):
    methods=("lru","fifo_reinsertion","sieve","mlp_h4_s7_argmin",
             "deepsets_h4_s7_argmin","deepsets_h8_s7_argmin","deepsets_h8_s7_margin_lru")
    return [(name,f,c,warm,end,None,"original",split) for f in families for c in caps for name in methods]


def tasks_for(runs,families,caps,warm,end,split,variant="streaming"):
    return [(name,f,c,warm,end,path,variant,split) for f in families for c in caps for name,path,*_ in runs]


def development(root,pool,args,env):
    train_families=tuple(f for f in PROTOCOL if f not in DEVELOPMENT)
    caps=(32,128)
    train=make_blocks(train_families,caps,"train",128,PILOT_END,8,5)
    val=make_blocks(DEVELOPMENT,caps,"development",PILOT_WARM,PILOT_END,5,5)
    compute_feature_audit(root,{**train,**val})
    provenance=dict(**env,train=list(train_families),validation=list(DEVELOPMENT),test=[],capacities=list(caps),
                    train_start=128,score_start=PILOT_WARM,end=PILOT_END,stimulus="LRU",train_blocks=8,
                    validation_blocks=5,sampled_decisions=5,split_seed=421)
    designs=[(Architecture(a),objective,16,"") for a in ARCHITECTURES for objective in ("T02","T03")]
    designs.extend((Architecture(a),"T01",h,"") for a,h in (("R02",16),("R05",4),("R05",16),("R05",32)))
    designs.extend((conf,"T03",16,suffix) for conf,suffix in (
        (Architecture("R05",reset_hidden=True),"_reset"),(Architecture("R05",order="permuted"),"_permuted"),
        (Architecture("R05",order="delayed"),"_delayed"),
        (Architecture("R05",global_hidden=32),"_global32"),
        (Architecture("R05",object_history=4),"_object4"),
        (Architecture("R07",request_history=32),"_history32"),
    ))
    trained=[]
    for conf,obj,h,suffix in designs:
        run=f"{conf.name}_{obj}_h{h}{suffix}"
        trained.append(fit_one(root,"development",run,train,val,conf,obj,args,provenance,horizon=h))
    state(root,"RUNNING_HEALTHY","development_replay",trained=len(trained))
    started=perf_counter()
    original=measure(root,pool,baseline_tasks(DEVELOPMENT,caps,PILOT_WARM,PILOT_END,"development"))
    results=measure(root,pool,tasks_for(trained,DEVELOPMENT,caps,PILOT_WARM,PILOT_END,"development"))
    scores = {}
    for name,path,*_ in trained:
        if name.split("_")[0] not in {"R03","R04","R05","R06","R07"} or len(name.split("_"))!=3:
            continue
        measured=[r for r in results if r["method"]==name]
        scores[name]=(sum(r["misses"] for r in measured),float(np.median([r["us_per_request"] for r in measured])))
    best=min(scores,key=scores.get)
    selected=next(x for x in trained if x[0]==best)
    ablation=("R05_T03_h16_recompute",*next(x for x in trained if x[0]=="R05_T03_h16")[1:])
    measure(root,pool,tasks_for([ablation],("brightkite","twemcache"),(32,),PILOT_WARM,PILOT_END,
                                "development",variant="recompute"))
    atomic_json(root/"pilot_selection.json",dict(selected=best,criterion="aggregate validation misses then median CPU cost",
                 candidates=scores, measured_set="development",replay_wall_seconds=perf_counter()-started))
    assemble(root)
    write_report(root)
    (root/"PILOT_COMPLETE").touch()
    if args.aggregation:
        aggregated=[]
        _,path,conf,obj,h=selected
        active=Stream(load(path))
        safe_families=("citibike","metakv")
        for kind in ("targeted","uniform"):
            extra=make_blocks(safe_families,(32,),"train_aggregation",1024,PILOT_WARM,5,5,policy=active,selection=kind)
            combined={**train,**{f"learned/{k}":v for k,v in extra.items()}}
            tag=f"{best}_aggregate_{kind}"
            aggregated.append(fit_one(root,"development",tag,combined,val,conf,obj,args,
                              dict(**provenance,aggregation=dict(policy=best,selection=kind,families=safe_families,
                                   capacity=32,end=PILOT_WARM)),horizon=h,initial=path,epochs=2,lr=.0003))
        measure(root,pool,tasks_for(aggregated,DEVELOPMENT,caps,PILOT_WARM,PILOT_END,"development"))
        assemble(root)
        write_report(root)
        (root/"AGGREGATION_COMPLETE").touch()
    return results


def full(root,pool,args,env):
    archive = list(archived_rows())
    atomic_json(root/"archived_baselines.json",archive)
    caps=(32,64,128)
    tasks = [task for task in baseline_tasks(PROTOCOL,caps,FULL_WARM,FULL_END,"test")
             if task[0] in {"lru","fifo_reinsertion","sieve"}]
    baseline = measure(root,pool,tasks)
    expected={(r["method"],r["family"],r["capacity"]):r["misses"] for r in archive}
    for r in baseline:
        if r["misses"] != expected[(r["method"],r["family"],r["capacity"])]:
            raise ValueError("Full matched native baseline drift")
    # The candidate roster is frozen independently of the development ranking.
    roster=(("R02","T03"),("R08","T03"),("R05","T02"),("R05","T03"),("R06","T03"))
    for family,spec in PROTOCOL.items():
        val_family=spec["validation"]
        train_families=tuple(spec["train"])
        if set(train_families)&{family,val_family} or family==val_family:
            raise ValueError("Invalid nested split")
        state(root,"RUNNING_HEALTHY",f"full/{family}")
        train=make_blocks(train_families,caps,"train",128,FULL_END,12,6,legacy=True)
        val=make_blocks((val_family,),caps,"validation",FULL_WARM,FULL_END,5,6,legacy=True)
        prov=dict(**env,train=list(train_families),validation=val_family,test=family,capacities=list(caps),
                  train_start=128,score_start=FULL_WARM,end=FULL_END,stimulus="LRU",
                  train_blocks=12,validation_blocks=5, sampled_decisions=6,split_seed=421)
        trained=[fit_one(root,family,f"{a}_{o}_h16",train,val,Architecture(a),o,args,prov,epochs=args.full_epochs)
                 for a,o in roster]
        val_rows=measure(root,pool,tasks_for(trained,(val_family,),caps,FULL_WARM,FULL_WARM+4096,
                                            f"nested-{family}"))
        def metric(run):
            rows=[r for r in val_rows if r["method"]==run[0]]
            return (sum(r["misses"] for r in rows),float(np.median([r["us_per_request"] for r in rows])))
        best=min([r for r in trained if r[0].startswith(("R05","R06"))],key=metric)
        rest=[]
        for seed in (19,42):
            n,_,conf,obj,h=best
            rest.append(fit_one(root,family,f"{n}_s{seed}",train,val,conf,obj,args,prov,seed=seed,epochs=args.full_epochs))
        tests=trained+rest
        selection={r[0]:json.loads((r[1].parent/"FIT_COMPLETE.json").read_text())["selected_sha256"] for r in tests}
        freeze=root/"fits"/family/"FREEZE.json"
        decision=dict(test_family=family,validation_family=val_family,selection=selection,
                      winner=best[0],criterion="validation closed-loop misses then median CPU cost")
        if freeze.exists() and json.loads(freeze.read_text())!=decision:
            raise ValueError("Frozen fold cannot be changed")
        atomic_json(freeze,decision)
        test=make_blocks((family,),caps,"test",FULL_WARM,FULL_END,8,6,legacy=True)
        for run,path,_,obj,h in tests:
            path_result=path.parent/"result.json"
            result=json.loads(path_result.read_text())
            if "test" not in result:
                result["test"]=evaluate(load(path,args.device),test,args.device,obj,h)
                atomic_json(path_result,result)
        chosen=[("R_SELECTED_s7",*best[1:]),*( (f"R_SELECTED_s{seed}",*r[1:]) for seed,r in zip((19,42),rest))]
        measure(root,pool,tasks_for(trained+chosen,(family,),caps,FULL_WARM,FULL_END,"test"))
        assemble(root)
        write_report(root)
    (root/"FULL_COMPLETE").touch()


def report_lines(root):
    table=root/"baseline_comparisons.csv"
    if not table.exists():
        return []
    import csv
    rows=list(csv.DictReader(table.open()))
    return [f"| {r['split']} | {r['method']} | {r['cells']} | {r['misses']} | {r['base_misses']} | {float(r['relative_reduction'])*100:+.3f}% | {r['wins']}/{r['ties']}/{r['losses']} |"
            for r in rows if r["baseline"]=="lru" and r["method"].startswith("R") and r["relative_reduction"]]


def write_report(root):
    data=report_lines(root)
    (root/"RESULTS.md").write_text("# Recurrent eviction results\n\n"
          "This table uses recorded matched cells. Development and nested validation results carry their split names. "
          "The definitive execution state is RUN_STATUS.json. Full retrospective coverage requires FULL_COMPLETE.\n\n"
          "| Split | Model | Cells | Misses | LRU misses | Relative reduction | W/T/L |\n"
          "|---|---|---:|---:|---:|---:|---:|\n" + "\n".join(data)+"\n")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=["pilot","full","auto","reduce"])
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    parser.add_argument("--epochs",type=int,default=5)
    parser.add_argument("--full-epochs",type=int,default=8)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--aggregation",action="store_true")
    args=parser.parse_args()
    root=args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    (root/"replays").mkdir(exist_ok=True)
    if args.stage=="reduce":
        assemble(root)
        write_report(root)
        return
    torch.set_num_threads(1)
    start=perf_counter()
    def stopped(signum, frame):
        state(root,"CANCELLED","signal",signal=signum)
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, stopped)
    signal.signal(signal.SIGINT, stopped)
    state(root,"RUNNING_HEALTHY","verify_inputs")
    try:
        source=subprocess.check_output(["git","rev-parse","HEAD"],cwd=HERE,text=True).strip()
        config={k:v for k,v in vars(args).items() if k not in {"out"}}
        env=dict(source=source,python=sys.version.split()[0],torch=str(torch.__version__),numpy=np.__version__,
                 platform=platform.platform(),cuda=torch.version.cuda,device=args.device,
                 gpu=torch.cuda.get_device_name(0) if args.device=="cuda" else None,
                 trace_hashes={f:x["sha256"] for f,x in PROTOCOL.items()})
        atomic_json(root/"environment.json",dict(**env,launch=config,launched_at=now()))
        atomic_json(root/"control_verification.json",verify_archive())
        with ProcessPoolExecutor(args.workers,mp_context=mp.get_context("spawn")) as pool:
            if args.stage in {"pilot","auto"}:
                start_pilot=perf_counter()
                pilot=development(root,pool,args,env)
            if args.stage in {"full","auto"}:
                if args.stage=="auto":
                    local_cell_seconds=max(float(np.quantile([r["seconds"] for r in pilot],.95)),.15)
                    replay_core_hours=local_cell_seconds/4096*50000*7*3*12/3600
                    prep_core_hours=8.0
                    fit = [json.loads(f.read_text())["training_seconds"] for f in (root/"fits").glob("*/*/FIT_COMPLETE.json")]
                    gpu_hours=float(np.quantile(fit,.95))*7*8*2/3600
                    estimates=dict(replay_cpu_core_hours=replay_core_hours,prep_cpu_core_hours=prep_core_hours,
                                   projected_gpu_hours=gpu_hours,pilot_wall_seconds=perf_counter()-start_pilot,
                                   cpu_limit=32,gpu_limit=8,method="p95 pilot cost extrapolation with 12 policy passes per full cell and 8h prep reserve")
                    atomic_json(root/"RESOURCE_PROJECTION.json",estimates)
                    if replay_core_hours+prep_core_hours > 32 or gpu_hours > 8:
                        state(root,"COMPLETED_SUCCESS","pilot_budget_gate_stopped_full",**estimates)
                        (root/"COMPLETED_SUCCESS").touch()
                        return
                full(root,pool,args,env)
        assemble(root)
        write_report(root)
        state(root,"COMPLETED_SUCCESS",args.stage,elapsed_seconds=perf_counter()-start)
        (root/"COMPLETED_SUCCESS").touch()
    except Exception as error:
        state(root,"FAILED","exception",error=repr(error),elapsed_seconds=perf_counter()-start)
        traceback.print_exc()
        raise


if __name__=="__main__":
    main()
