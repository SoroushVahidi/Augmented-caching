"""Resumable Phase 2A audit with immutable per-unit evidence."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
import csv
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
from time import perf_counter,process_time
import traceback

import numpy as np
from cell_audit import run_cell
from census import run_census
from compression import compression_pilot
from phase1_final import markdown,validate
from redundancy import spectrum
from release_audit import public_partition,public_split_census
from sources import (CAPACITIES,GLOBAL_NAMES,LAFC,NOVEMBER,PHASE1,PROTOCOL,REPO,SEED,STATIC_NAMES,
                     TIE_DB,TRACES,digest,load_cell)

CSV_NAMES=dict(redundancy="DECISION_GROUP_REDUNDANCY",near="NEAR_DUPLICATE_ANALYSIS",features="FEATURE_RANK_AUDIT",
              alias="FEATURE_ALIASING_ANALYSIS",horizon="SUPERVISION_HORIZON_ANALYSIS",optimal="OPTIMAL_CONTINUATION_REGRET",
              paired="PAIRED_ROLLOUT_RESULTS",shortlist="SHORTLIST_ORACLE_RESULTS",censor="CENSORING_AUDIT",
              actions="ACTION_LEVEL_EVIDENCE",state="REUSE_DELAY_EVIDENCE",distribution="LABEL_DISTRIBUTIONS",
              cost="UNIT_COSTS",cross="CROSS_SPLIT_OVERLAP")


def now():
    return datetime.now(timezone.utc).isoformat()


def jsonable(x):
    if isinstance(x,np.generic):
        return jsonable(x.item())
    if isinstance(x,np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x,dict):
        return {str(k):jsonable(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):
        return [jsonable(v) for v in x]
    if isinstance(x,Path):
        return str(x)
    return x


def atomic(path,data):
    t=path.with_suffix(path.suffix+".tmp")
    with t.open("w") as fd:
        json.dump(jsonable(data),fd,allow_nan=False,separators=(",",":"))
        fd.write("\n")
    os.replace(t,path)


def dump_csv(path,rows):
    columns=list(dict.fromkeys(key for row in rows for key in row))
    temp=path.with_suffix(".tmp")
    with temp.open("w",newline="") as fd:
        writer=csv.DictWriter(fd,columns or ["status"],lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(jsonable(v),separators=(",",":")) if isinstance(v,(dict,list,tuple)) else v
                             for k,v in row.items()})
    os.replace(temp,path)


def version(repo):
    return subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()


def tasklist(args):
    cells=[(f,c) for f in PROTOCOL for c in CAPACITIES]
    lafc=sorted((LAFC/"data/candidate_rows").glob("**/*.parquet"))
    return ([("state",f"{f}-{c}",(f,c,args.decisions,args.oracle,args.enriched)) for f,c in cells]
            +[("census",f"{f}-{c}",(f,c)) for f,c in cells]
            +[("public",str(i).zfill(3),(p,args.public_per_rowgroup)) for i,p in enumerate(lafc)])


def unit(out,task):
    kind,key,params=task
    folder=out/"units"/kind
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/f"{key}.json"
    if path.exists():
        saved=json.loads(path.read_text())
        if saved["parameters"]!=jsonable(params):
            raise ValueError(f"Existing unit has different parameters: {path}")
        return kind,key,"reused"
    cpu=process_time()
    start=perf_counter()
    if kind=="state":
        rows,matrix,sample,signatures=run_cell(*params)
        values=dict(rows=rows,sample=sample,signatures=signatures)
        tmp=folder/f"{key}.tmp.npz"
        np.savez_compressed(tmp,**matrix)
        os.replace(tmp,folder/f"{key}.npz")
    elif kind=="census":
        values=run_census(*params)
    else:
        values=public_partition(*params)
    atomic(path,dict(kind=kind,parameters=params,completed_at=now(),source_revision=version(REPO),cpu_seconds=process_time()-cpu,
                     wall_seconds=perf_counter()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**values))
    return kind,key,"completed"


def provenance(out):
    import duckdb,numba,pyarrow,scipy
    manifests={}
    for f in PROTOCOL:
        for c in CAPACITIES:
            _,_,_,m=load_cell(f,c)
            folder=NOVEMBER/"runs/campaign/data"/f"{f}-{c}"
            manifests[f"{f}/{c}"]=dict(**m,manifest_sha256=digest(folder/"manifest.json"),
                                        byte_counts={a:(folder/a).stat().st_size for a in ("x.npy","y.npy","t.npy")})
    phase1_manifest=PHASE1/"checkpoint_manifest.json"
    project=REPO.parents[1]
    kbs=project/"worktrees/kbs-second-revision"
    cited=["src/lafc/offline/belady_uniform.py","src/lafc/policies/offline_belady.py",
           "src/lafc/evict_value_v2_rollout.py","src/lafc/reuse_tail_horizon.py","src/lafc/lrb_model.py",
           "src/lafc/policies/evict_value_v1_selective.py","src/lafc/policies/guard_wrapper.py",
           "src/lafc/policies/sentinel_budgeted_guard_v2.py"]
    source_assets={n:dict(path=kbs/n,sha256=digest(kbs/n)) for n in cited if (kbs/n).is_file()}
    with duckdb.connect(str(TIE_DB),read_only=True,config={"threads":1}) as db:
        tie_tables=[x[0] for x in db.execute("show tables").fetchall()]
    phase1_terminal=validate()
    (out/"PHASE1_FINAL_AUDIT.md").write_text(markdown(phase1_terminal))
    atomic(out/"PHASE1_TERMINAL.json",phase1_terminal)
    sha={f:digest(TRACES/f/"trace.jsonl") for f in PROTOCOL}
    if sha!={f:v["sha256"] for f,v in PROTOCOL.items()}:
        raise ValueError("KBS trace changed")
    details=dict(source=version(REPO),base="015aa0d6065aa23a1b73a0d5bc48f352b0424cdb",seed=SEED,
       environment=dict(python=sys.version.split()[0],numpy=np.__version__,numba=numba.__version__,scipy=scipy.__version__,
                        pyarrow=pyarrow.__version__,duckdb=duckdb.__version__,host=platform.node(),logical_cpus=os.cpu_count()),
       november=dict(path=NOVEMBER,source="6c010cb2b8983f5b845bd05e75658962e655db60",horizons=[4,8,16],
                     roles=PROTOCOL,trace_sha256=sha,stimulus="full-miss LRU; forced victim then LRU",
                     continuation_time="starts at t+1 and truncates at trace end",cell_manifests=manifests,
                     old_sampler_seed=421),
       phase1=dict(path=PHASE1,source="fae8c8f5d5654b3d41537f3d19284d48c9240a55",
                   native_inputs="expanded 50 candidate and 14 causal request columns",
                   checkpoint_manifest_sha256=digest(phase1_manifest) if phase1_manifest.exists() else None,
                   retrospective_trace_lineage=sha,stage=phase1_terminal["status"]),
       lafc=dict(path=LAFC,repository_commit=version(LAFC.parents[1]),release_manifest_sha256=digest(LAFC/"metadata/release_manifest.json"),
                 row_counts=json.loads((LAFC/"metadata/release_manifest.json").read_text())["row_counts"],
                 scope="five-family temporal Parquet release on five shared KBS trace identifiers, distinct from KBS LOFO"),
       tie_database=dict(path=TIE_DB,bytes=TIE_DB.stat().st_size,tables=tie_tables,opened_read_only=True,
                         scope="preserved LAFC joined and selector evidence, no training use"),
       kbs_support=dict(commit=version(kbs),files=source_assets,linked_prior_analyses=[
                    str(kbs/"analysis/practical_significance_ablation_v1"),
                    str(kbs/"analysis/continuation_policy_causal_ablation_production_v1")]),
       confirmation="Protected. No paths, traces or outcomes read.")
    atomic(out/"DATA_PROVENANCE_MANIFEST.json",details)


def reduce(out):
    result=defaultdict(list)
    usage=[]
    rank_source=defaultdict(list)
    lafc_scopes=defaultdict(set)
    sampled={}
    sample_receipts={}
    for f in sorted((out/"units/state").glob("*.json")):
        d=json.loads(f.read_text())
        for key,rows in d["rows"].items():
            result[key].extend(rows)
        family,cap,*_=d["parameters"]
        sampled[(family,cap)]=set(d["signatures"]["november"])
        sample_receipts[f"{family}/{cap}"]=d["sample"]
        with np.load(f.with_suffix(".npz")) as block:
            for name in block.files:
                for scope in ("pooled","family_"+family,"capacity_"+str(cap)):
                    rank_source[(name,scope)].append(block[name])
        usage.append([d["kind"],d["cpu_seconds"],d["wall_seconds"],d["peak_rss_kib"]])
    for f in sorted((out/"units/census").glob("*.json")):
        d=json.loads(f.read_text())
        result["redundancy"].extend(d["rows"])
        result["cross"].extend(dict(surface="November cross-role counterfactual within family",**x) for x in d["cross"])
        usage.append([d["kind"],d["cpu_seconds"],d["wall_seconds"],d["peak_rss_kib"]])
    for f in sorted((out/"units/public").glob("*.json")):
        d=json.loads(f.read_text())
        result["redundancy"].extend(d["rows"])
        result["alias"].extend(d["alias"])
        for r in d["rows"]:
            lafc_scopes[(r["family"],r["capacity"],r["horizon"],r["role"])].update(d["signatures"])
        usage.append([d["kind"],d["cpu_seconds"],d["wall_seconds"],d["peak_rss_kib"]])
    for (f,c,h,a),keys in sorted(lafc_scopes.items()):
        for b in ("val","test") if a=="train" else ("test",) if a=="val" else ():
            other=lafc_scopes.get((f,c,h,b))
            if other:
                result["cross"].append(dict(surface="LAFC stratified feature equality sample",family=f,capacity=c,horizon=h,
                                            role_a=a,role_b=b,unique_a=len(keys),unique_b=len(other),
                                            feature_intersection=len(keys&other)))
    for f,spec in PROTOCOL.items():
        for c in CAPACITIES:
            if not all((x,c) in sampled for x in PROTOCOL):
                continue
            train=set.union(*(sampled[(k,c)] for k in spec["train"]))
            for a,b,x,y in (("train","test",train,sampled[f,c]),("train","validation",train,sampled[spec["validation"],c]),
                            ("validation","test",sampled[spec["validation"],c],sampled[f,c])):
                result["cross"].append(dict(surface="November matched sample across LOFO families",family=f,capacity=c,
                    role_a=a,role_b=b,unique_a=len(x),unique_b=len(y),feature_intersection=len(x&y),
                    actual_within_fold_provenance_overlap=0))
    names=load_cell("brightkite",32)[3]["columns"]
    for (name,scope),blocks in rank_source.items():
        columns={"november":names,"expanded":STATIC_NAMES,"expanded_request":GLOBAL_NAMES}[name]
        z=np.concatenate(blocks)
        result["features"].append(dict(representation=name,**spectrum(z,columns,scope),
                                       matrix_sample="balanced row subsample per cell"))
    for key,filename in CSV_NAMES.items():
        dump_csv(out/(filename+".csv"),result[key])
    atomic(out/"SAMPLE_RECEIPTS.json",sample_receipts)
    atomic(out/"UNIT_RESOURCE_ROWS.json",usage)


def command():
    cli=argparse.ArgumentParser()
    cli.add_argument("stage",choices=("all","one","reduce","phase1"))
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--workers",type=int,default=6)
    cli.add_argument("--decisions",type=int,default=320)
    cli.add_argument("--oracle",type=int,default=56)
    cli.add_argument("--enriched",type=int,default=24)
    cli.add_argument("--public-per-rowgroup",type=int,default=28)
    cli.add_argument("--unit",type=int)
    args=cli.parse_args()
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=="reduce":
        reduce(out)
        return
    if args.stage=="phase1":
        data=validate()
        (out/"PHASE1_FINAL_AUDIT.md").write_text(markdown(data))
        atomic(out/"PHASE1_TERMINAL.json",data)
        return
    tasks=tasklist(args)
    if args.stage=="one":
        if args.unit is None:
            cli.error("one requires --unit")
        print(unit(out,tasks[args.unit]),flush=True)
        return
    start=perf_counter()
    status=out/"RUN_STATUS.json"
    def set_status(state,**more):
        atomic(status,dict(status=state,at=now(),pid=os.getpid(),source=version(REPO),**more))
    set_status("RUNNING_HEALTHY",stage="input_validation",units=len(tasks))
    try:
        provenance(out)
        atomic(out/"EXECUTION_CONFIG.json",dict(arguments=vars(args),tasks=tasks,seed=SEED))
        completed=0
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            pending=[pool.submit(unit,out,t) for t in tasks]
            for f in as_completed(pending):
                kind,name,state=f.result()
                completed+=1
                print(now(),kind,name,state,flush=True)
                if completed%10==0:
                    set_status("RUNNING_HEALTHY",stage="audit_units",completed=completed,units=len(tasks))
        set_status("RUNNING_HEALTHY",stage="reduce",completed=completed,units=len(tasks))
        atomic(out/"LAFC_SPLIT_CENSUS.json",public_split_census())
        atomic(out/"COMPRESSION_VALIDATION.json",compression_pilot())
        reduce(out)
        elapsed=perf_counter()-start
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        own=resource.getrusage(resource.RUSAGE_SELF)
        atomic(out/"RESOURCE_ACCOUNTING.json",dict(local_wall_seconds=elapsed,
             local_process_cpu_seconds=own.ru_utime+own.ru_stime+usage.ru_utime+usage.ru_stime,
             largest_child_peak_rss_kib=usage.ru_maxrss,supervisor_peak_rss_kib=own.ru_maxrss,
             local_gpu_training_seconds=0,whitelisted_workloads=list(PROTOCOL),
             wulver_status="Authentication verification blocked. No jobs submitted.",
             wulver_job_ids=[],workers=args.workers,planned_units=len(tasks),completed_units=completed))
        set_status("COMPLETED_SUCCESS",stage="bounded_audit",completed=completed,units=len(tasks),wall_seconds=elapsed)
        (out/"COMPLETED_SUCCESS").touch()
    except Exception as error:
        set_status("FAILED",stage="audit_exception",error=repr(error))
        traceback.print_exc()
        raise


if __name__=="__main__":
    command()
