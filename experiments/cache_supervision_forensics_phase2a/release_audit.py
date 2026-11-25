"""Sample complete public-release candidate groups and check physical split provenance."""
from collections import defaultdict
from pathlib import Path
import time
import zlib

import duckdb
import numpy as np
import pyarrow.parquet as pq

from redundancy import conditional,redundancy,exact_group_keys
from sources import LAFC,SEED,load_cell

PHYSICAL=("trace_name","trace_family","dataset_source","capacity","horizon","decision_id","decision_t","decision_chunk_id")


def public_partition(path,per_rowgroup=28):
    start=time.perf_counter()
    pf=pq.ParquetFile(path)
    params=dict(point.split("=",1) for point in path.parent.relative_to(LAFC/"data/candidate_rows").parts)
    family=params["trace_family"]; cap=int(params["capacity"]); horizon=int(params["horizon"])
    features=load_cell(family,32)[3]["columns"]
    selected=[]
    rng=np.random.default_rng(SEED+cap+horizon)
    rowgroups=np.unique(np.linspace(0,pf.num_row_groups-1,min(3,pf.num_row_groups),dtype=int))
    edge_discards=0
    for rg in rowgroups:
        table=pf.read_row_group(int(rg),columns=[*PHYSICAL,"candidate_page_id","y_loss",*features],use_threads=False)
        key=table["decision_id"].to_numpy()
        edges=np.r_[0,np.flatnonzero(key[1:]!=key[:-1])+1,len(key)]
        groups=[]
        for lo,hi in zip(edges[:-1],edges[1:]):
            if hi-lo!=cap:
                edge_discards+=1
                continue
            groups.append((lo,hi))
        ix=rng.choice(len(groups),min(per_rowgroup,len(groups)),replace=False)
        for k in sorted(ix):
            lo,hi=groups[k]
            g=table.slice(int(lo),int(hi-lo))
            physical=tuple(g[x][0].as_py() for x in PHYSICAL)
            if any(not np.all(g[x].to_numpy()==physical[i]) for i,x in enumerate(PHYSICAL)):
                raise ValueError(f"Physical group collision {physical}")
            if len(set(g["candidate_page_id"].to_pylist()))!=cap:
                raise ValueError(f"Repeated candidate ID {physical}")
            x=np.column_stack([g[f].to_numpy() for f in features])
            y=g["y_loss"].to_numpy()
            ordered=np.argsort(x[:,features.index("candidate_recency_rank")],kind="stable")
            selected.append((physical,x[ordered],y[ordered]))
    selected=sorted({k:(k,x,y) for k,x,y in selected}.values(),key=lambda e:(int(e[0][6]),str(e[0])))
    if not selected:
        return dict(path=str(path),population_rows=pf.metadata.num_rows,rows=[],alias=[],cpu_wall_seconds=time.perf_counter()-start)
    keys,x,y=zip(*selected)
    t=np.asarray([k[6] for k in keys],dtype=np.int64)
    x=np.asarray(x); y=np.asarray(y)
    b,j=redundancy(x,y[...,None],t)
    meta=dict(family=family,capacity=cap,horizon=horizon,role=params["split"],
              surface="LAFC v1.0 row-group stratified sample",representation="public_legacy_features",
              partition=path.relative_to(LAFC).as_posix(),population_candidate_rows=pf.metadata.num_rows,
              sampled_row_groups=len(rowgroups),partial_or_noncapacity_groups_skipped=edge_discards)
    a=conditional(exact_group_keys(x),y)
    return dict(path=str(path),population_rows=pf.metadata.num_rows,rows=[dict(**meta,**b,**j[0])],
                alias=[dict(**meta,signature="exact",**a)],
                signatures=[zlib.compress(k,1).hex() for k in exact_group_keys(x)],cpu_wall_seconds=time.perf_counter()-start)


def public_split_census():
    path=LAFC/"data/decision_view/decision_view.parquet"
    cols=", ".join(PHYSICAL)
    with duckdb.connect(":memory:",config={"threads":2}) as db:
        physical="SELECT "+cols+", count(distinct split) n FROM read_parquet(?) GROUP BY "+cols
        p=db.execute("SELECT count(*),count(*) FILTER (WHERE n>1) FROM ("+physical+")",[str(path)]).fetchone()
        # Slot identity disregards horizon and nominal generated identifiers.
        stmt="""SELECT count(*),count(*) FILTER (WHERE n>1) FROM
          (SELECT trace_name,trace_family,dataset_source,capacity,decision_t,count(distinct split) n
           FROM read_parquet(?) GROUP BY ALL)"""
        t=db.execute(stmt,[str(path)]).fetchone()
        totals=db.execute("SELECT split,count(*) FROM read_parquet(?) GROUP BY split ORDER BY split",[str(path)]).fetchall()
        timing=db.execute("""SELECT trace_family,trace_name,capacity,horizon,split,min(decision_t),max(decision_t),count(*)
                           FROM read_parquet(?) GROUP BY ALL ORDER BY ALL""",[str(path)]).fetchall()
    return dict(source=str(path),exact_physical_groups=p[0],cross_split_physical_keys=p[1],
                horizon_agnostic_timestamps=t[0],cross_split_horizon_agnostic_timestamps=t[1],
                decisions_by_split=dict(totals),window_intervals=timing)
