"""Exact within-partition candidate census for the public LAFC release."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import json
from pathlib import Path
import resource
from time import perf_counter, process_time

import numpy as np
import pyarrow.parquet as pq

from census import compact_view
from sources import LAFC, load_cell
from run import atomic


def canonical_order(t, rank, cap):
    """Recover complete decisions, including those split by physical row order."""
    if np.any(rank < 0) or np.any(rank >= cap) or np.any(rank != rank.astype("i8")):
        raise ValueError("Invalid candidate recency ranks")
    order = np.argsort(t * cap + rank.astype("i8"), kind="stable")
    times = t[order].reshape(-1, cap)
    if not np.all(times == times[:, :1]) or not np.all(rank[order].reshape(-1, cap) == np.arange(cap)):
        raise ValueError("Partition fails complete and uniquely ranked decision check")
    return order, times[:, 0].copy()


def exact_partition(path):
    start, cpu = perf_counter(), process_time()
    relative = path.relative_to(LAFC)
    spec = dict(part.split("=", 1) for part in relative.parent.parts if "=" in part)
    cap, horizon, family = int(spec["capacity"]), int(spec["horizon"]), spec["trace_family"]
    names = load_cell(family, 32)[3]["columns"]
    physical_names = ["decision_id", "trace_name", "dataset_source", "trace_family", "split"]
    parquet = pq.ParquetFile(path, read_dictionary=physical_names)
    table = parquet.read(columns=[*physical_names, "decision_t", "decision_chunk_id", "capacity",
                                  "horizon", "y_loss", "y_value", *names], use_threads=False)
    t, rank = (table[k].to_numpy() for k in ("decision_t", "candidate_recency_rank"))
    order, times = canonical_order(t, rank, cap)
    n = len(times)
    for key in ("trace_name", "dataset_source", "trace_family", "split"):
        if len(table[key].combine_chunks().dictionary) != 1:
            raise ValueError(f"{relative} has multiple {key} values")
    for key,value in (("capacity",cap),("horizon",horizon)):
        if not np.all(table[key].to_numpy() == value):
            raise ValueError(f"{relative} has inconsistent {key}")
    ids = table["decision_id"].combine_chunks().indices.to_numpy()[order].reshape(n, cap)
    if not np.all(ids == ids[:, :1]) or len(np.unique(ids[:, 0])) != n:
        raise ValueError(f"{relative} has decision identity collisions")
    chunks = table["decision_chunk_id"].to_numpy()[order].reshape(n, cap)
    if not np.all(chunks == chunks[:, :1]):
        raise ValueError(f"{relative} has decisions crossing provenance chunks")
    del ids,chunks
    varied = []
    constants = []
    for name in names:
        column = table[name].to_numpy()
        if not np.isfinite(column).all():
            raise ValueError(f"{relative} contains nonfinite {name}")
        (constants if (column == column[0]).all() else varied).append(name)
    z=np.empty((len(t),len(varied)),dtype="<f8")
    for j,name in enumerate(varied):
        z[:,j]=table[name].to_numpy()[order]
    z[z==0]=0
    loss = table["y_loss"].to_numpy()[order].reshape(n, cap)
    target_value = table["y_value"].to_numpy()[order].reshape(n, cap)
    if not np.isfinite(loss).all() or not np.array_equal(-loss,target_value):
        raise ValueError(f"{relative} target identities differ")
    if not np.all((loss>=0)&(loss<=255)&(loss==loss.astype(np.uint8))):
        raise ValueError(f"{relative} target cannot be represented exactly as a byte")
    del table, target_value
    candidate_count = len(np.unique(compact_view(z)))
    _,keys,counts=np.unique(compact_view(z.reshape(n,cap,-1)),return_inverse=True,return_counts=True)
    joint=np.empty((n,8+cap),dtype=np.uint8)
    joint[:,:8]=np.ascontiguousarray(keys,dtype="<u8").view("u1").reshape(n,8)
    joint[:,8:]=loss
    joint_groups=np.unique(compact_view(joint),return_index=True)[1]
    classes,labels=np.unique(keys[joint_groups],return_counts=True)
    conflicts=classes[labels>1]
    adjacent=(keys[1:]==keys[:-1])&(times[1:]==times[:-1]+1)
    return dict(partition=relative.as_posix(),family=family,capacity=cap,horizon=horizon,role=spec["split"],
                candidate_rows=len(t),groups=n,unique_candidate_vectors=candidate_count,
                unique_input_groups=len(counts),input_duplicate_rate=1-len(counts)/n,
                unique_input_target_groups=len(joint_groups),input_target_duplicate_rate=1-len(joint_groups)/n,
                conflicting_input_groups=len(conflicts),groups_in_conflicting_buckets=int(counts[conflicts].sum()),
                within_alias_rows=0,within_alias_groups=0,constant_columns=constants,
                successive_request_duplicate_pairs=int(adjacent.sum()),
                longest_successive_request_run=int(np.diff(np.r_[0,np.flatnonzero(~adjacent)+1,n]).max()),
                multiplicities=dict(sorted(Counter(map(int,counts)).items())),
                wall_seconds=perf_counter()-start,cpu_seconds=process_time()-cpu,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def run(out,workers):
    with (out/"EXECUTION_CONFIG.json").open() as src:
        json.load(src)
    partitions=sorted((LAFC/"data/candidate_rows").rglob("*.parquet"))
    receipts=out/"units"/"public_exact"
    receipts.mkdir(parents=True,exist_ok=True)
    start=perf_counter()
    atomic(out/"PUBLIC_EXACT_STATUS.json",dict(state="RUNNING_HEALTHY",partitions=len(partitions)))
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending={pool.submit(exact_partition,p):i for i,p in enumerate(partitions)
                     if not (receipts/f"{i:03d}.json").exists()}
            for future in as_completed(pending):
                i=pending[future]
                atomic(receipts/f"{i:03d}.json",future.result())
                print(i,partitions[i].parent.relative_to(LAFC),flush=True)
        records=[json.loads((receipts/f"{i:03d}.json").read_text()) for i in range(len(partitions))]
        physical=json.loads((out/"LAFC_SPLIT_CENSUS.json").read_text())
        if sum(x["groups"] for x in records)!=physical["exact_physical_groups"]:
            raise ValueError("Candidate and decision view group census differ")
        with (out/"PUBLIC_EXACT_CENSUS.csv").open("w") as dest:
            csvout=csv.DictWriter(dest,fieldnames=list(records[0]),lineterminator="\n")
            csvout.writeheader()
            for row in records:
                csvout.writerow({k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()})
        summary={key:sum(x[key] for x in records) for key in (
            "candidate_rows","groups","unique_input_groups","unique_input_target_groups","groups_in_conflicting_buckets","cpu_seconds")}
        summary.update(state="COMPLETED_SUCCESS",partitions=len(partitions),workers=workers,
                       largest_worker_peak_rss_kib=max(x["peak_rss_kib"] for x in records),wall_seconds=perf_counter()-start)
        atomic(out/"PUBLIC_EXACT_STATUS.json",summary)
    except Exception as error:
        atomic(out/"PUBLIC_EXACT_STATUS.json",dict(state="FAILED",error=str(error),
                    completed=len(list(receipts.glob("*.json"))),wall_seconds=perf_counter()-start))
        raise


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,default=Path("runs/phase2a"))
    parser.add_argument("--workers",type=int,default=5)
    args=parser.parse_args()
    run(args.out,args.workers)
