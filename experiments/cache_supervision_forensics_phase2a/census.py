"""Exact November census using byte-verified canonical group equality."""
from collections import Counter,defaultdict
from time import perf_counter,process_time
import resource

import numpy as np

from sources import load_cell,PROTOCOL


def compact_view(a):
    a=np.ascontiguousarray(a)
    flat=a.reshape(len(a),-1)
    return flat.view(np.dtype((np.void,flat.dtype.itemsize*flat.shape[1]))).reshape(len(a))


def sample_indices(family,cap,t,role):
    begin=np.searchsorted(t,10000) if role=="test" else 0
    idx=np.arange(begin,len(t))
    max_groups={"train":2048,"validation":1024,"test":4096}[role]
    rng=np.random.default_rng(np.random.SeedSequence([421,list(PROTOCOL).index(family),cap]))
    return np.sort(rng.choice(idx,max_groups,replace=False)) if len(idx)>max_groups else idx


def run_census(family,cap):
    wall,cpu=perf_counter(),process_time()
    x,y,t,meta=load_cell(family,cap)
    names=meta["columns"]
    n=len(x)
    base=x[0,0].copy()
    variable=np.zeros(len(base),dtype=bool)
    rec=names.index("candidate_recency_rank")
    for i in range(0,n,4096):
        b=x[i:i+4096]
        if not np.array_equal(b[:,:,rec],np.broadcast_to(np.arange(cap),b.shape[:2])):
            raise ValueError("Archived canonical order does not match explicit recency")
        if not np.isfinite(b).all():
            raise ValueError("Nonfinite November features")
        for j in np.flatnonzero(~variable):
            variable[j]=np.any(b[:,:,j]!=base[j])
    z=np.ascontiguousarray(x[:,:,variable],dtype="<f8")
    z[z==0]=0
    compact=compact_view(z)
    _,keys=np.unique(compact,return_inverse=True)
    _,cand=np.unique(compact_view(z.reshape(-1,len(np.flatnonzero(variable)))),return_inverse=True)
    cand=cand.reshape(n,cap)
    joint=[]
    for h in range(y.shape[-1]):
        # Identity of the canonical feature blob plus ordered target vector.
        m=np.empty((n,8+cap),np.uint8)
        m[:,:8]=np.ascontiguousarray(keys,dtype="<u8").view("u1").reshape(n,8)
        m[:,8:]=y[:,:,h]
        _,numeric=np.unique(compact_view(m),return_inverse=True)
        joint.append(numeric)
    scopes={"all":np.arange(n),**{f" November_{role}_sample".strip():sample_indices(family,cap,t,role)
            for role in ("train","validation","test")}}
    result=[]
    for name,ix in scopes.items():
        source=keys[ix]
        unique,multiplicities=np.unique(source,return_counts=True)
        feature_unique=len(np.unique(cand[ix]))
        same=source[1:]==source[:-1]
        breaks=np.r_[0,np.flatnonzero(~same)+1,len(ix)]
        hist={str(k):v for k,v in sorted(Counter(map(int,multiplicities)).items())}
        common=dict(surface="November exact census",representation="november",family=family,capacity=cap,
                    role=name,groups=len(ix),candidate_rows=len(ix)*cap,unique_candidate_vectors=feature_unique,
                    unique_input_groups=len(unique),input_duplicate_rate=1-len(unique)/len(ix),
                    within_alias_groups=0,within_alias_rows=0,adjacent_duplicate_pairs=int(same.sum()),
                    longest_run=int(np.diff(breaks).max()),multiplicities=hist,
                    recency_order_verified=True,constant_columns=[a for a,b in zip(names,variable) if not b],
                    actual_role_split_collision=0,
                    role_note="Each family occupies one role per fold. Role samples here are counterfactual across folds.")
        for h,j in zip((4,8,16),joint):
            pairs=np.unique(np.stack((source,j[ix]),axis=1),axis=0)
            ff,fc=np.unique(pairs[:,0],return_counts=True)
            conflicting=ff[fc>1]
            result.append(dict(**common,horizon=h,unique_input_target_groups=len(pairs),
                        input_target_duplicate_rate=1-len(pairs)/len(ix),conflicting_input_groups=len(conflicting),
                        groups_in_conflicting_buckets=int(np.isin(source,conflicting).sum())))
    cross=[]
    for lhs,rhs in (("train","validation"),("train","test"),("validation","test")):
        a,b=(scopes[f"November_{r}_sample"] for r in (lhs,rhs))
        cross.append(dict(family=family,capacity=cap,role_a=lhs,role_b=rhs,
                          same_family_probe_feature_intersection=len(np.intersect1d(keys[a],keys[b])),
                          same_family_probe_timestamps_intersection=len(np.intersect1d(t[a],t[b])),
                          actual_within_fold_provenance_overlap=0))
    return dict(rows=result,cross=cross,cost=dict(family=family,capacity=cap,cpu_seconds=process_time()-cpu,
                wall_seconds=perf_counter()-wall,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                source_bytes=x.nbytes+y.nbytes+t.nbytes))
