"""Exact and approximate redundancy, conditional uncertainty, and spectra."""
from collections import Counter, defaultdict
import math

import numpy as np
from scipy.spatial import cKDTree

from math_core import ExactIndex, canonical


def sorted_rows(x,recency_column):
    """Align semantic recency for comparisons. Break residual ties by features."""
    return np.asarray([g[np.lexsort(tuple(g[:,i] for i in range(g.shape[1]-1,-1,-1))+
                                      (g[:,recency_column],))] for g in x])


def summarize_labels(y):
    y=np.asarray(y)
    n,c=y.shape
    best=y.min(1)
    opt=y==best[:,None]
    size=opt.sum(1)
    # Compare ordered pairs for strict differences, excluding self-pairs.
    distinct=np.unique(y)
    strict=n*c*c-sum(int(np.square((y==v).sum(1)).sum()) for v in distinct)
    gaps=y[:,0]-best
    return dict(groups=n,candidate_rows=n*c,all_tied=float((size==c).mean()),unique_optimal=float((size==1).mean()),
                mean_optimal_set_size=float(size.mean()),optimal_set_p50=float(np.median(size)),
                random_optimal_accuracy=float(size.mean()/c),lru_optimal_accuracy=float(opt[:,0].mean()),
                mean_spread=float((y.max(1)-best).mean()),lru_mean_regret=float(gaps.mean()),
                lru_positive_regret=float((gaps>0).mean()),lru_regret_p95=float(np.quantile(gaps,.95)),
                strict_pair_fraction=float(strict/(n*c*max(1,c-1))))


def redundancy(x,targets,times,split=None):
    input_keys=ExactIndex()
    joint=[ExactIndex() for _ in range(targets.shape[-1])]
    cid=[]
    labels=[defaultdict(set) for _ in joint]
    candidate=set()
    candidate_alias=0
    within=0
    same_adjacent=0
    longest=0
    run=0
    old=-1
    for g, y in zip(x,targets):
        key=canonical(g)
        i=input_keys.add(key)
        cid.append(i)
        if i==old:
            same_adjacent+=1
            run+=1
        else:
            run=1
        old=i
        longest=max(longest,run)
        normalized=np.array(g,dtype="<f8",copy=True)
        normalized[normalized==0]=0
        vectors={row.tobytes() for row in normalized}
        within+=len(vectors)<len(g)
        candidate_alias+=len(g)-len(vectors)
        candidate.update(vectors)
        for h,idx in enumerate(joint):
            j=idx.add(canonical(g,y[:,h]))
            labels[h][i].add(j)
    dup=Counter(input_keys.count.values())
    base=dict(groups=len(x),candidate_rows=int(x.shape[0]*x.shape[1]),unique_candidate_vectors=len(candidate),
              unique_input_groups=len(input_keys.count),input_duplicate_rate=1-len(input_keys.count)/len(x),
              within_alias_groups=within,within_alias_rows=candidate_alias,adjacent_duplicate_pairs=same_adjacent,
              longest_run=longest,first_t=int(times[0]),last_t=int(times[-1]),
              multiplicities={str(k):v for k,v in sorted(dup.items())})
    if split is not None:
        seen=defaultdict(set)
        for i,s in zip(cid,split):
            seen[i].add(s)
        base["cross_phase_duplicate_groups"]=sum(len(i)>1 for i in seen.values())
    return base,[dict(unique_input_target_groups=len(j.count),
                      input_target_duplicate_rate=1-len(j.count)/len(x),
                      conflicting_input_groups=sum(len(v)>1 for v in a.values()),
                      groups_in_conflicting_buckets=sum(input_keys.count[i] for i,v in a.items() if len(v)>1))
                 for j,a in zip(joint,labels)]


def spectrum(x,axis_names,scope):
    x=np.asarray(x,dtype=np.float64)
    mu=x.mean(0)
    sd=x.std(0)
    centered=x-mu
    covariance=centered.T@centered
    singular=np.sqrt(np.maximum(0,np.linalg.eigvalsh(covariance)[::-1]))
    prob=singular**2/(np.square(singular).sum() or 1)
    entropy=-float(np.sum(prob[prob>0]*np.log(prob[prob>0])))
    scale=singular[0] if len(singular) else 0
    nz=sd>1e-10
    corr=covariance/np.maximum(1e-300,len(x)*sd[:,None]*sd)
    i,j=np.triu_indices(len(axis_names),1)
    valid=nz[i]&nz[j]
    alias=[(axis_names[a],axis_names[b]) for a,b in zip(i[valid],j[valid]) if np.array_equal(x[:,a],x[:,b])]
    high=[(axis_names[a],axis_names[b],round(float(corr[a,b]),7)) for a,b in zip(i[valid],j[valid])
          if abs(corr[a,b])>=.999]
    return dict(scope=scope,rows=len(x),features=len(axis_names),numerical_rank_1e6=int((singular>scale*1e-6).sum()),
                numerical_rank_1e3=int((singular>scale*1e-3).sum()),
                entropy_rank=float(np.exp(entropy)) if scale else 0,
                stable_rank=float(np.square(singular).sum()/scale**2) if scale else 0,
                constant_columns=[k for k,v in zip(axis_names,sd) if v<=1e-10],
                near_constant=[k for k,v in zip(axis_names,sd) if 1e-10<v<=1e-3],aliases=alias,
                high_corr=high,singular_values=singular.tolist())


def exact_group_keys(x):
    return [canonical(g) for g in x]


def approximate(x, recency, times, late, seed):
    """Fit scales on the chronological reference subset; query on the later subset."""
    g=sorted_rows(x,recency)
    k=g.shape[1]
    ref=g[~late].reshape(-1,g.shape[-1])
    sd=ref.std(0)
    variable=sd>1e-8
    sd[~variable]=1
    scaled=(g-ref.mean(0))/sd
    # Preserve exact normalized distance over all aligned candidates after a summary index search.
    summary=np.concatenate((scaled.mean(1), scaled.std(1),
                            scaled[:,np.linspace(0,k-1,min(8,k),dtype=int),:].reshape(len(g),-1)),axis=1)
    target=np.flatnonzero(late)
    train=np.flatnonzero(~late)
    near=[]
    if len(train) and len(target):
        _, ids=cKDTree(summary[train]).query(summary[target],k=min(12,len(train)),workers=1)
        ids=np.asarray(ids).reshape(len(target),-1)
        for l,cands in zip(target,ids):
            pool=train[cands]
            dist=np.sqrt(np.square(scaled[pool]-scaled[l]).mean(axis=(1,2)))
            j=int(np.argmin(dist))
            near.append((int(l),int(pool[j]),float(dist[j])))
    return scaled,near


def conditional(keys,y,next_at=None,optimal_regret=None,paired_delta=None):
    groups=defaultdict(list)
    for i,key in enumerate(keys):
        groups[key].append(i)
    repeats=[v for v in groups.values() if len(v)>1]
    denom=sum(map(len,repeats))
    variance=entropy=ceiling=opt_disagreement=empty_intersection=0.
    centered_variance=preference_flip=0.
    identified={"reuse":[],"optimal":[],"paired":[]}
    for ids in repeats:
        z=np.asarray(y[ids],dtype=np.float64)
        opts=z==z.min(1,keepdims=True)
        p=opts.mean(0)
        variance+=float(z.var(0).mean()*len(ids))
        centered_variance+=float((z-z[:,0:1]).var(0).mean()*len(ids))
        controls=z-z[:,0:1]
        n,c=z.shape
        h=0.
        for i in range(c):
            probs=np.unique(z[:,i],return_counts=True)[1]/n
            h-=float(np.sum(probs*np.log2(probs)))
        entropy+=h/c*n
        ceiling+=float((z.mean(0).min()-z.min(1).mean())*n)
        opt_disagreement+=float((2*p*(1-p)).mean()*n)
        empty_intersection+=float(not opts.all(0).any())*n
        preference_flip+=float(((controls.min(0)<0)&(controls.max(0)>0)).mean()*n)
        for name,a in (("reuse",next_at),("optimal",optimal_regret),("paired",paired_delta)):
            if a is not None:
                v=np.asarray(a[ids],dtype=np.float64)
                finite=np.isfinite(v).all(0)
                if finite.any():
                    identified[name].append((float(v[:,finite].var(0).mean()),n))
    result=dict(signatures=len(groups),repeated_signatures=len(repeats),rows_in_repeated_signatures=denom)
    for name,x in (("conditional_label_variance",variance),("conditional_entropy_bits",entropy),
                   ("in_sample_min_regret",ceiling),("candidate_optimal_membership_disagreement",opt_disagreement),
                   ("incompatible_optimal_set_fraction",empty_intersection),
                   ("baseline_advantage_variance",centered_variance),("baseline_preference_flip_fraction",preference_flip)):
        result[name]=x/denom if denom else None
    for name,records in identified.items():
        result[name+"_conditional_variance"]=sum(x*n for x,n in records)/sum(n for _,n in records) if records else None
    return result


def predict_conditioned(keys,y,late):
    known=defaultdict(list)
    for i in np.flatnonzero(~late):
        known[keys[i]].append(i)
    test=np.flatnonzero(late)
    supported=0
    lru=0.
    regret=0.
    for i in test:
        ids=known.get(keys[i])
        selected=int(np.argmin(y[ids].mean(0))) if ids else 0
        supported+=bool(ids)
        best=y[i].min()
        regret+=y[i,selected]-best
        lru+=y[i,0]-best
    return dict(heldout_groups=len(test),signature_hits=supported,
                heldout_mean_regret=float(regret/len(test)),heldout_lru_regret=float(lru/len(test)))
