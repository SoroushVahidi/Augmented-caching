"""Exact stateless input-target compression and weighted-objective equivalence."""
from collections import Counter
from time import perf_counter

import numpy as np

from math_core import canonical
from sources import load_cell,PROTOCOL,CAPACITIES


def compress(x,y):
    seen={}
    where=[]
    groups=[]
    counts=[]
    for i,(features,target) in enumerate(zip(x,y)):
        k=canonical(features,target)
        group=seen.get(k)
        if group is None:
            group=len(groups)
            seen[k]=group
            groups.append(i)
            counts.append(0)
        counts[group]+=1
        where.append(group)
    return np.asarray(groups),np.asarray(counts),np.asarray(where)


def check_equivalence():
    import torch
    rng=np.random.default_rng(20261124)
    x=rng.normal(size=(13,7,4))
    y=rng.integers(0,3,size=(13,7))
    multiplicities=np.arange(1,14)
    values=np.repeat(x,multiplicities,axis=0)
    labels=np.repeat(y,multiplicities,axis=0)
    # Reverse alternate candidate rows to validate permutation-invariant loss and grouping.
    values[::2]=values[::2,::-1]
    labels[::2]=labels[::2,::-1]
    ix,weights,_=compress(values,labels)
    theta=torch.tensor([.1,.3,-.6,.2],dtype=torch.float64,requires_grad=True)
    def objectives(a,b,w):
        a=torch.from_numpy(a)
        b=torch.from_numpy(b)
        w=torch.from_numpy(w).double()
        pred=a@theta
        logits=-pred
        optimum=b.amin(1,keepdim=True)
        mask=b==optimum
        ranking=torch.logsumexp(logits,1)-torch.logsumexp(logits.masked_fill(~mask,-torch.inf),1)
        expected=((b-optimum)*logits.softmax(1)).sum(1)
        pref=b[:,:,None]<b[:,None,:]
        pairs=torch.nn.functional.softplus(pred[:,:,None]-pred[:,None,:])*pref
        pair=pairs.sum((1,2))/pref.sum((1,2)).clamp_min(1)
        regression=(pred-b).square().mean(1)
        out={}
        for n,v in (("pointwise_MSE",regression),("optimal_set_CE",ranking),("pairwise_logistic",pair),
                    ("expected_regret",expected),("decision_normalized_joint",regression+ranking+.5*expected)):
            loss=(v*w).sum()/w.sum()
            g,=torch.autograd.grad(loss,theta,retain_graph=True)
            out[n]=(float(loss.detach()),g.detach().numpy())
        return out
    full=objectives(values,labels,np.ones(len(values),dtype=np.int64))
    reduced=objectives(values[ix],labels[ix],weights)
    result={n:dict(loss_delta=abs(full[n][0]-reduced[n][0]),gradient_max_abs_delta=float(np.max(
            np.abs(full[n][1]-reduced[n][1])))) for n in full}
    if any(max(v.values())>1e-10 for v in result.values()):
        raise ValueError("Weighted objective mismatch")
    return result


def compression_pilot(max_per_cell=700):
    records=[]
    for family in PROTOCOL:
        for cap in CAPACITIES:
            x,y,_,_=load_cell(family,cap)
            ids=np.linspace(0,len(x)-1,min(max_per_cell,len(x)),dtype=int)
            X=np.asarray(x[ids]); Y=np.asarray(y[ids,:,2])
            start=perf_counter()
            ix,weight,_=compress(X,Y)
            elapsed=perf_counter()-start
            raw=X.nbytes+Y.nbytes
            packed=X[ix].nbytes+Y[ix].nbytes+weight.nbytes
            def read_equivalent(z,w):
                begin=perf_counter()
                out=0.
                for _ in range(15):
                    out+=float(np.dot(z[:,:,4].sum(1),w))
                return perf_counter()-begin,out
            a,av=read_equivalent(X,np.ones(len(X)))
            b,bv=read_equivalent(X[ix],weight)
            if not np.isclose(av,bv,rtol=1e-12):
                raise ValueError("Weighted group scan disagrees")
            records.append(dict(family=family,capacity=cap,rows=len(ids),unique=len(ix),raw_bytes=raw,
                                compressed_payload_bytes=packed,construction_seconds=elapsed,
                                uncompressed_memory_scan_seconds=a,compressed_weighted_scan_seconds=b,
                                kish_unique_weight_ess=float(weight.sum()**2/np.square(weight).sum())))
    return dict(horizon=16,tests=check_equivalence(),cell_pilots=records,
                measurement="In-memory replay of aligned candidate-rank sums, no serialized data-loading benchmark",
                passed=True,optimizer_note="Full-batch empirical equivalence does not preserve SGD trajectories.")


def sampling_arms(multiplicity, family, density, baseline_regret, *, beta=0.99):
    """Return aggregate weights per compressed row for unlaunched stateless arms.

    Density counts derive from training-only aligned decision signatures. Returning
    aggregate weights lets a sampler apply each arm directly to compressed records.
    """
    count = np.asarray(multiplicity, dtype=float)
    family = np.asarray(family)
    density = np.asarray(density, dtype=float)
    regret = np.asarray(baseline_regret, dtype=float)
    if count.ndim != 1 or count.size == 0 or any(
        a.shape != count.shape for a in (family, density, regret)
    ):
        raise ValueError("Require equally sized nonempty decision arrays")
    if (not np.isfinite(count).all() or not np.isfinite(density).all()
        or not np.isfinite(regret).all() or np.any(count < 1) or np.any(density < count)
        or np.any(regret < 0) or not 0 < beta < 1):
        raise ValueError("Invalid multiplicity, density, regret or beta")
    _, labels = np.unique(family, return_inverse=True)
    totals = np.bincount(labels, weights=count)
    effective_count = np.expm1(density * np.log(beta)) / (beta - 1)
    weights = {
        "natural_frequency": count,
        "multiplicity_preserving": count,
        "regret": count * (1 + regret),
        "inverse_density": count / density,
        "effective_sample": count / effective_count,
        "family_balanced": count / totals[labels],
    }
    return {name: arm / arm.sum() for name, arm in weights.items()}


def kish_n(weights):
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.any(weights > 0):
        raise ValueError("Require finite nonnegative weights with positive sum")
    return float(weights.sum()**2 / np.square(weights).sum())
