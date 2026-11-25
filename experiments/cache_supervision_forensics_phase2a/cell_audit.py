"""A01–A09 on paired causal KBS decisions and stratified forced-action states."""
from collections import Counter
import zlib
from time import perf_counter,process_time
import resource

import numpy as np
from scipy.stats import spearmanr

from math_core import canonical,censored_reuse,forced_belady,initial_next,paired_lru,shortlist_hindsight
from redundancy import approximate,conditional,exact_group_keys,predict_conditioned,redundancy,spectrum,summarize_labels
from sources import (GLOBAL_NAMES,STATIC_NAMES,SEED,frozen_predictions,load_cell,observed_states,
                     stratified,stream_for)

FIXED=(4,8,16,32)
SCALES=(1,2,4,8,16,32)
SHORTLISTS=(2,4,8,12,16,32)
AUDIT_KEYS={"redundancy":[],"near":[],"features":[],"alias":[],"horizon":[],"optimal":[],
            "paired":[],"shortlist":[],"censor":[],"cost":[],"actions":[],"state":[],"distribution":[]}


def make_lists(x,state,rng):
    t=perf_counter()
    n=len(x)
    rank=np.arange(n)
    # Baseline's actual victim stays in every override shortlist.
    candidates={
        "lru_tail":rank,
        "sieve_derived":np.lexsort((rank,state["sieve_flags"])),
        "low_frequency":np.lexsort((rank,x[:,1])),
        "recency_frequency":np.argsort(rank/max(1,n-1)+x[:,7],kind="stable"),
        "ghost_informed":np.lexsort((rank,x[:,1],x[:,10])),
        "causal_mixture":np.argsort(rank/max(1,n-1)+x[:,7]+x[:,8]/3,kind="stable"),
        "uniform":rng.permutation(n),
    }
    if state["native_sieve"]>=0:
        q=state["native_sieve"]
        candidates["sieve_derived"]=np.r_[q,candidates["sieve_derived"][candidates["sieve_derived"]!=q]]
    return {m:np.r_[0,v[v!=0]] for m,v in candidates.items()},(perf_counter()-t)*1e6


def row_features(x,g,names,meta):
    flat=x.reshape(-1,x.shape[-1])
    # Fit preprocessing from the chronological reference side of the same cell.
    limit=max(1,int(len(x)*.65))*x.shape[1]
    ref=flat[:limit]
    std=ref.std(0,dtype=np.float64)
    std[std<1e-8]=1
    normalized=(flat-ref.mean(0,dtype=np.float64))/std
    between=x.mean(1)
    within=(x-between[:,None]).reshape(flat.shape)
    values=[spectrum(flat,names,"cell_raw"),spectrum(normalized,names,"cell_early_standardized"),
            spectrum(within,names,"within_decision_centered"),spectrum(between,names,"decision_means")]
    if meta["representation"]=="expanded":
        names_log={"age","count","gap_mean","gap_variance","gap_min","gap_max","decayed_popularity",
                   "evictions","reinsertions","age_since_eviction",*(f"gap_{i}" for i in range(1,17))}
        ix=[i for i,n in enumerate(names) if n in names_log]
        unlog=flat.astype(np.float64)
        unlog[:,ix]=np.expm1(unlog[:,ix])
        values.append(spectrum(unlog,names,"cell_inverse_log"))
        for q in values:
            q["lag1_missing"]=float(1-flat[:,34].mean())
            q["lag8_missing"]=float(1-flat[:,41].mean())
            q["lag16_missing"]=float(1-flat[:,49].mean())
    return [dict(**meta,**v) for v in values]


def alias_by_rep(x,y,reuse,times,recency,meta,rows):
    late=times>=int(times.min()+.65*(times.max()-times.min()))
    keys=exact_group_keys(x)
    for i,h in enumerate((4,8,16)):
        labels=y[:,:,i]
        q=conditional(keys,labels,reuse)
        q.update(predict_conditioned(keys,labels,late))
        rows["alias"].append(dict(**meta,horizon=h,signature="exact",**q))
    z,near=approximate(x,recency,times,late,SEED)
    data=y[:,:,2]
    for width in (.05,.15,.35,.75):
        quant=np.rint(z/width).astype(np.int32)
        qkeys=[row.tobytes() for row in quant]
        rows["alias"].append(dict(**meta,horizon=16,signature=f"quantized_{width}",
                                  **conditional(qkeys,data,reuse),**predict_conditioned(qkeys,data,late)))
    for bound in (.01,.05,.15,.35):
        p=[(i,j,d) for i,j,d in near if d<=bound]
        membership=[]; incompatible=[]; regret=[]; variance=[]; reuse_diff=[]
        for i,j,d in p:
            a,b=data[i],data[j]
            oa=a==a.min(); ob=b==b.min()
            membership.append(np.mean(oa!=ob))
            incompatible.append(not (oa&ob).any())
            regret.append(float(a[np.argmin(b)]-a.min()))
            variance.append(float(np.square(a.astype(float)-b).mean()/4))
            obs=np.isfinite(reuse[i])&np.isfinite(reuse[j])
            if obs.any():
                reuse_diff.append(float(np.abs(reuse[i,obs]-reuse[j,obs]).mean()))
        distribution={str(k):v for k,v in sorted(Counter(j for i,j,d in p).items())}
        # Count neighbors with adequate separation and those in the same or adjacent 4096 chunk.
        rows["near"].append(dict(**meta,threshold=bound,heldout=len(near),accepted=len(p),
                                  unique_reference_neighbors=len(distribution),neighbor_loads=list(distribution.values()),
                                  nearest_distance_median=float(np.median([d for _,_,d in near])) if near else None,
                                  median_request_separation=float(np.median([times[i]-times[j] for i,j,_ in p])) if p else None,
                                  same_or_adjacent_chunk=sum(times[i]//4096-times[j]//4096<=1 for i,j,_ in p),
                                  mean_optimal_membership_disagreement=float(np.mean(membership)) if p else None,
                                  incompatible_optimal_fraction=float(np.mean(incompatible)) if p else None,
                                  paired_label_variance=float(np.mean(variance)) if p else None,
                                  mean_next_log_gap=float(np.mean(reuse_diff)) if reuse_diff else None,
                                  heldout_neighbor_regret=float(np.mean(regret)) if p else None,
                                  matched_lru_regret=float(np.mean([int(data[i,0])-int(data[i].min()) for i,_,_ in p])) if p else None))


def run_cell(family,cap,n_audit,n_oracle,n_enriched):
    begin=perf_counter(); cpu=process_time()
    rows={k:[] for k in AUDIT_KEYS}
    x,y,t,m=load_cell(family,cap)
    stream,objects=stream_for(family)
    rng=np.random.default_rng(np.random.SeedSequence([SEED,cap,sum(map(ord,family))]))
    ix=stratified(np.arange(len(t)),n_audit,rng)
    times=np.asarray(t[ix])
    r,states,next_at=observed_states(stream,objects,cap,times)
    legacy=np.asarray(x[ix])
    expanded=np.stack([states[int(a)]["x"] for a in times])
    globals_=r[times]
    target=np.asarray(y[ix])
    future=np.stack([states[int(a)]["next"] for a in times])
    censored=np.where(future<len(stream),np.log1p(future-times[:,None]),np.nan)
    old,new,old_hash,new_hash=frozen_predictions(family,cap,legacy,r,times,states)
    prefix=dict(family=family,capacity=cap)
    late=np.where(times<10000,"warmup","scored_window")
    for name,v,columns,recency in (("november",legacy,m["columns"],int(m["columns"].index("candidate_recency_rank"))),
                                  ("expanded",expanded,STATIC_NAMES,6),
                                  ("expanded_with_request",np.concatenate((expanded,np.broadcast_to(
                                      globals_[:,None,:],(len(times),cap,len(GLOBAL_NAMES)))),axis=-1),
                                   (*STATIC_NAMES,*GLOBAL_NAMES),6)):
        scope=dict(**prefix,surface="KBS matched LRU decision sample",representation=name)
        a,b=redundancy(v,target,times,late)
        for h,q in zip((4,8,16),b):
            rows["redundancy"].append(dict(**scope,horizon=h,**a,**q,
                                           phase_definition="KBS warmup and scored window, not training split",
                                           actual_role_split_collision=0))
        if name!="expanded_with_request":
            rows["features"].extend(row_features(v,globals_,columns,scope))
        alias_by_rep(v,target,censored,times,recency,scope,rows)
    for timing,mat in (("all_real_requests",r),("decision_requests",globals_)):
        rows["features"].append(dict(**prefix,representation="expanded_request",**spectrum(mat,GLOBAL_NAMES,timing)))
    # Full population for legacy horizons, plus a matched sample at all scales below.
    for i,h in enumerate((4,8,16)):
        zz=np.asarray(y[:,:,i])
        scope=dict(**prefix,sample="November all LRU decisions",horizon=h)
        rows["horizon"].append(dict(**scope,**summarize_labels(zz)))
        for what,values in (("lru_regret",zz[:,0]-zz.min(1)),("optimal_set_size",(zz==zz.min(1,keepdims=True)).sum(1))):
            v,n=np.unique(values,return_counts=True)
            rows["distribution"].extend(dict(**scope,metric=what,value=int(a),count=int(b)) for a,b in zip(v,n))
    population=stratified(np.arange(len(times)),n_oracle,rng)
    eligible=np.flatnonzero((old!=0)|((new>=0)&(new!=0)))
    eligible=np.setdiff1d(eligible,population)
    enriched=np.sort(rng.choice(eligible,min(n_enriched,len(eligible)),replace=False)) if len(eligible) else []
    horizons=np.unique(np.asarray((*FIXED,*(cap*k for k in SCALES)),dtype=np.int64))
    sample_times={}; aligned={}
    for sampling,indices in (("temporal_stratified",population),("learner_disagreement_enriched",enriched)):
        sample_times[sampling]=[int(times[i]) for i in indices]
        opt_regrets=[]; advantages=[]; reuse_vectors=[]; oracle_targets=[]
        for j,index in enumerate(indices):
            t0=int(times[index]); st=states[t0]
            group=dict(**prefix,sample=sampling,t=t0)
            future_stream=stream[t0+1:]
            start=perf_counter()
            costs,oracle=forced_belady(stream,next_at,t0+1,st["ids"],stream[t0],objects)
            oracle_s=perf_counter()-start
            reg=costs-costs.min()
            if reg.max()>1 or reg[oracle]!=0:
                raise RuntimeError("Forced MIN or unit bound failed")
            start=perf_counter()
            d,ad,length,status,base=paired_lru(st["ids"],stream[t0],future_stream,horizons,int(horizons.max()))
            paired_s=perf_counter()-start
            if not np.array_equal((base[None]-d)[:,[list(horizons).index(h) for h in (4,8,16)]],target[index]):
                raise RuntimeError("Contemporary forced LRU labels differ from November")
            if j < 2:
                _,full,_,rfull,_=paired_lru(st["ids"],stream[t0],future_stream,horizons,len(future_stream))
                if not (rfull>0).all() or not np.array_equal(ad[status>0],full[status>0]):
                    raise RuntimeError("Capped rollout coalescence diverges from full suffix")
                full_disagreements=int(((ad!=full)&(status==0)).sum())
            else:
                full_disagreements=None
            opt_regrets.append(reg)
            advantages.append(np.where(status>0,ad,np.nan))
            reuse_vectors.append(censored[index])
            oracle_targets.append(target[index,:,2])
            positions=st["next"]
            top=positions==positions.max()
            if not np.all(reg[top]==0):
                raise RuntimeError("Farthest-next ties included nonoptimal actions")
            old_i=int(old[index]); new_i=int(new[index]); sieve=int(st["native_sieve"])
            rows["optimal"].append(dict(**group,candidates=cap,optimal_count=int((reg==0).sum()),
                    full_tie=bool(reg.max()==0),unique_optimal=bool((reg==0).sum()==1),
                    farthest_next_count=int(top.sum()),opt_outside_farthest_next=int(((reg==0)&~top).sum()),
                    short_h16_opt_overlap=int(((reg==0)&(target[index,:,2]==target[index,:,2].min())).sum()),
                    lru_regret=int(reg[0]),sieve_regret=int(reg[sieve]) if sieve>=0 else None,
                    november_regret=int(reg[old_i]),phase1_regret=int(reg[new_i]) if new_i>=0 else None,
                    belady_suffix_misses=int(costs.min()),oracle_seconds=oracle_s,november_sha256=old_hash,
                    phase1_sha256=new_hash,sieve_shared_state=sieve>=0))
            alt=np.arange(cap)!=0
            exact=status>0
            joint=alt&exact
            rows["paired"].append(dict(**group,alternatives=cap-1,coalesced=int((alt&(status==1)).sum()),
                    exact_at_trace_end=int((alt&(status==2)).sum()),unresolved=int((alt&~exact).sum()),
                    profitable_resolved=int((joint&(ad>0)).sum()),harmful_resolved=int((joint&(ad<0)).sum()),
                    mean_effective_length=float(length[alt].mean()),p95_effective_length=float(np.quantile(length[alt],.95)),
                    max_effective_length=int(length.max()),full_suffix_unresolved_delta_disagreements=full_disagreements,
                    paired_seconds=paired_s,states_replay_cap=int(horizons.max()),
                    baseline_state_bytes=int(st["ids"].nbytes)))
            for h,k in zip(horizons,range(len(horizons))):
                yy=base[k]-d[:,k]
                q=summarize_labels(yy[None])
                informative=joint
                q.update(compared_to_exact_advantage=int(informative.sum()),
                         fixed_vs_exact_sign_disagreement=int(((np.sign(d[:,k])!=np.sign(ad))&informative).sum()),
                         fixed_vs_exact_value_disagreement=int(((d[:,k]!=ad)&informative).sum()),
                         fixed_vs_optimal_label_optimal_overlap=float(((yy==yy.min())&(reg==0)).sum()/(yy==yy.min()).sum()),
                         sieve_optimal_accuracy=float(yy[sieve]==yy.min()) if sieve>=0 else None,
                         november_optimal_rate=float(yy[old_i]==yy.min()),
                         phase1_optimal_rate=float(yy[new_i]==yy.min()) if new_i>=0 else None,
                         distinct_objects_in_horizon=len(set(future_stream[:int(h)])),
                         reuse_observed_through_horizon=float(((positions<len(stream))&(positions<=t0+h)).mean()),
                         baseline_misses=int(base[k]))
                rows["horizon"].append(dict(**group,horizon=int(h),**q))
                for what,values in (("lru_regret",[q["lru_mean_regret"]]),("optimal_set_size",[q["mean_optimal_set_size"]])):
                    rows["distribution"].append(dict(**group,horizon=int(h),metric=what,value=int(values[0]),count=1))
            fixed_target=target[index,:,2].astype(float)
            for h in sorted({max(1,int(cap*f)) for f in (.25,.5,1,2,4,8,16,32)}):
                duration,event,valid,stop=censored_reuse(positions,t0,len(stream),h)
                rows["censor"].append(dict(**group,horizon=h,administrative_horizon=h,trace_end=len(stream),
                     available_requests=len(stream)-1-t0,censor_at=stop,candidates=cap,events=int(event.sum()),
                     known_no_event_by_horizon=int((valid&~event).sum()),
                     unknown_due_to_trace_end=int((~valid).sum()),no_future_event_in_trace=int((positions==len(stream)).sum()),
                     h16_optimal_has_no_observed_reuse=float((~event)[fixed_target==fixed_target.min()].mean()),
                     exact_optimal_has_no_observed_reuse=float((~event)[reg==0].mean()),
                     resolved_profitable_event_count=int((event&exact&(ad>0)).sum()),
                     resolved_profitable_candidates=int((exact&(ad>0)).sum())))
            ranks,usec=make_lists(st["x"],st,rng)
            for which,order in ranks.items():
                for k in sorted(set((*SHORTLISTS,cap))):
                    if k>cap:
                        continue
                    ids=order[:k]
                    better=exact&(ad>0)
                    rows["shortlist"].append(dict(**group,kind=which,k=k,cover_next_use=bool(top[ids].any()),
                         cover_exact_optimal=bool((reg[ids]==0).any()),best_regret=int(reg[ids].min()),
                         profitable_available=int(better.sum()),profitable_in_shortlist=int(better[ids].sum()),
                         unresolved_alternatives=int((~exact).sum()),beneficial_override_found=bool(better[ids].any()),
                         best_resolved_gain=int(ad[ids[exact[ids]]].max()),construction_us_shared=usec))
            for i in range(cap):
                rows["actions"].append(dict(**group,candidate_recency_order=i,next_position=int(positions[i]) if positions[i]<len(stream) else None,
                     right_censor_requests=len(stream)-1-t0,h16_loss=int(fixed_target[i]),forced_belady_regret=int(reg[i]),
                     lru_paired_advantage=int(ad[i]),resolution=int(status[i]),rollout_length=int(length[i]),
                     is_lru=i==0,is_sieve=i==sieve,is_november=i==old_i,is_phase1=i==new_i))
            rows["state"].append(dict(**group,observed_next_delays=(positions[positions<len(stream)]-t0).tolist(),
                 longest_observed_reuse=int((positions[positions<len(stream)]-t0).max(initial=0))))
        if len(indices)>1:
            subset=np.asarray(indices)
            for form,xx in (("november",legacy),("expanded_with_request",np.concatenate((expanded,np.broadcast_to(
                    globals_[:,None,:],(len(times),cap,len(GLOBAL_NAMES)))),axis=-1))):
                ks=exact_group_keys(xx[subset])
                q=conditional(ks,np.asarray(oracle_targets),np.asarray(reuse_vectors),np.asarray(opt_regrets),
                              np.asarray(advantages))
                rows["alias"].append(dict(**prefix,surface=sampling,representation=form,horizon=16,
                                           signature="exact_oracle_subset",**q))
    for k in (1,4,8,cap):
        start=perf_counter()
        mm=shortlist_hindsight(stream,next_at,cap,k,10000)
        rows["shortlist"].append(dict(**prefix,sample="actual closed-loop hindsight",kind="lru_tail_next_use",
                                      k=k,scored_requests=40000,misses=mm,seconds=perf_counter()-start))
    def summary_link(a,b):
        if len(set(a))<2 or len(set(b))<2:
            return None
        return float(spearmanr(a,b).statistic)
    for dim,i in (("age",0),("count",1),("gap_mean",2),("frequency_rank",7),
                  ("decayed_popularity",8),("reinsertions",10),("frequency_1024",17)):
        z=expanded[:,:,i]
        rows["features"].append(dict(**prefix,representation="expanded",scope="within_decision_label_association",
                 investigated_feature=dim,groups=len(z),correlation_spread_with_h16_lru_regret=summary_link(
                 np.ptp(z,axis=1),target[:,0,2]-target[:,:,2].min(1)),
                 fraction_groups_feature_varies=float((np.ptp(z,axis=1)>0).mean()),
                 mean_within_decision_spearman_with_h16_loss=float(np.mean([q for a,b in zip(z,target[:,:,2])
                    if (q:=summary_link(a,b)) is not None])) if any(len(set(a))>1 and len(set(b))>1
                    for a,b in zip(z,target[:,:,2])) else None))
    rows["cost"].append(dict(**prefix,wall_seconds=perf_counter()-begin,cpu_seconds=process_time()-cpu,
                    peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    decision_sample=len(times),oracle_states=sum(map(len,sample_times.values())),
                    source_bytes=int(x.nbytes+y.nbytes+t.nbytes)))
    # Bounded row samples for pooled/family/capacity rank, stored independently of per-cell evidence.
    sampled=np.linspace(0,legacy.size//legacy.shape[-1]-1,min(3600,len(legacy)*cap),dtype=int)
    keep=dict(november=legacy.reshape(-1,legacy.shape[-1])[sampled],expanded=expanded.reshape(-1,expanded.shape[-1])[sampled],
              expanded_request=r[np.linspace(0,len(r)-1,2800,dtype=int)])
    signatures={"november": [zlib.compress(canonical(group),1).hex() for group in legacy]}
    return rows,keep,sample_times,signatures
