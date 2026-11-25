"""Verify producer receipts, then reduce grouped and design-weighted evidence."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import re
import sys
from shutil import copyfile
from zipfile import ZipFile, ZIP_DEFLATED
from time import perf_counter, process_time
import resource

from numba import njit
import numpy as np

from campaign import (ALPHAS, BUDGETS, CAPS, DIAGNOSTIC, FEATURES, FEATURE_GROUPS, FIXED, GLOBAL_NAMES,
                      KS, MECHANISMS, PROTOCOL, SCHEMA, STATIC_NAMES, STRATUM_WIDTH, UNIFORM,
                      dump, horizon_keys, sha)


def write_csv(root, name, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    path = root / name
    temp = path.with_suffix(".tmp")
    with temp.open("w",newline="") as f:
        writer = csv.DictWriter(f, columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def values(vector, cols):
    return dict(zip(cols, (float(x) for x in vector)))


@njit(cache=True)
def distribution(labels, resolved):
    n,k = labels.shape
    metrics=np.empty((n,12),dtype=np.float64)
    for g in range(n):
        row=np.sort(labels[g])
        best = row[-1]
        optimal = ties = kinds = 0
        entropy = 0.
        q = 0
        while q < k:
            end = q+1
            while end<k and row[end]==row[q]:
                end+=1
            count = end-q
            if row[q]==best:
                optimal = count
            kinds+=1
            ties+=count*(count-1)
            p=count/k
            entropy-=p*np.log2(p)
            q=end
        positive = harmful = differs = sign_change = 0
        for index in range(k):
            a,b = labels[g,index],resolved[g,index]
            positive += a>0
            harmful += a<0
            differs += a!=b
            sign_change += (0 if a==0 else 1 if a>0 else -1) != (0 if b==0 else 1 if b>0 else -1)
        metrics[g,0] = kinds==1
        metrics[g,1] = ties/(k*(k-1))
        metrics[g,2] = kinds
        metrics[g,3] = entropy
        metrics[g,4] = optimal/k
        metrics[g,5] = best==0
        metrics[g,6] = positive/(k-1)
        metrics[g,7] = harmful/(k-1)
        metrics[g,8] = best
        metrics[g,9] = differs/k
        metrics[g,10] = sign_change/k
        metrics[g,11] = best>0
    return metrics


H_METRICS = ("all_tie_decision_rate","pairwise_tie_rate","mean_distinct_labels","mean_entropy_bits",
             "optimal_set_fraction","lru_optimality_rate","beneficial_nonbaseline_fraction",
             "harmful_nonbaseline_fraction","mean_max_advantage","value_disagreement_rate",
             "sign_disagreement_rate","profitable_decision_rate")


def weighted_correlation(x, y, w):
    w = w/w.sum()
    x = x - w @ x
    y = y - w @ y
    variance = (w @ (x*x)) * (w @ (y*y))
    if variance <= 1e-18:
        return ""
    return float((w @ (x*y))/np.sqrt(variance))


class Cell:
    def __init__(self, receipt):
        self.info = receipt
        self.cap = receipt["capacity"]
        self.family = receipt["family"]
        self.length = receipt["groups"]
        self.primary = self.cap in CAPS
        self.hnames = [*horizon_keys(self.cap)[0],"resolved"]
        self.horizons = defaultdict(lambda: np.zeros(len(H_METRICS)+6))
        self.shortcuts = defaultdict(lambda: np.zeros(11))
        self.adequacy = defaultdict(lambda: np.zeros(15))
        self.decomp = np.zeros(11)
        self.degeneracy = defaultdict(list)
        self.sample = defaultdict(list)
        self.index = 0
        self.features_at = set(np.linspace(0,max(0,self.length-1),min(self.length,96 if self.primary else 32),dtype=int))
        self.degeneracy_at = set(np.linspace(0,max(0,self.length-1),min(self.length,512),dtype=int))
        self.before = -1
        self.gain_hist = defaultdict(Counter)

    def accept(self, z):
        n = len(z["position"])
        if int(z["position"][0]) <= self.before or np.any(np.diff(z["position"].astype(np.int64))<=0):
            raise ValueError("Duplicate or unsorted positions")
        self.before = int(z["position"][-1])
        if not np.array_equal(z["baseline"],z["resident"][:,0]):
            raise ValueError("Baseline alignment mismatch")
        if np.any(z["incoming"][:,None]==z["resident"]) or np.any(np.diff(np.sort(z["resident"],axis=1),axis=1)==0):
            raise ValueError("Invalid ordered residents")
        adv, resolved = z["advantage"], z["resolved"]
        if not np.array_equal(z["miss_counts"][:,:,:1].astype(np.int32)-z["miss_counts"],adv):
            raise ValueError("Absolute/relative count mismatch")
        if not np.array_equal(z["prevented"].astype(np.int32)-z["additional"],resolved) or int(resolved.min()) < -1:
            raise ValueError("Decomposition mismatch")
        if not np.array_equal(z["resolved_length"],z["branch_lengths"].max(axis=1)):
            raise ValueError("Resolution length mismatch")
        if not np.array_equal(z["trace_end_uncoalesced"],(z["branch_reasons"]==2).any(axis=1)):
            raise ValueError("Resolution reason mismatch")
        keep = z["budget_included"]
        assert keep[:,-1].all() and not (keep[:,:-1] & ~keep[:,1:]).any()
        w = 1/z["inclusion_probability"][:,-1]
        legal = z["shortlist"]
        assert (legal[:,:,0]==0).all() and (legal<self.cap).all()
        short = legal[:,MECHANISMS.index("low_frequency")]
        best = resolved.max(axis=1)
        bestk = np.take_along_axis(resolved,short[:,:8],axis=1).max(axis=1)
        gain_pos = best > 0
        alt = resolved[:,1:]
        counts = np.stack([gain_pos,bestk>0,(alt>0).sum(axis=1),(alt==0).sum(axis=1),(alt<0).sum(axis=1),
                           best,bestk,z["trace_end_uncoalesced"],z["within_stratum_all_labels"],best>1],axis=1)
        for k,budget in enumerate(BUDGETS):
            mask=keep[:,k]
            if mask.any():
                iw=1/z["inclusion_probability"][mask,k]
                selected=counts[mask]
                self.adequacy[budget] += np.r_[mask.sum(),iw.sum(),selected[:,:3].sum(axis=0),iw @ selected]
                gains = best[mask]
                for gain in np.unique(gains):
                    if gain > 0:
                        self.gain_hist[budget][int(gain)] += float(iw[gains==gain].sum())
        for name, order in zip(MECHANISMS, np.moveaxis(legal,1,0)):
            for k in KS:
                ix=order[:,:k]
                d=np.take_along_axis(resolved,ix,axis=1)
                selected_best=d.max(axis=1)
                record=(selected_best>0)
                self.shortcuts[name,k] += np.array([n,w.sum(), gain_pos.sum(), record.sum(),
                    w @ gain_pos, w @ record,w @ best,w @ selected_best,
                    w @ (d<0).sum(axis=1),w @ (d>0).sum(axis=1),record.sum()])
        self.decomp += np.array([n,(self.cap-1)*n,np.count_nonzero(z["additional"]),int(z["prevented"].sum()),
                                (alt>0).sum(),(alt<0).sum(),(alt==0).sum(),
                                w @ z["trace_end_uncoalesced"],w @ z["resolved_length"],
                                (z["branch_reasons"][:,1:]==2).sum(),(best>1).sum()])
        diagnostic = np.asarray([i for i in range(n) if i+self.index in self.degeneracy_at],dtype=np.intp)
        for j,name in enumerate(self.hnames):
            if name == "resolved":
                a = resolved
                effective = z["resolved_length"]
                unique = None
                truncated = z["trace_end_uncoalesced"]
            else:
                a = adv[:,j]
                effective = z["actual_requests"][:,j]
                unique = z["distinct_objects"][:,j]
                truncated = z["truncated"][:,j]
                check = (a==a.max(axis=1,keepdims=True)).sum(axis=1)
                assert np.all(check>=np.maximum(1,self.cap-effective.astype(np.int32)))
            for mode,k in [("all",self.cap), *(("low_frequency", k) for k in KS if k < self.cap)]:
                if mode=="all":
                    actions, truth = a, resolved
                else:
                    ix=short[:,:k]
                    actions=np.take_along_axis(a,ix,axis=1)
                    truth=np.take_along_axis(resolved,ix,axis=1)
                m = distribution(actions,truth)
                c_ratio=self.cap/np.maximum(1,effective)
                k_ratio=k/np.maximum(1,effective)
                turn=np.zeros(n) if unique is None else unique/self.cap
                # Average Ratios are explicit weighted means of per-decision ratios.
                self.horizons[name,mode,k] += np.r_[n,w.sum(),w @ effective,w @ truncated,w @ c_ratio,w @ k_ratio,w @ m]
                if name!="resolved" and len(diagnostic):
                    self.degeneracy[mode,k].append(np.stack(
                        (np.full(n,j),c_ratio,k_ratio,turn,m[:,0],m[:,1],w),axis=1)[diagnostic])
        sampled = [i for i in range(n) if i+self.index in self.features_at]
        if sampled:
            for name in ("legacy","expanded","workload","decay256","resolved"):
                self.sample[name].append(z[name][sampled])
        self.index += n

    def collect(self):
        base=dict(family=self.family,capacity=self.cap,diagnostic=not self.primary,
                  eligible=self.info["eligible"])
        adeq=[]
        for budget,s in self.adequacy.items():
            keys=("sampled_decisions","design_weight_sum","sampled_positive","sampled_positive_k8","sampled_beneficial_actions",
                  "estimated_positive","estimated_positive_k8","estimated_beneficial_actions","estimated_neutral_actions",
                  "estimated_harmful_actions","estimated_attainable_gain","estimated_attainable_k8_gain",
                  "estimated_uncoalesced_decisions","estimated_temporally_guarded_decisions","estimated_max_gain_above_one")
            info = values(s,keys)
            info["estimated_positive_prevalence"] = float(s[5]/s[1])
            info["estimated_positive_k8_prevalence"] = float(s[6]/s[1])
            gain_dist = self.gain_hist[budget]
            top = 0.1 * sum(gain_dist.values())
            retained = 0.
            for gain in sorted(gain_dist,reverse=True):
                count = min(top,gain_dist[gain])
                retained += gain*count
                top -= count
            info["top_decile_profitable_gain_share"] = retained/s[10] if s[10] else ""
            adeq.append(dict(base,budget=budget,role="family_capacity",**info))
        horizon=[]
        for (name,mode,k),s in self.horizons.items():
            row=values(s[6:]/s[1],H_METRICS)
            horizon.append(dict(base,horizon=name,candidate_set=mode,k=k,groups=int(s[0]),
                              weighted_population=s[1],mean_effective_requests=s[2]/s[1],
                              trace_end_truncation_rate=s[3]/s[1],mean_c_over_h=s[4]/s[1],mean_k_over_h=s[5]/s[1],
                              random_uniform_optimal_accuracy=row["optimal_set_fraction"],
                              shared_label_seconds_per_group=self.info["label_seconds"]/self.length,**row))
        short=[]
        for (name,k),s in self.shortcuts.items():
            short.append(dict(base,mechanism=name,k=k,groups=int(s[0]),sampled_profitables=int(s[2]),
                              sampled_covered=int(s[3]),estimated_profitables=s[4],estimated_covered=s[5],
                              estimated_gain=s[6],estimated_captured_gain=s[7],weighted_groups=s[1],
                              covered_profitables_pct=100*s[5]/s[4] if s[4] else "",
                              captured_gain_pct=100*s[7]/s[6] if s[6] else "",
                              harmful_alternatives_rate=s[8]/(s[1]*(k-1)),beneficial_alternatives_rate=s[9]/(s[1]*(k-1)),
                              selection_us=1e6*self.info["shortlist_seconds"][name]/self.length,
                              metadata=SHORT_META.get(name,"ordered residents and seeded RNG")))
        dep=[]
        n,alternatives,caused,prevented,benefit,harm,neutral,censored,length,branches,mult=self.decomp
        if n:
            dep=[dict(base,decisions=int(n),alternatives=int(alternatives),reference_misses_prevented=int(prevented),
                      additional_misses=int(caused),positive=int(benefit),harmful=int(harm),neutral=int(neutral),
                      design_estimate_uncoalesced_groups=censored,design_estimate_rollout_requests=length,
                      trace_end_branches=int(branches),groups_best_gain_above_one=int(mult),reconstruction_errors=0)]
        diag=[]
        for (mode,k),pieces in self.degeneracy.items():
            d=np.concatenate(pieces)
            for scope,mask in (
                ("all_horizons",np.ones(len(d),dtype=bool)),
                ("fixed_requests",d[:,0]<len(FIXED)),
                ("scaled_requests",(d[:,0]>=len(FIXED))&(d[:,0]<len(FIXED)+len(ALPHAS))),
                ("distinct_object",d[:,0]>=len(FIXED)+len(ALPHAS))):
                x=d[mask]
                for name,j in (("capacity_over_requests",1),("shortlist_over_requests",2),("distinct_turnover_fraction",3)):
                    diag.append(dict(base,candidate_set=mode,k=k,horizon_scope=scope,predictor=name,
                                     unique_groups=len(x)//len(np.unique(x[:,0])),horizon_observations=len(x),
                                     all_tie_weighted_pearson=weighted_correlation(x[:,j],x[:,4],x[:,6]),
                                     pairwise_tie_weighted_pearson=weighted_correlation(x[:,j],x[:,5],x[:,6]),
                                     interpretation="descriptive repeated horizons on a matched group sample"))
        if self.degeneracy:
            pooled=np.concatenate([np.concatenate(pieces) for pieces in self.degeneracy.values()])
            for name,j in (("capacity_over_requests",1),("shortlist_over_requests",2),("distinct_turnover_fraction",3)):
                diag.append(dict(base,candidate_set="varying_k",k="",horizon_scope="all_horizons",
                                 predictor=name,unique_groups=len(self.degeneracy_at),horizon_observations=len(pooled),
                                 all_tie_weighted_pearson=weighted_correlation(pooled[:,j],pooled[:,4],pooled[:,6]),
                                 pairwise_tie_weighted_pearson=weighted_correlation(pooled[:,j],pooled[:,5],pooled[:,6]),
                                 interpretation="within-cell variation across nested K and horizons; descriptive"))
        sample={k:np.concatenate(v) for k,v in self.sample.items()}
        return adeq,horizon,short,dep,diag,sample


SHORT_META = {
    "lru_tail":"resident order",
    "low_frequency":"resident order and bounded-history per-object request count",
    "sieve_derived":"parallel SIEVE order, hand and visit bits plus resident order",
    "ghost_informed":"resident order, frequency and ghost reinsertion history",
    "recency_frequency":"resident order and frequency rank",
    "causal_mixture":"resident order, frequency rank and decayed popularity",
}

FEATURE_METADATA = {
    "F0":"legacy cache metadata and last 64 requests/hits",
    "F1":"training-only cleaned legacy mask, retaining legacy maintenance",
    "F2":"resident plus bounded ghost frequency, last request and up to three gaps",
    "F3":"F2 plus lazy exponential counts at 64 and 256 requests",
    "F4":"existing CausalHistory residents, up to 2048 ghosts, windows and 16 gaps",
    "F5":"resident and bounded ghost eviction/reinsertion state",
    "F6":"request-side causal history and cache/request windows",
}


def representation(sample, group, mask=()):
    if group in ("F0","F1"):
        x=sample["legacy"]
        return x if group=="F0" else x[:,:,list(mask)]
    if group=="F6":
        return np.broadcast_to(sample["workload"][:,None,:],
                               (*sample["legacy"].shape[:2],len(GLOBAL_NAMES)))
    names=FEATURE_GROUPS[group]
    if group=="F3":
        return np.concatenate([sample["expanded"][:,:,[STATIC_NAMES.index(name) for name in names[:-1]]],
                               sample["decay256"][:,:,None]],axis=2)
    return sample["expanded"][:,:,[STATIC_NAMES.index(name) for name in names]]


def clean_masks(samples):
    masks = {}
    # Detect exact constants and duplicates against training decision groups only.
    for f,spec in PROTOCOL.items():
        x=np.concatenate([s["legacy"].reshape(-1,len(FEATURES))
                         for (family,c),s in samples.items() if family in spec["train"] and c in CAPS],axis=0)
        live=[]
        for i in range(x.shape[1]):
            v=x[:,i]
            if np.ptp(v)==0 or any(np.array_equal(v,x[:,j]) for j in live):
                continue
            live.append(i)
        masks[f]=tuple(live)
    return masks


def feature_analysis(x, target, names, common):
    n,c,p=x.shape
    flat=np.array(x.reshape(n*c,p),order="C",copy=True)
    flat[flat==0]=0  # Canonicalize signed zero.
    spread=flat.std(axis=0,dtype=np.float64)
    columns={}
    columns["constant_columns"]=" ".join(name for i,name in enumerate(names) if np.ptp(flat[:,i])==0)
    columns["near_constant_columns"]=" ".join(name for i,name in enumerate(names) if 0<spread[i]<=1.e-6)
    duplicates=[]
    affine=[]
    for j in range(p):
        for k in range(j):
            a,b=flat[:,j],flat[:,k]
            if np.array_equal(a,b):
                duplicates.append(f"{names[k]}={names[j]}")
            elif spread[j]>0 and spread[k]>0:
                aa=(a.astype(np.float64)-float(a.mean(dtype=np.float64)))/spread[j]
                bb=(b.astype(np.float64)-float(b.mean(dtype=np.float64)))/spread[k]
                if np.allclose(aa,bb,rtol=1e-6,atol=1e-6) or np.allclose(aa,-bb,rtol=1e-6,atol=1e-6):
                    affine.append(f"{names[k]}~{names[j]}")
    keep=spread>1e-12
    if np.any(keep):
        normalized=(flat[:,keep].astype(np.float64)-flat[:,keep].mean(axis=0,dtype=np.float64))/spread[keep]
        singular=np.linalg.svd(normalized,compute_uv=False)
        s=singular[singular>singular[0]*1e-10]
        strength=s*s
        strength/=strength.sum()
        rank=len(s)
        effective=float(np.exp(-(strength*np.log(strength)).sum()))
    else:
        rank=effective=0
    missing={name:float(1-flat[:,i].mean()) for i,name in enumerate(names)
             if re.fullmatch(r"known_gap_\d+|known_eviction|has_gap|known",name)}
    features=dict(common,decisions=n,candidates_per_decision=c,columns=p,rank_1e_10=rank,effective_rank=effective,
                  exact_duplicate_pairs=" ".join(duplicates),affine_pairs=" ".join(affine),
                  within_decision_variation=json.dumps(dict(zip(names,np.mean(np.ptp(x,axis=1)>0,axis=0).tolist()))),
                  missing_indicator_rates=json.dumps(missing),**columns)

    d=defaultdict(lambda:[0.,0.,0.,0.,0.,0.])
    group_keys=set()
    repeats=within=conflict_groups=0
    for row,y in zip(flat.reshape(n,c,p),target):
        keys=[v.tobytes() for v in row]
        g=tuple(sorted(keys))
        repeats+=int(g in group_keys)
        group_keys.add(g)
        existing={}
        has_conflict=False
        for key,adv in zip(keys,y):
            if key in existing:
                within+=1
                has_conflict |= existing[key] != adv
            existing[key]=adv
            v=d[key]
            v[0]+=1
            v[1]+=int(adv)
            v[2]+=int(adv)**2
            v[3+int(np.sign(adv))+1]+=1
        conflict_groups+=int(has_conflict)
    duplicate_rows=conflicting_rows=0
    var=repeat_var=conditional_entropy=0.
    repeated_weight=0
    class_count=np.zeros(3)
    for count,total,squares,neg,neutral,pos in d.values():
        classes=np.array([neg,neutral,pos])
        class_count+=classes
        variance=max(0.,squares/count-(total/count)**2)
        var+=variance*count
        nonzero=classes[classes>0]/count
        conditional_entropy-=count*float(nonzero @ np.log2(nonzero))
        if count>1:
            duplicate_rows+=int(count)-1
            repeated_weight+=count
            repeat_var+=variance*count
            conflicting_rows+=int(count)*(np.count_nonzero(classes)>1 or variance>0)
    pclasses=class_count[class_count>0]/class_count.sum()
    prior=-float(pclasses @ np.log2(pclasses))
    alias=dict(common,decisions=n,candidates_per_decision=c,unique_candidate_inputs=len(d),
               repeated_candidate_rows=duplicate_rows,repeated_whole_decisions=repeats,
               within_decision_duplicate_rows=within,within_decision_conflict_groups=conflict_groups,
               conflicting_candidate_rows=conflicting_rows,
               conditional_advantage_variance=var/(n*c),
               conditional_variance_on_repeated_inputs=repeat_var/max(1,repeated_weight),
               repeat_input_candidate_fraction=repeated_weight/(n*c),
               class_counts_neg_neutral_pos=json.dumps(class_count.astype(int).tolist()),
               class_prior_entropy_bits=prior,conditional_class_entropy_bits=conditional_entropy/(n*c),
               empirical_mutual_information_bits=prior-conditional_entropy/(n*c),
               information_scope="exact input partition descriptive plug-in; rare unique inputs can inflate estimate")
    return features, alias


def feature_rows(samples, receipts):
    masks=clean_masks(samples)
    info=[]
    aliases=[]
    for (family,cap),s in samples.items():
        r=receipts[family,cap]
        group_variants=[(g,(),[]) for g in ("F0","F2","F3","F4","F5","F6")]
        variants=defaultdict(list)
        for f,mask in masks.items():
            variants[mask].append(f)
        group_variants.extend(("F1",mask,folds) for mask,folds in variants.items())
        for group,mask,folds in group_variants:
            x=representation(s,group,mask)
            names=[FEATURES[j] for j in mask] if group=="F1" else FEATURE_GROUPS[group]
            stages={"F0":"legacy","F1":"legacy","F2":"expanded","F3":"expanded_plus_decay",
                    "F4":"expanded","F5":"expanded","F6":"workload"}
            parts=r.get("feature_stage_seconds",{})
            observed=(parts.get("expanded",0)+parts.get("decay",0) if group=="F3" else parts.get(stages[group]))
            common=dict(family=family,capacity=cap,feature_group=group,
                        mask_folds=";".join(folds),metadata=FEATURE_METADATA[group],feature_names=";".join(names),
                        measured_generation_scope=stages[group],
                        generation_us_per_group=1e6*observed/r["groups"] if observed is not None else "")
            a,b=feature_analysis(x,s["resolved"],names,common)
            info.append(a)
            aliases.append(b)
    return info,aliases,{f:[FEATURES[i] for i in m] for f,m in masks.items()}


def reduce(out):
    started,cpu=perf_counter(),process_time()
    evidence=out/"evidence"
    plan=json.loads((out/"PLAN.json").read_text())
    expected_jobs=json.loads((evidence/"WULVER_JOB_RECEIPTS.json").read_text())
    expected_array=str(expected_jobs["producer_array_job_id"])
    receipts={}
    manifest=[]
    sampling=[]
    adequacy=[]
    horizon=[]
    shortcuts=[]
    decomp=[]
    degeneracy=[]
    samples={}
    resources=[]
    capacity=[]
    for f in PROTOCOL:
        p=out/"receipts"/f"{f}.json"
        for r in json.loads(p.read_text()):
            key=(r["family"],r["capacity"])
            if str(r["producer_array_job_id"])!=expected_array or r["producer_array_index"]!=list(PROTOCOL).index(f):
                raise ValueError(f"Receipt changed producer identity for {key}")
            if key in receipts or r["family"]!=f:
                raise ValueError(f"Unexpected producer receipt {key}")
            receipts[key]=r
    expected={(r["family"],r["capacity"]):r for r in plan}
    assert set(receipts)==set(expected)
    for key,r in receipts.items():
        f,c=key
        for k in ("quotas","eligible","trace_sha256","diagnostic","distinct_trace_objects"):
            assert r[k]==expected[key][k]
        assert r["groups"]==r["quotas"][-1] and r["schema"]==SCHEMA
        cell=Cell(r)
        for part in r["partitions"]:
            path=out/"data"/f/str(c)/part["file"]
            if path.stat().st_size!=part["bytes"] or part["sha256"]!=sha(path):
                raise ValueError(f"Partition checksum mismatch {path}")
            with np.load(path,allow_pickle=False) as source:
                z={k:source[k] for k in source.files}
            assert len(z["position"])==part["groups"]
            cell.accept(z)
            manifest.append(dict(family=f,capacity=c,diagnostic=r["diagnostic"],trace_sha256=r["trace_sha256"],
                                 version=SCHEMA,path=str(path.relative_to(out)), **part))
        assert cell.index==r["groups"]
        a,h,s,d,g,sample=cell.collect()
        for i,b in enumerate(BUDGETS):
            s_count=sum(row[f"n_{b}"] for row in r["strata"])
            assert s_count == r["quotas"][i]
            lengths=[int(x["sampled_decisions"]) for x in a if x["budget"]==b]
            assert sum(lengths)==s_count
        adequacy.extend(a)
        sampling.extend(r["strata"])
        horizon.extend(h)
        shortcuts.extend(s)
        decomp.extend(d)
        degeneracy.extend(g)
        if sample:
            samples[key]=sample
        resources.append({k:v for k,v in r.items() if k.endswith(("_seconds","_kib")) and isinstance(v,(float,int))} |
                         dict(family=f,capacity=c,groups=r["groups"],peak_ghosts=r.get("peak_ghosts",0),
                              stored_bytes=sum(x["bytes"] for x in r["partitions"]),
                              raw_array_bytes=sum(x["raw_bytes"] for x in r["partitions"])))
        capacity.append(dict(family=f,capacity=c,diagnostic=r["diagnostic"],eligible=r["eligible"],
                             full_occupancy_scored_requests=r["full_occupancy_scored_requests"],
                             distinct_trace_objects=r["distinct_trace_objects"],groups=r["groups"],
                             **{k:d[0][k] for k in ("positive","neutral","harmful","groups_best_gain_above_one")} if d else {}))
        print(f"VERIFIED_CELL {f}/{c} groups={cell.index}",flush=True)
    for budget in BUDGETS:
        base=[x for x in adequacy if x["budget"]==budget and not x["diagnostic"]]
        assert sum(x["sampled_decisions"] for x in base)==budget
        for fold,spec in PROTOCOL.items():
            for role,families in (("train",spec["train"]),("validation",[spec["validation"]]),("test",[fold])):
                subset=[x for x in base if x["family"] in families]
                keys=[k for k in subset[0] if k.startswith(("sampled_","estimated_")) and not k.endswith("prevalence")]
                population=sum(x["eligible"] for x in subset)
                row={k:sum(x[k] for x in subset) for k in keys}
                adequacy.append(dict(fold=fold,role=role,budget=budget,eligible=population,
                                     estimated_positive_prevalence=row["estimated_positive"]/population,
                                     estimated_positive_k8_prevalence=row["estimated_positive_k8"]/population,**row))
    info,alias,masks=feature_rows(samples,receipts)
    seeds=[]
    for f in ("all",*PROTOCOL):
        for cap in ("all",*CAPS,*DIAGNOSTIC):
            for k in KS:
                rs=[x for x in shortcuts if x["k"]==k and x["mechanism"].startswith("uniform_")
                    and ((not x["diagnostic"]) if cap=="all" else x["capacity"]==cap)
                    and (f=="all" or x["family"]==f)]
                if not rs:
                    continue
                rates=[]
                for seed in UNIFORM:
                    m=[r for r in rs if r["mechanism"]==f"uniform_{seed}"]
                    total=sum(x["estimated_profitables"] for x in m)
                    gain_total=sum(x["estimated_gain"] for x in m)
                    coverage=sum(x["estimated_covered"] for x in m)/total if total else None
                    gain=sum(x["estimated_captured_gain"] for x in m)/gain_total if gain_total else None
                    if coverage is not None:
                        rates.append(coverage)
                    seeds.append(dict(family=f,capacity=cap,k=k,seed=seed,weighted_coverage=coverage,weighted_gain=gain))
                if rates:
                    seeds.append(dict(family=f,capacity=cap,k=k,seed="summary",coverage_mean=np.mean(rates),
                                      coverage_sd=np.std(rates,ddof=1),coverage_min=min(rates),coverage_max=max(rates)))
    resources.append(dict(stage="reducer",cpu_seconds=process_time()-cpu,wall_seconds=perf_counter()-started,
                          peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    for filename,rows in dict(DATASET_MANIFEST=manifest,SAMPLING_RECEIPTS=sampling,SAMPLE_ADEQUACY=adequacy,
                              HORIZON_SENSITIVITY=horizon,CAPACITY_SENSITIVITY=capacity,LABEL_DEGENERACY_ANALYSIS=degeneracy,
                              SHORTLIST_COVERAGE=shortcuts,SHORTLIST_SEED_VARIABILITY=seeds,ADVANTAGE_DECOMPOSITION=decomp,
                              FEATURE_INFORMATION_ANALYSIS=info,FEATURE_ALIASING_ANALYSIS=alias,RESOURCE_PROFILE=resources).items():
        write_csv(evidence,filename+".csv",rows)
    dump(evidence/"TRAINING_FEATURE_MASKS.json",masks)
    summarize(evidence,adequacy,horizon,shortcuts,capacity,resources)
    files=sorted(p for p in evidence.iterdir() if p.is_file() and p.name!="MANIFEST.sha256")
    (evidence/"MANIFEST.sha256").write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
    archive = pack_review(out,manifest)
    dump(out/"receipts"/"REDUCER_SUCCESS.json",dict(cells=len(receipts),primary_groups=BUDGETS[-1],
                                                  diagnostic_groups=sum(x["groups"] for x in capacity if x["diagnostic"]),
                                                  partitions=len(manifest),manifest_sha256=sha(evidence/"MANIFEST.sha256"),
                                                  private_review_archive_sha256=sha(archive)))



def pack_review(out, manifest):
    """Prepare a compact permitted review without candidate data or partition positions."""
    source=out/"evidence"
    review=out/"private-review"
    review.mkdir(mode=0o700,exist_ok=True)
    names=(
        "EXPERIMENT_PROTOCOL.md","SOURCE_PROVENANCE.json","LABELER_PLATFORM_PARITY.json","DATASET_SCHEMA.md",
        "SAMPLING_RECEIPTS.csv","SAMPLE_ADEQUACY.csv","HORIZON_SENSITIVITY.csv","CAPACITY_SENSITIVITY.csv",
        "LABEL_DEGENERACY_ANALYSIS.csv","SHORTLIST_COVERAGE.csv","SHORTLIST_SEED_VARIABILITY.csv",
        "FEATURE_INFORMATION_ANALYSIS.csv","FEATURE_ALIASING_ANALYSIS.csv","ADVANTAGE_DECOMPOSITION.csv",
        "RESOURCE_PROFILE.csv","WULVER_JOB_RECEIPTS.json","EXPERIMENT_CONCLUSIONS.md","TRAINING_FEATURE_MASKS.json",
    )
    for name in names:
        copyfile(source/name,review/name)
    groups=defaultdict(Counter)
    for r in manifest:
        key=(r["family"],r["capacity"],r["diagnostic"],r["trace_sha256"])
        groups[key].update(dict(groups=r["groups"],partitions=1,compressed_bytes=r["bytes"],raw_bytes=r["raw_bytes"]))
    write_csv(review,"DATASET_MANIFEST.csv",
              [dict(family=f,capacity=c,diagnostic=d,trace_sha256=t,**v) for (f,c,d,t),v in groups.items()])
    allow=[*names,"DATASET_MANIFEST.csv"]
    (review/"MANIFEST.sha256").write_text("".join(f"{sha(review/name)}  {name}\n" for name in sorted(allow)))
    archive=out/"phase2ca-private-review.zip"
    with ZipFile(archive,"w",compression=ZIP_DEFLATED) as zipfile:
        for name in [*allow,"MANIFEST.sha256"]:
            zipfile.write(review/name,name)
    with ZipFile(archive) as zipfile:
        if zipfile.testzip() is not None:
            raise ValueError("Private review ZIP verification failed")
    return archive

def summarize(out,adequacy,horizon,shortcuts,capacity,resources):
    def primary(b):
        fold_train=[x["sampled_positive_k8"] for x in adequacy if x["budget"]==b and x.get("role")=="train"]
        fold_val=[x["sampled_positive_k8"] for x in adequacy if x["budget"]==b and x.get("role")=="validation"]
        return f"| {b:,} | {int(min(fold_train))}–{int(max(fold_train))} | {int(min(fold_val))}–{int(max(fold_val))} |"
    r=[x for x in shortcuts if x["mechanism"]=="low_frequency" and x["k"]==8 and not x["diagnostic"]]
    covered=sum(x["estimated_covered"] for x in r)
    positive=sum(x["estimated_profitables"] for x in r)
    attainable=sum(x["estimated_gain"] for x in r)
    captured=sum(x["estimated_captured_gain"] for x in r)
    observed=""
    for metric in ("H8","alpha_1","distinct_1","resolved"):
        a=[x for x in horizon if not x["diagnostic"] and x["horizon"]==metric and x["candidate_set"]=="all"]
        p=sum(x["weighted_population"]*x["all_tie_decision_rate"] for x in a)/sum(x["weighted_population"] for x in a)
        observed+=f"- {metric} design-weighted all-tie rate {p:.2%}\n"
    diagnostics=sum(x["groups"] for x in capacity if x["diagnostic"])
    size=sum(x.get("stored_bytes",0) for x in resources)
    text=f"""# Phase 2C-A results

All family producers and 42 cell receipts passed reducer verification. The dataset contains 100,000 nested primary decision groups and {diagnostics:,} separate large-capacity diagnostic groups. Compressed partitions total {size/2**30:.3f} GiB.

| Nested decision budget | Sampled K=8 positive decisions per training fold | Sampled K=8 positive decisions per validation fold |
|---:|---:|---:|
{chr(10).join(primary(b) for b in BUDGETS)}

A fold with very few sampled validation positives needs qualification before any advantage gate is calibrated. Family-capacity population estimates and chronological guard coverage are in the adequacy table.

{observed}
Low-frequency K=8 covered {covered/max(1.,positive):.2%} of design-estimated profitable decisions and {captured/max(1.,attainable):.2%} of attainable one-action miss savings on the established capacities. The ten seeded uniform comparisons are in the shortlist tables.

The campaign fitted no eviction policy. These measurements describe fixed-state labels on the development traces. Group alias and entropy summaries use a bounded, equally spaced sample of decision groups within each cell. Candidate rows within a decision share context. The exact-feature plug-in mutual information can be inflated by unique feature inputs.
"""
    (out/"EXPERIMENT_CONCLUSIONS.md").write_text(text)


if __name__=="__main__":
    reduce(Path(sys.argv[1]))
