"""Causal replay and comparable CPU measurements through one cache controller."""
from collections import Counter, OrderedDict, deque
import hashlib
import math
import resource
import sys
from time import perf_counter, process_time_ns
from types import SimpleNamespace

import numpy as np

from assets import END, FEATURE_NAMES, GLOBAL_NAMES, HISTORY, PROTOCOL, STATIC_NAMES
from campaign import MultiDecay
from data import matrix
from exact_lru import rollout
from lafc.policies.sieve import SievePolicy
from learning import Scorer, derived


class Frequencies:
    """Count-only specialization with the same 2,048-entry ghost eviction order."""
    def __init__(self):
        self.residents={}
        self.ghosts=OrderedDict()

    def count(self,p):
        return self.residents[p]

    def after(self,p,t,hit,victim):
        if hit:
            self.residents[p]+=1
            return
        old=self.ghosts.pop(p,0)
        if victim is not None:
            self.ghosts[victim]=self.residents.pop(victim)
            if len(self.ghosts)>2048:
                self.ghosts.popitem(last=False)
        self.residents[p]=old+1


class Window:
    def __init__(self):
        self.history=deque()
        self.count=Counter()

    def add(self,p):
        if len(self.history)==64:
            old=self.history.popleft()
            self.count[old]-=1
            if not self.count[old]:
                del self.count[old]
        self.history.append(p)
        self.count[p]+=1


class Inputs:
    def __init__(self,cap,feature,shortlist,family,pages,reference=False):
        self.cap,self.feature,self.shortlist,self.family=cap,feature,shortlist,family
        self.reference=reference
        self.history=MultiDecay(cap) if feature not in ("none","F0","F1") or shortlist=="ghost_informed" else Frequencies()
        self.sieve=SievePolicy() if shortlist=="sieve_derived" else None
        if self.sieve:
            self.sieve.reset(cap,pages)
        self.uses_legacy=feature in ("F0","F1")
        self.req=Window() if self.uses_legacy else None
        self.hits=Window() if self.uses_legacy else None
        self.bucket={}
        self.confidence={}
        self.template=None
        self.buffers={}
        self.rank=np.arange(cap)
        self.uniform_seed=int(shortlist.split("_")[-1]) if shortlist.startswith("uniform_") else None
        self.family_seed=list(PROTOCOL).index(family)

    def count(self,p):
        return self.history.count(p) if isinstance(self.history,Frequencies) else self.history.residents[p].count

    def see_metadata(self,request):
        if not self.uses_legacy:
            return
        if request.metadata.get("bucket") is not None:
            self.bucket[request.page_id]=int(request.metadata["bucket"])
        if request.metadata.get("confidence") is not None:
            self.confidence[request.page_id]=max(0.,min(1.,float(request.metadata["confidence"])))

    def snapshot_sieve(self,request,candidates):
        if not self.sieve:
            return None,-1
        flags=np.asarray([self.sieve._visited.get(p,True) for p in candidates],dtype=bool) if candidates else None
        common=bool(candidates) and set(self.sieve._order)==set(candidates)
        event=self.sieve.on_request(request)
        return flags,candidates.index(event.evicted) if common and event.evicted is not None else -1

    def ranks(self,candidates,t,k,flags,native):
        mode=self.shortlist
        if mode in ("full","lru_tail"):
            return self.rank[:k]
        if self.uniform_seed is not None:
            order=np.random.default_rng(np.random.SeedSequence([self.uniform_seed,self.family_seed,self.cap,t])).permutation(self.cap)
        elif mode=="sieve_derived":
            order=np.lexsort((self.rank,flags))
            if native>0:
                order=np.r_[native,order[order!=native]]
        else:
            # Phase 2C-A sorts the float32 log1p counts, not unrounded counts.
            counts=np.asarray([math.log1p(self.count(p)) for p in candidates],dtype=np.float32)
            if mode=="ghost_informed":
                ghost=np.asarray([math.log1p(self.history.residents[p].reinsertions) for p in candidates],dtype=np.float32)
                order=np.lexsort((self.rank,counts,ghost))
            else:
                order=np.lexsort((self.rank,counts))
        output=np.empty(k,dtype=np.intp)
        output[0]=0
        output[1:]=order[order!=0][:k-1]
        return output

    def full_legacy(self,request,candidates):
        state=SimpleNamespace(_bucket_by_page=self.bucket,_confidence_by_page=self.confidence,
                              _recent_req_hist=self.req.history,_recent_hit_hist=self.hits.history)
        return matrix(state,request,candidates)

    def x(self,request,candidates,rank,profile=None):
        if self.uses_legacy:
            if self.reference or self.bucket or self.confidence or request.metadata:
                return self.full_legacy(request,candidates)[rank]
            if self.template is None:
                self.template=matrix(SimpleNamespace(_bucket_by_page={},_confidence_by_page={},
                    _recent_req_hist=(),_recent_hit_hist=()),request,candidates)
            if len(rank) not in self.buffers:
                self.buffers[len(rank)]=np.empty((len(rank),self.template.shape[1]),np.float64)
            buf=self.buffers[len(rank)]
            a=process_time_ns() if profile is not None else 0
            np.take(self.template,rank,axis=0,out=buf)
            if profile is not None:
                profile["feature_buffer_gather"]+=process_time_ns()-a
            rd,hd=max(1,len(self.req.history)),max(1,len(self.hits.history))
            for i,j in enumerate(rank):
                p=candidates[j]
                buf[i,24],buf[i,25]=self.req.count[p]/rd,self.hits.count[p]/hd
            return buf
        if self.feature=="F6":
            return self.history.candidates(candidates,request.t)[rank]
        live=[candidates[i] for i in rank]
        hs=self.history
        t=request.t
        result=np.zeros((len(live),len(FEATURE_NAMES[self.feature])),np.float32)
        extra=self.feature!="F2"
        for i,p in enumerate(live):
            e=hs.residents[p]
            age=t-e.last
            result[i,0]=math.log1p(age)
            result[i,1]=math.log1p(e.count)
            for j,gap in enumerate(reversed(e.gaps)):
                if j==3:
                    break
                result[i,2+j]=math.log1p(gap)
                result[i,5+j]=1.
            if extra:
                result[i,8]=math.log1p(e.popularity*math.exp(-age/64))
                cnt,last=hs.decay256[p]
                result[i,9]=math.log1p(cnt*math.exp(-(t-last)/256))
            if self.feature=="F4":
                result[i,10:14]=(math.log1p(e.evictions),math.log1p(e.reinsertions),
                                  math.log1p(t-e.last_eviction) if e.last_eviction>=0 else 0.,e.last_eviction>=0)
        if self.feature=="F5":
            z=hs.before(request.page_id,t,False)
            result[:,10:]=[z[GLOBAL_NAMES.index(name)] for name in FEATURE_NAMES["F5"][10:]]
        return result

    def after(self,request,hit,victim):
        p=request.page_id
        self.history.after(p,request.t,hit,victim)
        if self.uses_legacy:
            self.req.add(p)
            if hit:
                self.hits.add(p)

    def metadata(self):
        h=self.history
        rr,gg=h.residents,h.ghosts
        def entry(x):
            return sys.getsizeof(x)+(sys.getsizeof(x.gaps) if hasattr(x,"gaps") else 0)
        return dict(resident_entries=len(rr),ghost_entries=len(gg),
            resident_metadata_shallow_bytes=sys.getsizeof(rr)+sum(entry(x) for x in rr.values()),
            ghost_metadata_shallow_bytes=sys.getsizeof(gg)+sum(entry(x) for x in gg.values()),
            per_resident_shallow_bytes=(sum(entry(x) for x in rr.values())/max(1,len(rr))),
            per_ghost_shallow_bytes=(sum(entry(x) for x in gg.values())/max(1,len(gg))))


def replay(data,family,c,artifact=None,control=None,*,profile=False,audit=False,reference=False,
           end=END,signature=False,verify_callback=None):
    req,pages,encode,stream,decode=data
    spec=artifact["spec"] if artifact is not None else {}
    policy=control or {"rule":"lru"}
    rule=policy["rule"]
    rate=policy.get("rate",0.)
    value=policy.get("value")
    seed=policy.get("seed",0)
    live=rule not in ("lru","random_full") and not (rule in ("gate","preliminary","risk","random_gate") and value is None)
    kind=policy.get("shortlist",spec.get("shortlist","low_frequency"))
    k=c if kind=="full" else int(policy.get("k",spec.get("k",8)))
    if rule=="low_frequency_budget":
        kind="low_frequency"
        k=2
    need_score=live and rule not in ("random_shortlist","low_frequency_budget")
    inp=Inputs(c,spec.get("features","F0") if need_score else "none",kind,family,pages,reference) if live else None
    if verify_callback is not None:
        inp=Inputs(c,spec["features"],kind,family,pages,reference)
    scorer=Scorer(artifact,reference) if need_score else None
    rng=np.random.default_rng(np.random.SeedSequence([seed,list(PROTOCOL).index(family),c])) if rule.startswith("random") else None
    order=OrderedDict()
    records=[]
    times=Counter()
    clocks=0
    misses=evictions=overrides=0
    debt=0.
    events=[] if signature else None
    started_cpu=started_wall=None
    for request in req[:end]:
        p,t=request.page_id,request.t
        scored=t>=HISTORY
        if t==HISTORY:
            started_cpu,started_wall=process_time_ns(),perf_counter()
        pstart=process_time_ns() if profile and scored and t%83==0 else 0
        if pstart:
            clocks+=1
        hit=p in order
        ascertained=process_time_ns() if pstart else 0
        if inp:
            inp.see_metadata(request)
        maintain=process_time_ns() if pstart else 0
        victim=None
        candidates=None
        idx=0
        full=not hit and len(order)==c
        if full and (inp or rule=="random_full" or audit or signature):
            candidates=list(order)
        flag,native=inp.snapshot_sieve(request,candidates) if inp and inp.sieve else (None,-1)
        cand_ready=process_time_ns() if pstart else 0
        if full:
            if verify_callback is not None:
                verify_callback(inp,request,candidates,flag,native)
            if rule=="random_full":
                if rate>0 and rng.random()<rate:
                    idx=int(rng.integers(1,c))
            elif live:
                if rule=="random_shortlist":
                    fired=rate>0 and rng.random()<rate
                elif rule=="low_frequency_budget":
                    debt+=rate
                    fired=debt>=1
                    if fired:
                        debt-=1
                else:
                    fired=rule!="preliminary" or inp.count(candidates[0])>=2
                if fired:
                    b=process_time_ns() if pstart else 0
                    rank=inp.ranks(candidates,t,k,flag,native)
                    a=process_time_ns() if pstart else 0
                    if pstart:
                        times["shortlist"]+=a-b
                    if need_score:
                        x=inp.x(request,candidates,rank,times if pstart else None)
                        s=process_time_ns() if pstart else 0
                        score,risk=scorer(x,times if pstart else None)
                        g=process_time_ns() if pstart else 0
                        pick=derived(score,rank,risk,rule,value,policy.get("risk_cap",1.))
                        if rule=="random_gate" and pick:
                            pick=int(rng.integers(1,k))
                        if pstart:
                            times["feature_vector_total"]+=s-a
                            times["estimator_api_or_packed_score"]+=g-s
                            times["gate"]+=process_time_ns()-g
                    else:
                        pick=int(rng.integers(1,k)) if rule=="random_shortlist" else 1
                    idx=int(rank[pick])
        decided=process_time_ns() if pstart else 0
        if hit:
            order.move_to_end(p)
        else:
            if full:
                if idx:
                    victim=candidates[idx]
                    del order[victim]
                else:
                    victim,_=order.popitem(last=False)
            order[p]=None
        changed=process_time_ns() if pstart else 0
        if inp:
            inp.after(request,hit,victim)
        featured=process_time_ns() if pstart else 0
        if scored:
            misses+=not hit
            evictions+=full
            overrides+=idx!=0
            if audit and idx and t%79==0:
                records.append((tuple(encode[v] for v in candidates),encode[p],t,idx))
        if events is not None:
            events.append((hit,encode[victim] if victim is not None else -1))
        if pstart:
            times["membership"]+=ascertained-pstart
            times["request_metadata"]+=maintain-ascertained
            times["candidate_list_and_sieve"]+=cand_ready-maintain
            times["decision_total"]+=decided-cand_ready
            times["cache_state"]+=changed-decided
            times["feature_state_maintenance"]+=featured-changed
            times["counters_and_diagnostics"]+=process_time_ns()-featured
            times["sampled_total"]+=process_time_ns()-pstart
    cpu=(process_time_ns()-started_cpu)/1e9
    wall=perf_counter()-started_wall
    diagnostics=Counter()
    for residents,incoming,t,i in records:
        seq=(stream[j] for j in range(t+1,end))
        _,value=rollout(residents,incoming,seq,stop_when_resolved=True)
        delta=value.advantage[i]
        diagnostics["beneficial" if delta>0 else "harmful" if delta<0 else "neutral"]+=1
    digest=hashlib.sha256(np.asarray(events,dtype=np.int32).tobytes()).hexdigest() if events is not None else None
    columns=("beneficial","harmful","neutral")
    return dict(family=family,capacity=c,trace_sha256=PROTOCOL[family]["sha256"],misses=misses,requests=end-HISTORY,
                evictions=evictions,overrides=overrides,override_rate=overrides/max(1,evictions),cpu_seconds=cpu,
                wall_seconds=wall,cpu_us_per_request=1e6*cpu/(end-HISTORY),requests_per_cpu_second=(end-HISTORY)/cpu,
                profile_samples=clocks,profile_ns=dict(times),trace_decode_seconds=decode,
                peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,sampled_overrides=len(records),
                **{f"sampled_{field}":diagnostics[field] for field in columns},
                reference=reference,eviction_sha256=digest,metadata=inp.metadata() if inp else {})
