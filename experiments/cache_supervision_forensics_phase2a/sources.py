"""Read immutable research assets and form a deterministic common decision sample."""
from collections import OrderedDict
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROJECT = Path(os.environ.get("PHASE2A_PROJECT", REPO.parents[1]))
WORKTREES = PROJECT/"worktrees"
NOVEMBER = Path(os.environ.get("PHASE2A_NOVEMBER", WORKTREES/"neural-eviction-20261121"/"experiments/neural_eviction_20261121"))
PHASE1 = Path(os.environ.get("PHASE2A_PHASE1", WORKTREES/"recurrent-neural-eviction"/"experiments/recurrent_neural_eviction/runs/phase1"))
TRACES = Path(os.environ.get("PHASE2A_TRACES", PROJECT/"repo/data/processed"))
LAFC = Path(os.environ.get("PHASE2A_LAFC", PROJECT.parent/"lafc-evict-dataset/repo/release/lafc-evict-v1.0"))
TIE_DB = Path(os.environ.get("PHASE2A_TIE_DB","lafc_tie_audit.duckdb"))
PROTOCOL = json.loads((REPO/"experiments/neural_eviction_20261121/protocol.json").read_text())
SEED = 20261124
CAPACITIES = (32,64,128)

sys.path.insert(0,str(REPO/"src"))
sys.path.append(str(REPO/"experiments/recurrent_neural_eviction"))
sys.path.append(str(REPO/"experiments/neural_eviction_20261121"))

from features import CausalHistory, GLOBAL_NAMES, STATIC_NAMES  # noqa: E402
from lafc.policies.sieve import SievePolicy  # noqa: E402
from lafc.types import Page, Request  # noqa: E402
from math_core import next_positions  # noqa: E402


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def load_cell(family,cap):
    source=NOVEMBER/"runs/campaign/data"/f"{family}-{cap}"
    m=json.loads((source/"manifest.json").read_text())
    if (m["trace_sha256"],m["family"],m["capacity"],m["horizons"],m["continuation"]) != (
            PROTOCOL[family]["sha256"],family,cap,[4,8,16],"LRU"):
        raise ValueError(source)
    x,y,t=[np.load(source/(n+".npy"),mmap_mode="r",allow_pickle=False) for n in ("x","y","t")]
    if len(x)!=m["groups"] or x.shape[:2]!=y.shape[:2] or len(t)!=len(x) or not np.all(np.diff(t)>0):
        raise ValueError(f"Invalid archived coverage {source}")
    return x,y,t,m


def stream_for(family):
    path=TRACES/family/"trace.jsonl"
    if digest(path)!=PROTOCOL[family]["sha256"]:
        raise ValueError(f"Trace digest mismatch {path}")
    identities={}
    stream=[]
    with path.open() as source:
        for line in source:
            x=json.loads(line)
            pid=x["item_id"]
            stream.append(identities.setdefault(pid,len(identities)))
    if len(stream)!=50000:
        raise ValueError(path)
    return np.asarray(stream,dtype=np.int64),len(identities)


def stratified(times,n,rng):
    """Preserve temporal spread with an approximately uniform sample of decisions."""
    return np.asarray([rng.choice(block) for block in np.array_split(times,min(n,len(times))) if len(block)])


def observed_states(stream,objects,capacity,times):
    wanted=set(map(int,times))
    history=CausalHistory(capacity)
    lru=OrderedDict()
    req_x=np.empty((len(stream),len(GLOBAL_NAMES)),dtype=np.float32)
    sieve=SievePolicy()
    sieve.reset(capacity,{p:Page(p,1) for p in range(objects)})
    last={}
    data={}
    nxt=next_positions(stream,objects)
    for t,p in enumerate(stream):
        p=int(p)
        hit=p in lru
        req_x[t]=history.before(p,t,hit)
        native_members=frozenset(sieve._order) if t in wanted else None
        # Snapshot bounded controller state before the actual request mutates it.
        visit=sieve._visited.copy() if t in wanted else None
        hand=sieve._hand if t in wanted else None
        sn=sieve.on_request(Request(t,p))
        victim=None
        if hit:
            lru.move_to_end(p)
        elif len(lru)==capacity:
            victim=next(iter(lru))
            if t in wanted:
                ids=np.asarray(list(lru),dtype=np.int64)
                x=history.candidates(ids,t)
                shared=native_members==frozenset(ids)
                sv=int(np.flatnonzero(ids==sn.evicted)[0]) if shared and sn.evicted is not None else -1
                data[t]=dict(ids=ids,x=x,next=nxt[[last[i] for i in ids]],native_sieve=sv,
                             sieve_present=int(p not in native_members),
                             sieve_flags=np.asarray([visit.get(int(i),True) for i in ids]),
                             sieve_hand=hand)
            del lru[victim]
        if not hit:
            lru[p]=None
        history.after(p,t,hit,victim)
        last[p]=t
    if wanted!=set(data):
        raise ValueError("Matched LRU timestamps did not reproduce every candidate group")
    return req_x,data,nxt


def frozen_predictions(family,cap,x,req_x,times,states):
    import torch
    torch.set_num_threads(1)
    from models import load as legacy_load, predict
    from fit import load as phase1_load
    legacy_dir=NOVEMBER/"runs/campaign"/"h8"/family
    met=json.loads((legacy_dir/"offline.json").read_text())
    old=next(m for m in met["models"] if m["method"]=="deepsets" and m["seed"]==7)
    path=legacy_dir/old["file"]
    if digest(path)!=old["sha256"]:
        raise ValueError(path)
    old_net,_=legacy_load(path,family,8)
    with torch.inference_mode():
        _,scores=predict(old_net,x)
    old_choices=np.argmax(scores<=scores.min(1,keepdims=True)+.05,axis=1)
    freeze=PHASE1/"fits"/family/"FREEZE.json"
    if not freeze.exists():
        return old_choices,np.full(len(times),-1),old["sha256"],None
    winner=json.loads(freeze.read_text())
    tag=winner["winner"]
    path=freeze.parent/tag/"selected.pt"
    if digest(path)!=winner["selection"][tag]:
        raise ValueError(path)
    net=phase1_load(path)
    if net.config.order!="ordered" or net.attention or net.history_mlp:
        raise ValueError("Frozen-state scorer supports the full-fold recurrent roster only")
    with torch.inference_mode():
        hidden,_=net.global_outputs(torch.from_numpy(req_x[None]))
        candidates=torch.from_numpy(np.stack([states[int(t)]["x"] for t in times]))
        mask=torch.ones(candidates.shape[:2],dtype=torch.bool)
        logits=net.scores(candidates,mask,hidden[0,times])[...,0]
        chosen=logits.argmax(1).numpy()
    return old_choices,chosen,old["sha256"],winner["selection"][tag]
