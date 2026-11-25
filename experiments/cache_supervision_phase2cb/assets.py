"""Verify immutable Phase 2C-A inputs and load whole training/validation groups."""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.append(str(HERE.parent/"cache_supervision_phase2ca"))
from campaign import (CAPS, BUDGETS, END, FEATURES, FEATURE_GROUPS, GLOBAL_NAMES, HISTORY, MECHANISMS,
                      PROTOCOL, ROOT, SCHEMA, STATIC_NAMES, dump, horizon_keys, sha)

EXPECTED_CA="648a9fd9a967b6509b845f1ae49681eeeb0c9b8c"
EXPECTED_MANIFEST="4b0370d91572c2a358d750fe74574a47682f892759fb2899df9c50bd24a8cd58"
FEATURE_NAMES={
    "F0": list(FEATURES), "F1": list(FEATURES),
    "F2": FEATURE_GROUPS["F2"],
    "F3": FEATURE_GROUPS["F3"],
    "F4": FEATURE_GROUPS["F3"]+FEATURE_GROUPS["F5"],
    "F5": FEATURE_GROUPS["F3"]+["unique_rate64","hit_rate64","hit_rate_shift","eviction_rate64"],
    "F6": list(STATIC_NAMES),
}


def rows(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def check(path, expected):
    actual=sha(path)
    if actual!=expected:
        raise ValueError(f"Hash mismatch for {path}")
    return actual


def verify(dataset, pilot_root, destination, **extras):
    start=perf_counter()
    e=dataset/"evidence"
    terminal=json.loads((dataset/"receipts/REDUCER_SUCCESS.json").read_text())
    provenance=json.loads((e/"SOURCE_PROVENANCE.json").read_text())
    if provenance["production_revision"]!=EXPECTED_CA or terminal["manifest_sha256"]!=EXPECTED_MANIFEST:
        raise ValueError("Unexpected reducer/source identity")
    check(e/"MANIFEST.sha256",EXPECTED_MANIFEST)
    for line in (e/"MANIFEST.sha256").read_text().splitlines():
        digest,name=line.split("  ")
        check(e/name,digest)
    assert provenance["feature_schema"]==SCHEMA and provenance["groups"]["F0"]==list(FEATURES)
    assert set(provenance["folds"])==set(PROTOCOL)
    for f,spec in PROTOCOL.items():
        for k in ("trace","sha256","train","validation"):
            assert provenance["folds"][f][k]==spec[k]
        assert set(spec["train"])==set(PROTOCOL)-{f,spec["validation"]}
    sampling={(r["family"],int(r["capacity"]),int(r["stratum"])):r for r in rows(e/"SAMPLING_RECEIPTS.csv")}
    by_stratum=defaultdict(lambda: np.zeros(3,dtype=int))
    totals=Counter()
    last={}
    checked=0
    manifest=rows(e/"DATASET_MANIFEST.csv")
    for item in manifest:
        f,c=item["family"],int(item["capacity"])
        path=dataset/item["path"]
        key=f,c
        assert item["trace_sha256"]==PROTOCOL[f]["sha256"] and item["version"]==SCHEMA
        assert path.stat().st_size==int(item["bytes"])
        check(path,item["sha256"])
        with np.load(path,allow_pickle=False) as z:
            t=z["position"].astype(np.int64)
            n=len(t)
            assert n==int(item["groups"]) and t[0]>last.get(key,HISTORY-1) and (np.diff(t)>0).all()
            assert z["legacy"].shape==(n,c,len(FEATURES)) and z["expanded"].shape==(n,c,len(STATIC_NAMES))
            assert z["workload"].shape==(n,len(GLOBAL_NAMES)) and z["resolved"].shape==(n,c)
            assert np.array_equal(z["resident"][:,0],z["baseline"]) and (z["shortlist"][:,:,0]==0).all()
            a,miss=z["advantage"],z["miss_counts"]
            assert np.array_equal(miss[:,:,:1].astype(np.int32)-miss,a)
            assert np.array_equal(z["prevented"].astype(np.int32)-z["additional"],z["resolved"])
            assert (z["additional"]<=1).all() and (z["resolved"]>=-1).all()
            assert (z["resolved"][:,0]==0).all() and (t<END).all()
            assert np.array_equal(z["resolved_through"].astype(np.int64),t+z["resolved_length"])
            obs=z["all_labels_through"]
            assert (obs<END).all() and (obs>=z["resolved_through"]).all()
            assert (obs[:,None]>=t[:,None]+z["actual_requests"]).all()
            stratum=(t-HISTORY)//5000
            assert np.array_equal(stratum,z["stratum"])
            assert np.array_equal(obs<HISTORY+5000*(stratum+1),z["within_stratum_all_labels"])
            levels=z["budget_included"]
            assert levels[:,-1].all() and not (levels[:,:-1] & ~levels[:,1:]).any()
            for s in np.unique(stratum):
                ix=stratum==s
                record=sampling[f,c,int(s)]
                required=np.array([float(record[f"p_{b}"]) for b in BUDGETS])
                np.testing.assert_array_equal(z["inclusion_probability"][ix],np.broadcast_to(required,(ix.sum(),3)))
                by_stratum[f,c,int(s)]+=levels[ix].sum(axis=0)
            assert np.isfinite(z["legacy"]).all() and np.isfinite(z["expanded"]).all()
            last[key]=int(t[-1])
            if c in CAPS:
                for b,count in zip(BUDGETS,levels.sum(axis=0)):
                    totals[b]+=int(count)
            else:
                totals["diagnostic"]+=n
        checked+=1
    for key,r in sampling.items():
        assert tuple(by_stratum[key])==tuple(int(r[f"n_{b}"]) for b in BUDGETS)
    assert checked==terminal["partitions"]==420 and len(last)==terminal["cells"]==42
    assert totals==Counter({20000:20000,50000:50000,100000:100000,"diagnostic":2688})
    pilot_manifest=json.loads((pilot_root/"MANIFEST.json").read_text())
    expected_pilot={(f,c) for f in PROTOCOL for c in CAPS}
    assert set((r["family"],r["capacity"]) for r in pilot_manifest)==expected_pilot
    for r in pilot_manifest:
        assert r["trace_sha256"]==PROTOCOL[r["family"]]["sha256"]
        p=pilot_root/f"{r['family']}-{r['capacity']}.npz"
        check(p,r["sha256"])
        with np.load(p,allow_pickle=False) as z:
            assert z["x"].shape==(56,8,len(FEATURES)) and z["resolved"].shape==(56,8)
            assert np.array_equal(z["full_x"][np.arange(56)[:,None],z["short"]],z["x"])
            assert np.array_equal(np.take_along_axis(z["full_advantage"],z["short"],axis=1),z["resolved"])
            assert (np.diff(z["time"])>0).all() and (z["short"][:,0]==0).all()
    payload=dict(status="VERIFIED",phase2ca_source=EXPECTED_CA,phase2ca_manifest_sha256=EXPECTED_MANIFEST,
                 partitions=checked,cells=len(last),budgets={str(k):v for k,v in totals.items()},
                 pilot_groups=1176,pilot_manifest_sha256=sha(pilot_root/"MANIFEST.json"),
                 trace_and_role_contract="seven fixed development families",feature_schema=SCHEMA,
                 temporal_guard="actual label endpoints checked; no within-family role split",
                 candidate_alignment="verified baseline, shortlist rows, exact advantages and current feature parity smoke",
                 seconds=perf_counter()-start,**extras)
    dump(destination,payload)
    return payload


def selected_features(z, ix, ranks, group):
    def select(name):
        return z[name][ix][np.arange(len(ranks))[:,None],ranks]
    if group in ("F0","F1"):
        return select("legacy")
    ex=select("expanded")
    if group=="F6":
        return ex
    names=FEATURE_NAMES[group]
    parts=[]
    static=[STATIC_NAMES.index(x) for x in names if x in STATIC_NAMES]
    if group=="F4":
        compact=[STATIC_NAMES.index(x) for x in FEATURE_NAMES["F3"] if x in STATIC_NAMES]
        static=compact
    parts.append(ex[:,:,static])
    if group!="F2":
        parts.append(select("decay256")[:,:,None])
    if group=="F4":
        parts.append(ex[:,:,[STATIC_NAMES.index(x) for x in FEATURE_GROUPS["F5"]]])
    if group=="F5":
        workload=z["workload"][ix][:,[GLOBAL_NAMES.index(x) for x in names if x in GLOBAL_NAMES]]
        parts.append(np.broadcast_to(workload[:,None,:],(*ranks.shape,workload.shape[-1])))
    return np.concatenate(parts,axis=2)


class GroupData:
    def __init__(self, dataset, old, family_sets):
        self.root=dataset
        self.old=old
        self.family_sets=family_sets
        self.files=defaultdict(list)
        for r in rows(dataset/"evidence/DATASET_MANIFEST.csv"):
            key=r["family"],int(r["capacity"])
            if key[0] in family_sets and key[1] in CAPS:
                self.files[key].append(dataset/r["path"])

    def read(self, families, budget, group, target, shortlist, k, limit=None):
        if not set(families)<=self.family_sets:
            raise ValueError("Attempt to load outer-family supervised inputs")
        buffer=defaultdict(list)
        for f in families:
            for c in CAPS:
                if budget==1176:
                    assert (group,shortlist,k)==("F0","low_frequency",8)
                    with np.load(self.old/f"{f}-{c}.npz",allow_pickle=False) as z:
                        fields=dict(x=z["x"],y=z[target],resolved=z["resolved"],weight=np.ones(len(z["x"])),
                                    cap=np.full(len(z["x"]),c),position=z["time"],rank=z["short"])
                    for name,value in fields.items():
                        buffer[name].append(value[:limit])
                    continue
                index=BUDGETS.index(budget)
                obtained=0
                for file in self.files[f,c]:
                    with np.load(file,allow_pickle=False) as z:
                        ix=np.flatnonzero(z["budget_included"][:,index])
                        if limit is not None:
                            ix=ix[:max(0,limit-obtained)]
                        if not len(ix):
                            continue
                        n=len(ix)
                        if shortlist=="full":
                            rank=np.broadcast_to(np.arange(c),(n,c))
                        else:
                            rank=z["shortlist"][ix,MECHANISMS.index(shortlist),:k]
                        def target_rows(name):
                            return z[name][ix][np.arange(n)[:,None],rank]
                        truth=target_rows("resolved")
                        if target in ("resolved","two_part"):
                            y=truth
                        else:
                            name={"h8":"H8","h32":"H32","hc":"alpha_1"}[target]
                            slot=horizon_keys(c)[0].index(name)
                            y=z["advantage"][ix,slot][np.arange(n)[:,None],rank]
                        fields=dict(x=selected_features(z,ix,rank,group),y=y,resolved=truth,rank=rank,
                                    weight=1/z["inclusion_probability"][ix,index],cap=np.full(n,c),
                                    position=z["position"][ix])
                        if target=="two_part":
                            fields.update(prevented=target_rows("prevented"),additional=target_rows("additional"))
                        for name,value in fields.items():
                            buffer[name].append(value)
                        obtained+=n
        if shortlist=="full":
            raise ValueError("Full-candidate diagnostics load capacity blocks separately")
        return {name:np.concatenate(v) for name,v in buffer.items()}
