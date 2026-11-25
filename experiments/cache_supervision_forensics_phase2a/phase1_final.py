"""Validate the terminal Phase 1 campaign without running any model."""
import json
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median,mean

from sources import PHASE1,PROTOCOL,CAPACITIES,digest

ROSTER=("R02_T03_h16","R08_T03_h16","R05_T02_h16","R05_T03_h16","R06_T03_h16")
SELECTED=("R_SELECTED_s7","R_SELECTED_s19","R_SELECTED_s42")
NATIVE=("lru","fifo_reinsertion","sieve")


def validate():
    status=json.loads((PHASE1/"RUN_STATUS.json").read_text())
    if status["status"]!="COMPLETED_SUCCESS" or any(not (PHASE1/name).is_file()
            for name in ("PILOT_COMPLETE","AGGREGATION_COMPLETE","FULL_COMPLETE","COMPLETED_SUCCESS")):
        return dict(status="RUNNING_HEALTHY",phase1_state=status)
    raw={}
    for path in (PHASE1/"replays").glob("*.json"):
        r=json.loads(path.read_text())
        key=r["split"],r["method"],r["family"],r["capacity"]
        assert key not in raw
        assert r["requests"]==r["hits"]+r["misses"]==r["score_end"]-r["score_start"]
        assert r["trace_sha256"]==PROTOCOL[r["family"]]["sha256"]
        assert sum(r[f"phase{i}_misses"] for i in range(1,5))==r["misses"]
        raw[key]=r
    archive={(r["method"],r["family"],r["capacity"]):r for r in json.loads(
        (PHASE1/"archived_baselines.json").read_text())}
    found={}
    for p in (PHASE1/"fits").glob("*/*/FIT_COMPLETE.json"):
        j=json.loads(p.read_text())
        for q in ("selected","latest"):
            assert digest(p.with_name(f"{q}.pt"))==j[f"{q}_sha256"]
        result=json.loads(p.with_name("result.json").read_text())
        key=(p.parent.parent.name,p.parent.name)
        found[key]=j|{"validation":result["validation"],"test":result.get("test",{})}
    assert len(found)==77
    for (split,m,f,c),r in raw.items():
        if split=="development" and r.get("checkpoint_sha256"):
            expected=found["development",m.removesuffix("_recompute")]["selected_sha256"]
            assert r["checkpoint_sha256"]==expected
    development=Counter(k[1] for k in raw if k[0]=="development")
    assert development.pop("R05_T03_h16_recompute")==2 and all(n==6 for n in development.values())
    assert len(development)==35
    per_fold={}
    for f,spec in PROTOCOL.items():
        freeze=json.loads((PHASE1/"fits"/f/"FREEZE.json").read_text())
        v=spec["validation"]
        assert freeze["test_family"]==f and freeze["validation_family"]==v and v not in spec["train"]
        all_candidates={}
        for q in ROSTER:
            rr=[raw["nested-"+f,q,v,c] for c in CAPACITIES]
            assert all(r["checkpoint_sha256"]==found[f,q]["selected_sha256"] for r in rr)
            assert all(r["score_start"]==10000 and r["score_end"]==14096 for r in rr)
            all_candidates[q]=[sum(r["misses"] for r in rr),median(r["us_per_request"] for r in rr)]
        eligible=ROSTER[2:]
        expected=min(eligible,key=all_candidates.get)
        assert freeze["winner"]==expected
        assert set(freeze["selection"])==set(ROSTER)|{f"{expected}_s19",f"{expected}_s42"}
        for q in (*ROSTER,*SELECTED):
            model=q if q not in SELECTED else expected+({"R_SELECTED_s7":"","R_SELECTED_s19":"_s19","R_SELECTED_s42":"_s42"}[q])
            digest_=found[f,model]["selected_sha256"]
            assert digest_==freeze["selection"][model]
            assert set(found[f,model]["test"])=={f"{f}/{c}" for c in CAPACITIES}
            for cap in CAPACITIES:
                d=raw["test",q,f,cap]
                assert d["checkpoint_sha256"]==digest_ and (d["score_start"],d["score_end"])==(10000,50000)
        for q in NATIVE:
            for c in CAPACITIES:
                d=raw["test",q,f,c]
                assert d["misses"]==archive[q,f,c]["misses"] and d["requests"]==40000
        per_fold[f]=dict(validation=v,all_candidates=all_candidates,overall_best=min(ROSTER,key=all_candidates.get),
                         winner=expected,selected_hash=freeze["selection"][expected])
    assert len(raw)==548
    for (m,f,c),r in archive.items():
        raw.setdefault(("test",m,f,c),r)
    cells=[(f,c) for f in PROTOCOL for c in CAPACITIES]
    data=[]
    for m in sorted({k[1] for k in raw if k[0]=="test"}):
        own=[raw["test",m,f,c] for f,c in cells]
        delta=[r["misses"]-raw["test","lru",f,c]["misses"] for r,(f,c) in zip(own,cells)]
        total=sum(r["misses"] for r in own)
        lru=sum(raw["test","lru",f,c]["misses"] for f,c in cells)
        byfam={f:sum(raw["test",m,f,c]["misses"] for c in CAPACITIES) for f in PROTOCOL}
        report=dict(method=m,misses=total,miss_ratio=total/840000,relative_delta_lru=(total-lru)/lru,
                    families=byfam,wins=sum(d<0 for d in delta),ties=delta.count(0),losses=sum(d>0 for d in delta))
        for k in ("us_per_request","recurrent_us_per_request","eviction_p95_us","tensor_state_peak_bytes","metadata_peak_shallow_bytes"):
            values=[r[k] for r in own if r.get(k) is not None]
            report[k]=median(values) if values else None
        if m in (*ROSTER,*SELECTED):
            vals=[]
            for f in PROTOCOL:
                tag=per_fold[f]["winner"]
                tag={"R_SELECTED_s7":tag,"R_SELECTED_s19":tag+"_s19","R_SELECTED_s42":tag+"_s42"}.get(m,m)
                vals.extend(found[f,tag]["test"].values())
            report["offline_discriminative_macro"]=mean(x["discriminative_accuracy"] for x in vals if x["discriminative_accuracy"] is not None)
            report["offline_cf_regret_macro"]=mean(x["mean_cf_regret"] for x in vals)
        data.append(report)
    return dict(status="COMPLETED_SUCCESS",phase1_state=status,verified_selected_and_latest=2*len(found),
                development_replays=212,outer_replays_native=63,outer_replays_new=168,nested_replays=105,
                per_fold=per_fold,results=sorted(data,key=lambda r:r["misses"]))


def markdown(d):
    if d["status"]!="COMPLETED_SUCCESS":
        return "# Phase 1 terminal audit\n\nPhase 1 still lacked terminal evidence at the audit snapshot.\n"
    lines=["# Phase 1 terminal audit","",
           "Source `fae8c8f5d5654b3d41537f3d19284d48c9240a55`. Verified seven frozen folds, 21 cells, 77 completed fits and 154 selected/latest hashes. All 548 measured replay records reconcile. Native classical controls matched all 63 archived counts.",
           "",f"Terminal timestamp `{d['phase1_state']['updated_at']}`. Wall time {d['phase1_state']['elapsed_seconds']:.2f} seconds.",
           "", "The three seeded finalist rows use the design selected separately in each fold. The seed-7 finalist is also replayed in the base roster under its method name. These duplicate labels are matched replays, not independent fits.",
           "", "| Outer test policy | Misses | Δ vs LRU | W/T/L | BrightKite / Citi Bike / Alibaba / MetaCDN / MetaKV / Twemcache / Wikimedia | CPU µs/request | Offline discr. accuracy |",
           "|---|---:|---:|---:|---|---:|---:|"]
    for r in d["results"]:
        t=r["us_per_request"]
        a=r.get("offline_discriminative_macro")
        lines.append(f"| {r['method']} | {r['misses']:,} | {r['relative_delta_lru']:+.2%} | {r['wins']}/{r['ties']}/{r['losses']} | "+"/".join(f"{x:,}" for x in r["families"].values())+f" | {'—' if t is None else round(t,2)} | {'—' if a is None else format(a,'.2%')} |")
    lines.extend(["", "## Nested validation checks","",
                  "| Held out | Validation | R02 T03 | R08 T03 | R05 T02 | R05 T03 | R06 T03 | Recurrent winner | Overall winner |",
                  "|---|---|---:|---:|---:|---:|---:|---|---|"])
    for f,q in d["per_fold"].items():
        lines.append(f"| {f} | {q['validation']} | "+" | ".join(str(v[0]) for v in q["all_candidates"].values())+
                     f" | {q['winner']} | {q['overall_best']} |")
    lines.extend(["","MetaCDN is expected under the code's eligibility filter. The MetaKV validation winner among all candidates was R08 T03 (9,798 misses). The best eligible recurrent model was R05 T02 (9,827). Its seed-7 checkpoint and both replications match the frozen information and their own replay rows.",
                  "", "Runtime is median across cells and only for local measurements. PHASE1_TERMINAL.json adds memory, eviction latency, update cost, all frozen decisions, and offline regret. The terminal outcome uses the full requested KBS grid and remains development-lineage evidence."])
    return "\n".join(lines)+"\n"
