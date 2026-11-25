"""Extract a minimal local Phase 2A alignment fixture for private Wulver parity."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

from campaign import CAPS, PROTOCOL


def run(phase2a, november, output):
    fixture, wanted = {}, {}
    for family in PROTOCOL:
        for cap in CAPS:
            key=f"{family}_{cap}"
            receipt=json.loads((phase2a/"units/state"/f"{family}-{cap}.json").read_text())
            times=np.array([receipt["sample"]["temporal_stratified"][i] for i in (5,27,50)])
            d=november/f"{family}-{cap}"
            full=np.load(d/"t.npy",mmap_mode="r")
            ix=np.searchsorted(full,times)
            np.testing.assert_array_equal(full[ix],times)
            for name in ("x","y"):
                fixture[key+"_"+name]=np.load(d/f"{name}.npy",mmap_mode="r")[ix]
            fixture[key+"_t"]=times
            fixture[key+"_resolved"]=np.full((len(times),cap),9999,dtype=np.int32)
            fixture[key+"_next"]=np.full((len(times),cap),-1,dtype=np.int32)
            wanted.update({(family,cap,int(t)):(key,i) for i,t in enumerate(times)})
    with (phase2a/"ACTION_LEVEL_EVIDENCE.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            if r["sample"]!="temporal_stratified":
                continue
            match=wanted.get((r["family"],int(r["capacity"]),int(r["t"])))
            if match is None:
                continue
            key,i=match
            assert int(r["resolution"])>0
            idx=int(r["candidate_recency_order"])
            fixture[key+"_resolved"][i,idx]=int(r["lru_paired_advantage"])
            fixture[key+"_next"][i,idx]=int(r["next_position"] or 50000)
    for k,x in fixture.items():
        if k.endswith("_resolved"):
            assert not (x==9999).any()
        if k.endswith("_next"):
            assert not (x<0).any()
    np.savez_compressed(output,**fixture)


if __name__=="__main__":
    run(*map(Path,sys.argv[1:]))
