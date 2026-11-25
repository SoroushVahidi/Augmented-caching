"""Check feature and chronological eviction parity on verified Wulver partitions."""
from pathlib import Path
import sys

import joblib
import numpy as np

from assets import FEATURE_NAMES, MECHANISMS, dump, selected_features
from campaign import trace_data
from deploy import replay


def verify(dataset,out,trace_root):
    result=[]
    for family,cap in (("cloudphysics",32),("metakv",128)):
        data=trace_data(trace_root,family)
        with np.load(dataset/"data"/family/str(cap)/"groups-0000.npz",allow_pickle=False) as source:
            z={k:source[k] for k in source.files}
        indices=np.linspace(0,len(z["position"])-1,12,dtype=int)
        lookup={int(z["position"][i]):j for j,i in enumerate(indices)}
        modes=[(g,"low_frequency") for g in FEATURE_NAMES]
        modes.extend((("F0","ghost_informed"),("F0","sieve_derived"),("F0","uniform_7")))
        for group,mode in modes:
            shortlist=z["shortlist"][indices,MECHANISMS.index(mode),:8]
            expected=selected_features(z,indices,shortlist,group)
            checked=[]
            def compare(inp,request,residents,flags,native):
                if request.t not in lookup:
                    return
                j=lookup[request.t]
                np.testing.assert_array_equal([data[2][r] for r in residents],z["resident"][indices[j]])
                ranks=inp.ranks(residents,request.t,8,flags,native)
                np.testing.assert_array_equal(ranks,shortlist[j])
                np.testing.assert_array_equal(inp.x(request,residents,ranks),expected[j])
                checked.append(request.t)
            ev=replay(data,family,cap,artifact={"spec":{"features":group,"shortlist":mode,"k":8}},
                    control={"rule":"lru"},end=max(lookup)+1,verify_callback=compare)
            assert len(checked)==len(lookup)
            result.append(dict(family=family,capacity=cap,features=group,shortlist=mode,checked=len(checked)))
    final=[]
    for name in ("A_ridge_100000_h8","A_hgb_100000_resolved"):
        art=joblib.load(out/"models"/f"brightkite-{name}.joblib")
        v=trace_data(trace_root,"citibike")
        a=replay(v,"citibike",64,art,{"rule":"gate","value":0.},end=14000,signature=True,profile=True)
        b=replay(v,"citibike",64,art,{"rule":"gate","value":0.},end=14000,signature=True,profile=True,reference=True)
        assert a["eviction_sha256"]==b["eviction_sha256"] and a["misses"]==b["misses"]
        final.append(dict(model=name,reference=b,optimized=a))
    dump(out/"evidence"/"OPTIMIZED_INFERENCE_VALIDATION.json",dict(
        scope="bounded preflight; CPU on Wulver",feature_checks=result,
        chronological_parity=final,all_checks_match=True))


if __name__=="__main__":
    verify(*map(Path,sys.argv[1:]))
