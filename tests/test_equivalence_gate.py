from __future__ import annotations

import sys
sys.path.insert(0, "src")

import collections
import pandas as pd
from lafc.evict_value_wulver_v1 import WulverDatasetConfig, iter_candidate_rows, load_trace_from_any

def test_equivalence_gate():
    # Load canonical heavy_r1 row data for metacdn, capacity=32, horizon=16 (Train split only)
    canonical_path = "/home/soroush/projects/lafc-evict-dataset/repo/release/lafc-evict-v0.1-open/data/candidate_rows/split=train/trace_family=metacdn/capacity=32/horizon=16/candidate_rows.parquet"
    print(f"Loading canonical Parquet: {canonical_path}")
    df_canon = pd.read_parquet(canonical_path)
    
    # Filter canonical data to decisions t < 10000 to keep it lightweight for checking
    df_canon = df_canon[df_canon["decision_t"] < 10000]
    print(f"Filtered canonical rows: {len(df_canon)}")
    
    # Construct a lookup dictionary of canonical rows: (decision_t, candidate_page_id) -> row
    canon_by_key = {}
    for _, row in df_canon.iterrows():
        key = (int(row["decision_t"]), str(row["candidate_page_id"]))
        canon_by_key[key] = row
        
    # Load trace
    trace_path = "/home/soroush/projects/augmented-caching/repo/data/processed/metacdn/trace.jsonl"
    print(f"Loading local trace: {trace_path}")
    reqs, _pages, inferred_source = load_trace_from_any(trace_path)
    
    # Slice trace to match the t < 10000 range.
    reqs = reqs[:15000]
    
    cfg = WulverDatasetConfig(
        horizons=(16,),
        chunk_size=4096,
        split_mode="trace_chunk",
        split_seed=7,  # Ensure seed is 7, same as heavy_r1
    )
    
    # Generate local rows
    print("Generating local rows...")
    local_rows = []
    for r in iter_candidate_rows(
        requests=reqs,
        capacity=32,
        trace_name="metacdn_cdn_202303_head_50k",
        dataset_source="metacdn",
        trace_family="metacdn",
        cfg=cfg,
    ):
        if r["decision_t"] < 10000:
            local_rows.append(r)
            
    print(f"Generated local rows: {len(local_rows)}")
    
    # Verify exact match
    mismatch_count = 0
    compared_count = 0
    missing_in_local = 0
    
    for l_row in local_rows:
        if l_row["split"] != "train":
            # Skip non-train split rows since our canonical parquet is train split only
            continue
            
        key = (int(l_row["decision_t"]), str(l_row["candidate_page_id"]))
        if key not in canon_by_key:
            print(f"Key mismatch: local key {key} (split=train) not found in canonical rows.")
            mismatch_count += 1
            continue
            
        c_row = canon_by_key[key]
        
        # Compare y_loss
        if float(l_row["y_loss"]) != float(c_row["y_loss"]):
            print(f"y_loss mismatch at t={key[0]}, candidate={key[1]}: local={l_row['y_loss']}, canon={c_row['y_loss']}")
            mismatch_count += 1
            
        # Compare y_value
        if float(l_row["y_value"]) != float(c_row["y_value"]):
            print(f"y_value mismatch at t={key[0]}, candidate={key[1]}: local={l_row['y_value']}, canon={c_row['y_value']}")
            mismatch_count += 1
            
        compared_count += 1
        
    print(f"Compared {compared_count} rows. Total mismatches: {mismatch_count}")
    assert mismatch_count == 0, f"Encountered {mismatch_count} mismatches between local and canonical data!"
    assert compared_count == len(df_canon), f"Compared row count ({compared_count}) does not match canonical row count ({len(df_canon)})!"
    print("LABEL_EQUIVALENCE_GATE = PASS")

if __name__ == "__main__":
    test_equivalence_gate()
