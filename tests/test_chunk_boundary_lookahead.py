from __future__ import annotations

import collections
from lafc.types import PageId, Request
from lafc.evict_value_wulver_v1 import WulverDatasetConfig, iter_candidate_rows
from lafc.simulator.request_trace import build_requests_from_lists
from lafc.evict_value_dataset_v1 import _simulate_lru_misses

def test_chunk_boundary_lookahead():
    # Construct a trace of 5000 items (spanning across 4096 boundary)
    # The split chunk boundary is at 4096. Let's make sure we have eviction decisions
    # within 128 requests of 4096 (e.g. between 3968 and 4096, and after).
    
    # We will generate a trace of repeating sequence to keep it deterministic but causing misses
    # Let's use 50 unique items in rotation. Capacity is 10.
    page_ids = [f"P_{i % 50}" for i in range(5000)]
    prediction_records = [{"bucket": 1, "confidence": 0.8}] * 5000
    
    reqs, _ = build_requests_from_lists(page_ids=page_ids, prediction_records=prediction_records)
    
    capacity = 10
    horizons = (16, 32, 64, 128)
    cfg = WulverDatasetConfig(
        horizons=horizons,
        chunk_size=4096,
        split_mode="trace_chunk",
    )
    
    # Generate rows
    rows = list(
        iter_candidate_rows(
            requests=reqs,
            capacity=capacity,
            trace_name="test_boundary_trace",
            dataset_source="test_source",
            trace_family="test_family",
            cfg=cfg,
        )
    )
    
    # Index the generated rows by (decision_t, candidate_page_id, horizon) for O(1) lookup
    rows_by_key = {
        (r["decision_t"], r["candidate_page_id"], r["horizon"]): r
        for r in rows
    }
    
    # Now let's implement a simple independent reference replay to check correctness.
    # We will manually simulate the cache to find the decisions, and compute y_loss for each.
    order = collections.OrderedDict()
    decisions_checked = 0
    
    for t, req in enumerate(reqs):
        pid = req.page_id
        hit = pid in order
        if hit:
            order.move_to_end(pid)
            continue
        if len(order) < capacity:
            order[pid] = None
            continue
            
        # This is a miss and cache is full, so it's an eviction decision
        candidates = list(order.keys())
        
        future = reqs[t + 1 :]
        
        for candidate in candidates:
            for h in horizons:
                fut_h = future[:h]
                after = [p for p in candidates if p != candidate] + [pid]
                expected_loss = _simulate_lru_misses(after, fut_h, capacity=capacity)
                
                key = (t, candidate, h)
                assert key in rows_by_key, f"Expected 1 matching row for t={t}, cand={candidate}, h={h}"
                actual_loss = rows_by_key[key]["y_loss"]
                assert actual_loss == expected_loss, f"Mismatch at t={t}, cand={candidate}, h={h}: expected {expected_loss}, got {actual_loss}"
                decisions_checked += 1
                
        # Simulate LRU eviction for the reference cache
        lru_victim = next(iter(order))
        order.pop(lru_victim)
        order[pid] = None
        
    print(f"Verified {decisions_checked} candidate-horizon label pairs across chunk boundaries.")
    assert decisions_checked > 0
