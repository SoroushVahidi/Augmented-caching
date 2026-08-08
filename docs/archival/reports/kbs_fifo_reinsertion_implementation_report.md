# FIFO-Reinsertion implementation report (2026-06-19)

Zero-heavy-compute pass, done in parallel with the running
`kbs_full_policy_comparison_cap32_with_sieve` tmux job (PID 191758,
confirmed still running at time of writing — not touched). Implements the
baseline requested in R3-Issue3/recommended in
`reports/kbs_baseline_gap_action_plan.md` once its definition was pinned
down in `reports/kbs_halp_fifo_source_verification.md` §2.

## 1. What was implemented

`src/lafc/policies/fifo_reinsertion.py` — `FIFOReinsertionPolicy`
(registry name `fifo_reinsertion`), the classical CLOCK / Second-Chance
algorithm, built by directly mirroring `src/lafc/policies/sieve.py`'s
state representation (one FIFO queue as an ordered dict + one visited bit
per resident page) and `BasePolicy` interface usage. The only behavioral
difference from SIEVE: on an eviction scan, a *visited* object encountered
at the tail is cleared and **physically reinserted at the head** (mixed
back in with newly-inserted objects), rather than left in place behind a
separate hand pointer. This removes the need for SIEVE's persistent
`_hand` state — FIFO-Reinsertion's tail is always the true next eviction
candidate after any scan.

Algorithm (see the module docstring for the full citation trail):

1. **Hit**: `visited[pid] = True`. No reordering.
2. **Miss, cache full**: while the object at the tail is visited, clear its
   bit and move it to the head (second chance); once an unvisited object is
   found at the tail, evict it.
3. **Insertion**: new objects go to the head with `visited = False`.

Complexity: O(1) amortized per request — identical complexity class to
SIEVE and LRU (each reinsertion is one dict pop + one dict insert; the
number of reinsertions across the cache's lifetime is bounded by the
number of hits, same amortization argument as SIEVE's hand-scan).

## 2. Wiring

Added to both registries that SIEVE was added to, identically:

- `scripts/run_policy_comparison_wulver_v1.py`: new import
  (`from lafc.policies.fifo_reinsertion import FIFOReinsertionPolicy`) and
  new `POLICIES` dict entry `"fifo_reinsertion": lambda _:
  FIFOReinsertionPolicy()`.
- `src/lafc/runner/run_policy.py`: new import, new `POLICY_REGISTRY` entry
  `"fifo_reinsertion": FIFOReinsertionPolicy()`, and the module docstring's
  supported-`--policy`-values list updated to include `fifo_reinsertion`.

## 3. Tests

`tests/test_fifo_reinsertion.py` — 8 new unit tests, mirroring
`tests/test_sieve.py`'s structure exactly:

1. `test_fifo_reinsertion_hit_sets_visited_without_moving`
2. `test_fifo_reinsertion_new_insertions_go_to_the_head`
3. `test_fifo_reinsertion_evicts_unvisited_tail_without_reinsertion`
4. `test_fifo_reinsertion_reinserts_visited_survivors_at_head_then_evicts_first_unvisited`
   — the key behavioral-difference-from-SIEVE test: confirms visited
   survivors are physically moved to the head (`_residents_oldest_to_newest`
   order changes), not left in place.
5. `test_fifo_reinsertion_capacity_one`
6. `test_fifo_reinsertion_runs_via_run_policy_helper`
7. `test_runner_policies_dict_accepts_fifo_reinsertion`
8. `test_runner_policy_registry_accepts_fifo_reinsertion`

All 8 pass. Full suite: **299/299 passing** (up from 291/291 before this
pass — the +8 SIEVE tests from the prior pass plus these +8 new ones,
net change from the pre-SIEVE baseline of 283 is +16).

```
$ python -m pytest tests/test_fifo_reinsertion.py -q
........                                                                [100%]
8 passed in 0.11s

$ python -m pytest -q
299 passed, 12 warnings in 10.96s
```

## 4. Smoke test (non-canonical, tiny sample — same pattern as SIEVE's)

Ran the same tiny smoke-test shape used to validate SIEVE (1 trace, 1000
requests, capacity 32), writing only to new, non-canonical filenames:

```
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 --max-traces 1 --max-requests-per-trace 1000 \
  --policies lru,sieve,fifo_reinsertion \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke.md
```

Result (1 trace, family `brightkite`, capacity 32, 1000 requests, exit code 0):

| Policy | Mean misses | vs. LRU |
|---|---|---|
| lru | 503.0 | 0.00% |
| sieve | 496.0 | +1.39% (fewer misses) |
| fifo_reinsertion | 504.0 | -0.20% (1 more miss) |

This is directionally plausible and not a red flag: FIFO-Reinsertion is a
strictly weaker discipline than SIEVE on this one tiny sample — it mixes
retained objects back into the same FIFO order as new insertions (passive
demotion), so a retained-but-not-recently-useful object can still occupy a
"protected" position for a full cycle before being re-evaluated, whereas
SIEVE's hand keeps scanning forward without artificially refreshing recency
order. On a single 1000-request/1-trace sample, a 1-miss difference from
LRU is well within noise — this smoke test exists only to confirm the
registry wiring and algorithm don't crash or behave nonsensically, **not**
as a citable empirical result (same caveat already stated for SIEVE's
smoke test).

## 5. What is explicitly NOT done in this pass

- **Not** run through any canonical capacity chunk (cap32, cap32_with_sieve,
  cap64, cap128, cap256). It has zero canonical empirical numbers. Adding it
  to a canonical sweep is a separate, larger decision with its own runtime
  cost — same status as SIEVE before its own cap32_with_sieve decision.
- **No bib entry added to `refs.bib`** — see
  `kbs_halp_fifo_source_verification.md` §2.4 for the two BibTeX entries
  needed (`zhang2024sieve`, `yang2023s3fifo`) if/when FIFO-Reinsertion
  results are discussed in the manuscript. Not added here because no
  manuscript edit was made.
- **No canonical-chunk relaunch was triggered or considered** by writing
  this code — this pass is pure zero-compute-except-tiny-smoke-test, per
  the explicit constraint "do not launch any multi-hour experiment."

## 6. Files changed/added this pass

- Added: `src/lafc/policies/fifo_reinsertion.py`
- Added: `tests/test_fifo_reinsertion.py`
- Modified: `scripts/run_policy_comparison_wulver_v1.py` (+2 lines: import, registry entry)
- Modified: `src/lafc/runner/run_policy.py` (+3 lines: import, registry entry, docstring list)
- Added (smoke-test output, non-canonical):
  `analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke.csv`,
  `analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke.md`
