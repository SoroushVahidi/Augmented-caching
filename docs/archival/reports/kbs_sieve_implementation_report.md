# SIEVE implementation report (2026-06-19)

Manuscript: KNOSYS-D-26-07461, in response to Reviewer #3's R3-Issue3/R3-Rec2
(see `reports/kbs_real_reviewer_comments.md`). This report documents the
implementation that followed the zero-compute verification pass in
`reports/kbs_sieve_source_verification.md`. cap64/cap128/cap256 were **not**
launched in producing this report; the only compute performed was the single
authorized tiny smoke test in §5 (1 trace, 1000 requests, capacity 32, 2
policies).

## 1. Sources used

- Official NSDI'24 paper, Algorithm 1 (§3.1, p.1232), downloaded directly
  from `https://www.usenix.org/system/files/nsdi24-zhang-yazhuo.pdf` and read
  in full for the relevant sections.
- Official reference implementation, `libCacheSim/cache/eviction/Sieve.c`
  (`https://github.com/1a1a11a/libCacheSim`), fetched independently and
  cross-checked line-for-line against Algorithm 1 — no discrepancy found.
- Full verification detail (citation, URLs, exact pseudocode, prose
  semantics, independent code cross-check) is in
  `reports/kbs_sieve_source_verification.md`, items 1-8. That report was
  completed and reviewed for ambiguity *before* any implementation code was
  written, per the gating instruction for this task.

## 2. Exact semantics implemented

Implemented Algorithm 1 verbatim, with no behavioral deviation:

- One FIFO-ordered resident structure (`self._order`, a `dict` used as an
  ordered set) giving O(1) membership, O(1) append-at-head, and O(1)
  remove-from-anywhere.
- One `visited: Dict[PageId, bool]` bit per resident page, defaulting to
  `False` at insertion (Algorithm 1 line 16).
- One `hand` cursor (`self._hand`), initialized to `None` (paper's NULL).
- **Cache hit**: set `visited[pid] = True`. Nothing else — no reordering, no
  move-to-head/move-to-end (the one rule most likely to be copied wrong from
  an LRU-style implementation; `src/lafc/policies/lru.py`'s hit path calls
  `move_to_end`, which SIEVE's hit path must not and does not replicate).
- **Cache miss, cache full**: scan starts at the hand (or the tail/oldest end
  if the hand is unset or no longer resident), walks toward the head/newest
  end, clears `visited=True -> False` for every object passed, wraps from
  past-the-head back to the tail if the scan runs off the newest end, and
  evicts the first object found with `visited = False`.
- **Hand update after eviction**: set to the evicted object's predecessor
  toward the head (`p <- o.prev`), exactly as Algorithm 1 line 13.
- **Insertion**: every newly admitted page goes to the head with
  `visited = False`.

This is a literal, unweighted-paging-native implementation: no
reinterpretation was needed, because (per source-verification report §7)
Algorithm 1's own cache-full check (`|T| = C`) is already an object-count
check, which is exactly what this repo's `CacheState.is_full()` already
does.

## 3. Files changed

- **`src/lafc/policies/sieve.py`** (new file) — `SievePolicy(BasePolicy)`,
  `name = "sieve"`. Implements `reset()` and `on_request()` per the
  `BasePolicy` interface (`src/lafc/policies/base.py`), using the existing
  `_add()`/`_evict()`/`_record_hit()`/`_record_miss()` helpers exactly as
  `lru.py` and `marker.py` do.
- **`scripts/run_policy_comparison_wulver_v1.py`** — added
  `from lafc.policies.sieve import SievePolicy` and one `POLICIES` dict entry,
  `"sieve": lambda _: SievePolicy()`. This is the canonical heavy_r1 pipeline;
  no other change was needed since `--policies` already does generic
  dict-key filtering against `POLICIES`.
- **`src/lafc/runner/run_policy.py`** — added the same import and one
  `POLICY_REGISTRY` entry, `"sieve": SievePolicy()`, plus added `sieve` to
  the module docstring's list of supported `--policy` CLI values. This is
  the older/general runner shared by most existing unit tests (including the
  new SIEVE tests, via `run_policy()`); kept in sync so `sieve` is usable
  identically from either entry point and so no test infra needed
  special-casing.
- **`tests/test_sieve.py`** (new file) — see §4.

No other files were modified. No existing policy file, registry entry, or
CLI flag was changed beyond the additions above. No canonical analysis
output, dataset, model, or `.bib`/`.tex` file was touched.

## 4. Tests and results

Added `tests/test_sieve.py` with 8 tests covering exactly the 6 requested
behaviors (the eviction-scan behavior split into two tests for clarity):

1. `test_sieve_hit_sets_visited_without_moving` — a hit sets `visited=True`
   and leaves queue order unchanged (no move-to-head).
2. `test_sieve_new_insertions_go_to_the_head` — a newly inserted page is the
   newest/head-most element, with `visited=False`.
3. `test_sieve_eviction_skips_and_clears_visited_then_evicts_first_unvisited`
   — with A and B re-accessed (visited=True) and C untouched, a full-cache
   miss clears A and B's visited bits and evicts C, matching a hand-traced
   walk through Algorithm 1 lines 5-14.
4. `test_sieve_eviction_removes_first_unvisited_item_found_by_hand` — with no
   page re-accessed, the scan stops immediately at the tail and evicts the
   oldest page.
5. `test_sieve_capacity_one` — every miss at capacity 1 evicts the
   immediately preceding resident page; a hit on the sole resident page
   causes no eviction.
6. `test_sieve_runs_via_run_policy_helper` — integration smoke check that
   `SievePolicy` runs cleanly through the shared `run_policy()` driver.
7. `test_runner_policies_dict_accepts_sieve` — confirms
   `scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES["sieve"]`
   resolves to a working `SievePolicy` instance (the canonical pipeline's
   `--policies sieve` integration point), with zero trace I/O.
8. `test_runner_policy_registry_accepts_sieve` — same check against
   `src/lafc/runner/run_policy.py`'s `POLICY_REGISTRY`.

Ran exactly the requested command, using the existing `.venv_kbs_heavy_r1`
virtualenv (an editable `lafc` install pointing at this checkout, left over
from the prior phase's heavy_r1 work):

```
pytest tests/ -k "sieve or policy_comparison or lru or marker" -v
```

Result: **45 passed, 246 deselected**, 0 failed. All 8 new SIEVE tests
passed, alongside all pre-existing LRU/Marker/Predictive-Marker/Atlas/
Baseline-4 tests matched by the same `-k` filter (no regression).

As an extra (not requested but low-cost) check, also ran the full suite:

```
pytest tests/ -q
```

Result: **291 passed**, 0 failed, 12 warnings (all pre-existing PuLP
deprecation warnings, unrelated to this change). This is 8 more than the
283 documented in `AGENTS.md`, exactly matching the 8 new SIEVE tests added.

## 5. Smoke result

Ran exactly the requested tiny smoke-test command (1 trace, 1000 requests,
capacity 32, policies `lru,sieve`), writing only to the two new,
non-canonical files named explicitly in the instruction:

```
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 --max-traces 1 --max-requests-per-trace 1000 \
  --policies lru,sieve \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_sieve_smoke.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_sieve_smoke.md
```

Result (1 trace, family `brightkite`, capacity 32, 1000 requests):

| Policy | Mean misses |
|---|---|
| lru | 503.0 |
| sieve | 496.0 |

SIEVE produced **1.39% fewer misses than LRU** on this single tiny sample —
directionally consistent with the published SIEVE result (SIEVE is reported
to be competitive with or better than LRU on most workloads) and, more
importantly for this report's purpose, evidence that the implementation
runs cleanly end-to-end through the canonical pipeline, produces a
plausible (non-degenerate, non-error) miss count, and does not crash, hang,
or evict the entire cache on every step. This is a sanity check on a single
tiny sample, not a claim about SIEVE's performance at scale or across the
full trace/capacity grid — no such claim is made here.

No canonical output file was overwritten: confirmed via
`git status --short -- analysis/` before and after the run, which shows only
the two new smoke-test files as untracked additions; the one pre-existing
modified file in `analysis/` (`evict_value_v1_wulver_dataset_summary_heavy_r1.md`)
predates this task and was not touched by it.

## 6. Whether SIEVE is ready for cap64/cap128/cap256

**Yes.** The implementation is deterministic, faithful to the verified
Algorithm 1 semantics (§2), O(1) amortized per request (each request does
O(1) work on a hit; a full-cache miss does amortized O(1) work across the
hand-scan, since each scanned object's visited bit is cleared at most once
per "lap" of the hand around the queue, the same amortized argument the
paper itself makes), passes all 8 targeted unit tests plus the full 291-test
suite with no regressions, and produces a plausible result in the tiny
smoke test. It is registered in the canonical heavy_r1 pipeline's `POLICIES`
dict under the exact CLI name `sieve`, so `--policies ...,sieve,...` (or
omitting `--policies` to run all available policies) will include it in any
future cap64/cap128/cap256 launch with no further code changes required.

## 7. Whether cap32 must be rerun with SIEVE

**Recommended, but not forced by this report alone — a decision for the
user.** Per `reports/kbs_before_cap64_baseline_decision_memo.md` §5, the
canonical pipeline writes its CSV/MD exactly once at the end of a run and
has no mechanism to append a policy column to an already-completed chunk.
cap32 (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv`)
was run before SIEVE existed and therefore has no SIEVE column. cap32's own
runtime was ~5:13:02 for 7 traces x 7 policies at one capacity — substantially
cheaper than any of the still-unlaunched cap64/128/256 chunks — so re-running
it to add SIEVE (and any other newly added baseline) is the cheapest point
at which to absorb that cost, but this report does not launch that rerun;
it only confirms the engineering precondition (a working `sieve` policy) is
now satisfied.

## 8. Remaining ambiguity

**None that blocks SIEVE itself.** The source-verification report
(`reports/kbs_sieve_source_verification.md` §7-8) already concluded there is
no semantic ambiguity in adapting Algorithm 1 to this repo's unweighted
paging simulator, and the implementation in this report followed that
conclusion literally with no deviation. The only open items are scope/
scheduling decisions outside SIEVE's own correctness:

- Whether and when to rerun cap32 with SIEVE included (§7, user decision).
- FIFO-Reinsertion's exact intended variant remains undefined (separate
  baseline, unrelated to SIEVE; tracked in
  `reports/kbs_before_cap64_baseline_decision_memo.md` §2 and unaffected by
  this report).
- Whether to add a `.bib` citation entry for `zhang2024sieve` and a Table 2
  roster row in the manuscript — a manuscript-editing task, not a code or
  semantics question, out of scope for this zero-compute-except-smoke-test
  report.
