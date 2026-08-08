# Overhead and scalability evidence (2026-06-19)

Read-only evidence-mining pass against existing logs/reports. **No new jobs
were run to produce this report** — every number below is either read
directly from an existing log/timestamp or computed from such numbers.
Where no measurement exists, that is stated explicitly rather than estimated
as if it were one.

## Part 1 — Offline cost / label construction

### Dataset build (label construction) wall-clock and size

- **Size on disk: 96G**, **662 shard files**, 7 trace families × 4 capacities
  (32/64/128/256) × 3 horizons (4/8/16) worth of finite-horizon eviction-cost
  labels (`data/derived/evict_value_v1_wulver_heavy_r1/manifest.json`).
- **Measured wall-clock, core shard-generation step**: first shard written
  2026-06-17 23:21:23, last shard + `split_summary.csv` written 2026-06-18
  09:39:04 → **10:17:41 (~10.3 hours)**. This bound comes directly from shard
  file mtimes (`find ... -type f | xargs stat`), not from a log with explicit
  timing — `logs/kbs_heavy_r1/build_dataset.log` itself has no per-line
  timestamps, only per-trace×capacity `[done] ... shards=N` lines.
- Per-trace shard counts vary 4–23 per (trace, capacity) combination,
  increasing with capacity (more candidate windows per capacity) — e.g.
  `brightkite_50k__cap32 shards=4` vs. `brightkite_50k__cap256 shards=23`;
  `cloudphysics_alibaba_block_head_50k__cap256 shards=72` is the largest
  single combination.
- A separate post-step, `dataset_summary_extended_heavy_r1.json` generation,
  finished at 10:30:02 (50m58s after the core build) — this is dataset
  *analysis*, not part of label construction itself, and should not be
  counted in the build-cost number above.

### Training cost

- **Measured wall-clock: ~7–8 minutes** for all 9 model fits (3 horizons ×
  3 models: ridge, random_forest, hist_gb) — bounded between
  `started_at.txt` (10:29:52) for the combined summary+train+validation
  session and `train_run.log`'s mtime (10:37:52). This includes a brief
  dataset-summary regeneration step at the start, so the pure training time is
  a slight overestimate of the true figure, not an underestimate.
- This is fast because it trains on a **capped, sampled validation/training
  set** (`--max-val-rows 250000`, with the known early-stop sampling artifact
  documented in `reports/kbs_revision_gap_tracker.md` — only ~4,572 rows
  actually used for validation/horizon selection, ~3.2% of the ~141,000
  available), not on the full 96G of shards directly. Training cost itself is
  **not** the scalability concern; label construction (the 10.3h step above)
  is.

### What label-construction scalability evidence is and isn't available

- **Available**: total wall-clock (10:17:41) and total output size (96G,
  662 shards) for the full 7-trace × 4-capacity × 3-horizon sweep — a real,
  measured data point for "how long does building the training data take at
  this scale."
- **Not available**: a per-trace or per-capacity breakdown of build time
  (the build log has no per-line timestamps, only a final `[done]` marker per
  combination with shard counts, not durations). Cannot currently say "cap256
  took Xh vs cap32 took Yh" for the *build* step the way we can for the
  *eval* step (Part 2 has this breakdown).
- **Scalability limitation, as already self-acknowledged in the manuscript**:
  `main.tex`'s Limitations section already discusses that the learning target
  requires offline counterfactual/finite-horizon replay to construct, which is
  inherently more expensive than supervised targets that don't require
  simulating future cache state — this acknowledgment already exists in
  prose; what's missing is a concrete number to back it (now available above:
  ~10.3h / 96G for this dataset's scale, which can be cited directly).

## Part 2 — Online decision overhead

### Theoretical complexity (code-verified)

- **`evict_value_v1`** (`src/lafc/policies/evict_value_v1.py:179-200`,
  `_choose_victim`): on every cache miss when the cache is full, it iterates
  over **all `capacity` candidates currently resident**, builds a feature
  vector for each, and calls `predict_loss_one` once per candidate. This is
  **O(capacity) model-inference calls per miss**, confirmed directly from the
  loop body (`for cand in candidates: ... pred_losses[cand] =
  self._scorer.predict_loss_one(feat)`).
- **`lru`** (`src/lafc/policies/lru.py:47`): eviction is
  `collections.OrderedDict.popitem(last=False)` — **O(1) per miss**, no
  per-candidate scoring.
- **SIEVE** (not implemented in this repo — see
  `reports/kbs_baseline_gap_action_plan.md`): published design uses a single
  moving "hand" pointer over a circular list with O(1) amortized eviction,
  i.e. also O(1)-class, not O(capacity) — cited here only as the textbook
  complexity claim from the SIEVE paper, **not independently re-verified
  against any implementation in this repo**, since none exists yet.
- `trust_and_doubt`, `predictive_marker`, `blind_oracle_lru_combiner` were
  **not** re-audited line-by-line in this pass (out of scope for this
  read-only mining task); the execution plan's existing note that they
  "likely have similar capacity-dependent eviction-scan costs" stands as an
  unverified assumption carried forward, not newly confirmed here.

### Measured wall-clock (what exists today)

Three independent timing anchors exist, **all confounded by differing trace
mixes and/or request counts** — useful as rough magnitude evidence, not as a
clean controlled scaling curve:

| Run | Scope | Total wall-clock | Per (trace, policy) pair |
|---|---|---|---|
| cap32 chunk | 7 trace families × 7 policies × capacity 32 × 50k req/trace | 5:13:02 (`/usr/bin/time` not used; tmux `date` markers) | **383.3 s/pair** (average across 7 different traces) |
| Validation pass | 1 trace (brightkite) × 7 policies × capacity 256 × 50k req | 4:02:57 (`/usr/bin/time -v`, "Elapsed (wall clock) time") | **2,082.4 s/pair** (single trace only) |
| Validation-sanity | 1 trace (brightkite) × 2 policies (lru, evict_value_v1) × capacity 32 × **5k req** | 2:14.73 (`/usr/bin/time -v`) | 67.4 s/pair (10x fewer requests than the other two rows) |

- Naive ratio of the first two rows: 2,082.4 / 383.3 ≈ **5.4x** for a nominal
  8x capacity increase (32→256) — *lower* than a clean O(capacity) prediction
  would suggest, but this is **not a controlled comparison**: the cap32
  figure averages over 7 different trace families (some cheap, some
  expensive) while the cap256 figure is brightkite only. It is not valid
  evidence that the real scaling is sub-linear — it's simply not the same
  experiment run twice at two capacities.
- No existing log records timing **per individual (trace, capacity, policy)
  triple** — only per-chunk totals (one wall-clock number for an entire
  multi-trace, multi-policy run). This is the actual gap: the infrastructure
  needed to produce a clean overhead table doesn't exist yet, not just the
  data.

### What still needs an actual timing benchmark

To get a manuscript-ready overhead table (e.g., "ms per decision, `lru` vs.
`evict_value_v1`, as a function of capacity"), the following is **not yet
measured and would need new, lightweight instrumentation**:

1. Per-policy, per-capacity wall-clock or per-decision latency, holding the
   trace fixed (single-trace, multi-capacity sweep with timing printed per
   policy, not just per chunk).
2. Ideally measured with `time.perf_counter()` around just the eviction-
   decision call (excluding I/O/trace-loading), to isolate the policy's own
   cost from the harness's fixed overhead.
3. This does **not** require the full 7-trace canonical sweep — a single
   trace (e.g. brightkite, already used for both prior timing anchors) run
   once per capacity with per-policy timing added would likely take well
   under an hour and would directly answer the overhead-analysis concern.

### Suggested lightweight timing benchmark (proposed, not run)

```bash
# Illustrative only — not executed as part of this audit.
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --max-traces 1 \
  --capacities 32,64,128,256 \
  --max-requests-per-trace 50000 \
  --policies lru,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv /tmp/overhead_probe.csv --out-md /tmp/overhead_probe.md
```

This would need a small code change (per-policy timer around the
eviction-decision call, written to a side-channel timing column) to be more
informative than the existing chunk-level wall-clock alone — flagged as a
small, scoped addition to `scripts/run_policy_comparison_wulver_v1.py`, not
attempted in this read-only pass.

## Summary: measured vs. proposed

| Claim | Status |
|---|---|
| Dataset build took ~10.3h for 96G/662 shards | **Measured** (shard mtimes) |
| Training took ~7–8 min for 9 model fits | **Measured** (log/started_at mtimes) |
| `evict_value_v1` is O(capacity) per miss; `lru` is O(1) | **Measured / code-verified** (direct source read) |
| cap32 chunk averaged 383s per (trace,policy); cap256/brightkite took 2,082s per (trace,policy) | **Measured**, but confounded — not a controlled scaling curve |
| Real per-decision latency table (ms/decision vs. capacity, controlled) | **Not measured** — proposed benchmark above, not run |
| SIEVE is O(1) amortized | **Cited from the published design**, not verified against any implementation in this repo (none exists) |
