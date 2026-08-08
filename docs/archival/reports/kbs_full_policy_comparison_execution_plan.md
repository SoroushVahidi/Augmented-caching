# KBS heavy_r1 full policy-comparison: execution plan (decision report — nothing launched)

Status: validation pass done (see `reports/kbs_validation_eval_report.md`). This
document answers whether/how to run the full 7-trace × 4-capacity × 7-policy
canonical sweep, and provides exact commands. **Nothing in this document has
been executed.** Supersedes the "estimate" sections of
`reports/kbs_policy_comparison_launch_plan.md` (written before training/
validation completed) with real numbers; that file's chunking rationale
(cost scales with capacity) is confirmed correct by source inspection below
and carried forward here.

## 1. Full canonical command (option A — single shot)

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32,64,128,256 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md
```

No `--max-traces` → all 7 traces. Expected output: 7 × 4 × 7 = 196 data rows.

## 2. Why chunking is the real question, not just speed

Read `scripts/run_policy_comparison_wulver_v1.py:101-131`: all
`(trace, capacity, policy)` results accumulate in one in-memory list
(`rows_out`), and the CSV/MD are written **exactly once**, after the entire
triple-nested loop finishes (`args.out_csv.open("w", ...)` at line 127). There
is:

- **No checkpoint file.**
- **No incremental write.**
- **No flag to skip already-computed combinations.**
- **No resume mechanism of any kind.**

Any interruption — kill, crash, reboot, OOM, an unhandled exception on one
trace late in the run — loses **100%** of the run's progress. You get the old
output file untouched (if interrupted before line 127) or nothing at all; you
never get partial credit.

This already happened once for real: `reports/kbs_revision_repo_audit.md:101`
documents that the exact same full canonical command (job 908326, 16 dedicated
Wulver CPUs) **exceeded a 24h walltime limit before producing any output**,
and the resubmission at 72h walltime never confirmed completion either (it was
blocked by a maintenance window before getting compute time, so 72h is not
established as sufficient — only that 24h was confirmed insufficient). The
model-load bugfix since then (`b9df6f1`) fixes a fast-failing `pickle`/`joblib`
mismatch, not a performance issue, so it doesn't explain or discount that
timeout — the compute genuinely takes >24h at this scope.

## 3. Cost model — why capacity is the dominant variable

`src/lafc/policies/evict_value_v1.py:179-200` (`_choose_victim`): on **every
cache miss**, it iterates over **all candidates currently in the cache** (up
to `capacity` items), builds features and calls `predict_loss_one` once per
candidate. Cost per miss is `O(capacity)` model-inference calls, not O(1).
Other content-aware policies (`trust_and_doubt`, `predictive_marker`,
`blind_oracle`) likely have similar capacity-dependent eviction-scan costs but
were not individually re-verified in this session — treat the capacity
scaling below as confirmed for `evict_value_v1` and a reasonable assumption
for the rest.

## 4. Runtime estimate (from today's validation data point)

Validation: 1 trace × capacity 256 × 7 policies = 14,577s (4:02:57) →
**~2,082s per (trace, policy) at capacity 256.**

Two extrapolation models, both anchored to that one data point:

| Model | Assumption | cap32 chunk (7 traces × 7 pol) | cap64 | cap128 | cap256 | **Total (4 chunks)** |
|---|---|---|---|---|---|---|
| Capacity-proportional (optimistic, matches §3's confirmed O(capacity) cost) | cost ∝ capacity | ~3.5h | ~7.1h | ~14.2h | ~28.3h | **~53h (~2.2 days)** |
| Capacity-independent (conservative) | every capacity costs like cap256 | ~28.3h | ~28.3h | ~28.3h | ~28.3h | **~113h (~4.7 days)** |

**Uncertainty is wide and should be stated as such**: this is extrapolated
from a single `(trace, capacity)` data point (`brightkite` only). The 7 trace
families differ hugely in unique-page count (citibike: 1,917 → wiki2018:
50,000 of 50,000 requests, i.e. no repeats at all), which can swing
per-trace cost in either direction for content-aware policies. Treat the
table above as a planning estimate with a realistic ±2x band, not a
commitment.

**Historical corroboration:** the conservative end of this range
(~4.7 days, or possibly more) is consistent with the prior Wulver job
exceeding 24h on comparable (16-core dedicated) hardware for the same full
scope.

## 5. Chunking: what the CLI actually supports

1. **One capacity at a time** — yes, `--capacities` takes a single value or a
   comma-list; this is the **recommended primary chunk axis** (§3: cost scales
   with capacity, so chunks are naturally graduated in size, cheapest first).
2. **One trace at a time** — only via `--max-traces N` (first-N-rows prefix of
   whatever manifest you pass), or by handing it a custom subset manifest CSV
   for an arbitrary single trace. No native "trace name" selector. Useful as a
   **secondary** split inside the cap256 chunk (the expensive one) to bound
   worst-case loss to ~4h instead of ~28h — optional, not required.
3. **Policy subsets** — yes, `--policies` is already a comma-list.
4. **Separate output files per chunk** — yes, `--out-csv`/`--out-md` are fully
   controllable per invocation.
5. **Resume if interrupted** — **no**, confirmed in §2. Chunking doesn't add
   resume *within* a chunk; it bounds how much work a single failure can cost.
6. **Skip completed rows vs. overwrite** — always recomputes everything in
   memory; never skips. Each chunk run is a fresh, independent computation.
7. **Risk of corrupting the final CSV if interrupted** — low. The output file
   is only opened in `"w"` mode and written near the very end of each
   invocation; an interruption before that point leaves any pre-existing file
   completely untouched. The real risk is total loss of that chunk's compute,
   not a corrupted/half-written file (and even that would be obviously
   short-row-count, not silently wrong).

## 6. Recommended chunk commands (capacity-primary split)

Each writes its own CSV/MD and log; cheapest first so you get an early signal.

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
```

**Chunk 1 — capacity 32 (~3.5h optimistic / ~28h conservative):**
```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.md
```

**Chunk 2 — capacity 64:** same command with `--capacities 64` and
`_cap64` in both output paths.

**Chunk 3 — capacity 128:** `--capacities 128`, `_cap128`.

**Chunk 4 — capacity 256 (the expensive one, ~28.3h optimistic / same
conservative):** `--capacities 256`, `_cap256`. If you want a tighter
loss-bound on this specific chunk, further split it by trace using a
single-row subset manifest per trace (7 sub-chunks of ~4h each, matching
today's validation measurement directly) instead of running all 7 traces in
one ~28h shot.

## 7. tmux wrapper pattern (per chunk, not yet launched)

```bash
tmux new -s kbs_full_policy_comparison_cap32
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
mkdir -p logs/kbs_full_policy_comparison
set -o pipefail
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-requests-per-trace 50000 \
  --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap32.log
echo "CAP32_EXIT=${PIPESTATUS[0]}" | tee -a logs/kbs_full_policy_comparison/cap32.log
```

Repeat in separate named tmux sessions (`..._cap64`, `..._cap128`,
`..._cap256`) with the corresponding `--capacities` value and output paths.

**Optional acceleration (machine is idle):** `nproc`=20, `uptime` load average
0.22, 62GB RAM with 59GB available; the script is single-core/~99%-CPU bound
with no internal parallelism, so the 4 capacity chunks (or the 7 trace
sub-chunks of cap256) can run **concurrently** in separate tmux sessions
without resource contention. Running all 4 capacity chunks at once would cut
wall-clock from the ~53–113h sequential sum down to roughly the cost of the
single slowest chunk (cap256: ~28h). This is optional and only relevant once
you approve a launch — flagging it here so the choice is available, not
recommending it outright over the safer sequential/staged approach below.

## 8. Merge command (after all 4 capacity chunks finish)

```bash
python - <<'PY'
import csv
from pathlib import Path

chunks = [
    "analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv",
    "analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64.csv",
    "analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128.csv",
    "analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap256.csv",
]
rows, fieldnames = [], None
for c in chunks:
    with open(c, newline="", encoding="utf-8") as fh:
        r = csv.DictReader(fh)
        fieldnames = fieldnames or r.fieldnames
        rows.extend(r)

out = "analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv"
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(rows)
print(f"merged {len(rows)} rows from {len(chunks)} chunk files -> {out}")
PY
```

This produces the canonical CSV at the exact expected path. The aggregate
`.md` summary (mean misses, %-vs-LRU, per-family win/tie/loss) is **not**
produced by this merge step — it requires re-running the same aggregation
logic as `run_policy_comparison_wulver_v1.py:132-175` against the merged CSV.
That's a small follow-up script (not written yet, out of scope for this
decision report) — flagging it as a known remaining step before
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` exists, even
after the chunked CSVs are merged.

## 9. Recommendation: **Option B (chunked), not A or C**

- **Not A (single-shot, ~2.2–4.7 days, zero checkpointing):** the exact same
  command already failed to produce output once before on comparable hardware
  (§2/§4). Betting a multi-day blind run on a script with no checkpoint and no
  resume is an unforced risk — any single failure mode (one bad trace, one
  OOM, one accidental session loss) loses everything, including the days
  already spent.
- **Not C (skip the full eval):** the reviewers' stated concern is explicitly
  "lack of end-to-end miss-ratio evaluation" — this is the one concern that
  literally cannot be addressed without this exact artifact. There is no
  resource or time case for skipping it: `reports/kbs_revision_repo_audit.md`
  states a **July 8, 2026** revision deadline (~20 days from today,
  2026-06-18), the machine is idle (load 0.22 / 20 cores / 59GB free), and
  even the conservative 4.7-day estimate fits comfortably with chunking.
- **B (chunked by capacity, cheapest first):** reaches the identical canonical
  output as A, at the identical (or better, if run concurrently) wall-clock
  cost, but bounds the blast radius of any failure to a single chunk
  (worst case ~28h instead of ~113h) and surfaces results incrementally —
  including an early read on whether the validation pass's worrying
  `evict_value_v1` result (§4 of the validation report: 39% *more* misses than
  LRU on `brightkite`) is an isolated artifact or a systematic pattern, well
  before the full sweep finishes.

**Sanity-audit update (2026-06-18, `reports/kbs_validation_sanity_audit.md`,
conclusion A — unblocked):** the gating concern from §10 below has been
resolved. Code audit + 58 passing targeted tests found no bug in
`evict_value_v1`'s feature construction, eviction direction, or
miss-accounting; the model/config used for validation is confirmed fresh.
A second, independent data point at capacity 32 / 5,000 requests
(`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation_sanity.csv`)
again shows `evict_value_v1` worse than LRU (+11.81% misses, vs +39.13% at
cap256/50k) — directionally consistent, not a fluke, and consistent with a
short-training-horizon (h=4) explanation rather than a wiring defect.
Separately, `blind_oracle`'s poor number was traced to a real configuration
gap (the comparison script never attaches real predictions to it, so it
degenerates to evicting the smallest page_id) — but `blind_oracle` is not
in `TABLE3_POLICIES`, so this does not block or taint any manuscript-citable
result. **Recommendation unchanged and now confirmed: proceed with Option B
(chunked sweep), starting with the cap32 chunk in §6.**

## 10. Risks (carry into the final summary)

- Estimate is anchored to one `(trace, capacity)` data point; real total could
  reasonably be ±2x the table in §4.
- No checkpointing inside a chunk — interrupting a ~28h cap256 chunk still
  loses that whole chunk (mitigate by sub-splitting cap256 by trace, §6).
- `evict_value_v1` underperforming LRU on the one trace tested is a *research*
  risk, not just a compute risk — full results might show the model needs
  rework, independent of whether the eval completes cleanly. **Confirmed real**
  by the sanity audit (two independent data points, same direction); not a
  reason to delay the sweep, but expect the full-sweep table to need an
  honest narrative about this rather than a clean "we win" result.
- `blind_oracle` scoring worse than LRU was a known configuration gap (no
  real predictions attached to it by the comparison script), now root-caused
  by the sanity audit. **Not a blocker** — `blind_oracle` is outside
  `TABLE3_POLICIES` — but do not cite its number from this script as a
  meaningful baseline in any narrative.
- Running multiple chunks concurrently (optional §7 acceleration) increases
  total CPU draw on a shared machine — currently idle, but worth confirming no
  other load is expected before launching several at once.
