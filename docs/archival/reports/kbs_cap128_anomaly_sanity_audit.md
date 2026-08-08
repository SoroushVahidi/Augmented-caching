# cap128 anomaly sanity/root-cause audit (2026-06-21)

**Purpose:** Low-cost, read-mostly sanity check on the cap128 reversal —
`evict_value_v1`'s gap vs LRU widened sharply (cap32 +4.84% → cap64 +2.26% →
cap128 +11.76%), concentrated in `brightkite` and `citibike` — to determine
whether this is real model/policy behavior or a bug/artifact, **before**
deciding on cap256.

**Scope discipline:** No cap256 launch. No full capacity sweep. No
overwrite of any existing cap32/cap64/cap128 canonical output. No
push/commit/merge/delete/rename. One check (full-scale 2-trace targeted
recheck) was estimated to exceed the ~1h budget for this pass and was
**not run** — see §3.

---

## 1. Raw cap128 row verification

Read directly from
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv/.md`:

- 56 data rows total = 7 trace families × 8 policies, no duplicates, no
  missing (trace, policy) pairs.
- `capacity` column is uniformly `128` on every row.
- All 8 canonical policies present for every trace: `lru`, `sieve`,
  `fifo_reinsertion`, `predictive_marker`, `blind_oracle_lru_combiner`,
  `trust_and_doubt`, `rest_v1`, `evict_value_v1`.
- `misses` + `hit_rate` are internally consistent with 50,000
  requests/trace for every row (`hit_rate == 1 - misses/50000` to 5 dp).
- Spot-checked families (brightkite, citibike, metacdn, twemcache) plus
  the remaining three (metakv, cloudphysics, wiki2018) all confirmed
  consistent. wiki2018 is degenerate (50000/50000 misses, 0.0 hit rate)
  for **all 8 policies** — a saturated working-set artifact already known
  from cap32/cap64, not new at cap128, and uninformative for trend
  purposes (documented in `kbs_post_cap128_decision_template.md` §7).

The two anomaly traces specifically, at capacity 128:

| trace | lru misses | evict_value_v1 misses | gap vs LRU |
|---|---|---|---|
| brightkite_50k | 15,479 | 23,360 | +50.9% |
| citibike_202401_50k | 17,124 | 24,914 | +45.5% |

No structural defect found. Rows are clean.

## 2. Cross-capacity structural/schema verification

Ran the user-supplied exact command:

```
python scripts/paper/verify_kbs_policy_chunks.py \
    --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
             analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
             analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv \
    --expected-capacities 32,64,128 \
    --expected-policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1
```

Result: **PASSED, exit 0, no warnings, no errors.** All three chunks share
identical columns, capacities/policies/traces are exactly as expected, no
duplicate `(trace_name, capacity, policy)` keys across or within files,
`misses` are non-negative ints, `hit_rate` in `[0,1]` for all 168 rows.
This script checks structure/schema/numeric-sanity only — it does not (and
cannot) detect a semantic regression like the one under investigation, but
it rules out file corruption, truncated writes, or a schema drift between
chunks.

## 3. Targeted reproducibility check

**Full canonical-scale recheck (brightkite + citibike only, 50,000
requests/trace, capacity 128, all 8 policies) was not run.** Extrapolating
from the completed cap128 chunk's total wall-clock (22.14h for 7 traces),
and assuming roughly even per-trace cost (`evict_value_v1`'s O(capacity)
candidate scan dominates total runtime across all 8 policies), 2 of 7
traces is estimated at **~6.3h** — well above the ~1h budget for this pass.
Per the explicit rule for this audit, that run was not started; the exact
command is staged below for separate approval if wanted:

```
source .venv_kbs_heavy_r1/bin/activate
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --max-traces 2 \
  --capacities 128 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_brightkite_citibike_recheck.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_brightkite_citibike_recheck.md
```
(Estimated ~6.3h, **not run, not approved**.)

**Substitute: small-scale (1,000 requests/trace) probes**, run outside the
repo in `/tmp/cap128_probe/` (throwaway, not tracked, not part of any
deliverable), each completing in well under a minute:

| probe | trace(s) | capacity | requests/trace | evict_value_v1 | lru | gap vs LRU |
|---|---|---|---|---|---|---|
| run6_cap32 | brightkite | 32 | 1,000 | 570 | 503 | −13.32% |
| run5_cap64 | brightkite | 64 | 1,000 | 495 | 445 | −11.24% |
| run2 / run3 | brightkite | 128 | 1,000 | 553 | 410 | **−34.88%** |
| run4_2traces | brightkite + citibike | 128 | 1,000 | 408.5 (mean) | 302 (mean) | −35.26% (citibike alone: −36.07%) |

Two findings from these probes:

1. **Determinism confirmed.** `probe_run2.csv` and `probe_run3.csv` (identical
   parameters, separate invocations) are byte-identical via `diff`. The
   policy is deterministic, not flaky/seed-sensitive.
2. **The anomaly's direction reproduces at 1/50th scale.** Even with only
   1,000 requests/trace, the brightkite gap vs LRU widens sharply from
   cap64 (−11.24%) to cap128 (−34.88%) — the same qualitative reversal seen
   in the full 50,000-request canonical run. This is strong evidence the
   effect is not a 50k-scale-only fluke, not corrupted output, and not
   randomness — it is reproducible, deterministic behavior of the trained
   policy at capacity 128 on these two traces, at any tested scale.

## 4. Code-path / decision-logic audit

Inspected `src/lafc/policies/evict_value_v1.py`,
`src/lafc/evict_value_features_v1.py`, `src/lafc/evict_value_model_v1.py`,
`src/lafc/evict_value_dataset_v1.py`, `src/lafc/learned_gate/features.py`,
`src/lafc/policies/base.py`, `src/lafc/simulator/cache_state.py`. Grepped
`src scripts tests` for any capacity==128 / >=128 / <=128 / cache_size==128
branching.

Findings:

- **No capacity-specific branching anywhere** — zero grep matches for any
  literal-128 condition in the codebase.
- Victim selection (`_choose_victim`) scans **all** cache-resident
  candidates and picks the minimum predicted-loss one — generic,
  capacity-agnostic logic. This is O(capacity) per eviction, which is why
  wall-clock scales with capacity (confirmed cap32=5.27h → cap64=10.07h
  [1.91x] → cap128=22.14h [2.20x]), but cost scaling is not the same thing
  as a correctness bug.
- Feature construction (`compute_candidate_features_v1`,
  `compute_lru_scores`, `compute_predictor_scores`) is rank/percentile
  normalized by `len(candidates)-1` or `len(uniq)-1` — designed to be
  scale-invariant across capacities. One exception:
  `cache_unique_bucket_count` is a **raw, unnormalized** count of distinct
  buckets currently cached — this is a plausible (unconfirmed) contributor
  to capacity-scale sensitivity, since its raw magnitude necessarily grows
  with cache capacity while every other feature is rank-normalized to stay
  in a fixed range.
- Model loading (`EvictValueV1Model.predict_loss_one`) builds feature
  vectors **by persisted column name**, robust against train/serve
  column-order mismatches.
- The offline label-builder (`build_evict_value_examples_v1` in
  `evict_value_dataset_v1.py`) imports and reuses the *same*
  `compute_candidate_features_v1` as the online policy — no feature-code
  divergence between train and serve.
- **Confirmed methodological detail (not a bug, but the leading
  behavioral hypothesis):** per-candidate training labels are
  single-step LRU-continuation counterfactuals — `y_loss` for evicting
  candidate `c` is computed by simulating `h` future misses assuming LRU
  continues after evicting `c`. But the actual labeling-time trace walk,
  regardless of which candidate's label was just computed, always evicts
  the **true** LRU victim (`candidates[0]`) before moving to the next
  request. This means training labels are anchored to "what if I evict X
  and then LRU forever," while the **deployed** policy recursively evicts
  its own (non-LRU) choices, so its real trajectory increasingly diverges
  from the LRU-continuation assumption baked into every label as a trace
  progresses. This is the existing, documented v1 limitation (see
  `docs/evict_value_v1_method_spec.md`) — not new, not specific to cap128,
  but a structurally plausible mechanism for *compounding* error that
  could plausibly get worse with a larger candidate pool per decision
  (more candidates to mis-rank per eviction) over the same 50,000-request
  trajectory length.

## 5. Model / training-distribution check

1. **Was cap128 included in training?** Yes. The dataset summary
   (`analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`) lists
   87,004,032 rows at capacity 128 (confirmed independently in this audit
   by summing actual shard-file row counts: 5,894,784 + 6,526,464 +
   19,150,848 + 11,978,880 + 14,536,704 + 10,630,272 + 18,286,080 =
   87,004,032 — matches exactly).
2. **How many cap128 rows for brightkite/citibike specifically?**
   brightkite: 5,894,784 rows (12 shards); citibike: 6,526,464 rows (14
   shards) — both large in absolute terms.
3. **Does cap128 use the same model as cap32/cap64?** Yes — all three
   canonical runs used the identical `--evict-value-model
   models/evict_value_wulver_v1_best_heavy_r1.pkl`, and this is a single
   model trained jointly across capacities 32/64/128/256 pooled together
   (not a separate model per capacity). Also confirmed
   `models/evict_value_wulver_v1_best.pkl` and
   `models/evict_value_wulver_v1_best_heavy_r1.pkl` are byte-identical
   (matching md5sum `bf02b48891420c911fb2f6de436f1757`) — a naming
   duplicate, not a provenance discrepancy.
4. **Are brightkite/citibike represented in training?** Yes, but they are
   the two **least**-represented families at every capacity. At cap128:
   brightkite = 6.77% of rows, citibike = 7.50% — the two smallest shares
   of all 7 families (next-smallest is metacdn at 12.22%). At cap32 the
   same two families are also smallest (brightkite 7.44%, citibike
   8.23%) — i.e. **this under-representation is a constant, capacity-independent
   property of the dataset, not something that gets relatively worse at
   cap128 specifically.**
5. **Evidence of trace-family/capacity imbalance?** Yes, real and
   confirmed, but it does not by itself explain the cap128-specific
   widening: since brightkite/citibike's *relative* share of training data
   is essentially flat across cap32/64/128, a pure under-representation
   story predicts a roughly *constant* relative disadvantage for these two
   families at every capacity — not a sudden ~5x widening of the gap
   specifically between cap64 and cap128. Under-representation is a
   credible **aggravating** factor (less signal for the model to learn
   these families' dynamics well in absolute terms) but not, on its own, a
   sufficient explanation for the capacity-specific reversal.

## 6. Conclusion

**Likely real model/policy behavior, not a bug or data artifact.** Six
independent checks (raw-row integrity, cross-capacity schema/structural
verification, small-scale determinism, small-scale qualitative
reproducibility, full code-path audit, training-distribution audit) all
came back clean — no corruption, no schema drift, no capacity-specific
code branching, no train/serve feature mismatch, no flakiness. The
anomaly reproduces deterministically even at 1/50th of the canonical
request count, which rules out a 50k-run-specific fluke.

The precise causal mechanism for *why* the reversal concentrates so
sharply at cap128 on these two traces remains **unconfirmed**. The leading
hypothesis — compounding distribution shift from single-step
LRU-continuation training labels, possibly aggravated by (a) the one raw
unnormalized feature (`cache_unique_bucket_count`) and (b) brightkite/
citibike's constant relative under-representation in the shared training
set — is structurally plausible and consistent with all evidence gathered,
but was not directly proven (that would require either a full-scale
targeted rerun with per-decision trajectory logging, or a small retraining
ablation, both out of scope for this ~1h-budget pass).

## 7. Recommendation

**Do not launch cap256 yet.** Given the evidence leans toward "real
behavior" rather than "bug," cap256 is not being held back out of suspicion
of corrupted results — it is being held back because:

- A reversal this sharp, if it is genuinely capacity-dependent and
  non-monotonic, is exactly the kind of result that needs to be reported
  and discussed (as a capacity-sensitivity finding tied to the documented
  single-step-counterfactual limitation), not extended further compute
  into, until the existing 3-point curve is itself written up honestly.
- The next cheapest step to raise confidence is the already-staged ~6.3h
  brightkite+citibike-only recheck in §3 (not yet approved) — far cheaper
  and lower-risk than a full 7-trace cap256 sweep (~30–40h estimated, no
  mid-run checkpointing, and a prior single-trace cap256 validation point
  already showed a similar-direction reversal once before).
- If approved, the §3 recheck would help distinguish "true capacity-128
  dynamics" from "something specific to running brightkite+citibike in
  isolation vs. alongside 5 other traces" — though given determinism is
  already confirmed and the per-trace policy/cache state has no
  cross-trace coupling in the codebase, the isolation variable is expected
  to be a non-factor; this recheck is mainly useful as an independent
  confirmation, not because cross-trace contamination is suspected.

**Suggested framing for the manuscript/response:** report cap32→cap64→cap128
as a genuine non-monotonic capacity-sensitivity curve for `evict_value_v1`,
explicitly tied to the known single-step LRU-continuation labeling
limitation already documented in `docs/evict_value_v1_method_spec.md`, with
brightkite/citibike flagged as the concentration point and their
relatively thinner training representation noted as a contributing (not
sole) factor. This is an honest, defensible negative/mixed result, not a
result to suppress or wait out.

---
_Audit performed 2026-06-21. No cap256 launch. No canonical cap32/cap64/cap128
output modified. No commit/push/merge/delete/rename performed. All new
artifacts: this report, and throwaway timing probes in `/tmp/cap128_probe/`
(outside the repo, not tracked by git)._
