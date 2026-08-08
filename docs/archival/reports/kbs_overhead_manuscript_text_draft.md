# Overhead and Scalability — manuscript text draft (2026-06-19)

Directly answers **R2-MC2** (offline label-construction/preprocessing
cost), **R3-Issue5/R3-Rec3** ("no wall-clock time analysis, no complexity
comparison against O(1) baselines like LRU or SIEVE, no discussion of
production overhead" / "Report computational overhead"), and AE's
"missing computational overhead analysis." Built entirely on top of the
existing zero-compute evidence-mining pass in
`reports/kbs_overhead_and_scalability_evidence.md` — no new jobs run here
either, except re-checking the two newly-implemented O(1) baselines'
source code directly (SIEVE, FIFO-Reinsertion — both added earlier today,
see `reports/kbs_sieve_implementation_report.md` and
`reports/kbs_fifo_reinsertion_implementation_report.md`), which is a
source-code read, not a benchmark run. Done in parallel with the running
`kbs_full_policy_comparison_cap32_with_sieve` tmux job (not touched).

## 1. Update to the underlying evidence: SIEVE and FIFO-Reinsertion are now implemented, not just cited

`kbs_overhead_and_scalability_evidence.md`'s Part 2 (written before SIEVE
existed in this repo) could only cite SIEVE's *published* O(1)-amortized
complexity claim, explicitly flagged as "not independently re-verified
against any implementation in this repo, since none exists yet." That gap
is now closed for both SIEVE and FIFO-Reinsertion by direct source
inspection of this repo's own implementations:

- **SIEVE** (`src/lafc/policies/sieve.py`): one hand pointer over an
  ordered dict; a hit is O(1) (one dict write, no reordering); a miss's
  eviction scan is O(1) **amortized** — each `visited` bit is cleared at
  most once before being evicted, so the total work across the cache's
  lifetime is bounded by the number of hits, the standard CLOCK-family
  amortization argument.
- **FIFO-Reinsertion** (`src/lafc/policies/fifo_reinsertion.py`): identical
  state shape; a hit is O(1); a miss's eviction scan is also O(1)
  amortized by the same argument (each reinsertion clears a bit that must
  be re-set by a future hit before it can cost work again).
- Both are now **measured-from-source**, not merely cited, for this
  manuscript's own baseline pool.

## 2. Draft "Overhead and Scalability" manuscript subsection

Recommended insertion point: as a new subsection within either the Method
section (after the algorithm/diagnostics description) or the Discussion/
Limitations section — recommend Discussion, since it synthesizes both the
offline (label-construction) and online (per-decision) cost story together
rather than fragmenting it across Method.

> **Overhead and scalability.** The proposed method has two distinct
> computational costs: an offline label-construction cost incurred once
> per dataset, and an online per-decision cost incurred on every full-cache
> miss during replay or deployment. We report both, separating measured
> evidence from complexity analysis.
>
> *Offline cost.* Constructing the finite-horizon eviction-value dataset
> used in this paper — 7 trace families, 4 cache capacities, 3 replay
> horizons — took approximately 10.3 hours of wall-clock time and produced
> 96 GB across 662 shard files. This cost is dominated by the counterfactual
> replay required to compute each candidate's finite-horizon label, as the
> reviewer correctly notes: unlike supervision targets that can be read
> directly off the trace (e.g., next-arrival time), our label requires
> simulating a fixed continuation policy forward for H steps from every
> eviction decision considered. Model training itself is comparatively
> cheap: fitting all nine (horizon × model-class) configurations used for
> model and horizon selection took approximately 7-8 minutes, because
> training operates on a bounded sample of the constructed dataset rather
> than on the full 96 GB directly.
>
> *Online cost.* At each full-cache miss, `evict_value_v1` constructs a
> feature vector for every one of the `k` candidates currently resident in
> the cache and scores each with the trained model, giving **O(k)
> model-inference calls per miss**, where `k` is the cache capacity. This
> contrasts with the **O(1)**-per-miss classical baselines in our
> comparison: LRU evicts via a single deque/ordered-dict pop, and the
> CLOCK-family algorithms in our baseline pool — SIEVE and
> FIFO-Reinsertion — evict via an amortized O(1) hand-scan or
> reinsertion-scan respectively, with no per-candidate model inference.
> This is the expected and inherent cost of candidate-level learned
> scoring relative to structural eviction rules, and it grows linearly with
> cache capacity rather than remaining constant.
>
> *What remains to be measured.* The complexity analysis above is verified
> directly against our implementation's source code. We have not yet
> produced a controlled, capacity-isolated wall-clock benchmark (i.e., a
> single trace replayed at multiple capacities with per-policy timing
> instrumentation) that would convert this asymptotic claim into a
> concrete latency table; the wall-clock figures available from our
> existing experimental runs are confounded by differing trace mixes across
> capacities and are not suitable for this purpose. We report this as an
> explicit limitation rather than presenting confounded timing numbers as a
> controlled benchmark.

## 3. Draft Limitations sentence (pairs with §2, makes the gap explicit
in one place readers are likely to check)

> Our overhead analysis is grounded in direct complexity analysis of our
> implementation (O(k) candidate scoring per miss for the learned policy
> versus O(1) for the classical and CLOCK-family baselines) and in measured
> wall-clock figures for offline dataset construction and model training.
> We have not yet produced a controlled online latency benchmark isolating
> per-decision cost as a function of capacity alone; the timing data
> available from our existing experimental runs conflates capacity with
> differing trace mixes and request counts, and we do not present it as
> such a benchmark.

## 4. Draft response-to-reviewers paragraphs

### 4.1 R2-MC2 (offline cost)

> We thank the reviewer for raising this point, which we agree was
> underspecified in the original submission. We now report that
> constructing the finite-horizon eviction-value dataset used in this paper
> (7 trace families × 4 capacities × 3 horizons) took approximately 10.3
> hours of wall-clock time and produced 96 GB across 662 shard files, and
> that this cost is dominated by the counterfactual replay step the
> reviewer identifies — each candidate's label requires simulating a fixed
> continuation policy forward for H steps, which is inherently more
> expensive than constructing a supervision target that can be read
> directly from the trace. By contrast, model training itself is cheap
> (~7-8 minutes for all nine horizon/model configurations used for
> selection), since it operates on a bounded sample of the constructed
> dataset. We have added this quantitative breakdown to [Section X] of the
> revised manuscript.

### 4.2 R3-Issue5/R3-Rec3 (online overhead and O(1)-baseline comparison)

> We thank the reviewer for this concrete request. We have added a direct
> complexity comparison: our learned policy performs O(k) model-inference
> calls per full-cache miss (one per resident candidate, k = cache
> capacity), versus O(1) per miss for LRU and for the two CLOCK-family
> baselines now included in our comparison, SIEVE and FIFO-Reinsertion
> (both verified directly against our own implementations' source code, not
> merely cited from their respective papers). We have not yet produced a
> controlled wall-clock latency benchmark isolating this cost as a function
> of capacity alone — the timing data available from our existing
> experimental runs conflates capacity with differing trace composition —
> and we report this as an explicit limitation rather than present
> confounded numbers as if they were a clean benchmark. We view a
> capacity-isolated timing benchmark as a concrete, scoped follow-up that
> does not require new trace data or model retraining.

## 5. Honesty checklist (per the "be honest" instruction for this pass)

| Claim | Status |
|---|---|
| Dataset build ~10.3h / 96G / 662 shards | **Measured** (carried over, unchanged) |
| Training ~7-8 min for 9 fits | **Measured** (carried over, unchanged) |
| `evict_value_v1` is O(k) per miss | **Code-verified** (carried over, unchanged) |
| LRU is O(1) per miss | **Code-verified** (carried over, unchanged) |
| SIEVE is O(1) amortized per miss | **Code-verified against this repo's own implementation** (upgraded this pass — previously only cited from the paper) |
| FIFO-Reinsertion is O(1) amortized per miss | **Code-verified against this repo's own implementation** (new this pass — policy did not exist before today) |
| Controlled, capacity-isolated latency benchmark (ms/decision vs. k) | **Not measured** — explicitly flagged as a proposed, not-yet-run benchmark in both the manuscript and limitations draft text above |

No claim above states or implies a controlled timing benchmark exists when
it does not. The manuscript and response text in §2-4 are written to make
this distinction explicit to a reader, per R3-Issue5's specific complaint
that the original submission provided "no wall-clock time analysis, no
complexity comparison... and no discussion of production overhead" — this
draft provides the complexity comparison and the offline wall-clock
numbers, and explicitly flags rather than hides the absence of an online
controlled benchmark.
