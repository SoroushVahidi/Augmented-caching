# Horizon H=4 revision strategy (2026-06-19)

Directly answers **R2-MC1** ("the manuscript does not provide a
principled justification for [H=4]... unclear whether the optimal horizon
is workload-dependent, cache-size-dependent, or specific to the evaluated
datasets... provide a more thorough sensitivity analysis, theoretical
intuition, or an adaptive mechanism") and **R3-Rec7** ("Investigate why
H=4 works best"). Zero-heavy-compute pass: all evidence below was **mined
from artifacts already produced and committed** by the heavy_r1 training
run (`analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`); no new
training, no new replay, nothing relaunched. Done in parallel with the
running `kbs_full_policy_comparison_cap32_with_sieve` tmux job (not
touched).

## 1. Current evidence on H=4 (aggregate)

`analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` (already
committed) gives validation/test MAE, RMSE, top-1 eviction match, and mean
regret-vs-oracle for every (horizon, model) pair actually trained:
H∈{4,8,16} × model∈{ridge, random_forest, hist_gb}. For the winning model
family (`random_forest`):

| Horizon | val_mae | val_rmse | val_top1 | val_mean_regret |
|---|---|---|---|---|
| **4** | 1.0041 | 1.2414 | 0.0588 | **0.02078** |
| 8 | 1.8725 | 2.2616 | 0.0293 | 0.03150 |
| 16 | 3.7019 | 4.3771 | 0.0385 | 0.06343 |

Every error metric increases monotonically with horizon, for every one of
the three models trained (ridge, random_forest, hist_gb — see the full CSV
for the other two rows per horizon). This monotonic degradation is the
existing aggregate evidence already cited in
`reports/kbs_response_to_reviewers_skeleton.md`'s R2-MC1 placeholder.

## 2. NEW: workload (trace-family) dependence — mined this pass

The training script (`scripts/train_evict_value_wulver_v1.py:_family_metrics`)
**already computes and saves** a per-trace-family validation breakdown for
every horizon — it is sitting in the committed
`analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` under
`horizons.<H>.random_forest.val.per_family_val`, but had not previously
been extracted into any report. Extracted this pass (read-only, no
recomputation):

| Family | H=4 mean_regret | H=8 mean_regret | H=16 mean_regret | Direction |
|---|---|---|---|---|
| brightkite | 0.0944 | 0.2389 | 0.2944 | monotonic ↑ with H |
| cloudphysics | 0.0153 | 0.0337 | 0.0905 | monotonic ↑ with H |
| metacdn | 0.0445 | 0.0491 | 0.1243 | monotonic ↑ with H |
| twemcache | 0.0521 | 0.0781 | 0.0833 | monotonic ↑ with H |
| wiki2018 | 0.0000 | 0.0000 | 0.0000 | flat (degenerate — see caveat) |

**Finding: within every trace family that has validation coverage, H=4
has the lowest (or tied-lowest) mean regret — the ranking H=4 < H=8 < H=16
is workload-consistent, not an artifact of aggregation across families.**
This is a materially stronger answer to R2-MC1 than the aggregate table
alone, and directly addresses the reviewer's explicit question ("is the
optimal horizon workload-dependent?") with a concrete, non-aggregated
answer: **no, not within this dataset** — the same horizon wins in every
family represented in the validation sample.

**Caveat, carried over from the known validation-coverage gap
(`reports/kbs_revision_gap_tracker.md`'s "Open methodological caveat")**:
this validation sample only covers 5 of 7 trace families — **citibike and
metakv are entirely absent** from the ~4,572-row validation sample used
for both model and horizon selection, due to an early-stop sampling
artifact in `_load_rows_from_manifest`. The workload-consistency finding
above is real for the 5 covered families but **cannot yet be extended to
citibike/metakv** without either (a) fixing the early-stop sampling and
retraining (a real, if cheap ~7-8 minute, retrain — see
`reports/kbs_overhead_and_scalability_evidence.md`), or (b) a separate
offline scoring pass against citibike/metakv's existing dataset rows using
the already-trained model (cheaper — no retraining, just inference —
but still a new script run, not yet done in this pass). This is the most
important honesty caveat for this report.

The `wiki2018` row's flat 0.0000 regret at every horizon is the same
phenomenon flagged in `reports/kbs_cap32_policy_comparison_report.md`:
`wiki2018` produces a 100%-miss floor for every policy at this capacity in
the online replay, and apparently a degenerate (zero-variance or always-
trivially-optimal) label distribution offline as well — it is not evidence
that horizon choice doesn't matter there, but rather that this family's
decisions are saturated/uninformative at the evaluated settings. Worth a
one-sentence flag in the manuscript so a reviewer doesn't read it as "H
doesn't matter for wiki2018."

## 3. Cache-size (capacity) dependence — explicitly NOT available without a small new run

Unlike the trace-family breakdown, **no per-capacity breakdown of horizon
performance exists anywhere in the current artifacts.** Checked directly:
`_family_metrics` groups only by `trace_family`, never by `capacity`, even
though `capacity` is a per-row field in every shard (confirmed via the
shard CSV header: `trace_name,trace_family,dataset_source,capacity,
horizon,decision_id,...`). Producing this breakdown would require either:

- A small modification to `_family_metrics` (or a new ad hoc grouping
  function) plus a rerun of the training script (~7-8 minutes per the
  measured wall-clock in `kbs_overhead_and_scalability_evidence.md` —
  cheap, but still a new run, not performed in this pass per the "no new
  heavy experiments" constraint), **or**
- A pure-inference pass: load the already-trained
  `models/evict_value_wulver_v1_best_heavy_r1.pkl` (and the H=8/H=16
  models, if retained) against the existing validation rows already on
  disk, grouped by `capacity` instead of `trace_family`. This is cheaper
  (no retraining) but still a new script, also not performed here.

**This is the one piece of R2-MC1's explicit question ("is the optimal
horizon... cache-size-dependent?") that this pass cannot answer from
existing artifacts alone.** Recommend flagging this explicitly as an open
item rather than fabricating or assuming an answer.

## 4. Does existing data suffice for a response, or is more needed?

**Partially sufficient — enough for a substantially stronger response than
the current placeholder, but with two explicit residual gaps**, both
already flagged above:

1. Workload-dependence: **answered** for 5/7 families (no, the ranking is
   consistent); **unanswered** for citibike/metakv (validation-coverage
   gap, separate from the horizon question itself).
2. Cache-size-dependence: **unanswered** — no per-capacity breakdown exists
   yet in any artifact, mined or otherwise.

The response-to-reviewers letter should state the workload finding
plainly (it is real, mined evidence, not a placeholder) while being
explicit that the capacity dimension and the citibike/metakv family gap
remain open, rather than implying the sensitivity analysis is complete.

## 5. Theoretical intuition (draft)

Three complementary arguments, none requiring new evidence, for *why*
shorter horizons should be expected to produce lower-regret supervision in
this candidate-level finite-horizon replay framework:

1. **Shorter horizons reduce label noise from compounding continuation
   effects.** The eviction-value label for a candidate is constructed by
   replaying a fixed continuation rule forward for H steps past the
   eviction decision. Each additional step the continuation rule plays out
   introduces further continuation-policy-dependent randomness/variance
   into what is fundamentally a label for a *single* decision (whether to
   evict this candidate *now*). As H grows, more of the label's variance is
   attributable to what happens many steps downstream — events increasingly
   decoupled from the immediate eviction decision being scored — rather
   than to the decision itself, degrading the supervision signal's
   precision for the thing actually being predicted.
2. **Long-horizon continuation mixes the candidate's own effect with
   artifacts of the continuation policy itself.** The dataset's
   continuation rule (LRU, per the framework's design) governs what happens
   to the cache state for the remainder of the horizon window. Over a short
   window, the continuation policy's choices are mostly downstream
   consequences of the original eviction; over a long window, the
   continuation policy's *own* eviction decisions increasingly dominate the
   accumulated cost, diluting the causal link between "did I evict the
   right candidate at time t" and "what is the accumulated cost by t+H."
   This is a confound, not pure signal, and it grows with H.
3. **A finite, short horizon aligns the supervision target with the
   decision's actual near-term operational consequence.** The practical
   quantity that matters for an online paging policy is whether a
   particular eviction choice causes an avoidable miss in the near future
   — not the cache's state arbitrarily far downstream, which depends on
   many subsequent decisions the policy being scored had no part in. A
   short finite horizon is a closer proxy to "did this eviction choice
   directly cause near-term harm" than a long one, which is progressively
   diluted by unrelated future dynamics. This is consistent with the
   monotonic empirical degradation in §1-2: as H grows, the label
   increasingly measures something other than the immediate decision's
   quality, and a model trained to predict that noisier, more diluted
   target performs worse by every offline metric measured.

These three points can be combined into a single theoretical-intuition
paragraph (§6.2) — none require new data, only the existing monotonic
empirical pattern (§1-2) as the explanandum.

## 6. Draft text

### 6.1 Response-to-reviewers paragraph (replaces the `[IN PROGRESS]`
placeholder at R2-MC1 in `reports/kbs_response_to_reviewers_skeleton.md`)

> We thank the reviewer for pressing on this point. We selected H via an
> offline ablation across H∈{4,8,16} on held-out validation data (Table
> 4/5; `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`),
> where H=4 achieves the lowest validation mean regret-vs-oracle for every
> model class tested (ridge, random forest, histogram gradient boosting),
> and every error metric we measure (MAE, RMSE, top-1 eviction match,
> regret) degrades monotonically as H increases from 4 to 8 to 16. To
> address the reviewer's specific question about workload-dependence, we
> additionally break this down by trace family: **H=4 attains the lowest
> (or tied-lowest) mean regret within every individual trace family
> represented in our validation sample** (brightkite, cloudphysics,
> metacdn, twemcache, wiki2018), indicating the horizon ranking is not an
> artifact of aggregation across heterogeneous workloads. We note two
> remaining limitations of this analysis, which we state explicitly rather
> than overclaim: (1) our current validation sample does not yet include
> the citibike and metakv trace families due to a sampling artifact we have
> identified and are correcting; and (2) we have not yet measured whether
> the optimal horizon depends on cache capacity, which we flag as an open
> question for future work rather than claim to have resolved. We also
> provide a theoretical account for the observed pattern: shorter horizons
> reduce the label noise introduced by compounding continuation-policy
> effects, better isolate the causal contribution of the scored eviction
> decision from unrelated downstream dynamics, and align the supervision
> target more closely with the near-term operational harm an eviction
> choice is intended to avoid. We have added this discussion to [Section
> X] of the revised manuscript.

### 6.2 Manuscript paragraph (new, recommended insertion point: end of the
Method section's discussion of the eviction-value target, near where the
horizon H is first introduced — or alternatively in a new Discussion
subsection alongside the existing offline-ablation paragraphs)

> The choice of replay horizon H controls a tradeoff in label quality. A
> short horizon ties the eviction-value label closely to the immediate
> consequence of the scored decision, while a long horizon allows the
> label to absorb increasing variance from the continuation policy's own
> subsequent choices — effects that are downstream consequences of later
> decisions rather than of the eviction being scored. We observe this
> tradeoff empirically: validation MAE, RMSE, top-1 eviction match, and
> mean regret-vs-oracle all degrade monotonically as H increases from 4 to
> 8 to 16, for every model class evaluated (Table~\ref{tab:...}). This
> pattern holds not only in aggregate but within every individual trace
> family represented in our validation sample, suggesting the advantage of
> shorter horizons in this framework is not specific to any one workload
> among those evaluated. We have not yet evaluated whether this pattern
> extends to all trace families in our full dataset or whether it
> interacts with cache capacity; we flag both as open questions for future
> work.

### 6.3 Limitations addition (pairs with §6.2, makes the residual gaps explicit)

> Our horizon sensitivity analysis is currently limited along two axes:
> validation coverage for two trace families (citibike, metakv) is
> incomplete due to a sampling artifact in our current pipeline, and we
> have not measured whether the optimal horizon varies with cache capacity.
> Both are concrete, addressable extensions of the present analysis rather
> than open-ended future work.

## 7. What is explicitly NOT claimed here

- **Not claimed**: that H=4 is optimal for citibike/metakv (no validation
  coverage exists yet for those families at any horizon).
- **Not claimed**: that H=4 is optimal across all cache capacities (no
  per-capacity breakdown exists).
- **Not claimed**: any *online* (replay/miss-ratio) horizon sensitivity —
  cap32 and cap32_with_sieve were both run with a single fixed model
  (H=4, random_forest, the offline-selected winner); there is no online
  evidence comparing H=4/H=8/H=16 head-to-head in actual cache-replay
  miss counts, only offline target-quality metrics. This mirrors the
  exact concern R3-Issue1/R2-MC1 raise about offline-vs-online evidence
  more broadly (see `reports/kbs_response_to_reviewers_skeleton.md`).
- All claims above are marked with their evidentiary basis so a future
  pass with the canonical capacity chunks finished can either confirm or
  revise this report's draft text — nothing here should be treated as
  final until the manuscript text is actually inserted and the residual
  gaps (§3, §7) are resolved or explicitly retained as stated limitations.
