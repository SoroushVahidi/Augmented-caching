# Reviewer Fairness Protocol — audit and corrected comparison design

This document is the audit trail and protocol specification behind
`configs/reviewer_fairness_protocol.json`,
`src/lafc/experiments/reviewer_fairness_common.py`, and
`scripts/experiments/run_reviewer_fairness.py`. Every claim below was
verified directly against code, git history, and generated artifacts —
prior agent reports (including this repository's own `docs/*_method_spec.md`
files) were treated as hypotheses to re-check, not as ground truth.

## 1. Original paper protocol, reconstructed from executable evidence

**Reconstruction (from `manuscript_source/main.tex`, not from documentation
or memory):** the manuscript's current, published, end-to-end comparison
(Table `tab:available-capacities-trend`, Section on end-to-end capacities)
evaluates all eight then-existing policies (`evict_value_v1`, LRU, SIEVE,
FIFO-Reinsertion, REST, BO/LRU, T&D, PredMk) over the **full 0–50,000
request replay of each of the 7 canonical Wulver trace families, no
held-out window, no warm-up exclusion**. The manuscript text states this
explicitly: *"we report end-to-end online replay results for all eight
policies ... at 50,000 requests per trace"* and the table caption:
*"mean replay misses ... 50,000 requests/trace."* There is **no mention
anywhere in the manuscript text of a train/test split, held-out scoring
window, or warm-up-exclusion protocol for this table.**

**Discrepancy found:** none between manuscript claim and the executable
evidence for *this* table — the manuscript accurately describes what it
reports. The discrepancy is instead a **gap**: the manuscript never
addresses whether this full-stream protocol is fair to policies with
different in-trace training/adaptation footprints, because until the four
new external baselines were added, every non-`evict_value_v1` policy in
the table was training-free (LRU/SIEVE/FIFO/REST/PredMk) or effectively
static (BO/LRU, T&D), so the question did not arise.

**A second, separate and more serious discrepancy was found** (not in the
manuscript's *description* of its protocol, but in what that protocol
implies about the proposed method's own numbers) — see Section 8 below.

## 2. Two fairness questions, kept distinct

**A. Deployment fairness** (full stream, `[0, 50000)`): "if each policy
starts from the beginning of the same request stream under its intended
operational behavior, how many misses does it produce?" This is exactly
the manuscript's existing, already-published protocol. Retained here as
the **secondary** comparison (`policy_variant=deployment_full_stream`).

**B. Controlled test-window fairness** (`[10000, 50000)` after processing
`[0, 10000)`): "after an identical history/prefix, how do all policies
perform on exactly the same held-out request positions?" This isolates
each learned baseline's steady-state quality from cold-start/training-
footprint asymmetries that differ by construction between a pretrained
model (`evict_value_v1`, 0% in-trace cold start) and online-adaptive
baselines (LRB/3L-Cache: continuous retraining; HALP: one frozen split;
CACHEUS: continuous adaptation from a uniform prior). **Recommended as the
primary reviewer-facing comparison** for the four new external baselines
vs. `evict_value_v1` and the classical pool — see Section 4 for why 20%
was chosen, not assumed.

Both metrics are derived from a **single execution** per (trace, capacity,
policy): `run_policy()` already returns one `CacheEvent` per request, in
order (verified directly: `src/lafc/runner/run_policy.py`,
`for req in requests: events.append(policy.on_request(req))`), so slicing
`events[10000:]` and recounting hits/misses reconstructs the windowed
metric losslessly — no second simulation, no rerun needed merely to obtain
both numbers. See `lafc.experiments.reviewer_fairness_common.score_window`
and its tests (`tests/test_reviewer_fairness_common.py`) for the verified
primitive.

## 3. Why the 20%/80% split is not an arbitrary choice

Before picking a split, the existing external-baseline runners were
checked (not the other way around):

- `scripts/experiments/run_lrb_external_baseline.py`:
  `--validation-fraction` default **0.2**.
- `scripts/experiments/run_three_l_cache_comparison.py`:
  `--validation-fraction` default **0.2**.
- `src/lafc/policies/halp.py` / `docs/halp_method_spec.md`:
  `training_trigger` default **10,000 of 50,000 = 0.2**.

Three of the four new baselines had already, independently, converged on
a 20% leading fraction for their own validation/training-boundary
purposes, before this fairness audit existed. Adopting `history_end =
10000` reuses that convergent, pre-existing choice rather than
introducing a new number chosen to make any particular policy look better
— which is also why this document states the boundary **before** running
any comparison (predeclaration, per this protocol's own requirement).

CACHEUS has no trace-length-scaled parameter (its `history_size = capacity
// 2` is capacity-scaled, not request-count-scaled) and is compatible with
any history/score boundary without special-casing, since it adapts
continuously from `t=0` with no explicit "training phase."

## 4. Distinguishing warm-up, training, adaptation, validation, and test

| Concept | Definition used here |
|---|---|
| Cache warm-up | Requests processed to establish cache state before scoring (every policy processes `[0, 10000)` for this purpose in the primary window) |
| Model training | Requests/labels used to fit model parameters (HALP: once, at `t=10000`; `evict_value_v1`: entirely offline, see Section 8; LRB/3L-Cache: continuously, including during the scored suffix; CACHEUS: continuously, including during the scored suffix, via delayed ghost-history feedback) |
| Online adaptation | Policy updates continuing during evaluation (LRB, 3L-Cache, CACHEUS: yes; HALP: no, frozen after `t=10000`; `evict_value_v1`, classical baselines: no) |
| Hyperparameter validation | Data used to choose configuration (LRB/3L-Cache: their own `--validation-fraction` prefix, historically the *same* `[0,10000)` region now reused as history here; `evict_value_v1`: a separate offline val split — see Section 8) |
| Test | Requests contributing to the reported metric (`[10000, 50000)` for the primary comparison; `[0, 50000)` for the deployment comparison) |

A policy can process the common history prefix **without** "training" on
it (LRU, SIEVE, FIFO-Reinsertion: pure cache warm-up, no parameters to
fit). This distinction is preserved per-policy in the common schema's
`model_training_mode`/`online_adaptation_during_test` fields, not
collapsed into one "training" label.

## 5. HALP leakage audit (highest priority per task instructions)

Re-verified directly against `src/lafc/policies/halp.py` in this worktree
(commit `5a54b33`), not assumed from the prior HALP-branch report:

- Training trigger: `t >= self._config.training_trigger and not
  self._model_trained` (line 153) — single training event, at
  `training_trigger` (10,000 in this protocol's configuration, matching
  `history_end`).
- Label construction (`_train`, lines 232–262): iterates
  `self._recorded_events` — shortlists recorded **only** during
  `[0, training_trigger)` — and compares `self._actual_next.get(cand, ...)`
  for each candidate pair. `self._actual_next[pid] = request.actual_next`
  is updated on **every** `on_request` call, including future ones, but
  `_train()` is called exactly once, at `t = training_trigger`, and only
  ever reads `self._actual_next` values as they stand **at that single
  call time** — i.e., only for candidates whose `actual_next` was already
  known from requests observed in `[0, training_trigger)`. No request at
  `t >= training_trigger` is ever read before `_train()` executes, because
  `_train()` executes before any `t >= training_trigger` request is
  processed (both happen inside `on_request` at the same `t`, and
  `_train()` is called at the top of the branch, before any
  `self._actual_next` update for the *current*, later request would occur
  in a subsequent call).
- After `_model_trained = True`, the model is never refit
  (`if t >= ... and not self._model_trained` guards further calls) and no
  further `_recorded_events` are appended (recording only happens in the
  `if not self._model_trained:` branch).
- **Finding: no leakage.** Confirmed both by code inspection and by
  `tests/test_halp.py::test_no_future_leakage_from_next_arrival` (mutates
  `actual_next` for a post-training request and asserts identical
  decisions) — that test was re-run in this session as part of the full
  suite (see Section "Fairness harness tests" below) and still passes.
- **Fairness-relevant (not a leakage) finding:** HALP's reported
  `deployment_full_stream` miss count blends a pure-LRU cold-start phase
  (`[0, 10000)`) with a trained phase (`[10000, 50000)`) into one number —
  this is *not* a bug, but it is exactly the asymmetry the
  `primary_controlled_window` metric is designed to remove, since that
  metric excludes `[0, 10000)` from the count for every policy uniformly.

**No correction required for HALP.** Its existing implementation already
satisfies both fairness questions correctly; only the reporting split
needed to be added (done, via `score_window`).

## 6. `evict_value_v1` training provenance audit — CRITICAL FINDING

This is the single most important finding of this audit. Verified directly
against the actual generated artifacts in the primary checkout (not
inferred from documentation):

- `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` (the
  manifest for the dataset that trained `models/evict_value_wulver_v1_best.pkl`,
  the model referenced as canonical throughout `docs/baselines.md` and
  used by `scripts/experiments/run_lrb_external_baseline.py
  --evict-value-model`):
  ```
  "split_mode": "trace_chunk", "chunk_size": 4096,
  "max_requests_per_trace": 50000, "trace_count": 7,
  preflight.trace_stats[*].path: "data/processed/{brightkite,citibike,
    wiki2018,twemcache,metakv,metacdn,cloudphysics}/trace.jsonl"
  ```
  This is **the exact same 7 trace files, the exact same
  0–50,000-request range**, as `analysis/wulver_trace_manifest_full.csv`
  — the canonical manifest used for the manuscript's end-to-end evaluation
  and for every external-baseline comparison in this repository.
- `assign_split()` (`src/lafc/evict_value_wulver_v1.py`, lines 41–68):
  for `split_mode="trace_chunk"`, `chunk_id = t // chunk_size` and
  `key = f"trace={trace_name}|chunk={chunk_id}"`, then
  `bucket = _stable_bucket(key, seed)` decides train (70%) / val (15%) /
  test (15%) **per chunk, via a hash of the chunk's own key** — not a
  contiguous prefix/suffix split. With `chunk_size=4096` and 50,000
  requests/trace, each trace has ~13 chunks, pseudo-randomly and
  independently distributed across train/val/test, **scattered across the
  entire `[0, 50000)` range**, not confined to an early prefix.
- **Consequence:** the model that produces the manuscript's headline
  end-to-end numbers was fit on training examples drawn from
  approximately 70% of the chunks of each of the same 7 traces —
  distributed throughout the entire request range — that are then scored
  end-to-end, over their **full, unrestricted** `[0, 50000)` stream, for
  the manuscript's Table `tab:available-capacities-trend`. This is a
  direct train/test overlap for the proposed method, at the level of the
  primary reported evaluation.

**This is ranked CRITICAL**, not major: it directly affects the
scientific validity of the proposed method's own headline numbers, which
this repository's other baselines are being compared against.

**Mitigating context, stated for balance, not to excuse the finding:**
the manuscript's own honest conclusion is that `evict_value_v1`
*underperforms* LRU/SIEVE/FIFO-Reinsertion at every evaluated capacity
despite this potential advantage — so the leakage does not appear to have
been *exploited* to inflate a positive claim. That does not make the
protocol sound; it means the negative result is reported honestly despite
a compromised setup, which is better than the alternative but does not
resolve the fairness problem for any *new* comparison built on top of
these same numbers.

**Corrected protocol (recommended, not executed in this session — see
Section 9):**
1. Retrain on a corpus with **zero request-level or trace-level overlap**
   with the 7 canonical evaluation traces — either entirely different
   source traces, or (if reusing these 7 sources) a **temporally disjoint
   split** (e.g., train only on request ranges beyond 50,000 in the same
   raw source logs, if such data exists and was not itself used to build
   the 50k evaluation prefix) rather than a chunk-shuffled split within the
   same 50k window.
2. Re-run hyperparameter/model-family selection (`horizon`, `model`) on
   that same disjoint corpus, not reusing `evict_value_wulver_v1_best_config_heavy_r1.json`
   as-is, since that selection was itself made against the leaking split.
3. Re-run the full end-to-end evaluation with the retrained, non-leaking
   model.

This retraining is a substantial undertaking (rebuilding a ~289M-row
candidate dataset, refitting and reselecting a hist_gb model, re-running
the full offline ablation) and was **not executed in this session** — it
requires an explicit decision from the paper's author given how much it
could affect the paper's central narrative, and is out of scope for a
single fairness-audit-and-harness task. It is recorded here as the
highest-priority follow-up.

## 7. LRB fairness/efficiency audit

`scripts/experiments/run_lrb_external_baseline.py` bundles, in one script:
an LRB `memory_window`/`batch_size` grid search (6 combinations by
default) tuned on a `--validation-fraction` (default 0.2) prefix, an
"untuned paper-default" LRB run, the full classical baseline pool (LRU,
SIEVE, FIFO-Reinsertion, PredictiveMarker, BlindOracle,
BlindOracleLRUCombiner, RestV1, TrustAndDoubt), **and** `evict_value_v1` —
confirmed directly from the script's imports and `main()` body. This
directly explains why the currently-running LRB process (PID 113981 at
this audit's time) has taken multiple hours: it is not "LRB" alone, it is
LRB-tuning × (LRB validation grid + baseline-pool reproduction +
evict_value_v1 reproduction) × 7 traces × 3 capacities.

**This was not treated as wasteful in the sense of needing to be killed**
— the process was healthy and was not interfered with, per this task's
explicit instruction. It is, however, a real design inefficiency worth
fixing going forward: a `--skip-baselines` flag **already exists**
("Skip the lru/sieve/... baseline pool and evict_value_v1 (LRB-only smoke
run)") — a future corrected LRB rerun under this fairness protocol should
use `--skip-baselines` and be pointed at a script that also emits
per-request event logs (or reuse this protocol's `score_window`
reconstruction) rather than rebuilding the classical baseline pool a
second time.

**LRB was not rerun in this session** (per instruction: do not disturb a
currently healthy process, and do not duplicate expensive computation).
Once free, `--policy-only lrb`-equivalent execution is
`run_lrb_external_baseline.py --skip-baselines`, and its results should be
re-derived into this protocol's schema via the same `score_window`
technique if its own run captures per-request event logs, or rerun
directly against `run_reviewer_fairness.py`'s pattern (not yet
implemented for LRB in this session — flagged as follow-up work, not
executed).

## 8. 3L-Cache fairness audit

`ThreeLCacheConfig.batch_size` class default is **4096**, hardcoded in
`src/lafc/policies/three_l_cache.py` at implementation time — verified
directly, not reverse-engineered from the existing sensitivity grid's
outputs. This session's primary-comparison run uses that value uniformly
across all 7 traces × 3 capacities. This is defensible as *a* predeclared
value, but **not** the most rigorous available option: the existing
`run_three_l_cache_comparison.py` already validation-tunes `batch_size`
per trace/capacity on a `--validation-fraction 0.2` prefix — which, after
this audit, is now known to be the *same* `[0, 10000)` region adopted as
this protocol's common history window. A follow-up run that performs that
same per-trace/capacity grid search restricted to `[0, 10000)` only, then
freezes the selected `batch_size` for the `[10000, 50000)` scoring, would
be strictly more rigorous and is recommended as the next iteration. **Not
executed in this session** (flagged as a minor-to-major gap, not
critical, since a defensible predeclared default was used).

## 9. CACHEUS fairness audit

No changes from the CACHEUS-specific audit already completed in
`docs/cacheus_method_spec.md`/`docs/cacheus_provenance.md`: official,
unmodified source, hardcoded upstream seed 123 (not repository-
controlled, preserved as-is — **not** changed to 0, per this task's
explicit instruction), continuous online adaptation from a uniform prior
with no explicit training phase. Re-verified in this session:
`verify_official_source_integrity()` passes at commit
`1eec63ce166502be33ddd1f35bc041ed73a24f4d` in this worktree's fresh fetch,
and the deployment-window misses reproduced in this session's run are
bit-identical to the CACHEUS worktree's earlier full-campaign numbers
(e.g., brightkite@32: 18216 misses both times) — confirming determinism
across independent worktrees/fetches.

## 10. Classical baselines

LRU, SIEVE, and FIFO-Reinsertion are run identically in this protocol:
`reset()`, process the full request stream in order (including the common
history prefix), and the `primary_controlled_window` metric is derived by
slicing the resulting event log at `score_start` — **the cache state at
that boundary genuinely reflects having processed the history**, not a
reset to empty (verified by
`tests/test_reviewer_fairness_common.py::test_score_window_history_does_not_affect_state_after_it`).
No classical policy is initialized fresh at the score boundary while
learned methods carry warmed state — all policies share the identical
history-processing discipline by construction of `score_window`.

`blind_oracle_lru_combiner`, `rest_v1`, `trust_and_doubt`,
`predictive_marker` are in the manuscript's existing policy pool but were
**not** ported to `run_reviewer_fairness.py` in this session (time/scope
triage — none of them require prediction inputs unavailable in this
setting, so porting them is mechanical, not blocked; flagged as follow-up).

## 11. OPT/Belady oracle

`offline_belady` exists in the policy registry but was not run in this
session. Per this task's guidance: it is a theoretical lower bound, not an
implementable baseline, and by definition may use future information. If
included in a future fairness table, it must be labeled explicitly as
"offline oracle / theoretical lower bound," never merged into the learned-
baseline comparison rows, and its future-information use should be
recorded as `future_information="oracle_by_definition"` rather than
"none" or "leakage."

## 12. Common result schema

See `lafc.experiments.reviewer_fairness_common.COMMON_SCHEMA_FIELDS` for
the authoritative field list (28 fields: protocol version, policy
identity/provenance, trace identity/hash, capacity/object-size semantics,
window boundaries, scored counts, hits/misses/miss_ratio, warm-up/training/
adaptation/hyperparameter/seed/future-information metadata, runtime,
status/failure_reason). `validate_common_row()` rejects any row missing a
required field before it is written — a malformed row cannot silently
enter a comparison. `hits + misses == scored_requests` is guaranteed by
construction (`score_window` computes `hits = len(window) - misses`), not
asserted after the fact.

## 13. Existing-result reusability classification

| Artifact | Classification | Why |
|---|---|---|
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_*` (manuscript table source) | **B — deployment-only** | Matches the manuscript's own already-published full-stream protocol exactly; valid for that purpose. Its underlying model is train/test-compromised (Section 6), so any *new* fairness table built from it must carry that caveat, not present it as primary-comparison-eligible. |
| `analysis/external_learned_baselines/lrb/*` (in-progress) | **D — pending; not yet classifiable** | Run not finished as of this audit; its `--skip-baselines`-equivalent-free full run is a superset containing the needed rows, but was never captured with per-request event logs, so the windowed metric cannot be losslessly reconstructed from it once finished — a fresh `--skip-baselines` run with event logging is required for primary-comparison eligibility. |
| `analysis/external_learned_baselines/three_l_cache/*` (batch-size sensitivity grid) | **C — sensitivity/diagnostics only** | Valid as a sensitivity analysis; not directly reusable for the primary comparison because it reports only aggregate `deployment_full_stream`-style misses, not windowed, and predates this protocol's frozen split. |
| `analysis/external_learned_baselines/halp/*` (prior campaign, if any) | **D — rerun** (done in this session) | Prior sessions ran smoke tests only, not a full campaign; this session's `analysis/reviewer_fairness/policy_comparison_halp.csv` is the first full, windowed-schema run. |
| `analysis/external_learned_baselines/cacheus/*` (prior full campaign) | **A — directly reusable for deployment metric; windowed metric reconstructed fresh in this session** | Deterministic (hardcoded seed 123), confirmed bit-identical reproduction in this session; the prior campaign's aggregate CSV lacks per-request event logs, so `analysis/reviewer_fairness/policy_comparison_cacheus.csv` (this session) is the first artifact with the windowed metric. |
| This session's `analysis/reviewer_fairness/*` (lru, sieve, fifo_reinsertion, cacheus, halp, three_l_cache, evict_value_v1) | **A — primary comparison eligible**, except `evict_value_v1` which is **E — invalid for primary comparison** (Section 6) though computed for documentation | Fresh, single-execution-per-row, common schema, windowed and full-stream both derived losslessly. |

## 14. Fairness harness tests

`tests/test_reviewer_fairness_common.py` (5 tests, all passing): hand-
derived suffix miss count on a fully traceable LRU trace, full-range
window matches the underlying `run_policy()` full-stream result exactly,
out-of-range windows rejected, state-preservation across the history/score
boundary (a page warmed only in history and re-requested exactly at the
boundary must be a hit, not a miss — verifies the cache is *not* reset at
`score_start`), and common-schema validation rejecting a row missing a
required field.

## 15. Reviewer-facing terminology

For the primary (`primary_controlled_window`) comparison: *"Every policy
processes an identical 10,000-request history prefix under its own
legitimate online behavior (cache warm-up, and — for HALP — one-time model
training; for LRB/3L-Cache/CACHEUS — online adaptation, which continues
into the scored region per each method's own design), then is scored on
an identical held-out 40,000-request suffix using an identical miss
definition. evict_value_v1's numbers are reported for completeness but are
explicitly marked ineligible for this comparison due to a confirmed
train/test overlap in its training data (Section 6)."*
