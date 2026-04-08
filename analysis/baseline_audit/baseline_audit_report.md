# Baseline Audit Report (Full Baseline Pass)

Date: 2026-04-08  
Repo: `Augmented-caching`

## Scope and method

Audited baselines:
- Baseline 1: `la_det` / Learning-Augmented Weighted Paging
- Baseline 2: `marker`, `blind_oracle`, `predictive_marker`
- Baseline 3: `trust_and_doubt`
- Baseline 4: `blind_oracle_lru_combiner`
- sanity-reference: `lru`, `weighted_lru`, `advice_trusting`

Method used per request:
1. Read policy implementation(s).
2. Read docs/README claims.
3. Read runner + prediction/error metrics integration.
4. Read tests and evaluate coverage quality.
5. Run baseline-specific and integration tests.
6. Run smoke CLI traces.
7. Issue verdict on faithfulness, bug risk, and paper-comparison readiness.

## Commands run

- `pip install -e .`
- `pytest -q tests/test_policies.py tests/test_policies_baseline2.py tests/test_baseline3_trust_and_doubt.py tests/test_baseline4.py tests/test_runner.py tests/test_metrics.py`
- `python -m lafc.runner.run_policy --policy la_det --trace data/example.json --capacity 3 --output-dir /tmp/audit_la_det`
- `python -m lafc.runner.run_policy --policy marker --trace data/example_unweighted.json --capacity 3 --output-dir /tmp/audit_marker`
- `python -m lafc.runner.run_policy --policy blind_oracle --trace data/example_unweighted.json --capacity 3 --output-dir /tmp/audit_bo`
- `python -m lafc.runner.run_policy --policy predictive_marker --trace data/example_unweighted.json --capacity 3 --output-dir /tmp/audit_pm`
- `python -m lafc.runner.run_policy --policy trust_and_doubt --trace data/example_unweighted.json --capacity 3 --derive-predicted-caches --output-dir /tmp/audit_td`
- `python -m lafc.runner.run_policy --policy blind_oracle_lru_combiner --trace data/example_unweighted.json --capacity 3 --output-dir /tmp/audit_comb`

Test outcome: 116 tests passed.

---

## Baseline 1 — `la_det` (Learning-Augmented Weighted Paging)

### Findings

- The implementation itself explicitly documents that the eviction rule is an **interpretation** of paper structure, not a literal code translation of the paper’s primal-dual / water-filling mechanism.
- Current implementation uses score `predicted_next / weight` with deterministic tie-breaks; this is plausible and internally coherent.
- Weight classes are built and exposed, but class structure is not used in a way that mirrors explicit primal-dual state updates.
- README currently labels `la_det` as deterministic theorem baseline; that framing is stronger than what the implementation note itself supports.

### Faithfulness assessment

**Likely approximate / interpreted**, not high-confidence paper-faithful.

### Bug / mismatch risk

**Medium** risk for paper-faithfulness mismatch (not obvious runtime bug). I did not identify a concrete crash/logic bug in current code path.

### Tests adequacy

- Existing tests mostly check smoke behavior, cost accounting, and a few simple eviction outcomes.
- Missing tests for edge scenarios that would discriminate primal-dual-faithful behavior vs heuristic ranking behavior.

### Paper-comparison readiness

**Only with caveat**: use as an interpreted approximation baseline, not as a strict implementation of the deterministic theorem algorithm.

### Recommended action

Relabel as “interpreted approximation” in docs/README tables and comparison sections unless a fuller paper-faithful primal-dual implementation is added.

---

## Baseline 2 — `marker`, `blind_oracle`, `predictive_marker`

### Findings

- `marker`: phase logic and marking semantics are consistent with deterministic Marker-style behavior, but tie-breaking is deterministic (not random Marker from original randomized result).
- `blind_oracle`: follows straightforward naive prediction-trusting eviction (`argmax predicted_next`), online, unit-cost.
- `predictive_marker`: correctly differs from both by restricting to unmarked pages and using `argmax predicted_next` within that set; phase handling appears consistent.
- Clean-chain diagnostics use actual-next values, but only for diagnostics bookkeeping after eviction choice (no decision leakage observed).

### Faithfulness assessment

- `blind_oracle`: **high confidence faithful** to intended naive prediction-following baseline.
- `predictive_marker`: **plausible and mostly faithful**, with documented deterministic tie-break/diagnostic interpretations.
- `marker`: **plausible but interpreted** deterministic variant (not randomized theorem version).

### Bug / mismatch risk

**Low to medium** (mostly modeling choices vs randomized/theoretical versions). No obvious future-information leak in decisions.

### Tests adequacy

Strongest among audited groups:
- many hand-check traces,
- phase and clean-chain checks,
- runner output integration checks.

### Paper-comparison readiness

**Yes for `blind_oracle` and `predictive_marker` (with normal interpretation caveats).**  
`marker` is safe if clearly labeled deterministic marker variant.

### Recommended action

Keep as-is, but explicitly label deterministic Marker choice when reporting competitive-theory context.

---

## Baseline 3 — `trust_and_doubt`

### Findings

- Implementation clearly states interpretation choices for ambiguous “arbitrary” paper steps and lazy/non-lazy handling.
- Uses predicted cache states input (`predicted_cache`) consistent with algorithm framing.
- Complex state machine (`A`, `U`, `M`, `T`, `D`, `C`, per-page threshold bookkeeping) is present.
- Deterministic LRU substitutions and warm-up fallbacks are implemented where pseudocode is less explicit.
- No direct future-information leak found in control flow.

### Faithfulness assessment

**Plausible but partly interpreted** (complex specialization, nontrivial inferred details).

### Bug / mismatch risk

**Medium** due to complexity + relatively thin tests around detailed invariant transitions.

### Tests adequacy

Current tests are mostly smoke + determinism + I/O format; they do not deeply assert per-step set transitions or threshold-doubling invariants.

### Paper-comparison readiness

**Only with caveat** right now.

### Recommended action

Before paper-grade claims, add targeted invariant tests for:
- trust↔doubt transition points,
- threshold doubling intervals,
- real-cache vs simulated-cache lazy coupling,
- adversarial predictor-cache scenarios.

---

## Baseline 4 — `blind_oracle_lru_combiner`

### Findings

- Implementation is explicit that this is reconstructed/interpreted due no public reference code.
- Online-ness appears correct: comparison uses shadow miss counts before current request update.
- Shadow policies are updated on every request and kept independent.
- Combiner applies chosen rule to combiner cache (not copying shadow state), which is a sound online interpretation.

### Faithfulness assessment

**Plausible but partly interpreted** (reasonable operationalization of “follow whoever performed better so far”).

### Bug / mismatch risk

**Low to medium**. No obvious temporal leakage or state-copying bug found.

### Tests adequacy

Moderate-to-good:
- step-log checks,
- diagnostics checks,
- deterministic behavior checks,
- multiple sanity traces.

Could still use sharper adversarial traces that force policy switching and verify expected switch points.

### Paper-comparison readiness

**Only with caveat** (interpreted implementation).

### Recommended action

Document interpretation assumptions prominently in baseline table/reporting and add at least 2-3 forced-switch unit tests.

---

## Sanity-reference baselines (`lru`, `weighted_lru`, `advice_trusting`)

- Implementations are straightforward and coherent.
- Existing tests are adequate for sanity-reference use.
- Suitable as calibration baselines in comparison tables.

---

## Cross-cutting observations

1. `run_policy` and metrics wiring are generally coherent for all baselines audited.
2. `compute_cache_state_error` uses a specific distance interpretation (`|X \ Y|`) that should be explicitly kept as an interpretation in writeups.
3. README naming currently overstates Baseline 1 faithfulness relative to implementation comments.

---

## Final verdicts

### Trustworthy now (for practical baseline comparisons)

- `blind_oracle` (high confidence)
- `predictive_marker` (good with small interpretation caveats)
- `lru`, `weighted_lru`, `advice_trusting`

### Use with caution

- `marker` (deterministic variant caveat)
- `trust_and_doubt` (complex interpreted specialization, limited invariant tests)
- `blind_oracle_lru_combiner` (interpreted reconstruction)

### Not yet safe as paper-faithful claims without caveat

- `la_det` as currently labeled/theorem-faithful.

---

## Should atlas_v1 development pause?

**Recommendation: partial pause for paper-comparison claims, not full engineering pause.**

- If upcoming atlas_v1 evaluation depends on claiming strict paper-faithful baselines, pause and first relabel/fix baseline framing + add targeted tests (especially Baseline 1 and 3).
- If upcoming work is exploratory and caveat-labeled, atlas_v1 experimentation can continue in parallel.
