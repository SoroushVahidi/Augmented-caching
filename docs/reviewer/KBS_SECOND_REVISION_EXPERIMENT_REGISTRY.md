# KBS Second-Revision Experiment Registry

Status: authoritative canonical index of every reviewer-relevant experiment
touching the KBS second-revision line. Cross-references
[`KBS_SECOND_REVISION_REVIEWER_COVERAGE.md`](KBS_SECOND_REVISION_REVIEWER_COVERAGE.md)
(concern -> evidence) and
[`KBS_SECOND_REVISION_HYPOTHESIS_MAP.md`](KBS_SECOND_REVISION_HYPOTHESIS_MAP.md)
(mechanism -> evidence). This file indexes by *experiment*.

## Canonical status vocabulary

Docs on this branch have historically used inconsistent terms (`done`,
`complete`, `validated`, `final`, `ready`, `pilot`, ...). This registry (and
new writing going forward) uses exactly these:

| Term | Meaning |
|---|---|
| `FINAL_VALIDATED` | Execution complete, audited, no known integrity gaps, eligible to cite without further caveats |
| `COMPLETE_DIAGNOSTIC` | Execution complete for its intended (often single-cell or small) scope; explicitly a mechanistic diagnostic, not a primary-table result |
| `COMPLETE_PARTIAL_SCOPE` | Execution complete only for part of the intended campaign (e.g. 4/7 families); do not treat as the full result |
| `RUNNING` | Actively executing right now |
| `PENDING` | Not started; no blocker other than sequencing/priority |
| `IMPLEMENTATION_READY` | Source/tests/config exist and are frozen, but no scientific result (beyond smoke) exists yet |
| `SMOKE_ONLY` | A smoke-scale run exists and is explicitly non-canonical (too small to be evidence) |
| `BLOCKED` | Blocked on an external dependency (Wulver sync, another phase finishing, etc.) |
| `DEFERRED` | Intentionally deprioritized, not currently planned |
| `SUPERSEDED` | Replaced by a newer protocol; kept for provenance only |
| `INVALID_DO_NOT_USE` | Known-contaminated or incorrect; must never be cited as evidence |

Older status strings (`COMPLETE_VALIDATED`, `RUNNING_LOCAL`,
`DIAGNOSTIC_PARTIAL`, `LOCAL_FOUNDATION_ONLY`, `PENDING_CONTROLLED_RUN`,
`CONTAMINATED_DO_NOT_USE`, `HISTORICAL`, `BLOCKED_PENDING_SYNC`) still appear
in `kbs_second_revision_artifact_map.md` and are not being mechanically
rewritten there (see Pass #3 report); this registry uses the vocabulary
above going forward and gives the mapping inline per row where the two
differ.

**2026-08-11 note:** experiments #2 (distribution-shift), #3 (corrected
held-out replay), #5 (learned-baseline comparison), #7 (target-degeneracy),
#8 (historical-tail), #9 (learning convergence), #10 (continuation C1/C2),
and #11 (controlled timing) below now have additional, more current
Wulver-side facts than what's written in their individual rows (the corrected
held-out replay and controlled timing are now complete on Wulver; broad
degeneracy and historical-tail results exist at Wulver scale; distribution-
shift's merged state is 24/42, not the row's stated local checkpoint;
continuation C1/C2 has a concrete production blocker, not just a scale gap).
Rows below were not individually rewritten to avoid duplicating detail --
see [`../CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md`](../CROSS_ENVIRONMENT_EVIDENCE_MATRIX.md)
(authoritative for cross-environment status) and
[`../WULVER_TO_GITHUB_PROMOTION_QUEUE.md`](../WULVER_TO_GITHUB_PROMOTION_QUEUE.md)
(sync priority) before citing any of these rows' `Status` field as current.

## Evidence-strength hierarchy

| Class | Meaning |
|---|---|
| `PRIMARY_REVIEWER_EVIDENCE` | Directly answers a reviewer concern in the primary comparison table |
| `SUPPORTING_EVIDENCE` | Strengthens or contextualizes primary evidence but is not itself the primary table |
| `MECHANISTIC_DIAGNOSTIC` | Explains *why*, usually single-cell or small-scope; never a standalone claim |
| `EXPLORATORY_ASSOCIATION` | Correlational/exploratory only, explicitly not causal |
| `IMPLEMENTATION_ONLY` | Code/tests/config exist; no scientific weight yet |
| `HISTORICAL_SUPERSEDED` | Kept for provenance; not current evidence |

---

## Registry

### 1. Objective ablation
- Scientific question: which supervision-objective target (`eviction_loss`,
  `next_arrival`, `reuse_distance`, `objective_pairwise`) produces the best
  downstream caching performance?
- Reviewer concern: Reviewer #2 Major 2
- Machine: LOCAL
- Protocol: `docs/supervision_objective_ablation_protocol.md`, frozen, `MODEL_SELECTION_FROZEN=true`
- Scope: 7 families x 4 objectives x 3 capacities, 28-model registry, 84/84 rows
- Source entry point: `scripts/experiments/run_supervision_objective_ablation.py`
- Config: n/a (protocol doc + registry build script)
- Output path: `analysis/supervision_objective_ablation_v1/policy_comparison.csv`, `model_registry.json`
- Status: `FINAL_VALIDATED`
- Evidence strength: `SUPPORTING_EVIDENCE` (per artifact map: "usable supporting evidence", not the primary baseline table)
- Primary/diagnostic/supporting: supporting
- Next action: none required for this concern; already complete
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 2

### 2. Distribution-shift ablation
- Scientific question: does trajectory divergence between LRU-labeled and
  learned-deployed cache states (DAgger-style) explain part of the miss gap?
- Reviewer concern: Reviewer #2 Major 3 / Reviewer #3 continuation-mismatch
- Machine: LOCAL (4/7 families); seven-family continuation is Wulver-side per sync docs
- Protocol: `docs/distribution_shift_ablation_protocol.md`, frozen
- Scope: `metacdn` only locally audited in depth; 24/42 primary rows, 4/7 families
- Source entry point: `scripts/experiments/run_distribution_shift_ablation.py`, resumable via `scripts/experiments/resume_distribution_shift.py`
- Config: protocol doc-embedded, frozen condition matrix
- Output path: `analysis/distribution_shift_ablation_v1/`
- Status: `COMPLETE_PARTIAL_SCOPE` (`STOPPED_CLEANLY_PARTIAL` per `revision_status.py`)
- Evidence strength: `MECHANISTIC_DIAGNOSTIC`
- Primary/diagnostic/supporting: diagnostic
- Next action: seven-family continuation (Wulver-side, not locally launchable per sync docs)
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.6

### 3. Corrected held-out `evict_value_v1` replay
- Scientific question: does the primary method actually beat baselines under
  a corrected, non-contaminated, cross-family held-out protocol?
- Reviewer concern: Reviewer #2 Major 1 / R3-Issue1 (end-to-end evidence)
- Machine: LOCAL (partial) / BOTH for full completion
- Protocol: `docs/reviewer_fairness_cross_family_v1.md`
- Scope: `PARTIAL` -- not yet a usable primary table
- Source entry point: `scripts/experiments/run_evict_cross_family_pipeline.py`, `scripts/experiments/run_cross_family_heldout_eval.py`
- Config: `configs/fair_cross_family_v1/folds/*.json`
- Output path: `analysis/reviewer_fairness_cross_family_v1/`
- Status: `COMPLETE_PARTIAL_SCOPE`
- Evidence strength: `SUPPORTING_EVIDENCE` (not yet primary -- this is the fairness audit's top required fix)
- Primary/diagnostic/supporting: intended as primary, currently supporting-only
- Next action: complete cross-family held-out replay with frozen registry and local model hashes (fairness-audit required fix, cost `high`, reviewer_value `very_high`)
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 1; `analysis/kbs_comparison_fairness_audit.json`

### 4. Simple exact-protocol baselines (LRU / SIEVE / FIFO)
- Scientific question: how does the method compare to simple, exactly-specified,
  non-learned baselines under the frozen fairness protocol?
- Reviewer concern: R3-Issue3
- Machine: LOCAL
- Protocol: `docs/reviewer_fairness_protocol.md`, frozen
- Scope: full controlled-window rows, all families/capacities
- Source entry point: `scripts/experiments/run_reviewer_fairness.py`
- Config: `configs/reviewer_fairness_protocol.json`
- Output path: `analysis/reviewer_fairness/policy_comparison_{lru,sieve,fifo_reinsertion}.csv`
- Status: `FINAL_VALIDATED`
- Evidence strength: `PRIMARY_REVIEWER_EVIDENCE`
- Primary/diagnostic/supporting: primary
- Next action: none
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 1

### 5. Learned-baseline fairness comparison (LRB / 3L-Cache / CACHEUS / HALP)
- Scientific question: how does the method compare to modern learned caching
  baselines under a fair, controlled protocol?
- Reviewer concern: Reviewer #2 Major 1, R3-Issue2
- Machine: LOCAL
- Protocol: `docs/reviewer_fairness_protocol.md`, frozen
- Scope: full controlled-window rows, all families/capacities, per-baseline fidelity noted
- Source entry point: `scripts/experiments/run_reviewer_fairness.py`
- Config: `configs/reviewer_fairness_protocol.json`
- Output path: `analysis/reviewer_fairness/policy_comparison_{lrb,three_l_cache,cacheus,halp}.csv`
- Status: `FINAL_VALIDATED`
- Evidence strength: `PRIMARY_REVIEWER_EVIDENCE`, with fidelity caveats: CACHEUS `HIGH` fidelity (official source); LRB/3L-Cache `MEDIUM` (independent reimplementation); **HALP `LOW_TO_MEDIUM`** -- no public official HALP implementation exists, this is an adapted reimplementation of a continuous production algorithm against a frozen offline split, not the vendor's own code
- Primary/diagnostic/supporting: primary, with the HALP caveat always attached
- Next action: none planned; keep the fidelity caveats attached in any manuscript citation
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 1; `analysis/kbs_comparison_fairness_audit.json`

### 6. Exact-target oracle diagnostic
- Scientific question: does exact optimization of the frozen `eviction_loss`
  finite-horizon target beat LRU, and does the learned model reproduce or
  outperform that exact target?
- Reviewer concern: MC1 (horizon justification), R3-Issue4
- Machine: LOCAL
- Protocol: shares label semantics with `supervision_objective_ablation.py`'s shared kernel; horizon `H=4`
- Scope: one cell (`brightkite`, capacity `64`) -- `ONE_CELL_DIAGNOSTIC_COMPLETE / FULL_SWEEP_NOT_RUN`; parameterized (`--family`, `--capacity`, `--horizon`), not hardcoded to this cell
- Source entry point: `scripts/experiments/run_exact_target_oracle_diagnostic.py`
- Config: CLI args, defaults `--family brightkite --capacity 64 --horizon 4`
- Output path: `analysis/exact_target_oracle_diagnostic_v1/brightkite_cap64_h4/`
- Status: `COMPLETE_DIAGNOSTIC`
- Evidence strength: `MECHANISTIC_DIAGNOSTIC` (single cell)
- Primary/diagnostic/supporting: diagnostic
- Next action: rerun on more (family, capacity) cells to test generality (H2/H3 in the hypothesis map)
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` "Exact target oracle vs learned online policy"; `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` H2/H3

### 7. Target-degeneracy & longer-horizon tie-resolution diagnostic
- Scientific question: how degenerate (low-resolution) is the H=4
  `eviction_loss` target, and how much do longer horizons (H=8/16/32) resolve
  that degeneracy?
- Reviewer concern: MC1 (horizon justification)
- Machine: LOCAL
- Protocol: standalone math module over candidate values produced by the
  shared kernel; base `H=4`, extension horizons `8,16,32`
- Scope: one cell (`brightkite`, capacity `64`) -- same single-cell caveat as #6; parameterized, not hardcoded
- Source entry point: `scripts/experiments/analyze_eviction_loss_target_degeneracy.py`
- Config: CLI args, defaults `--family brightkite --capacity 64 --horizon 4 --long-horizons 8,16,32`
- Output path: `analysis/eviction_loss_target_degeneracy_v1/brightkite_cap64_h4/`
- Status: `COMPLETE_DIAGNOSTIC`
- Evidence strength: `MECHANISTIC_DIAGNOSTIC` (single cell) -- currently the **strongest-supported** mechanistic finding on this branch (H3 `STRONGLY_SUPPORTED`)
- Primary/diagnostic/supporting: diagnostic
- Next action: (a) replicate across families/capacities to test generality; (b) a genuine *base-horizon* sensitivity sweep (varying `H` itself, e.g. `H in {1,2,4,8,16}`, distinct from the already-run longer-horizon tie-*resolution* extension) remains `PENDING` -- this is the literal "horizon sensitivity" item still open, separate from the completed margin/tie-degeneracy result
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.5/9.5.1/9.5.2; `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` H3/H4/H9/H10

### 8. Historical-tail diagnostic
- Scientific question: does a finite-horizon target with an implicit
  zero-terminal-value beyond `H` undervalue candidates whose benefit occurs
  later (motivating a `Q_H + V_tail_hat` formulation)?
- Reviewer concern: MC1
- Machine: **WULVER** (per existing local docs; not independently verified this pass)
- Protocol: not locally defined
- Scope: `NOT_LOCAL`
- Source entry point: `KNOWN_WULVER_RESULT / NOT_LOCAL` -- no local file exists; do not fabricate a path
- Config: `NOT_LOCAL`
- Output path: `NOT_LOCAL`
- Status: `BLOCKED` (locally); Wulver-side status not asserted here
- Evidence strength: `IMPLEMENTATION_ONLY` at best, unverified locally
- Primary/diagnostic/supporting: diagnostic (planned)
- Next action: a historical-tail diagnostic must precede any new loss definition; local audit cannot start this without contacting Wulver or independently designing and implementing it
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.5.1; `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` H8

### 9. Learning-convergence (same-target scalar-vs-pairwise)
- Scientific question: does more training data close the offline-to-online
  gap, and does pairwise representation of the *same* eviction-loss target
  outperform scalar regression?
- Reviewer concern: MC3, R3-Issue1/4
- Machine: LOCAL
- Protocol: `configs/supervision_objective_learning_curve_v1.json`, frozen; `H=4`, 7 families, capacities `[32,64,128]`, seed `0`
- Scope: fractions tested `1%, 2%, 5%, 10%, 25%, 50%`; `25%` = 7/7
  families, 42/42 rows; `50%` = 7/7 families, 42/42 rows, all
  `status=ok`; `100%` intentionally not run due the H1 stopping rule, not
  missing required work.
- Source entry point: `scripts/experiments/run_supervision_objective_learning_curve.py`
- Config: `configs/supervision_objective_learning_curve_v1.json`
- Output path: `analysis/supervision_objective_learning_curve_v1/`
- Status: `FINAL_50PCT_VALIDATED` for the intended stopping-rule scope;
  H1 stopping decision `STOP_SAMPLE_SIZE_HYPOTHESIS`.
- Evidence strength: `MECHANISTIC_DIAGNOSTIC_HIGH` for the tested
  `1%-50%` range ("explanatory diagnostic only, not a primary reviewer
  comparison" per artifact map).
- Primary/diagnostic/supporting: diagnostic
- Next action: none for H1 under the current campaign scope. Do not launch
  `100%`; it is intentionally not run by stopping rule.
- Synthesis: `analysis/supervision_objective_learning_curve_v1/final_50pct_synthesis_20260811/`
- Source hash: `policy_comparison.csv`
  SHA-256 `5323eea6e3f6fb9a44b2fab2f6632f61917442ba239eababc1b2cda1fca8612a`
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.4; `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` H1

### 10. Continuation-policy causal ablation (C1/C2)
- Scientific question: does replacing the fixed LRU label-continuation with
  the already-learned frozen `pi1` continuation improve the next learned
  policy `pi2`?
- Reviewer concern: Reviewer #2 Major 3 / Reviewer #3 continuation-mismatch
- Machine: LOCAL implementation / WULVER for full campaign
- Protocol: `configs/continuation_policy_causal_ablation_v1.json`, `PROTOCOL_FROZEN_NO_RESULTS`
- Scope: implementation + tests complete; only `TINY_SMOKE_ONLY` executed (`decision_count=3`)
- Source entry point: `src/lafc/continuation_policy_ablation.py`, `scripts/experiments/run_continuation_policy_causal_ablation_smoke.py`
- Config: `configs/continuation_policy_causal_ablation_v1.json`
- Output path: no full-campaign output yet; smoke is stdout-only (`analysis/continuation_policy_causal_ablation*` explicitly excluded from sync)
- Status: `IMPLEMENTATION_READY`
- Evidence strength: `IMPLEMENTATION_ONLY`
- Primary/diagnostic/supporting: diagnostic (planned)
- Next action: full 7-family C1/C2 campaign at Wulver scale
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.7 (includes this pass's cross-reference to the older, distinct `evict_value_v2_rollout.py` continuation exploration); `docs/reviewer/local_to_wulver_continuation_sync_manifest.md`

### 11. Practical-significance / controlled timing
- Scientific question: is fine-grained candidate-level learned eviction
  computationally practical?
- Reviewer concern: MC2, R3-Issue5
- Machine: LOCAL
- Protocol: `docs/practical_significance_ablation_protocol.md`
- Scope: smoke-scale exact-optimization-equivalence check complete; controlled final timing not started
- Source entry point: `scripts/experiments/run_practical_significance_ablation.py` (smoke), `scripts/experiments/run_practical_significance_controlled.py` (controlled, not yet run)
- Config: protocol doc-embedded
- Output path: `analysis/practical_significance_ablation_v1/`
- Status: `SMOKE_ONLY` (exact-optimization-equivalence result); controlled timing `PENDING`
- Evidence strength: `IMPLEMENTATION_ONLY` for timing numbers; the exact-decision-preserving equivalence check itself (`all_variants_exact_across_all_trace_capacity_pairs=true`) is `SUPPORTING_EVIDENCE`
- Primary/diagnostic/supporting: supporting (equivalence check) / not yet usable (timing)
- Next action: controlled timing campaign, kept separate from smoke conclusions
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 4

### 12. Cross-cutting fairness audit
- Scientific question: how fair is the overall baseline/method comparison
  pool, and what needs fixing before primary claims can rely on it?
- Reviewer concern: cross-cutting (Reviewer #2 Major 1, R3-Issue1)
- Machine: LOCAL
- Protocol: n/a (audit of existing artifacts, not a new experiment)
- Scope: full audit of the reviewer-fairness pool, `overall_score=76`
- Source entry point: n/a (audit artifact, no dedicated runner script tracked locally)
- Config: n/a
- Output path: `analysis/kbs_comparison_fairness_audit.json`, `docs/reviewer/kbs_comparison_fairness_audit.md`
- Status: `FINAL_VALIDATED` (as an audit; the gaps it identifies in underlying experiments are separately tracked per-row above)
- Evidence strength: `SUPPORTING_EVIDENCE`
- Primary/diagnostic/supporting: supporting
- Next action: none for the audit itself; see its `required_fixes` list (tracked via experiment #3 above)
- Canonical documentation: `docs/reviewer/kbs_comparison_fairness_audit.md`

### 13. Continuation-policy light ablation (historical)
- Scientific question: (superseded) does an alternative label-rollout
  continuation choice (`lru`/`blind_oracle`/`fifo`) change label proxies or
  replay outcomes?
- Reviewer concern: none current (predates the R2 Major 3 / R3
  continuation-mismatch framing that motivated experiment #10)
- Machine: LOCAL (historical)
- Protocol: exploratory, not frozen
- Scope: tiny (4 traces, <=300 requests/trace)
- Source entry point: `scripts/experiments/run_continuation_policy_light_ablation.py`, `src/lafc/evict_value_v2_rollout.py`
- Config: n/a
- Output path: `analysis/continuation_policy_light/` (already git-tracked, historical)
- Status: `SUPERSEDED` by experiment #10's causally-cleaner C1/C2 formalization
- Evidence strength: `HISTORICAL_SUPERSEDED`
- Primary/diagnostic/supporting: historical
- Next action: none; do not conflate with experiment #10
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.7 (this pass's cross-reference note); `docs/internal_current_project_decisions.md` 6-7

---

## Notes on scope creep / redundancy (Section 8/9 cross-check)

No orphan experiments were found (every experiment above maps to at least one
reviewer concern or hypothesis). No unnecessary duplication was found:
experiment #6 (exact-target oracle) and #7 (target degeneracy) look similar
but answer different questions (deployable-vs-exact comparison vs.
information-content-of-the-target) and are explicitly documented as distinct
in `kbs_negative_results_interpretation.md`. Experiment #13 is redundant with
#10 in topic only; it is explicitly marked `SUPERSEDED`, not deleted, and the
two are cross-referenced against each other rather than left ambiguous.
