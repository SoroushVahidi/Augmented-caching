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
held-out replay and controlled timing are now complete on Wulver; R2 Major 1
baseline-side evidence is locally exact-protocol validated with per-baseline
caveats and final synthesis is pending only the corrected treatment sync; broad
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
- Machine: LOCAL; completion is active in the resumable local runner
- Protocol: `docs/distribution_shift_ablation_protocol.md`, frozen
- Scope: seven folds, 42 primary rows; read the live manifest for progress
- Source entry point: `scripts/experiments/run_distribution_shift_ablation.py`, resumable via `scripts/experiments/resume_distribution_shift.py`
- Config: protocol doc-embedded, frozen condition matrix
- Output path: `analysis/distribution_shift_ablation_v1/`
- Status: `RUNNING_LOCAL_TMUX`; use `analysis/distribution_shift_ablation_v1/campaign_state.json` and the live log for progress. Partial rows are not citable.
- Evidence strength: `MECHANISTIC_DIAGNOSTIC`
- Primary/diagnostic/supporting: diagnostic
- Next action: allow the local campaign to finish, then run the formal integrity audit
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.6

### 3. Corrected held-out `evict_value_v1` replay
- Scientific question: does the primary method actually beat baselines under
  a corrected, non-contaminated, cross-family held-out protocol?
- Reviewer concern: Reviewer #2 Major 1 / R3-Issue1 (end-to-end evidence)
- Machine: LOCAL (partial) / BOTH for full completion
- Protocol: `docs/reviewer_fairness_cross_family_v1.md`
- Scope: treatment artifact complete on Wulver, absent locally; baseline side locally exact-protocol validated
- Source entry point: `scripts/experiments/run_evict_cross_family_pipeline.py`, `scripts/experiments/run_cross_family_heldout_eval.py`
- Config: `configs/fair_cross_family_v1/folds/*.json`
- Output path: `analysis/reviewer_fairness_cross_family_v1/`
- Status: `WULVER_ONLY_VALIDATED` for treatment artifact; R2 Major 1 overall `EXPERIMENTALLY_COMPLETE_SYNTHESIS_PENDING`
- Evidence strength: `PRIMARY_REVIEWER_EVIDENCE` after sync/review; locally supporting-only until the verified treatment CSV is present
- Primary/diagnostic/supporting: intended as primary; final numeric table pending local synchronization/synthesis
- Next action: sync the corrected 42/42 Wulver CSV and provenance, verify SHA-256 `982bfdffdbd816b56c2eef86ecb730a1eb136b3f85e36ad533739e586fa0a296`, then run `scripts/analysis/prepare_r2_major1_evidence.py --treatment-csv ...`
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
- Status: `FINAL_VALIDATED` for baseline side; LRB `LOCAL_EXACT_PROTOCOL_VALIDATED`, 3L-Cache `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_CAVEAT`, CACHEUS `LOCAL_EXACT_PROTOCOL_VALIDATED_WITH_PROVENANCE_CAVEAT`
- Evidence strength: `PRIMARY_REVIEWER_EVIDENCE`, with fidelity caveats: CACHEUS `HIGH` fidelity (official source); LRB/3L-Cache `MEDIUM` (independent reimplementation); **HALP `LOW_TO_MEDIUM`** -- no public official HALP implementation exists, this is an adapted reimplementation of a continuous production algorithm against a frozen offline split, not the vendor's own code
- Primary/diagnostic/supporting: primary, with the HALP caveat always attached
- Next action: none for baseline compute; keep the fidelity/provenance caveats attached in any manuscript citation and use `analysis/kbs_r2_major1_evidence_prep_20260811/` for compact provenance
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 1; `analysis/kbs_comparison_fairness_audit.json`

### 6. Exact-target oracle replication
- Scientific question: does exact optimization of the frozen `eviction_loss`
  finite-horizon target beat LRU, and does the learned model reproduce or
  outperform that exact target?
- Reviewer concern: MC1 (horizon justification), R3-Issue4
- Machine: LOCAL
- Protocol: shares label semantics with `supervision_objective_ablation.py`'s shared kernel; horizon `H=4`
- Scope: 21 family-capacity cells, 42 required rows, H=4
- Source entry point: `scripts/experiments/run_exact_target_oracle_replication.py`
- Config: `configs/exact_target_oracle_replication_v1.json`
- Output path: `analysis/exact_target_oracle_replication_v1/`
- Status: `FINAL_VALIDATED`
- Evidence strength: `MECHANISTIC_DIAGNOSTIC_HIGH` (21-cell family/capacity scope)
- Primary/diagnostic/supporting: diagnostic
- Next action: none; do not rerun
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` "Exact target oracle vs learned online policy"; `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` H2/H3

### 7. Strict-preference/horizon diagnostic
- Scientific question: how degenerate (low-resolution) is the H=4
  `eviction_loss` target, and how much do longer horizons (H=8/16/32) resolve
  that degeneracy?
- Reviewer concern: MC1 (horizon justification)
- Machine: LOCAL
- Protocol: standalone math module over candidate values produced by the
  shared kernel; base `H=4`, extension horizons `8,16,32`
- Scope: 21 family-capacity cells; 63 H4→H8/H16/H32 comparison rows
- Source entry point: `scripts/experiments/run_strict_preference_horizon_diagnostic.py`
- Config: `configs/strict_preference_horizon_diagnostic_v1.json`
- Output path: `analysis/strict_preference_horizon_diagnostic_v1/`
- Status: `FINAL_VALIDATED`
- Evidence strength: `MECHANISTIC_DIAGNOSTIC_HIGH` (21-cell scope; H3 `STRONGLY_SUPPORTED`)
- Primary/diagnostic/supporting: diagnostic
- Next action: none for this revision; do not rerun. A separate base-horizon sweep remains optional future work.
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.5/9.5.1/9.5.2; `KBS_SECOND_REVISION_HYPOTHESIS_MAP.md` H3/H4/H9/H10

### 8. Learned/exact target agreement and regret
- Scientific question: is model-fitting failure a family-general explanation for the online gap?
- Reviewer concern: Reviewer #2 Major 3 / Reviewer #3
- Machine: LOCAL
- Protocol: `configs/learned_exact_target_agreement_v1.json`, frozen H=4 target and seven-model registry
- Scope: 21 family-capacity cells, 21 summaries, seven held-out models
- Source entry point: `scripts/experiments/run_learned_exact_target_agreement.py`
- Output path: `analysis/learned_exact_target_agreement_v1/`
- Status: `FINAL_VALIDATED`
- Evidence strength: `MECHANISTIC_DIAGNOSTIC_HIGH`; set-aware agreement must be read with degeneracy and regret metrics
- Primary/diagnostic/supporting: diagnostic
- Result: macro set-aware agreement ≈0.975301, positive-regret fraction ≈0.024699, learned misses 601,569 versus LRU 565,126
- Next action: none; do not rerun

### 9. Historical-tail diagnostic
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

### 10. Learning-convergence (same-target scalar-vs-pairwise)
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

### 11. Continuation-policy causal ablation (C0/C1/C2)
- Scientific question: does replacing the fixed LRU label-continuation with
  the already-learned frozen `pi1` continuation improve the next learned
  policy `pi2`?
- Reviewer concern: Reviewer #2 Major 3 / Reviewer #3 continuation-mismatch
- Machine: LOCAL tmux session `kbs_continuation_c0_c1_c2_production_resume2_retry_20260812`
- Protocol: `configs/continuation_policy_causal_ablation_v1.json`, `PROTOCOL_FROZEN_NO_RESULTS`
- Scope: 21 planned `(held_out_family, capacity)` production units; expected 63 policy rows, 21 label-agreement rows, 21 pi2-training rows
- Source entry point: `src/lafc/continuation_policy_ablation.py`, `scripts/experiments/run_continuation_policy_causal_ablation.py`
- Config: `configs/continuation_policy_causal_ablation_production_v1.json`
- Output path: `analysis/continuation_policy_causal_ablation_production_v1/`; models in `models/continuation_policy_causal_ablation_production_v1/`
- Status: `PRODUCTION_RUNNING_LOCAL_TMUX`
- Evidence strength: `RUNNING_NO_RESULTS_YET`
- Primary/diagnostic/supporting: diagnostic (planned)
- Next action: monitor/resume; do not relaunch duplicate campaign; do not cite outcomes until final integrity passes
- Launch provenance: launched 2026-08-11 from source SHA `a813617f36822f793b0e48b0ee3e6009d56ee324`, config SHA-256 `7556e120ead3b3e8a8c6d85ef7f800f2e8f1f1cb37800bde57b14d1a194d8670`, serial thread caps, `--resume --max-wall-hours 8`
- Canonical documentation: `docs/reviewer/kbs_negative_results_interpretation.md` 9.7 (includes this pass's cross-reference to the older, distinct `evict_value_v2_rollout.py` continuation exploration); `docs/reviewer/local_to_wulver_continuation_sync_manifest.md`

### 12. Practical-significance / controlled timing
- Scientific question: is fine-grained candidate-level learned eviction
  computationally practical?
- Reviewer concern: MC2, R3-Issue5
- Machine: LOCAL
- Protocol: `docs/practical_significance_ablation_protocol.md`
- Scope: local smoke-scale equivalence check complete; controlled 420/420 timing is Wulver-only and sync-pending
- Source entry point: `scripts/experiments/run_practical_significance_ablation.py` (smoke), `scripts/experiments/run_practical_significance_controlled.py` (controlled, not yet run)
- Config: protocol doc-embedded
- Output path: `analysis/practical_significance_ablation_v1/`
- Status: `SMOKE_ONLY` locally; controlled timing `WULVER_ONLY_VALIDATED` / `SYNC_PENDING`
- Evidence strength: `IMPLEMENTATION_ONLY` for timing numbers; the exact-decision-preserving equivalence check itself (`all_variants_exact_across_all_trace_capacity_pairs=true`) is `SUPPORTING_EVIDENCE`
- Primary/diagnostic/supporting: supporting (equivalence check) / not yet usable (timing)
- Next action: synchronize and audit the Wulver timing payload; do not substitute local smoke timings
- Canonical documentation: `docs/reviewer/kbs_second_revision_artifact_map.md` Reviewer #2 Major 4

### 13. Cross-cutting fairness audit
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
