# Manuscript consistency audit (zero-compute pass, 2026-06-19)

**Scope.** Read-only audit of the actual submission package
(`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`,
extracted read-only to a scratch temp dir, never modified) against current
repository state, while `cap64_with_sieve_fifo` continues running untouched
in the background. No experiments were run, no files inside the zip were
changed, and the running job was not inspected beyond a read-only status
check. This report is the deliverable for that audit; the manuscript
rewrite itself is **not** performed here — see §7/§8 and the tracker
updates below for what is queued.

**Source located.** No loose `main.tex`/`refs.bib`/cover-letter/highlights/
CRediT/author-agreement files exist anywhere in the repo outside the zip
(`find` over `.tex/.bib/.cls/.docx/.zip` and cover/highlight/credit/
interest/author-agreement filenames confirmed this). The zip is the single
source of truth for the submission package: `main.tex` (559 lines),
`refs.bib`, `cover-letter.tex`, `author-agreement.tex`, `elsarticle.cls` +
3 `.bst` files, and exactly two figures — `figures/method_overview.png` and
`figures/figure4_ablation.png`. No DOCX format exists anywhere.

---

## 1. Executive summary

- **cap64_with_sieve_fifo is running, untouched, not complete.** Re-verified
  at audit time (2026-06-19 22:51 EDT): tmux session
  `kbs_full_policy_comparison_cap64_with_sieve_fifo` alive, PID 261643
  alive, `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log` is
  0 bytes (no exit marker), started 22:29:17, ~22 minutes elapsed at audit
  time. Nothing in this pass touched it.
- **The manuscript prose itself is already conservatively worded.** A full
  read of `main.tex` plus targeted greps for the risk phrases the task asked
  about (`robust superiority`, `practical superiority`, `end-to-end
  improvement`, `lightweight production readiness`, `validated fallback`,
  `broad generality of H=4`) found **none of these phrases, or close
  paraphrases of them, anywhere in the current text.** The Abstract,
  Discussion, Limitations, and Conclusion sections already explicitly
  disclaim end-to-end superiority, deployment readiness, and universal
  validity of the fallback and of $H=4$. This is a materially different
  starting point than the task brief assumed — see §2 for what is actually
  unsafe versus what was checked and found already safe.
- **The real consistency problems are staleness and omission, not
  overclaiming:**
  1. SIEVE and FIFO-Reinsertion — implemented, unit-tested, and now backed
     by real cap32 numbers (and a running cap64 chunk) — are **absent from
     every part of `main.tex`**: not in Related Work, not in Table 2
     (`tab:main_policy_families`), not in Table 6
     (`table6_related_work_learned_caching.tex`), not anywhere.
  2. **Three different, mutually inconsistent policy rosters** exist across
     the repo: `main.tex`'s embedded Table 2 (7 entries, including a
     "Marker" and "R-FTP+Marker" row whose wiring status is unconfirmed),
     `tables/manuscript/table2_policy_roster.csv` (6 entries, includes REST
     but omits Marker/R-FTP/SIEVE/FIFO), and the actual canonical
     `--policies` list used in cap32/cap64 (8 entries: `lru, sieve,
     fifo_reinsertion, predictive_marker, blind_oracle_lru_combiner,
     trust_and_doubt, rest_v1, evict_value_v1`). None of the three match.
  3. The offline-ablation table (`tab:evict-value-ablation` in `main.tex`,
     mirrored in `tables/manuscript/table4_main_ablation.csv/.tex`) is
     confirmed stale relative to the current (uncommitted) heavy_r1 retrain
     — pinned to commit `53726ce`, per
     `reports/kbs_stale_artifact_refresh_plan.md`.
  4. There is **no online end-to-end results table or figure anywhere in
     `main.tex` today** — not stale, simply not yet written, because the
     canonical multi-capacity policy-comparison CSV
     (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`) that
     the manuscript builder gates on has never existed (per
     `reports/manuscript_artifacts/end_to_end_evidence_gap_report.md`) and,
     under the Option C decision, may never exist in that single-file form
     — see §4.
  5. Contribution #3 in the Introduction states the guarded-fallback
     extension in confident, completed-capability language
     ("We develop a practical robust extension...") that is **not matched**
     by the heavily hedged treatment of the same mechanism later in
     Discussion/Limitations ("should be interpreted as a practical
     robustness layer... rather than as a fully established empirical
     contribution on its own"). This asymmetry — strong claim up front,
     walked back later — is the single most reviewer-visible internal
     inconsistency in the current text, and lines up directly with
     R3-Issue6/Rec5.
- **cap128/cap256 remain not launched.** Nothing in this audit launched,
  planned, or recommended launching them; that decision stays pending
  review of the cap64 result per `reports/kbs_after_cap32_decision_memo.md`.
- **Nothing pushed, merged, deleted, or overwritten.** Only new files were
  written (this report plus tracker updates, all additive/append-only).

---

## 2. Claims that must be removed or softened

Checked explicitly for the phrase classes named in the task brief. Exact
finding for each:

| Risk phrase class | Found verbatim/paraphrased in `main.tex`? | Evidence |
|---|---|---|
| "robust superiority" | **No.** | `superior`/`superiority` appears 4 times (lines 423, 466, 468, 509), every instance phrased as a *disclaimer* ("does not by itself establish superiority...", "rather than as a claim of universal superiority"). |
| "practical superiority" | **No.** | Not present in any form. |
| "end-to-end improvement" | **No.** | `end-to-end` appears 6 times (lines 353, 423, 466, 468, 484, 505), every instance explicitly stating end-to-end evidence is *not yet shown* or *not established*. |
| "lightweight production readiness" | **No.** | `deploy`/`deployment` appears at lines 484, 511, 526, always hedged ("stronger claims about... guarded deployment... should be reserved for future evaluation"; "their value may therefore depend substantially on... deployment conditions"). No production-readiness claim exists. |
| "validated fallback" | **No** in the Discussion/Limitations text (which explicitly says the opposite — "the effectiveness of fallback control in deployment" is listed among claims *not* established). **Partially yes by omission** in the Introduction — see below. |
| "broad generality of H=4" | **No.** | Both $H=4$ discussions (lines 419, 462) are scoped with "within the current artifact set" / "among the evaluated settings" — not framed as general.

**The one real overclaiming risk found is structural, not a single
sentence to delete:**

> Introduction, Contribution #3 (`main.tex:75`): *"We develop a practical
> robust extension in which the learned candidate scorer can be combined
> with lightweight fallback behavior when recent online outcomes suggest
> locally unsafe decisions. This preserves the candidate-level structure of
> the method while providing a conservative control layer for imperfect
> predictive regimes."*

Read in isolation, this sentence asserts a working, beneficial mechanism.
Read against Limitations (`main.tex:507,511`) and Discussion
(`main.tex:466`), the same mechanism is described as unvalidated and
heuristic. **Recommended fix (zero-compute, available now):** apply the
already-drafted demotion language in
`reports/kbs_fallback_revision_strategy.md` to Contribution #3 — reframe it
from "we develop a practical robust extension" to something like "we also
study a lightweight guarded extension... as an exploratory robustness
mechanism, evaluated only at the level of behavioral diagnostics in this
revision" — so the front-matter claim matches the back-matter hedging. This
directly answers R3-Issue6/Rec5's "demote or validate" choice in favor of
demote, consistent with the strategy doc's conclusion (no fallback-specific
ablation artifact exists anywhere in the repo).

**Secondary, lower-severity item:** the Method section's Algorithmic
Workflow subsection (`main.tex:233-329`) and the guarded-decision Algorithm
box both present the fallback mechanism in full procedural detail as part
of "the proposed method," with no inline caveat at first introduction
(the caveats only appear later, in Discussion/Limitations). This is
consistent with normal manuscript structure (methods first, caveats in
discussion) and is **not** flagged as unsafe on its own — only the
Contribution #3 sentence needs softening, because contributions lists are
read by reviewers as the paper's claim ledger.

No other removal-worthy claim was found in `main.tex` or `cover-letter.tex`.
The cover letter's 3 keyword-grep matches (`robust`, `baseline`) are all
category-name usages ("robust learning-augmented caching," "robust
reference policies") rather than outcome claims, and need no change.

---

## 3. Claims that can remain as-is

- The decision-aligned/candidate-level target formulation (Introduction,
  Contributions #1-#2, Method §3.1-3.2) — methodologically scoped, not
  result-dependent, accurately describes what is implemented.
- The offline target-quality evaluation framing (§4.3, "Offline Ablation
  and Model Selection") — accurately scoped as offline-only; explicitly
  states it "does not by itself establish superiority in end-to-end online
  replay" (line 423). Safe even though the underlying numbers are stale
  (staleness is a data-refresh issue, not a claim-safety issue — see §4).
- The reproducibility-pipeline description (§3.4, Algorithmic Workflow) —
  accurate description of the actual five-stage pipeline; no result claims
  embedded.
- The deterministic SIEVE/FIFO-Reinsertion baseline descriptions — N/A to
  `main.tex` directly (they aren't mentioned at all, see §1), but the
  underlying implementation claims in `reports/kbs_sieve_implementation_report.md`
  and `reports/kbs_fifo_reinsertion_baseline_audit.md` are accurate and
  can be cited as-is once the manuscript text is added.
- The cap32 end-to-end audit framing in `reports/kbs_cap32_with_sieve_fifo_result_analysis.md`
  and `reports/kbs_cap32_policy_comparison_report.md` — both already state
  plainly that `evict_value_v1` loses to LRU on 6/7 trace families; nothing
  there needs softening, it is already conservative and accurate.
- Limitations §4.5 and Future Research §5.3 in `main.tex` — both already
  read as appropriately modest and forward-looking; no changes needed.
- Acknowledgements, Funding, Data/Code Availability, Declaration of
  Competing Interest, AI Declaration — all accurate as written, no
  conflicts with current repo state.

---

## 4. Tables/figures requiring regeneration (or net-new creation)

| Item | Current state | What's needed | Blocked on compute? |
|---|---|---|---|
| `tab:evict-value-ablation` (main.tex Table, = `tables/manuscript/table4_main_ablation.tex`) + Fig. 4 (`figure4_ablation.png`) | **Stale.** Pinned to commit `53726ce`; current uncommitted heavy_r1 retrain shows a different best-model selection (`random_forest` flip per `kbs_stale_artifact_refresh_plan.md`). | Run `python scripts/paper/build_kbs_main_manuscript_artifacts.py` against the current retrain, paste refreshed numbers into `main.tex`. | Minor/mechanical, not the cap64 sweep — **not executed in this zero-compute pass**, plan already exists (`kbs_stale_artifact_refresh_plan.md` §5-7). |
| `tables/manuscript/table5_offline_selection.tex` (+ Fig. 5, unreferenced in `main.tex`) | Same staleness family as Table 4; also not currently wired into `main.tex` at all. | Same regeneration script; decide whether to wire it into `main.tex` or leave as a repo-only artifact. | Same as above — mechanical, not a sweep. |
| Table 2 policy roster — **three disagreeing versions** (`main.tex` embedded table, `tables/manuscript/table2_policy_roster.csv`, actual 8-policy CLI roster) | **Inconsistent**, not just stale. | Reconcile to the actual canonical 8-policy set; resolve the REST_v1/"R-FTP+Marker" naming gap per `kbs_baseline_gap_action_plan.md` §5; add SIEVE/FIFO-Reinsertion rows. | **No — pure editing, zero compute, doable now.** |
| Table 6 related-work positioning table (`table6_related_work_learned_caching.tex`) | Missing SIEVE/FIFO-Reinsertion rows entirely; HALP row already self-flags as needing a wording check. | Add SIEVE (NSDI'24) and FIFO-Reinsertion (CLOCK/Second-Chance family) rows; add `zhang2024sieve` bib entry (currently absent from `refs.bib` — confirmed by grep). | **No — pure editing, zero compute.** |
| Table 3 / `table3_main_quantitative_comparison.csv/.tex` (main online-replay comparison) | Explicit, honest stub: `"NOT_VERIFIED... Canonical file... is absent."` Gated by `build_kbs_main_manuscript_artifacts.py` on a single canonical multi-capacity CSV that has never existed. | **Process note, not a bug**: under the Option C decision (cap64-only, cap128/256 deferred), the originally-envisioned single canonical multi-capacity file may never be built in that exact form. Whoever finalizes Table 3 will likely need to manually assemble it from the `_cap32_with_sieve_fifo` + `_cap64_with_sieve_fifo` chunk files (and any later-approved chunks) rather than waiting on the auto-gated pipeline. | **Yes — needs cap64 at minimum; full version needs a cap128/256 decision.** |
| Figures 2-3 (online comparison plots) | **Do not exist** (`manuscript_artifact_manifest.json` / `manuscript_artifact_report.md` record `policy_comparison_present: false`; builder leaves these as comment-only snippets). | Net-new creation once enough capacity data exists, same dependency as Table 3. | **Yes — same as Table 3.** |
| Figures 6/7/8 (`figure6_guard_wrapper_evict_value_v1`, `figure6_regret_vs_top1_alignment`, `figure7_continuation_policy_agreement`, `figure8_target_construction_concept`) | Exist as PNG/PDF in `figures/manuscript/`, but are **not referenced anywhere in `main.tex`** and are **not part of the submission zip** (zip contains only `method_overview.png` and `figure4_ablation.png`). | No action needed unless a future revision decides to add a guard-diagnostics or feature-construction figure to the paper. Currently correctly excluded given the fallback-demotion direction in §2. | No — these are dormant repo artifacts, not a manuscript inconsistency. |

---

## 5. Sections requiring rewrite — classification

Tags used (exactly as specified): **SAFE NOW**, **REWRITE NOW**, **WAIT FOR
CAP64**, **WAIT FOR FINAL SWEEP**, **REMOVE/DEMOTE**.

| Section | Tag | Rationale |
|---|---|---|
| Abstract | **SAFE NOW** | Already scoped to offline ablation + "practically grounded direction"; no overclaiming found (§2). Optional light touch-up after final sweep, not required. |
| Introduction / Contributions | **REWRITE NOW** (Contribution #3 only; rest is SAFE NOW) | Contribution #3's fallback wording is the one real claim/hedge asymmetry found (§2). Pure text edit, zero compute. |
| Related Work | **REWRITE NOW** | SIEVE and FIFO-Reinsertion are real, implemented, tested, and now results-backed baselines that are completely absent from this section and from Table 6. Adding them is zero-compute text + one bib entry. |
| Method / fallback mechanism (§3.2-3.4) | **REMOVE/DEMOTE** | Apply the drafted demotion from `reports/kbs_fallback_revision_strategy.md` consistently between Contributions and Method, so the fallback reads as an exploratory/secondary mechanism throughout rather than only in Limitations. |
| Experimental Setup (§4.1-4.2) | **REWRITE NOW** (policy-roster reconciliation) / **WAIT FOR CAP64** (results-dependent prose) | The Table 2 roster fix is pure editing and can happen today (§4). Any prose claiming what the "evaluated baseline set" empirically showed must wait for the cap64 numbers. |
| Results (§4.3, Offline Ablation) | **REWRITE NOW** (table refresh is mechanical, see §4) / **WAIT FOR CAP64** (new online-results subsection does not exist yet and cannot be written before cap64 lands) | Two separable pieces of work with different blockers; do not conflate "refresh stale offline table" with "add the still-nonexistent online table." |
| Discussion and Analysis (§4.4) | **SAFE NOW** | Already accurately scoped; will benefit from incorporating cap64 numbers once available but is not unsafe today. |
| Conclusions / Implications / Future Research (§5.1-5.3) | **SAFE NOW** | Same as above — already appropriately modest; "Future Research Directions" already states broader sweeps remain future work, consistent with the actual Option C decision. |
| Acknowledgements / Funding / AI Declaration / Data & Code Availability / Competing Interest | **SAFE NOW** | All accurate as written; no conflicts with current repo state. |
| Cover letter | **SAFE NOW** | No overclaiming found (§2); would only need a touch-up if the title/contribution framing changes per `kbs_manuscript_shortening_and_reframing_plan.md`'s alternate-title options, which is itself optional. |
| Response to reviewers (`reports/kbs_response_to_reviewers_skeleton.md`) | **Mixed — see §6** | Already uses its own disciplined placeholder-tag system (`[PENDING CAP64/CAP128/CAP256]`, `[PENDING BASELINE DECISION]`, `[PENDING MANUSCRIPT REWRITE]`) and was already updated for the cap64 launch in the prior pass. No section there is currently mistagged relative to the findings in this audit. |

---

## 6. Reviewer-comment coverage, mapped to real labels

Cross-checking `reports/kbs_response_to_reviewers_skeleton.md`'s existing
status tags against this audit's manuscript-text findings (no
contradictions found — listed for completeness and to confirm alignment):

| Label | Concern | Audit finding that bears on it | Skeleton's current status (unchanged by this audit) |
|---|---|---|---|
| AE | No end-to-end eval; insufficient baselines; unvalidated fallback; no overhead analysis | All four confirmed still true of `main.tex` as written: no online table exists (§4), SIEVE/FIFO missing from baselines (§1/§4), fallback claim/hedge asymmetry (§2), zero occurrences of "overhead"/"O("/"scalability" anywhere in `main.tex` (confirmed by grep). | `[PENDING MANUSCRIPT REWRITE]` |
| R2-MC1 | Replay horizon $H$ justification | Current text already hedges $H{=}4$ appropriately (§2); deeper sensitivity discussion still not inserted. | `[IN PROGRESS]` |
| R2-MC2 | Overhead of label construction | Confirmed: zero mention in `main.tex`; measured numbers exist in `kbs_overhead_and_scalability_evidence.md`/`kbs_overhead_manuscript_text_draft.md` but not yet inserted. | `[IN PROGRESS]` |
| R2-MC3 | Offline-vs-online gap; fallback under-demonstrated | Confirmed both halves: no online table (§4) and fallback asymmetry (§2). | `[PENDING CAP64/CAP128/CAP256]` + `[PENDING BASELINE DECISION]` |
| R3-Issue1 | No end-to-end miss-ratio results | Confirmed — no such table/figure exists in `main.tex` at all (§4), not merely outdated. | `[PENDING CAP64/CAP128/CAP256]` |
| R3-Issue2 | Insufficient HALP differentiation | Checked Related Work (`main.tex:89-91`) — HALP is already cited and analytically differentiated in prose; this section is in better shape than the other related-work gaps. | `[PENDING BASELINE DECISION]` |
| R3-Issue3 | Missing SIEVE/FIFO-Reinsertion comparisons | Confirmed absent from `main.tex` entirely (§1); canonical cap32 numbers exist, cap64 running. | `[IN PROGRESS]` |
| R3-Issue4 | Self-admitted weakness undermines contribution | The Contribution #3 asymmetry found in §2 is directly relevant — softening it now is one lever; the cap64/cap128/256 outcome is the other. | `[PENDING CAP64/CAP128/CAP256]` |
| R3-Issue5 | Computational cost not addressed | Confirmed zero mentions of overhead/complexity in `main.tex`; code-verified $O(\text{capacity})$ claim ready to insert (`run_policy.py`/`evict_value_v1.py` per skeleton). | `[IN PROGRESS]` |
| R3-Issue6 | Fallback unvalidated and oversold | This audit's single clearest finding (§2) — recommends the demote route, consistent with the skeleton's framing of options (a)/(b). | `[PENDING BASELINE DECISION]` |
| R3-Issue7 | Single authorship / AI-tool reliance | Not a manuscript-text-staleness issue; pure framing decision. | `[PENDING MANUSCRIPT REWRITE]` |
| R3-Minor8 | Verbosity/repetition | Confirmed by direct reading: Introduction §1.1/§1.2 and Discussion/Limitations repeat the same hedges (offline-vs-online, fallback caveats) 4-5 times each across sections — consistent with the reviewer's complaint and with `kbs_manuscript_shortening_and_reframing_plan.md`'s existing cut plan. | `[PENDING MANUSCRIPT REWRITE]` |
| R3-Minor9 | Missing workload-specific analysis | cap32 per-trace breakdown exists; cap64/full version still pending. | `[IN PROGRESS]` / `[PENDING CAP64/CAP128/CAP256]` |
| R3-Rec1-8 | (roll-up of above) | No new findings beyond Issues 1-6 above; see skeleton for the 1:1 mapping. | Per skeleton, unchanged. |

No relabeling of the skeleton's tags is recommended — this audit's findings
are consistent with, not corrective of, its existing status assignments.

---

## 7. Recommended immediate edits (zero-compute, available today)

1. **Demote fallback Contribution #3** to match Limitations' hedging
   (apply `reports/kbs_fallback_revision_strategy.md`'s draft language).
   Resolves the single clearest internal inconsistency found (§2) and
   directly addresses R3-Issue6/Rec5.
2. **Add SIEVE + FIFO-Reinsertion to Related Work and Table 6**, with a new
   `zhang2024sieve` bib entry (confirmed missing from `refs.bib`).
   Addresses R3-Issue3/Rec2 (SIEVE/FIFO half).
3. **Reconcile the three disagreeing policy rosters** into one canonical
   8-policy Table 2, matching the actual `--policies` CLI list. Fixes the
   REST_v1/"R-FTP+Marker" gap flagged in `kbs_baseline_gap_action_plan.md`
   §5 at the same time.
4. **Insert the overhead/scalability subsection draft**
   (`reports/kbs_overhead_manuscript_text_draft.md`) — currently zero
   presence of this topic anywhere in `main.tex` despite it being an
   explicit AE/R2-MC2/R3-Issue5/Rec3 concern with ready text.
5. **Insert the $H=4$ interpretation draft**
   (`reports/kbs_horizon_h4_revision_strategy.md`) to deepen the existing
   (already-safe, but thin) horizon discussion, addressing R2-MC1/R3-Rec7.
6. **Disclose the validation-set family-coverage caveat** (citibike/metakv
   absent from the ~4,572-row model-selection sample) wherever model
   selection is discussed — ties directly into item 1 of the Table 4
   refresh discussion in §4.
7. **Fix the four incomplete BibTeX entries** (`lykouris2018competitive`,
   `bansal2022weightedpaging`, `wei2020lacaching`,
   `chledowski2021robustlacaching`) in `refs/related_work_table6.bib`.
8. **Trim the repeated hedging** flagged by R3-Minor8 in Introduction
   §1.1-1.2 and Discussion/Limitations, per the existing
   `kbs_manuscript_shortening_and_reframing_plan.md` — a larger lift than
   items 1-7, but still zero-compute and independently actionable.

None of the above touch `cap64_with_sieve_fifo`, launch any experiment, or
require cap128/256.

---

## 8. Edits that must wait

1. **Any net-new online end-to-end results table/figure** (Table 3,
   Figs. 2-3) — does not exist today (§4); cannot be written responsibly
   before cap64 completes, and the *final* version needs a human decision
   on cap128/256.
2. **R3-Issue1/Rec1** (end-to-end miss-ratio results) and **R3-Issue4**
   (choosing response (a) "robust across conditions" vs. (b) "revised
   scope") — both depend on the cap64 outcome.
3. **R2-MC3/R3-Minor9/Rec8 full multi-capacity workload breakdown** — cap32
   alone exists; cap64 in progress; cap128/256 not launched.
4. **A real fallback-ablation alternative to demotion** (R3-Issue6/Rec5
   option (a)) — only required if a human decides to validate rather than
   demote the fallback; the demote route (item 1 of §7) needs no new
   compute and is recommended as the lower-risk default.
5. **Final reconciliation of Table 3's data source** — depends on whether
   a human ever decides to launch cap128/256, or whether the manuscript
   commits to a 2-capacity (32+64) online story instead. This is a
   decision to flag, not to make in this pass.

---

## Files read for this audit (no modifications)

`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
(read-only `unzip -l` + extraction to `/tmp/kbs_manuscript_audit_readonly`,
outside the repo, not committed); `main.tex`; `cover-letter.tex`;
`author-agreement.tex`; `refs.bib` (grep only); `tables/manuscript/*.csv`,
`*.tex`; `figures/manuscript/` (`ls` only); `reports/manuscript_artifacts/end_to_end_evidence_gap_report.md`;
`reports/manuscript_artifacts/reviewer_concern_gap_map.md` (header only,
already-existing supersession banner, not modified by this audit);
`reports/kbs_next_actions_without_rerun.md`;
`reports/kbs_response_to_reviewers_skeleton.md`.
