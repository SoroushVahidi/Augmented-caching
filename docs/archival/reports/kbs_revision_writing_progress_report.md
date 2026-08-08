# KBS revision-writing progress report — 2026-06-21

Manuscript: KNOSYS-D-26-07461. Decision: Revise. Due: 2026-07-08.

This report covers the revision-writing pass that converts the completed
cap32/cap64/cap128 end-to-end evidence and the cap128 anomaly sanity audit
(`reports/kbs_cap128_anomaly_sanity_audit.md`) into actual manuscript and
response-to-reviewers text. It was performed under explicit constraints:
**do not launch cap256; do not run heavy experiments; do not overwrite raw
CSV/MD outputs; do not push, commit, merge, delete, or rename anything;**
manuscript/report editing is allowed.

## 1. What changed

### 1.1 Capacity-explicit tables/figures (built first, used as the source of truth for all prose below)

- `scripts/paper/build_kbs_policy_trend_artifacts.py` re-run with
  `--inputs <cap32,cap64,cap128 with_sieve_fifo csvs> --capacities 32,64,128`,
  producing/refreshing:
  - `analysis/kbs_policy_trend_available_capacities.csv`
  - `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`
  - `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`
  All three are explicitly labeled "DRAFT / AVAILABLE CAPACITIES ONLY (cap32,
  cap64, cap128 — cap256 NOT run)" and are distinct filenames from the
  canonical `table3_main_quantitative_comparison.csv`, which remains
  untouched in its `NOT_VERIFIED` stub state.
- New script `scripts/paper/build_kbs_available_capacities_figure.py`
  (created this pass) reads the trend CSV and produces:
  - `figures/manuscript/figure_available_capacities_trend_DRAFT.pdf` / `.png`
  - `reports/manuscript_artifacts/latex_snippets/figure_available_capacities_trend_DRAFT_snippet.tex`
  Two-panel figure: (a) mean misses by capacity per policy, (b)
  `evict_value_v1`'s gap vs LRU/SIEVE/FIFO-Reinsertion by capacity. The PNG
  was also copied to `manuscript_source/figures/` to match the manuscript's
  flat figures-directory convention.
- This pathway is intentionally separate from
  `scripts/paper/build_kbs_main_manuscript_artifacts.py` (Table 3/Figure
  2/Figure 3), which is gated on a single capacity-blind merged canonical
  CSV that does not exist and uses a stale 6-policy roster missing SIEVE/
  FIFO-Reinsertion. That pathway was left untouched by design — it is out
  of scope for this capacity-explicit revision.

### 1.2 Manuscript (`manuscript_source/main.tex`)

Seven edits applied: Abstract; two new subsections ("End-to-End Online
Replay Evaluation Across Available Capacities," "Workload-Specific
Breakdown"); Discussion and Analysis; Limitations point 3; Summary of
Findings (opening and closing paragraphs); figure path fix. Full detail,
exact numbers, and rationale for each edit: `reports/kbs_manuscript_change_log_2026-06-21.md`.

Net effect: the manuscript now reports, in the reviewers' own terms, that
`evict_value_v1` does **not** beat LRU, SIEVE, or FIFO-Reinsertion at any of
the three evaluated capacities, that the gap widens non-monotonically at
cap128, and that this is analyzed (not hidden) and linked to the single-step
LRU-continuation labeling methodology. No claim of superiority appears
anywhere in the new or edited text. SIEVE and FIFO-Reinsertion are
consistently described as strong, low-overhead baselines.

### 1.3 Response to reviewers

- `reports/kbs_response_to_reviewers_skeleton.md` (authoritative tracker):
  11 edits — new banner; AE; R2-MC3; R3-Issue1; R3-Issue3 (update
  sub-paragraph); R3-Issue4; R3-Issue6 (update sub-paragraph); R3-Issue7;
  R3-Minor8; R3-Minor9; Recommended-Revisions items 1/2/4/6/8; and the
  bottom status-summary table/paragraph fully recomputed (see §2 below).
- `submission_kbs_revision_docx/response_to_reviewers_skeleton.md`
  (externally-facing letter rendering): mirrored the tracker's substantive
  content into letter prose — banner, AE, R2-MC2/MC3, R3-Summary,
  R3-Issue1/3/4/6, R3-Minor8/9, Recommended Revisions 1-8, and the closing
  "Do not submit as-is" caveat — per the file's own stated convention
  ("update the tracker first, then mirror relevant changes here"). The
  `.docx` render of this file was **not** regenerated in this pass (no
  pandoc re-run); it is now stale relative to the `.md` and is tracked as a
  remaining step (§3).

## 2. Status-summary recomputation (tracker bottom table)

Before this pass: `[DONE]`=0, `[IN PROGRESS]`=8, `[PENDING CAP128/CAP256]`=6,
`[PENDING BASELINE DECISION]`=5, `[PENDING MANUSCRIPT REWRITE]`=6,
`[PENDING DOCX PACKAGE]`=0.

After this pass: `[DONE]`=0, `[IN PROGRESS]`=13 (AE, R2-MC1, R2-MC2, R2-MC3
end-to-end half, R3-Issue1, R3-Issue3, R3-Issue4, R3-Issue5, R3-Issue7,
R3-Minor9, R3-Rec1, R3-Rec2 SIEVE/FIFO-Reinsertion half, R3-Rec3, R3-Rec4,
R3-Rec7, R3-Rec8), `[PENDING CAP128/CAP256]`=0 (cap128 is done; cap256 is
now a tracked scope decision, not a pending-compute blocker),
`[PENDING BASELINE DECISION]`=4 (R2-MC3 fallback half, R3-Issue2, R3-Issue6,
R3-Rec2 HALP half, R3-Rec5), `[PENDING MANUSCRIPT REWRITE]`=3 (R3-Summary,
R3-Minor8, R3-Rec6), `[PENDING DOCX PACKAGE]`=0.

No item reached `[DONE]` in this pass — every upgraded item still needs a
final cross-check (e.g., a clean `tectonic` compile, see §4) before it can
be called finished.

## 3. What remains before a final DOCX submission

1. **Fallback validate-or-remove decision** (R2-MC3 fallback half,
   R3-Issue6, R3-Rec5) — open author decision; no ablation has been run.
2. **HALP-reimplementation scope decision** (R3-Issue2, R3-Rec2 HALP half) —
   open author decision; analytical differentiation stands in its place for
   now.
3. **Manuscript-shortening pass** (R3-Minor8, R3-Rec6, R3-Summary) — not yet
   applied; the manuscript grew in this pass rather than shrinking, since
   the priority was adding the missing end-to-end evidence first.
4. **cap256 scope decision** — still on hold, not part of this revision;
   needs an explicit go/no-go separate from this writing pass.
5. **LaTeX compile verification** of the now heavily-edited `main.tex` (new
   labels/refs/tables/figure) — staged as the next step (§4), not yet
   confirmed in this pass.
6. **Regenerate the `.docx` render** of
   `submission_kbs_revision_docx/response_to_reviewers_skeleton.md` via
   `pandoc` to match its now-updated `.md` source — not done in this pass.
7. **Revised manuscript DOCX** (the submission-package item itself) —
   still blocked behind items 1-5 above; not started.

## 4. Verification status

Both lightweight checks were run and passed in this pass:

- `scripts/paper/verify_kbs_policy_chunks.py` on cap32+cap64+cap128
  together (`--expected-capacities 32,64,128`, full 8-policy roster):
  **PASSED**, zero errors/warnings.
- `tectonic main.tex`, clean rebuild from scratch: **exit 0**, `main.pdf`
  written (42 pages, 890,312 bytes), **zero undefined reference/citation
  warnings** in the final pass. Only pre-existing cosmetic overfull/
  underfull `\hbox` warnings remain (not introduced by this pass). All five
  new labels independently confirmed defined exactly once each.

Item 5 in §3 above is therefore resolved; items 1-4 and 6-7 remain open.

## 5. Constraints honored

- cap256: **not launched**. No file in this pass states or implies a
  cap256 result.
- No heavy experiment was run; only existing chunk CSVs were read, an
  existing trend-builder script was re-run on already-completed inputs, and
  one new lightweight (no-compute, pure plotting) figure script was added.
- No raw CSV/MD output was overwritten — the cap32/64/128 chunk files were
  only read (via direct Python computation for exact per-family
  percentages), never modified.
- No push, commit, merge, delete, or rename was performed.
