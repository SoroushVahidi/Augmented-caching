# Safe manuscript-source edits report (2026-06-19)

Zero-heavy-compute pass, done while `kbs_full_policy_comparison_cap64_with_sieve_fifo`
continued running, untouched, on the local/cloud machine (not Wulver, no Slurm). This
pass makes actual text edits to manuscript source — distinct from the earlier
**read-only** audit (`reports/kbs_manuscript_consistency_audit.md`), which inspected
the submission zip but changed nothing.

## 0. Where the manuscript source now lives

The only copy of `main.tex`/`refs.bib`/`cover-letter.tex`/`author-agreement.tex` in
this repo was, until this pass, sealed inside
`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
at the repo root, with no loose `.tex` source anywhere in the tracked tree (confirmed
again this pass via `find . -maxdepth 5 -type f \( -iname "main.tex" -o -iname
"refs.bib" -o -iname "*.tex" \)`, which returned only pre-existing
`tables/manuscript/*.tex` and `reports/manuscript_artifacts/latex_snippets/*.tex`
snippet/table files — not `main.tex` itself).

Since the user's instruction this pass explicitly authorizes "manuscript-source edits"
and asks for a "files edited" report, in-place editing of the original zip was **not**
an option (the safety rules forbid overwriting artifacts), so the zip was extracted
**read-write into a new, durable, repo-tracked directory**:

```
manuscript_source/
├── author-agreement.tex
├── cover-letter.tex
├── elsarticle.cls
├── elsarticle-harv.bst
├── elsarticle-num.bst
├── elsarticle-num-names.bst
├── figures/
│   ├── figure4_ablation.png
│   └── method_overview.png
├── main.tex      <- edited this pass
└── refs.bib      <- edited this pass
```

The original zip (`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`)
is **untouched** — `manuscript_source/` is a new, separate extraction, not an
overwrite. `manuscript_source/main.tex` and `manuscript_source/refs.bib` are now the
live, editable working copy; all edits below were made there.

## 1. Files edited

| File | Type of change |
|---|---|
| `manuscript_source/refs.bib` | Added 1 new BibTeX entry (`zhang2024sieve`) |
| `manuscript_source/main.tex` | 10 targeted text edits (Abstract, Contributions, Related Work, Method/guard section, Datasets/Baselines prose, Table 2, two new Discussion subsections) |
| `scripts/paper/build_kbs_main_manuscript_artifacts.py` | Added `sieve` and `fifo_reinsertion` rows to the hardcoded policy-roster generator list (`_build_table2_policy_roster`) |
| `tables/manuscript/table2_policy_roster.csv` | Regenerated (by rerunning only `_build_table2_policy_roster()`, no other artifact builder) to include the 2 new rows |
| `tables/manuscript/table2_policy_roster.tex` | Regenerated alongside the CSV |
| `reports/manuscript_artifacts/latex_snippets/table2_snippet.tex` | Regenerated alongside the CSV/tex (caption/label only, no row data) |

No file outside this list was modified. `manuscript_source/` itself (the new
directory) is new/untracked; everything else above already existed and was edited
in place.

## 2. Exact nature of edits

All edits are text-only, conservative, and independent of cap32/64/128/256 numeric
results. None inserts a final online-results claim, a cap64 number, or a new Table 3.

1. **Abstract** (`main.tex`): reworded the fallback clause from "we also examine
   lightweight guarded variants..." to "we also describe, but do not yet empirically
   validate, a lightweight guarded extension... we present this guard as a design
   extension for future empirical study rather than as a validated component of the
   present results."
2. **Contribution #3** (`main.tex`, numbered contributions list): reworded from "We
   develop a practical robust extension..." to explicitly call it "an optional guard,
   an implementation safeguard," state it is "evaluated only as an ablation and as a
   candidate for future robustness validation," and add "we do not claim that it
   improves end-to-end miss ratio unless future experiments support it (see
   Limitations)."
3. **Related Work** (`main.tex`): inserted a new paragraph (after the MUSTACHE
   paragraph, before the trace-families paragraph) introducing SIEVE and
   FIFO-Reinsertion as CLOCK/Second-Chance-family structural baselines, citing
   `zhang2024sieve` for both (the SIEVE paper itself frames the
   hand-advancement-vs-reinsertion contrast, per
   `reports/kbs_sieve_source_verification.md` §6).
4. **Method/guard section** (`main.tex`, "Robust Decision Mechanism" subsection): two
   edits — (a) reworded the guard's introductory sentence to call it "an implementation
   safeguard and a candidate design for future empirical validation rather than ... a
   demonstrated component of the present results (see Limitations)"; (b) appended an
   explicit empirical-status flag to the "two layers" summary paragraph: "We note that
   this second layer's empirical effect on end-to-end outcomes has not yet been
   measured in the present study; we discuss this gap directly in Limitations."
5. **Datasets/Baselines prose** (`main.tex`, §"Datasets, Baselines, and Evaluation
   Metrics"): rewrote the baseline-pool description paragraph to (a) drop "marker-style
   paging" and "robust follow-the-prediction with marker fallback" — neither is in the
   canonical `POLICIES` dict actually run for cap32/64 — and (b) add SIEVE,
   FIFO-Reinsertion, and REST, which are canonical but were previously either absent
   (SIEVE, FIFO-Reinsertion) or entirely unmentioned anywhere in `main.tex` (REST/
   `rest_v1` — confirmed via `grep -n "rest_v1\|ReST\|REST\b" main.tex refs.bib`
   returning zero hits before this edit, even though `rest_v1` is part of the actual
   canonical 8-policy comparison and already has its own row in
   `tables/manuscript/table2_policy_roster.csv`).
6. **Policy-roster cross-reference sentence** (`main.tex`, same subsection, the
   sentence beginning "Within the learned-policy family..."): added explicit naming of
   "a standalone marker-style policy" and "a robust follow-the-prediction combiner with
   marker fallback" to the existing "other policy families... retained for contextual
   analysis... not treated as central baselines unless explicitly reported" sentence,
   so the manuscript now explicitly states that these two ICML'21-style variants
   (`src/lafc/policies/marker.py`, `src/lafc/policies/robust_ftp_marker_combiner.py`)
   are implemented but non-canonical, rather than leaving that fact implicit. Also
   folded in the same guard-demotion language as edit 2/4.
7. **Table 2** (`main.tex`, `tab:main_policy_families`): replaced the "Marker" and
   "R-FTP+Marker" rows (non-canonical) with "SIEVE" and "FIFO-Reinsertion" rows
   (canonical), reordered rows to match the user-specified canonical order, and added a
   citation to "LRU" that was previously uncited. Added a caption clause noting that
   the diagnostic-only `blind_oracle` variant is intentionally excluded from the table.
   Final table now lists exactly the 8 canonical policies: LRU, SIEVE,
   FIFO-Reinsertion, PredMk, BO/LRU, T&D, REST, EV.
8. **Two new Discussion subsections** (`main.tex`, inserted at the end of "Discussion
   and Analysis," before the "Conclusions and Future Research" section, so nothing in
   Conclusions/Future Research is touched): "Overhead and Scalability" and "Replay
   Horizon Selection." Both use only already-measured, already-source-backed evidence
   mined in earlier zero-compute passes (`reports/kbs_overhead_manuscript_text_draft.md`,
   `reports/kbs_horizon_h4_revision_strategy.md`) — dataset-build wall-clock (~10.3h /
   96 GB / 662 shards), training wall-clock (~7-8 min), code-verified O(k) vs. O(1)
   per-miss complexity, and per-trace-family validation regret figures mined from the
   already-committed `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json`. Each
   subsection carries an explicit, commented-out `% TODO(revision): ...` source marker
   (not typeset, since no `todonotes`-style package is loaded in this manuscript) for a
   human reviser to revisit once cap64/128/256 complete.
9. **Policy-roster generator + generated table files**: added `sieve` and
   `fifo_reinsertion` rows to `_build_table2_policy_roster()` in
   `scripts/paper/build_kbs_main_manuscript_artifacts.py`, then regenerated only
   `tables/manuscript/table2_policy_roster.csv/.tex` and
   `reports/manuscript_artifacts/latex_snippets/table2_snippet.tex` by calling that one
   function directly (not the script's full `main()`, which also builds Table 3/4/5
   from policy-comparison CSVs that are mid-write or stale — calling only the
   roster-builder avoided touching anything cap64-dependent). This keeps the
   stand-alone roster table files (used outside `main.tex`, e.g. for
   response-to-reviewers text) consistent with `main.tex`'s own Table 2 and with the
   actual canonical 8-policy CLI list.

## 3. Citations added/verified

- **Added**: `zhang2024sieve` (NSDI'24, "SIEVE is Simpler than LRU"), full BibTeX entry
  added to `manuscript_source/refs.bib`, author/venue/page details taken directly from
  the already-completed source-verification pass
  (`reports/kbs_sieve_source_verification.md` §1-4, which fetched and read the official
  USENIX proceedings PDF). Used for both the SIEVE baseline and the
  FIFO-Reinsertion/CLOCK/Second-Chance-family discussion, since the SIEVE paper itself
  frames that exact contrast (§3.1, p.1232, per the verification report).
- **Deliberately not added**: `yang2023s3fifo` (S3-FIFO/"FIFO queues are all you need
  for cache eviction"). `reports/kbs_fifo_reinsertion_baseline_audit.md` §5 lists this
  as an *optional* secondary citation "if explaining the FIFO-family context... in more
  detail," with `zhang2024sieve` alone as the "practical recommendation." Unlike SIEVE,
  no dedicated zero-compute source-verification pass exists in this repo for S3-FIFO
  (no paper fetched/read, no author/page/venue cross-check), so fabricating a BibTeX
  entry from memory would risk an inaccurate citation (wrong page numbers, author list,
  or venue). Skipped as the conservative choice; flagged as a TODO below.
- **HALP** (`song2023halp`): verified already adequate, left untouched. The existing
  Related Work paragraph already differentiates HALP ("uses candidate-level preference
  learning from future re-access outcomes rather than direct action imitation") without
  claiming any empirical comparison against it, and HALP does not appear anywhere in
  the baseline-pool/Table 2 description. No edit was needed to satisfy "add HALP
  discussion if not already adequate, but do not claim empirical HALP comparison" —
  it was already adequate and already made no such claim.

## 4. Fallback demotion wording

Applied in 5 separate locations in `main.tex` (Abstract, Contribution #3, Method
section twice, Datasets/Baselines cross-reference sentence — see §2 items 1, 2, 4, 6
above), consistently using the user-specified vocabulary:

- "implementation safeguard"
- "optional guard"
- "evaluated only as an ablation"
- "a candidate for future robustness validation" / "a candidate design for future
  empirical validation"
- explicit "we do not claim that it improves end-to-end miss ratio unless future
  experiments support it" (appears twice, verbatim or near-verbatim)

This follows the already-drafted, already-reasoned strategy in
`reports/kbs_fallback_revision_strategy.md` §5 (written in an earlier zero-compute
pass but never applied to actual manuscript text until now). The pre-existing
Limitations-section hedging (`main.tex`, "the fallback mechanism is introduced as a
practical control layer rather than as a formally proved safeguard," and similar
sentences) was left untouched — it was already consistent with the demoted framing and
needed no change; this pass mainly fixed the **earlier, more confident** sections
(Abstract, Contributions, Method) that had been out of step with it.

## 5. Policy roster reconciliation

Final manuscript-citable baseline set, now consistently named and ordered everywhere
edited this pass:

```
lru, sieve, fifo_reinsertion, predictive_marker, blind_oracle_lru_combiner,
trust_and_doubt, rest_v1, evict_value_v1
```

- `main.tex` Table 2 (`tab:main_policy_families`): now exactly these 8 (as LRU, SIEVE,
  FIFO-Reinsertion, PredMk, BO/LRU, T&D, REST, EV).
- `main.tex` baseline-pool prose (§"Datasets, Baselines, and Evaluation Metrics"): now
  names all 8 (previously named only 6 of 8, omitted SIEVE/FIFO-Reinsertion entirely,
  and named 2 non-canonical policies instead).
- `tables/manuscript/table2_policy_roster.csv/.tex`: now exactly these 8 (previously 6
  of 8, missing SIEVE and FIFO-Reinsertion — same gap independently confirmed in
  `reports/kbs_fifo_reinsertion_baseline_audit.md` §7).
- **Diagnostic-only/excluded**: `blind_oracle` (the raw `BlindOraclePolicy`, distinct
  from `blind_oracle_lru_combiner`) does not appear in any of the above, and a Table 2
  caption clause now says so explicitly. No manuscript text anywhere claimed
  `blind_oracle` as a tested baseline before this pass either — this was already
  correct, only now made explicit rather than implicit.
- **Removed from Table 2 / baseline-pool prose, retained in repository-description
  prose as explicitly non-canonical**: "Marker" (`src/lafc/policies/marker.py`) and
  "R-FTP+Marker" (`src/lafc/policies/robust_ftp_marker_combiner.py`, an ICML'21
  "RobustFtP" implementation per its module docstring). Both are real, working code in
  this repo but are absent from the canonical `POLICIES` dict in
  `scripts/run_policy_comparison_wulver_v1.py` (verified directly against the source
  this pass — `marker` is present only in the broader `POLICY_REGISTRY` in
  `src/lafc/runner/run_policy.py`, used by the single-policy CLI, not the multi-policy
  comparison driver), so neither has ever been run in any cap32/64 chunk. `main.tex`
  now says this explicitly rather than listing them as if they were active baselines.
- **No final online-results table was created.** This pass only reconciled the
  *roster* (which policies are named, and as what), not any results table.

## 6. Sections deliberately left untouched (cap64/final results still pending)

- `main.tex`'s entire `\section{Conclusions and Future Research}` (Summary of
  Findings, Implications, Limitations, Future Research Directions) — not edited at
  all, per Step 4. The two new Discussion subsections (§2 item 8) were inserted
  **before** this section boundary specifically so nothing inside Conclusions was
  touched.
- No final Table 3 / online end-to-end results table exists anywhere in `main.tex`
  (confirmed again this pass via `grep -n "table3\|main_quantitative" main.tex` —
  zero hits) — there is nothing to mark with a TODO here because it was never written,
  consistent with the earlier audit's finding that it is gated on a canonical
  multi-capacity CSV that does not yet exist.
- No sentence in `main.tex` was changed to assert or imply any cap64, cap128, or
  cap256 numeric result, nor any claim of "broad generality of H=4" or "validated
  fallback" beyond what is explicitly hedged above.
- `tables/manuscript/table3_main_quantitative_comparison.tex`,
  `table4_main_ablation.tex`, `table5_offline_selection.tex`,
  `table6_related_work_learned_caching.tex`, and all `figures/manuscript/*` — not
  opened or modified this pass (out of scope for "policy roster"/"fallback"/
  "citations"/"overhead-and-horizon-skeleton," which is what was authorized).
- `manuscript_source/cover-letter.tex` and `manuscript_source/author-agreement.tex` —
  extracted alongside `main.tex`/`refs.bib` but not edited; out of scope for this pass.

## 7. Whether compile/checks were run

- `git diff --stat` — run; output recorded in the final summary below.
- `grep -Rni "SIEVE\|FIFO\|HALP\|fallback\|guard" tables/manuscript/... reports/manuscript_artifacts/latex_snippets/table2_snippet.tex` — run; confirmed the regenerated roster files contain the new SIEVE/FIFO-Reinsertion/REST rows.
- **LaTeX compile: run, and it succeeded.** `tectonic` (already installed at
  `~/.local/bin/tectonic`, invoked via the existing `latexmk` shim or directly) was
  used to compile `manuscript_source/main.tex` end-to-end into a scratch output
  directory (`/tmp/kbs_manuscript_compile_check*`, cleaned up afterward — not a repo
  artifact). Result: exit code 0, `main.pdf` produced (483,719 bytes), and a
  second verification pass with `--keep-logs` confirmed **zero** "undefined" citation
  or cross-reference warnings in the final `main.log` (the `zhang2024sieve` entry
  resolved correctly through `main.bbl`). Only pre-existing, harmless warnings remained
  (Underfull/Overfull hbox formatting warnings, and one unrelated UTF-8 byte warning in
  `algorithm.sty` itself, not in any manuscript text touched this pass). `pdflatex` and
  `bibtex` are not separately installed on this machine; `tectonic` is a self-contained
  modern TeX engine that handles bibliography resolution internally, so it served as
  the "known fast LaTeX compile command" referenced by Step 5.

## 8. Risks/TODOs

- **TODO**: if a future pass wants the broader S3-FIFO taxonomy citation
  (`yang2023s3fifo`), it needs its own zero-compute source-verification pass (fetch and
  read the actual paper, confirm author list/venue/pages) before adding a BibTeX entry,
  the same standard already applied to SIEVE. Not done this pass — flagged, not
  fabricated.
- **TODO** (already marked as LaTeX source comments in `main.tex`, not typeset): revisit
  the new "Overhead and Scalability" subsection once a controlled, capacity-isolated
  latency benchmark exists, and revisit the new "Replay Horizon Selection" subsection
  once citibike/metakv validation coverage and any horizon-by-capacity interaction are
  measured (both gaps were already known and documented in
  `reports/kbs_horizon_h4_revision_strategy.md` §2-3 before this pass; this pass only
  added the manuscript-facing TODO markers).
- **Risk, mitigated**: editing a manuscript that exists only as an extracted zip copy
  creates two copies of "the manuscript" (the original zip, untouched, and
  `manuscript_source/`, now the live edited copy). This is intentional and was the only
  way to satisfy both "do not overwrite artifacts" and "prepare manuscript-source
  edits" simultaneously, but a human collaborator should be told explicitly (this
  report does so) that `manuscript_source/` — not the zip — is now the current working
  copy, to avoid anyone re-extracting the stale zip over it later.
- **Risk, mitigated**: the LaTeX compile check is a strong build-correctness signal
  (no missing citations, no broken table syntax) but is not a substitute for human
  proofreading of the new prose for tone/flow: a human pass should still re-read the
  edited paragraphs before submission.
- **Not a risk requiring action**: `cap64_with_sieve_fifo` was confirmed still running
  and untouched at both the start (Step 1) and end (this section) of this pass; no
  cap128/cap256 job was launched; nothing was pushed or merged.
