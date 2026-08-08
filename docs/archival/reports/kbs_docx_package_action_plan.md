# DOCX submission package — action plan (2026-06-19)

Builds directly on the read-only inspection in
`reports/kbs_docx_submission_package_report.md` (confirmed again this pass:
zero `.docx` files exist anywhere in the repo; the only real submission
package is the LaTeX zip). This pass adds two things that report didn't
have: (1) `submission_kbs_revision_docx/` now exists as a tracked
placeholder directory with a README, and (2) a direct, reproducible test
of whether `pandoc` (confirmed installed, v3.9.0.2, at
`/home/soroush/bin/pandoc`) can actually convert this manuscript's LaTeX
source to DOCX with acceptable fidelity. Zero-heavy-compute pass, done in
parallel with the running `kbs_full_policy_comparison_cap32_with_sieve`
tmux job (PID 191758, reconfirmed running at the time of writing — not
touched). The test conversion itself was run entirely in `/tmp` scratch
space, never inside the repo.

## 1. Required DOCX files

Per the task's exact list (also now recorded verbatim in
`submission_kbs_revision_docx/README.md`):

1. Revised manuscript DOCX
2. Response to reviewers DOCX
3. Revised cover letter DOCX
4. Highlights DOCX
5. CRediT author statement DOCX
6. Declaration of interest DOCX
7. Author agreement DOCX/Elsevier form

All seven are currently **missing** — zero DOCX files exist anywhere in
this repository (reconfirmed by `find . -iname "*.docx"`, no results
outside this pass's own `/tmp` test artifacts, which were never copied
into the repo).

## 2. Which can be drafted now vs. which must wait

| # | File | Can draft now? | Why / why not |
|---|---|---|---|
| 1 | Revised manuscript DOCX | **No** | The underlying manuscript content itself is not yet revision-ready: no online-replay/Table 3 results for the final capacity sweep, SIEVE/FIFO-Reinsertion baseline gaps (closing, see §3 below), and several sections (fallback mechanism framing, H=4 justification, overhead text, shortening pass) have drafted replacement text in this pass's other reports but have **not been applied to `main.tex`** yet. Converting now would produce a DOCX of a manuscript that will need to change again once cap64+ finish — a second stale artifact, not a time-saver. |
| 2 | Response to reviewers DOCX | **Partially** | `reports/kbs_response_to_reviewers_skeleton.md` exists and now has substantially more drafted content after this pass (R2-MC1/H=4, R2-MC2/overhead, R3-Issue6/fallback, R3-Issue3/FIFO-Reinsertion all have draft paragraphs in dedicated reports — see Task #39 update). The full letter cannot be finalized until the final empirical results (cap64+) are known and a small number of `[IN PROGRESS]` placeholders are resolved, but a near-complete draft skeleton in DOCX form is achievable today if a human wants an early look. |
| 3 | Revised cover letter DOCX | **Yes, mostly** | `cover-letter.tex` exists but is written for an *initial* submission ("Please consider the enclosed manuscript... for publication"). Rewriting it for a revision (standard boilerplate: manuscript number, summary of changes, thanks to reviewers) does not depend on final results — only on the response-to-reviewers letter's summary points, which can be referenced in outline form even before every numeric result is final. |
| 4 | Highlights DOCX | **Partially** | Drafting candidate bullet points doesn't require final results (the framework/method-level highlights are stable), but bullets that claim a specific empirical outcome (e.g., "X% miss reduction") must wait for the final canonical sweep. Recommend drafting the method-level highlights now and leaving 1-2 result-dependent bullets as placeholders. |
| 5 | CRediT author statement DOCX | **Yes** | Single-author paper (per the existing `author-agreement.tex`/manuscript metadata) — this is a mechanical, fixed-format statement with no dependency on results. |
| 6 | Declaration of interest DOCX | **Yes** | An inline "Declaration of Competing Interest" section already exists in `main.tex`; extracting it into a standalone DOCX file is purely mechanical, no result dependency. |
| 7 | Author agreement DOCX/Elsevier form | **Yes** | `author-agreement.tex` already exists; converting/transcribing it into Elsevier's actual form (if a specific DOCX/PDF template is required by Editorial Manager) is mechanical, no result dependency. |

**Bottom line**: items 5, 6, 7 can be produced today with no blockers.
Item 3 is nearly unblocked (only needs the response letter's summary
points in outline form). Items 2 and 4 are partially blocked (drafts
possible now, finalization needs cap64+ results). Item 1 — the one that
matters most — is fully blocked until the canonical sweep finishes and the
drafted manuscript-text edits from this pass's other reports
(`kbs_fallback_revision_strategy.md`, `kbs_horizon_h4_revision_strategy.md`,
`kbs_overhead_manuscript_text_draft.md`,
`kbs_manuscript_shortening_and_reframing_plan.md`) are actually applied to
`main.tex`.

## 3. Does a LaTeX-to-DOCX conversion workflow exist?

**Technically yes, but it is not reliable for this manuscript without
substantial manual repair.** `pandoc` is installed on this machine
(v3.9.0.2) and successfully converts `main.tex` to `.docx` with exit code 0
(some non-fatal warnings). A direct test was run to check fidelity rather
than assume it:

```bash
cd /tmp/kbs_manuscript_inspect
pandoc main.tex -o main_test_conversion.docx --bibliography=refs.bib
pandoc main_test_conversion.docx -t plain -o main_test_conversion.txt
```

(All files kept in `/tmp/kbs_manuscript_inspect/` only — never copied into
the repo; this was a fidelity probe, not a deliverable.)

Converting the resulting `.docx` back to plain text and comparing against
the original `main.tex` source surfaced four concrete, reproducible fidelity
problems:

1. **Equations containing `\textsc` inside math mode fail to convert**
   (e.g. `\mathcal{M}_t \in \{\textsc{base}, \textsc{fallback}\}`). These
   render broken or are dropped in the output.
2. **Matrix/`cases`/`bmatrix` display environments fail** and render as
   broken raw TeX text rather than formatted equations.
3. **The Algorithm 2 pseudocode block (`algorithmicx`/`algorithm`
   environment) is dropped entirely.** Only narrative sentences that
   *mention* "Algorithm 2" survive in the converted text — none of the
   actual pseudocode content (steps, conditionals, the guard logic) is
   present anywhere in the output.
4. **Cross-references (`\ref{...}`) do not resolve** — they render as
   broken literal text instead of resolving to the referenced number.
5. **Tables silently do not survive.** `main.tex` contains 8
   `\begin{tabular}`/`\begin{table}` environments. Grepping the converted
   plain text for expected table content/keywords (`LRU`, `booktabs`,
   `tabular`) returns 8 hits — but every one of them is a narrative
   sentence ("Table 2 summarizes the workload families relevant to the
   present study," "Table 3 summarizes...," "Table 4 and Fig. 2
   summarize...") that merely *mentions* a table, never the table's actual
   data (no policy names, capacity values, or miss counts appear anywhere
   in the converted output). This indicates pandoc silently dropped all 8
   tables' content rather than converting it — and did so without raising
   an error or warning that would flag this loss to a user running the
   conversion.

## 4. Do Word equations/tables/figures need manual inspection?

**Yes, unambiguously, confirmed by direct test rather than assumption.**
Every one of the four content categories most load-bearing for a technical
manuscript — equations, the algorithm pseudocode block, cross-references,
and tables — either breaks visibly or (worse, for tables) is silently
dropped with no warning. A one-shot `pandoc main.tex -o manuscript.docx`
run is **not sufficient** to produce a submittable manuscript DOCX. The
practical workflow, if/when the manuscript DOCX is eventually produced,
must be:

1. Run pandoc to get a structural starting point (headings, section order,
   prose, citations — these convert reasonably).
2. Manually re-insert or rebuild every equation containing `\textsc` or a
   matrix/cases environment as a native Word equation object.
3. Manually re-create Algorithm 2 as a Word table or formatted
   monospace block, since none of its content survives conversion.
4. Manually fix every cross-reference (`\ref{...}` → resolved figure/table/
   equation number, or Word's native cross-reference field).
5. Manually re-insert all 8 tables' data as native Word tables — pandoc's
   silent table-content loss is the single highest-risk failure mode found
   in this test, since it produces no error to alert a user that data is
   missing.
6. Manually verify figures are present and correctly captioned (not
   specifically tested this pass, lower risk than tables/equations since
   figures are external image files pandoc typically references directly,
   but not verified — flagged as an open check for whoever performs the
   real conversion).

## 5. Recommendation

- Do not treat any future pandoc-based conversion as a finished artifact.
  Budget real manual-repair time (equations, algorithm block,
  cross-references, all 8 tables) for whoever performs the eventual
  conversion — this is not a one-shot automated step for this manuscript.
- Sequence the manuscript DOCX conversion **after** the canonical sweep
  finishes and the drafted text edits from this pass are applied to
  `main.tex` — converting now would mean repeating all of §4's manual
  repair work a second time once the content changes.
- The five items identified as drafting-now-ready in §2 (CRediT,
  Declaration of Interest, Author Agreement, cover letter, and a
  method-level-only Highlights draft) have no such conversion-fidelity
  risk, since they are short, prose-only, table/equation-free documents —
  pandoc (or even a plain Word rewrite from the existing `.tex`/inline
  text) should be reliable for these.

## 6. What this pass does NOT do

- Does not produce any actual `.docx` file inside the repository. The only
  `.docx` files created during this pass (`main_test_conversion.docx` and
  its companion `.txt`) live in `/tmp/kbs_manuscript_inspect/` only, purely
  as a fidelity probe, and are not part of any deliverable.
- Does not finalize, populate, or convert any of the 7 files now listed in
  `submission_kbs_revision_docx/README.md` — that directory remains an
  explicit placeholder, per the "do not treat this folder as final-ready"
  instruction baked into its README.
- Does not touch `kbs_full_policy_comparison_cap32_with_sieve` or launch
  any new run.
