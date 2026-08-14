# KBS manuscript compression audit — 2026-08-14

- Branch: `kbs/second-revision-science`
- HEAD before this pass: `c13fe4a03b9c2cff834fb0dd3439121f12a72ae4`
- No experiment was run.
- Scientific result files were not modified.

## Page count

- Starting (pre-edit compile of canonical `main.tex`): **50** pages
  - main text through front/back matter ~1–39
  - Appendix A historical ~40–43
  - Appendix B fallback ~44
  - Appendix C features + references start ~45
  - references ~45–50 (~6 pages)
- Final compile: **20** pages (references start p. 16; 5 reference pages)
- Pages saved: **30**
- Target `<=35` achieved: **yes** (20 < 35)

The first-round 30–40% shortening request is exceeded. Further padding toward
33–35 pages would reintroduce the verbosity that Reviewer #3 criticized.
Reviewer-critical tables and numbers remain.

## Appendices removed (from the journal PDF only)

- Historical Appendix A: removed as a multi-page appendix. Compact
  family-level LRU-gap table retained in §3.3 as Table 4 (historical/leaky,
  not primary). Underlying repository evidence untouched.
- Fallback Appendix B: removed. One-sentence unvalidated-guard note remains
  in Method and Limitations.
- Feature-group Appendix C: removed. Six feature groups summarized in §2.2.
  Full specification remains in the repository.

## Figures removed / retained

- Removed: offline-ablation Figure 2 (redundant with Table 3).
- Removed: historical capacity-trend figure (appendix).
- Retained: Figure 1 (method overview); it is the only workflow figure.

## Tables redesigned

- Table 1 (traces): booktabs, `\arraystretch{1.15}`, ragged-right `p{}` cells, `[2pt]` row gaps.
- Table 2 (policies): collapsed Category into group headings; two columns;
  group separators via `\addlinespace`; `\arraystretch{1.18}`; `[2pt]` gaps.
- Tables 3–8: booktabs + local `\arraystretch{1.15}` where not already present;
  shortened captions.

## Major prose compressed

- Abstract (~journal length)
- Introduction (merged problem/scope; shortened related work)
- Method (single `L_H` definition; duplicate argmin/matrix removed; fallback subsection removed)
- Results interpretation (one synthesis subsection)
- Practical significance, limitations (itemized), future work (one paragraph)
- Conclusions no longer restates the full mechanistic story

## Reviewer-critical evidence preserved

- Matched LRB / 3L-Cache / CACHEUS / HALP / LRU / SIEVE / FIFO-Reinsertion (Table 5)
- Pipeline objective comparison (Table 6) and Common-Model V2 (Table 7)
- Exact-target + tie-aware diagnosis (§3.6)
- Continuation C0/C1/C2 and DAgger (§3.7)
- Overhead timing (Table 8) + EV single-run caveat
- H=4 justification (Table 3)
- Workload-specific historical gaps (Table 4)
- Fallback: disclosed as unvalidated, not claimed

## Scientific values verified unchanged

- Common V2: 571,976 / 577,339 / 615,850 / 627,392
- Tie-aware: 0/3/18 +81,750; 16/5/0 −413; 0/3/18 +89,135; 1.0; 0.649; 0.991
- Matched baseline means and win/loss/tie counts unchanged
- Offline ablation numbers unchanged

## Response

- `submission_kbs_revision_final/02_Response_to_Reviewers.md` section/table/page
  pointers updated to the compressed manuscript (20 pages).

## Validation

- `tectonic main.tex`: exit 0
- no `??` in `pdftotext`
- `git diff --check`: clean
- no experiment rerun
