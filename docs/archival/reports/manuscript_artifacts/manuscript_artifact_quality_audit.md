# Manuscript artifact quality audit (figures, tables, LaTeX snippets)

**Scope:** Repository-only review of `figures/manuscript/`, `tables/manuscript/`, `reports/manuscript_artifacts/latex_snippets/`, and generation logic in `scripts/paper/build_kbs_main_manuscript_artifacts.py` (+ related helpers). **No** external `main.tex` was inspected.

**Actions taken (conservative):** caption and snippet wording polished for **journal-facing tone** (fewer raw filenames / repo paths in LaTeX captions), `\input` path for Related Work table clarified, Related Work intro wording softened, guard-wrapper caption de-internalized. **No** numeric experiment outputs were edited. Snippets tied to the main builder were refreshed by running `python scripts/paper/build_kbs_main_manuscript_artifacts.py` after code changes.

---

## 1. What was already good

| Artifact | Notes |
|----------|--------|
| **Figure 4 / Table 4 (offline ablation)** | Clear offline-only framing, consistent panels, booktabs tables, aligns with evidence map. |
| **Table 2 (policy roster)** | Compact, readable abbreviations, appropriate for Methods. |
| **Figure 2–3 snippets (stub)** | Comment-only stubs when policy CSV absent—honest gatekeeping. |
| **Matplotlib style** | `apply_manuscript_matplotlib_style()` (vector PDF, type 42 fonts) is consistent across generated figures. |
| **Table 6 (Related Work) + `refs/related_work_table6.bib`** | Appropriate caution language; kept as author-maintained asset. |

---

## 2. What was changed in this pass

| Item | Change |
|------|--------|
| **`build_kbs_main_manuscript_artifacts.py`** | Figure~1 caption: removed “canonical bundle here”; states optional guard vs.\ primary unguarded results. Fig.~4 / Fig.~5 / Table~5 captions: dropped long in-caption CSV/JSON filenames; refer to “heavy\_r1 training-run comparison” / “same offline experiment”. Table~1 caveat: “Main” column wording without embedding full CSV path in caption. Table~3 unavailable caption: removed pointer to `docs/...`; instructs regenerate after replay eval. Table~4 snippet: selection sentence uses “offline configuration” + rule, not `best_config_*.json` path. Offline-supplement placeholders: less “policy CSV” jargon. |
| **`table6_related_work_float.tex`** | `\input` path standardized to `tables/manuscript/...` with comment that LaTeX root should be repo root (matches other snippets). |
| **`related_work_table6_intro.tex`** | Replaced “repository's v1 label construction” with “v1 label construction used here”. |
| **`build_guard_wrapper_manuscript_figure.py`** | Caption: removed `\path{src/...}`; clarified shadow stepping in neutral language; re-ran script to refresh PDF/PNG/snippet. |
| **Regenerated snippets** | Ran main builder so `reports/manuscript_artifacts/latex_snippets/{figure1,figure4,table1,table3,table4,table5,...}.tex` match updated strings. |

---

## 3. Per-artifact verdict (post-pass)

| Artifact | Verdict | Notes |
|----------|---------|--------|
| **Fig.~1 method overview** | **Light polish (done)** | Panel text is still schematic (not metric); caption now manuscript-facing. |
| **Fig.~4 offline ablation** | **Light polish (done)** | Caption no longer cites raw CSV name. |
| **Fig.~5 offline Top-1** | **Light polish (done)** | Same as Fig.~4 when emitted; omitted placeholder comment improved when policy CSV present. |
| **Fig.~6 guard wrapper** | **Light polish (done)** | Single-column width; dense text—acceptable for control-flow; caption cleaned. |
| **Fig.~6 regret vs Top-1 / Fig.~7 continuation / Fig.~8 target** (supplemental) | **Good / separate track** | Produced by `build_additional_eviction_value_figures.py`; **no** matching `latex_snippets` in this folder—integration left to authors (not wrong, just incomplete packaging). |
| **Table~1 dataset summary** | **Good with caveat** | “Main” column depends on eval artifact—caption now states that generically. |
| **Table~2 roster** | **Good as-is** | — |
| **Table~3 (unavailable stub)** | **Real limitation** | Cannot be “pretty” until policy CSV exists; body still mentions canonical filename in **tabular** (intentional warning for authors). |
| **Table~4 ablation** | **Good as-is** | Numeric body unchanged. |
| **Table~5 / Fig.~5 when policy present** | **N/A (omitted)** | Placeholder comments only—correct behavior. |
| **Table~6 Related Work** | **Good; author-verify** | Bib keys and row summaries require primary-source checks (already noted in `table6_related_work_uncertainties.md`). |

---

## 4. What remains weak (not fully fixable in-repo)

1. **Main quantitative story (Table~3 / Fig.~2–3)** until `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` exists—artifacts are **stub/hidden by design**.  
2. **Figure numbering collision:** two different stems use the “figure6” prefix (`figure6_guard_wrapper_evict_value_v1` vs `figure6_regret_vs_top1_alignment`); authors must **renumber** in LaTeX.  
3. **Supplemental figures 6–8 (alignment / continuation / target)** lack ready-made `latex_snippets`—authors must wire paths manually or extend the builder.

---

## 5. Artifacts to **avoid** or use only with care in the journal paper

| Artifact | Guidance |
|----------|----------|
| **`table3_main_quantitative_comparison` stub body** | **Do not** present as results—status table for authors only. |
| **Offline-only Fig.~5 / Tab.~5** when end-to-end results exist | Builder **omits** them; if an old draft still includes them alongside Table~3, **drop** offline duplicates to avoid double narrative. |
| **`figure6_regret_vs_top1_alignment`**, **`figure7_*`**, **`figure8_*`** | Fine for appendix/supplement **if** narrative matches; not part of canonical `heavy_r1` quantitative bundle—caption discipline required. |

---

## Top 3 manuscript-support items that still most need improvement

1. **End-to-end quantitative bundle (Table~3, Fig.~2–3)** — blocked on completing canonical eval; no amount of LaTeX polish substitutes.  
2. **Packaging for supplemental figures (regret/Top-1 alignment, continuation, target concept)** — add optional `latex_snippets` or a short `docs/` note with recommended `\includegraphics` widths and placement.  
3. **Related Work Table~6** — content quality depends on **bibliography verification** and tightening row wording against primary papers (`table6_related_work_uncertainties.md`).
