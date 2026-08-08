# Table 6 (Related Work) — explicit uncertainty / verification flags

This note accompanies `tables/manuscript/table6_related_work_learned_caching.tex`. It is **not** a result artifact.

## Rows requiring primary-source verification before sharpening claims

| Row | Why |
|-----|-----|
| **HALP** | Internal repo notes flag HALP as a **major** neighbor for preference/heuristic-aided eviction learning. The table uses cautious language (“preference learning vs.\ miss regression”); **verify HALP’s exact supervision object and label construction** in the NSDI paper before any stronger contrast. |
| **MUSTACHE** | Abstract emphasizes **multi-step-ahead page request forecasting** to approximate Belady. Our v1 labels are **simulated miss counts** after hypothetical evictions on enumerated residents (see `docs/evict_value_v1_method_spec.md`). The distinction is conceptual; **do not over-interpret** without reading MUSTACHE’s training/inference interface in full. |
| **PARROT** | Summarized as imitation/oracle-style supervision at a high level. PARROT’s full pipeline (features, losses, deployment constraints) should be checked if the text makes **fine-grained** comparisons beyond positioning. |
| **Mockingjay / Raven / LRB** | One-line summaries omit architecture details (e.g., MDN reuse modeling, relaxed Belady boundary metrics). Expand only with paper-backed sentences. |

## BibTeX / citation hygiene

- Keys are defined in `refs/related_work_table6.bib` and (for systems papers) were aligned with `docs/internal_bibliography_gap_report.md` where possible.
- A few entries carry `note = {Verify ...}` for page/series details—re-export from DBLP/USENIX/PMLR for camera-ready.

## Repository positioning (unchanged)

This table supports the repo’s conservative novelty boundary: **candidate-level finite-horizon eviction-value supervision** and replay evaluation, **not** a claim that all prior learned eviction methods are superseded.
