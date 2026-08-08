# Reference usage note (repository-derived)

## Finding: minimal `.bib` for Related Work Table~6

The repository now includes **`refs/related_work_table6.bib`** (keys cited by `tables/manuscript/table6_related_work_learned_caching.tex`). There is still **no** single authoritative project-wide `.bib` for the full paper; merge or extend as needed for camera-ready.

**Practical substitute for “refs (1).bib”:** canonical **paper-to-code** citations live in **`docs/baselines.md`** (and scattered mentions in `docs/framework.md`, `README.md`). The groups below are derived from those docs plus which policies appear in the **KBS manuscript bundle** (`TABLE3_POLICIES` in `scripts/paper/build_kbs_main_manuscript_artifacts.py`).

---

## Group A — **Must-use** for the current KBS manuscript (evict_value_v1 + Wulver heavy_r1 story)

These ground **baseline identity** and LA-caching context when you discuss competitors that the eval actually runs (per `slurm/evict_value_v1_wulver_heavy_eval.sbatch` and Table 2/3 design).

| Reference (as in `docs/baselines.md`) | Exact role in the paper |
|---------------------------------------|-------------------------|
| **Antoniadis et al., ICML 2020** — *Online Metric Algorithms with Untrusted Predictions* (TRUST&DOUBT) | Define **`trust_and_doubt`** baseline: predicted-cache interface, trust/doubt mechanism; cite when naming T&D in Table 2/3. |
| **Wei, APPROX/RANDOM 2020** — *Better and Simpler Learning-Augmented Online Caching* | Define **`blind_oracle_lru_combiner`** (BO/LRU combiner / FTP-style LA baseline). |
| **Bansal et al., SODA 2022** — *Learning-Augmented Weighted Paging* | Optional **related-work** anchor for LA paging line (your method is not `la_det`, but readers expect LA context). |
| **LRU / Marker literature** (standard) | **`lru`** and **`predictive_marker`** — use your venue’s standard refs for classic paging and marking; repo implements them in `lafc/policies/` (see `baselines.md` for marker-related notes). |

**Repo note:** `rest_v1` is an **in-repo heuristic** (`lafc/policies/rest_v1.py`); cite **as implemented here** or add a short system description if no external paper is claimed.

---

## Group B — **Optional background** (cite **only** if the text explicitly discusses them)

| Reference / topic | When to cite |
|-------------------|--------------|
| **Pairwise / ranking line** (`docs/pairwise_*.md`, `analysis/pairwise_*`) | Only if introduction or appendix discusses exploratory pairwise work — **not** canonical KBS `heavy_r1` main numbers. |
| **Offline Belady / LP general caching** (`docs/offline_belady.md`, `docs/offline_general_caching_approx.md`) | Only if discussion touches offline optimum or general-cost caching (separate entry points in repo). |
| **Sentinel / guard designs** (`docs/sentinel_*`, `lafc/policies/guard_wrapper.py`) | Only if you discuss **guard** behavior; **no** canonical heavy_r1 artifact in manuscript builder for guarded metrics. |
| **Manuscript evidence map / TIST** (`docs/manuscript_evidence_map.md`) | Meta-repo claims, not user-facing KBS paper unless you frame as prior internal report. |

---

## Group C — **Still missing** for *stronger* KBS-style framing (add to `.bib` when you create one)

These are **not** supplied as bib entries in-repo; authors typically add them for a polished KBS submission:

| Gap | Why it helps |
|-----|----------------|
| **Knowledge-based / decision-support systems** survey or exemplar | Positions “explicit knowledge + learned decision rule” narrative for a KBS audience (`docs/kbs_knowledge_framing_note.md`). |
| **Caching / CDN / key-value trace literature** for **datasets** | Grounds Wulver-style traces (BrightKite, Wiki, etc.) as real domains — use dataset docs `docs/datasets*.md` to pick appropriate citations. |
| **Recent learned-cache / ML-for-systems surveys** (if allowed) | Situates `evict_value_v1` among learned policies (careful not to overclaim vs. your evidence). |
| **Statistical reporting** (confidence intervals, multiple comparisons) | If you add **new** experiments beyond what scripts output; not in current artifact bundle. |

---

## Action items

1. **Add a root-level or `paper/refs.bib`** when the LaTeX manuscript exists; seed Group A from `docs/baselines.md` biblio lines.
2. **Keep Group B** out of the main story unless explicitly needed.
3. **Fill Group C** from your reading list — the repo cannot invent these keys.
