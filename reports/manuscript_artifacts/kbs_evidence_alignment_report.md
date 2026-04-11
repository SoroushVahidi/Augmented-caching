# KBS manuscript evidence alignment (repository snapshot)

This report was generated to pair with `scripts/paper/build_kbs_main_manuscript_artifacts.py` and the canonical paths in `docs/evict_value_v1_kbs_canonical_artifacts.md` / `docs/kbs_manuscript_workflow.md`.

## 1. Canonical policy-comparison files

| File | Role |
|------|------|
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` | Primary multi-trace policy comparison for KBS main quantitative claims |
| `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.md` | Human-readable companion (optional; CSV is source of truth per workflow) |

**Repository status (verify after each pull):** if these files are absent, **Table 3** and **Figures 2–3** cannot be evidence-backed from the canonical heavy eval. The builder emits an explicit unavailable stub for Table 3 and does not build Fig. 2–3.

**Consistency:** Paths match `EVIDENCE_FILES["policy_comparison"]` in `build_kbs_main_manuscript_artifacts.py` and the tables in `docs/kbs_manuscript_workflow.md` / `docs/evict_value_v1_kbs_canonical_artifacts.md`.

---

## 2. Manuscript claims that could overstate available evidence (and mitigations)

| Location / topic | Risk | Mitigation applied or recommended |
|------------------|------|-------------------------------------|
| **Table 1 (dataset summary)** | Implies a completed or “locked” main eval roster | Caption softened to “offline dataset construction”; **Main** column caveated until policy CSV exists; capacities from dataset spec when policy CSV absent |
| **Figures 2–3 / Table 3** | Strong quantitative story without `policy_comparison_heavy_r1.csv` | Snippets for Fig. 2–3 are only written when the CSV exists; otherwise stubs / commented templates |
| **Offline ablation (Fig. 4, Table 4)** | Confused with end-to-end replay | Captions state offline training metrics; Fig. 4 / Table 4 do not claim policy-level misses without policy CSV |
| **New Fig. 5 / Table 5** (when policy CSV **absent**) | Could be read as main results | Labels and captions explicitly say **offline training only** and **not** end-to-end replay |
| **Eval driver vs. manuscript table** | Policy CSV from `heavy_eval.sbatch` can include `blind_oracle`; Table 3 uses a fixed six-policy subset | Manuscript figures 2–3 filter to the same subset as Table 3 (`MAIN_PERF_POLICIES`); extra CSV rows are ignored in the bundle |

---

## 3. When `policy_comparison_heavy_r1.csv` **exists**

The builder regenerates **Table 3** and **Figures 2–3** using **only** the manuscript policy subset (LRU, PredMk, T\&D, BO/LRU, REST, EV). Captions reference Table 3 for consistency.

**Offline-only supplements (Table 5 / Fig. 5)** are **not** emitted as primary assets: disk files are removed and LaTeX placeholders explain omission so the paper does not duplicate narratives.

---

## 4. When `policy_comparison_heavy_r1.csv` **does not** exist

The builder adds **artifact-backed offline supplements** (training evidence only):

- **Table 5** (`tables/manuscript/table5_offline_selection.*`): single row from `evict_value_wulver_v1_best_config_heavy_r1.json` joined with `evict_value_wulver_v1_model_comparison_heavy_r1.csv`.
- **Figure 5** (`figures/manuscript/figure5_offline_top1_ablation.*`): Top-1 error vs. horizon (validation / test panels), same CSV as the regret ablation.

These **do not** replace end-to-end results; captions state they are offline diagnostics.

---

## 5. Safe vs. pending vs. priority next steps

| Artifact | Safe without policy CSV? | Needs heavy eval? |
|----------|-------------------------|-------------------|
| Fig. 1 method overview | Yes | No |
| Table 1 (with caveats) | Yes | Partial: **Main** / capacities best confirmed after eval |
| Table 2 policy roster | Yes | No |
| Table 3 main comparison | **No** (stub only) | **Yes** |
| Fig. 2–3 | **No** (not built) | **Yes** |
| Table 4 / Fig. 4 offline ablation | Yes | No (training CSV) |
| Table 5 / Fig. 5 | Yes (offline-only narrative) | No |

**Highest-priority experiment for the main quantitative story:** complete **`slurm/evict_value_v1_wulver_heavy_eval.sbatch`** (after `heavy_train` and `models/evict_value_wulver_v1_best_heavy_r1.pkl` exist) so `policy_comparison_heavy_r1.csv` is produced; then rerun `build_kbs_main_manuscript_artifacts.py`.

**Secondary (not blocking main numbers):** optional `policy_comparison_heavy_r1.md` from the same eval; guarded/fallback diagnostics remain non-canonical unless a designated `*_heavy_r1` artifact exists (see builder notes).

---

## Related deliverables (repo-only audits)

- `reports/manuscript_evidence_audit.md` — claim vs. artifact table with defensible wording.
- `docs/evict_value_v1_method_spec.md` — full method specification from code/docs.
- `docs/kbs_knowledge_framing_note.md` — KBS “knowledge” framing without overclaiming.
- `reports/kbs_safe_paper_package.md` — which figures/tables are safe now and caption discipline.
