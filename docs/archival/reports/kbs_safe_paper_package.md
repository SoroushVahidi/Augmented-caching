# Best paper-artifact package **safe right now** (repo snapshot)

**Evidence gate:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` is **missing** here → **no** evidence-backed main quantitative table or main performance figures from the builder.

**Generator:** `scripts/paper/build_kbs_main_manuscript_artifacts.py` (+ `scripts/paper/regenerate_evidence_aligned_manuscript_figures.py` for Fig. 1 & 4 only).

---

## 1. Manuscript-safe **now** (canonical offline / schematic)

| Artifact | Files | Role | Caption / evidence-scope wording (repo-faithful) |
|----------|-------|------|---------------------------------------------------|
| **Fig. 1** — Method overview | `figures/manuscript/figure1_method_overview.{pdf,png}`, `latex_snippets/figure1_snippet.tex` | Conceptual pipeline | State that the figure is **schematic**; **do not** imply specific replay rankings. Snippet already describes offline vs. online panels. |
| **Table 1** — Trace roster | `tables/manuscript/table1_dataset_summary.{csv,tex}` | Lists manifest traces / capacities for heavy\_r1 **offline construction** | Use wording that **Main**/roster alignment with end-to-end eval is **pending** until `policy_comparison_heavy_r1.csv` exists (see current `table1_snippet.tex` caveat). |
| **Table 2** — Policy roster | `tables/manuscript/table2_policy_roster.{csv,tex}` | Intended comparison **design** | “Policies **included in the empirical comparison design**; numerical results require the canonical eval CSV.” |
| **Table 4** — Offline ablation | `tables/manuscript/table4_main_ablation.{csv,tex}` | From `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` | “**Offline training metrics** on heavy\_r1 shards; **lower is better** for regret and Top-1 as defined in training script; **not** end-to-end replay misses.” |
| **Fig. 4** — Offline regret | `figures/manuscript/figure4_ablation.{pdf,png}` | Same source as Table 4 | “Validation/test **mean regret vs. oracle** by horizon; **does not** depict policy-level replay performance.” |

---

## 2. Blocked on `policy_comparison_heavy_r1.csv`

| Artifact | Files | When safe |
|----------|-------|-----------|
| **Table 3** | `tables/manuscript/table3_main_quantitative_comparison.{csv,tex}` | Only after CSV exists and builder run; stub otherwise (`tab:main-comparison-unavailable`). |
| **Fig. 2** | `figures/manuscript/figure2_main_performance_comparison.{pdf,png}` | Same; snippet is commented template until built. |
| **Fig. 3** | `figures/manuscript/figure3_improvement_vs_lru.{pdf,png}` | Same. |

**Exact scope when present:** Builder filters to **`TABLE3_POLICIES`** (six policies); eval CSV may contain extra policies (e.g. `blind_oracle`)—figures/tables **ignore** extras.

---

## 3. Offline-only supplements — **temporary** when eval is missing

Emitted **only when** policy CSV is **absent**; **removed** when CSV appears (placeholders replace snippets—see builder).

| Artifact | Files | Wording discipline |
|----------|-------|-------------------|
| **Table 5** | `tables/manuscript/table5_offline_selection.{csv,tex}` | Single row from `best_config_heavy_r1.json` + `model_comparison_heavy_r1.csv`. Label explicitly **offline training only**; **not** replay misses. |
| **Fig. 5** | `figures/manuscript/figure5_offline_top1_ablation.{pdf,png}` | Top-1 error vs. horizon; **offline** only. |

**Recommendation:** Treat as **supplement / pre-eval bundle**; **drop or demote** once Table 3 / Fig. 2–3 are available so the paper centers on end-to-end evidence.

---

## 4. Cross-check list before submission

- [ ] `policy_comparison_heavy_r1.csv` present → rebuild bundle; verify Table 3 / Fig. 2–3 and **remove** reliance on Table 5 / Fig. 5 for main claims.
- [ ] If CSV still missing → **do not** claim multi-trace replay results; lean on Table 4 / Fig. 4 / Fig. 1 + Table 2 + caveated Table 1.
- [ ] Never cite unsuffixed `analysis/evict_value_wulver_v1_policy_comparison.csv` as KBS main line (`docs/wulver_heavy_evict_value_experiment.md`).

---

*See also: `reports/manuscript_evidence_audit.md`, `reports/manuscript_artifacts/kbs_evidence_alignment_report.md`, `docs/evict_value_v1_method_spec.md`, `docs/kbs_knowledge_framing_note.md`.*
