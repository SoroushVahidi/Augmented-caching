# Stale manuscript artifact refresh plan (2026-06-19)

Read-only inspection only. **No manuscript artifacts were regenerated in this
pass** — this is a plan, not an execution, per instruction.

## 1. Root cause (now pinned down precisely)

`git diff HEAD -- analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`
and `git diff HEAD -- analysis/evict_value_wulver_v1_best_config_heavy_r1.json`
confirm exactly what changed and when:

- The last **committed** version of these two files (commit `53726ce`, "Add
  current heavy_r1 training outputs and interim policy comparison artifacts")
  selected **H=4, hist_gb** (val_mean_regret 0.009142857142857144 — the lowest
  of the 9 horizon×model rows in that committed file).
- The **current, uncommitted, working-tree** version (this session's heavy_r1
  retrain, reflecting the twemcache-undercoverage fix referenced in the
  evidence ledger) now selects **H=4, random_forest** (val_mean_regret
  0.02077865266841645, vs. hist_gb's 0.02099737532808399 — a much closer
  margin than before, consistent with the gap tracker's prediction that the
  close hist_gb/random_forest margin "could plausibly flip").
- `tables/manuscript/table4_main_ablation.csv`, `table5_offline_selection.csv`,
  and `figures/manuscript/figure4_ablation.{pdf,png}` all have file mtime
  **2026-06-17 22:11** — i.e., they were built from the **committed
  `53726ce` numbers**, not the current working-tree retrain. `main.tex`
  (inside the manuscript zip, also committed at 22:08, one commit earlier)
  is consistent with that same committed snapshot.

**In short: nothing is mysteriously stale — the manuscript and its tables/
figures are self-consistent with the last commit, but the working tree has
since moved past that commit (uncommitted), and the manuscript artifacts
haven't been regenerated against the new numbers yet.**

## 2. Which tables/figures are stale

| Artifact | Built from | Status |
|---|---|---|
| `tables/manuscript/table4_main_ablation.csv/.tex` | `evict_value_wulver_v1_model_comparison_heavy_r1.csv` | **STALE** — matches committed `53726ce` values, not the current working-tree retrain. |
| `tables/manuscript/table5_offline_selection.csv/.tex` | `model_comparison_heavy_r1.csv` + `best_config_heavy_r1.json` | **STALE** — still shows the old `hist_gb` selection; current working tree selects `random_forest`. |
| `figures/manuscript/figure4_ablation.{pdf,png}` | `model_comparison_heavy_r1.csv` (via `make_offline_ablation_figure`) | **STALE** — same root cause as table4. |
| `figures/manuscript/figure5_offline_top1_ablation.{pdf,png}` | `model_comparison_heavy_r1.csv` + `best_config_heavy_r1.json` | **STALE** — same root cause; only emitted when end-to-end replay results are absent (per the build script's own comment), which is still the current state. |
| `main.tex`'s embedded `tab:evict-value-ablation` table (inside the zip) | hand-copied from the same committed numbers | **STALE** — same numbers as table4, same root cause. |

## 3. Which current analysis outputs supersede them

- `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` (working
  tree, modified, not committed) — supersedes the committed version used by
  all artifacts above.
- `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` (working tree,
  modified, not committed) — now says `"model": "random_forest"` instead of
  `"hist_gb"`.
- `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` (working tree,
  modified) — full per-horizon/per-model/per-family metrics behind the
  summary CSV above; also supersedes the committed version.

## 4. Which tables/figures are NOT affected by this particular staleness

- `table1_dataset_summary.*`, `table2_policy_roster.*` — descriptive
  (dataset/policy roster), not derived from model-selection metrics; not
  implicated by the hist_gb→random_forest flip. (`table1` carries its own
  **separate, pre-existing** caveat — `table1_capacities_from_dataset_
  summary_md_not_verified_against_eval_csv` — about capacities not yet
  verified against the canonical policy-comparison CSV, which is a different,
  already-known issue, not new.)
- `table6_related_work_learned_caching.tex` — related-work table, unrelated
  to training metrics.
- `figure1_method_overview.*` — method schematic, no numbers from training.

## 5. Exact artifact-builder script to run later (not run in this pass)

```bash
cd /home/soroush/Augmented-caching
source .venv_kbs_heavy_r1/bin/activate
python scripts/paper/build_kbs_main_manuscript_artifacts.py
```

- No CLI arguments — the script reads fixed paths (`EVIDENCE_FILES` dict at
  the top of `scripts/paper/build_kbs_main_manuscript_artifacts.py`),
  including `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`
  and `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` directly
  from the working tree, so re-running it now would naturally pick up the
  current (uncommitted) retrain numbers.
- **Why this looks lightweight/safe based on code inspection** (not executed
  to confirm): it only reads existing small CSV/JSON files and renders a
  handful of matplotlib figures — no trace replay, no model training, no
  multi-hour computation. The script already has guard logic for the missing
  canonical policy-comparison CSV (confirmed by the *current* stub outputs:
  `table3_main_quantitative_comparison.csv` reads `NOT_VERIFIED`, and
  `manuscript_artifact_manifest.json` has `policy_comparison_present: false`)
  — re-running it would refresh table4/table5/figure4/figure5 while
  correctly continuing to leave Table 3 and Figs. 2–3 as not-yet-verified
  stubs, since the canonical CSV is still absent.
- **Recommendation, not an action taken here**: this looks safe to run
  whenever a refresh is wanted, but per instruction this audit stops at
  the plan — it has not been executed.
- After running it, `main.tex`'s `tab:evict-value-ablation` table would
  still need a **manual** edit to copy the refreshed numbers in (the build
  script does not edit the manuscript zip's `main.tex` directly — it only
  produces `tables/manuscript/*.tex` snippets and `reports/manuscript_
  artifacts/latex_snippets/table4_snippet.tex` for `\input{}`-style
  inclusion, which `main.tex` does not currently use — it has the table
  hand-copied inline instead).

## 6. What should wait until the full canonical CSV exists

These are a **separate, larger** blocker, not fixed by the refresh above:

- `tables/manuscript/table3_main_quantitative_comparison.*` (online-replay
  main comparison table) — currently a `NOT_VERIFIED` stub; needs
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` (the merged
  cap32+64+128+256 canonical file), which does not exist yet.
- Figures 2–3 (main online-replay performance comparison figures) — not yet
  built at all (confirmed absent from `figures/manuscript/`); same blocker.
- The workload-specific breakdown table (matrix row 9 in
  `reports/kbs_revision_completion_audit.md`) — same blocker.
- Any narrative text about how `evict_value_v1` compares to LRU end-to-end —
  don't lock in wording until all 4 capacity chunks are merged (cap32 alone
  is only 1 of 4 capacities, even though its direction is consistent with 2
  other independent data points).

## 7. Recommended sequencing

1. Run the artifact-builder script (§5) now — it is independent of the
   canonical-CSV blocker and only needs a human decision to execute, not new
   compute. This fixes table4/table5/figure4/figure5 staleness immediately.
2. Manually copy the refreshed table4 numbers into `main.tex`'s
   `tab:evict-value-ablation` (the zip's LaTeX source isn't wired to the
   generated `.tex` snippets, so this step needs a manual paste either way).
3. Leave table3/figures2-3 untouched until the canonical CSV (cap64/128/256 +
   merge) is complete — running the builder script early for those would just
   reproduce the same `NOT_VERIFIED` stub, which is already the correct,
   honest current state.
