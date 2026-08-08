# KBS Revision Evidence Ledger — Source of Truth

**Purpose of this document:** single place to check, before spending any more compute or
cluster time, what has already been done for the KBS revision, what it proved, what it did
*not* prove, and what is still safe/unsafe to cite. Read this before rerunning anything.

**Machine context for everything in this ledger:** the local/cloud machine
(`/home/soroush/Augmented-caching`), **not** Wulver, **no** Slurm/sbatch/squeue/sacct.
Where a command below mirrors a `.sbatch` file, it was run directly as a plain
shell/python invocation, not submitted to a scheduler.

**Document generated:** 2026-06-18, by a read-only documentation pass (no experiments
launched, nothing deleted/overwritten/merged/pushed, no running jobs touched).
**Repo state at generation time:** branch `main`, HEAD `e9ac132c0aba12c640b0663af8c8bb6c946e81e1`.

---

## 1. Revision objective

The objective of this revision is to satisfy all Associate Editor (AE) and reviewer
comments for:

- **Manuscript ID:** KNOSYS-D-26-07461
- **Title:** "Decision-aligned eviction-value prediction for robust learning-augmented
  caching"
- **Journal:** Knowledge-Based Systems (Elsevier)
- **Decision:** Revise

---

## 2. Verified reviewer/AE comments

**UPDATE 2026-06-19 — superseded.** The full, real reviewer/AE comments were
supplied by the user in this session and are now recorded verbatim (as
supplied) in `reports/kbs_real_reviewer_comments.md`: an Associate Editor
summary, Reviewer #2 Major Comments 1–3, and Reviewer #3 (Summary, Issues
1–7, Minor Problems 8–9, Recommended Revisions 1–8). **Everything below this
line in this section describes the pre-2026-06-19 state and is kept for
historical/provenance reasons only — do not treat it as current.**

---

### Historical state (superseded 2026-06-19; kept for provenance)

**Only the following three concerns were verified as coming from the actual editorial
decision** (user-provided, confirmed) prior to 2026-06-19:

1. Lack of end-to-end miss-ratio evaluation.
2. Limited differentiation from existing/baseline approaches.
3. Need for stronger experimental support / positioning / reproducibility clarity.

**Full reviewer comments were, at that time, still missing.** No Editorial Manager decision letter or
notification email text had been retrieved into this repository or either of its two
worktrees. This was explicitly searched for and confirmed absent:
`reports/need_full_kbs_reviewer_comments.md` (exists only on branch
`kbs-revision-parallel-cleanup` / worktree `/home/soroush/Augmented-caching-kbs-parallel`,
not on `main`) documents an exhaustive grep across both worktrees and `~/projects`
(which doesn't exist on this machine) for `"KNOSYS-D-26-07461"`, `"Reviewers' comments"`,
`"Associate Editor:"`, etc. — zero hits beyond this project's own prior internal notes.
**This grep predates the user directly pasting the real comments into the chat in this
session — it was never going to find text that wasn't in any file at the time.**

**Any repo-local "sample review" concerns remain hypothetical/anticipatory, and are now
superseded by the real comments where they overlap.** The file
`reports/manuscript_artifacts/reviewer_concern_gap_map.md` (11 items, on `main`) self-labels
its input as the **"sample review"** (its own words, line 3) — it was this repo's own
plausible/anticipatory audit against repo evidence, **not** a transcription of the real
KNOSYS-D-26-07461 reviewer/AE text. It has been marked superseded at the top of that file.
Likewise the old `reports/kbs_complete_reviewer_comment_matrix.md`
(branch `kbs-revision-parallel-cleanup` only) explicitly tagged every row's source as either
"AE (user-provided, confirmed)" (rows 1–3, 15) or "Unknown / internal sample-review audit"
(everything else) — that file is a different, older, unreconciled draft on a branch that was
never merged; the current `main`-branch matrix of this name has been rebuilt from the real
comments (`reports/kbs_real_reviewer_comments.md`), see §15/§16 below.

**Do not present any "sample review" item to anyone (reviewers, co-authors, the editor) as
confirmed verbatim reviewer text — use `reports/kbs_real_reviewer_comments.md` instead.**

---

## 3. Evidence map by concern

| Concern | Needed evidence | Existing artifact | Status | Manuscript-usable? | Rebuttal-usable? | Missing / blocker |
|---|---|---|---|---|---|---|
| End-to-end miss-ratio evaluation | Multi-trace × multi-capacity replay miss/hit table for `TABLE3_POLICIES`, canonical `heavy_r1` tag | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv` (1 trace × 1 capacity × 7 policies) | **Validation/smoke only** — see §6 | No | No (cite only as "pipeline validated, full sweep pending") | Canonical `..._heavy_r1.csv` (7×4×7=196 rows) — not launched, see §7–§8 |
| Baseline differentiation | Empirical/positioning contrast vs. internal baselines and external systems-level prior work | `analysis/baseline_audit/baseline_audit_summary.csv` (9 internal baselines audited); `reports/kbs_baseline_positioning_plan.md` (branch only); `tables/manuscript/table6_related_work_learned_caching.tex` + `refs/related_work_table6.bib` (11 external citations present) | Internal baselines: **ready**. External systems: **positioning-only, not empirically compared** (by design — see §10) | Internal baselines: yes. External: positioning text only | Yes, with explicit scope language | None for internal baselines. External empirical comparison is out of scope this cycle (new implementation work, not a rerun) |
| Reproducibility | Clean-install + test suite; quick-start walkthrough without crashes; documented canonical/exploratory naming convention | 283/283 tests passing (re-verified today, see §11); two README/script bugs fixed on branch `kbs-revision-parallel-cleanup` (commit `64e39bb`) but **not on `main`** | **Partially on main** — fixes exist but uncommitted to `main` | Not yet (fixes not on `main`) | Can describe once merged | Decide whether/when to merge `64e39bb`/`9a1642a` into `main` |
| Experimental support | Dataset scale/composition, offline ablation, online replay | 323M-row heavy_r1 dataset (rebuilt today, see §4), offline model comparison (Table 4 equivalent, see §5), validation replay (§6) | Offline: **strong**. Online: **validation-only** | Offline: yes. Online: not yet | Offline: yes | Canonical online sweep (same blocker as row 1) |
| Full reviewer comments / point-by-point response | Real Editorial Manager decision letter / reviewer reports | None retrieved | **Hard blocker** | N/A | N/A | Must be retrieved from Editorial Manager or the notification email (Gmail/institutional inbox) — not recoverable from any local file |

---

## 4. Dataset provenance

- **Dataset directory:** `data/derived/evict_value_v1_wulver_heavy_r1`
- **Size:** **96G** (`du -sh`, confirmed today). **Never copy this directory** — only
  metadata files below were inspected/checksummed.
- **Manifest:** `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` — present,
  104K (106318 bytes), mtime 2026-06-18 09:39.
  sha256 `7e06f18d99313eb7f8ff1158719243f85230880f27dd395a237d6aed0648d6bf`
- **Split summary:** `data/derived/evict_value_v1_wulver_heavy_r1/split_summary.csv` —
  present, 8.1K (8269 bytes), mtime 2026-06-18 09:39.
  sha256 `4aebf011490b1279ee5f4072e11ffe7ad2d155e5d649228088b74fdf9a80b88b`
- **Build command** (recovered from `slurm/evict_value_v1_wulver_heavy_train.sbatch`
  defaults, run directly/locally — not via `sbatch` — and corroborated line-for-line by
  `logs/kbs_heavy_r1/build_dataset_preflight.log` / `build_dataset.log`):
  ```bash
  python scripts/build_evict_value_dataset_wulver_v1.py \
    --trace-manifest analysis/wulver_trace_manifest_full.csv \
    --capacities 32,64,128,256 \
    --horizons 4,8,16 \
    --split-mode trace_chunk \
    --chunk-size 4096 \
    --split-seed 7 \
    --max-rows-per-shard 500000 \
    --max-requests-per-trace 50000 \
    --out-dir data/derived/evict_value_v1_wulver_heavy_r1
  ```
- **Build log:** `logs/kbs_heavy_r1/build_dataset.log` (2504 bytes) +
  `logs/kbs_heavy_r1/build_dataset_preflight.log` (922 bytes). Preflight confirms all 7
  trace families valid (≥1000 requests, ≥100 unique pages); build log lists per-(trace,
  capacity) shard counts ending in `Wrote manifest=... summary=...` with no error lines.
- **Build exit code:** no explicit `BUILD_EXIT=` marker was logged for this stage (unlike
  the train/summary/validation stages below). Treated as **exit 0 / success**: the
  sbatch-equivalent pipeline this mirrors runs under `set -euo pipefail` with a hard
  `test -f manifest.json || exit 3` gate before training starts, and the downstream
  training stage (which requires this exit) ran and itself exited 0 (`TRAIN_EXIT=0`,
  §5) — so a non-zero build exit can be ruled out by the existence of everything after it.
- **Build start/end time:** acquisition phase started 2026-06-17 23:02 (first
  `acquire_*_download.log` mtime); preflight ran 2026-06-17 23:19; `build_dataset.log`
  finalized 2026-06-18 09:39:04 (mtime). **Wall clock for acquisition+build ≈ 10h37m**,
  run unattended overnight (corroborated independently by
  `/home/soroush/Augmented-caching-kbs-parallel/logs/kbs_overnight_revision/heavy_job_monitor.log`,
  a read-only 30-minute-interval monitor from a second worktree that observed this same
  job's directory size growing from 5.2GB at ~23:55 to 7.2GB at ~00:08 and onward).
- **Git commit at time of this build:** `e9ac132` (current `main` HEAD; dataset was built
  with this commit checked out, no code changes since).
- **Canonical heavy_r1 status:** **Yes, this dataset should be considered the current
  canonical heavy_r1 dataset** — but see the critical caveat immediately below.

### Critical provenance caveat: this dataset is NOT the one in `main`'s last commit

`git diff` against the committed (`main`, commit `53726ce`, 2026-04-10) version of
`analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md` shows the on-disk dataset
was **completely rebuilt today, locally**, and differs materially from what was committed:

| Field | Committed (2026-04-10, presumably Wulver) | Current on-disk (2026-06-18, local rebuild) |
|---|---|---|
| Total rows | 289,304,256 | 323,043,072 |
| Unique decisions | 2,524,527 | 2,767,149 |
| Shard count | 594 | 662 |
| twemcache rows | 10,591,392 | 44,330,208 |
| Best model (horizon 4) | `hist_gb` | `random_forest` |

The committed twemcache row count was roughly **4x smaller** than the current build —
consistent with an incomplete/earlier twemcache acquisition in the original (presumably
Wulver-side) run. The current local rebuild re-acquired all 7 traces from scratch
(`logs/kbs_heavy_r1/acquire_twemcache_*.log`) and produced full-scale twemcache coverage
(50,000 requests, 11,500 unique pages, matching the other 6 families' scale). **This
changed the best-model selection** (horizon=4 selects `random_forest` now, not `hist_gb`).
**Do not mix old committed numbers with new on-disk numbers in any manuscript table** —
the current on-disk artifacts (uncommitted, see `git status --short`) are the only
internally-consistent set and should be the ones committed and cited going forward.

---

## 5. Training provenance

- `analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json` — 17793 bytes, mtime
  2026-06-18 10:37. sha256 `1d924f14e39eb9d0372eda7c3b704bed95934176c88a3689fba2aa6cc05fa363`
- `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv` — 1624 bytes, mtime
  2026-06-18 10:37. sha256 `9deaa8a3d71eb4df1d636c361b15c2b52dad2a889cd5ec4795242203efc45ccd`.
  9 rows (3 horizons × 3 models: ridge, random_forest, hist_gb), columns
  `horizon,model,val_mae,val_rmse,test_mae,test_rmse,val_top1,test_top1,val_mean_regret,test_mean_regret`.
- `analysis/evict_value_wulver_v1_best_config_heavy_r1.json` — 192 bytes, mtime
  2026-06-18 10:37. sha256 `87493114203f8429a4c0d2c5e7554fe6ae58daf31d11becff424af34f306dd36`.
  Content:
  ```json
  {
    "horizon": 4,
    "model": "random_forest",
    "model_path": "models/evict_value_wulver_v1_best.pkl",
    "selection_rule": "minimize validation mean_regret_vs_oracle across (horizon, model)"
  }
  ```
- **Exact training command** (recovered from `slurm/evict_value_v1_wulver_heavy_train.sbatch`
  defaults, run directly/locally):
  ```bash
  python scripts/train_evict_value_wulver_v1.py \
    --manifest data/derived/evict_value_v1_wulver_heavy_r1/manifest.json \
    --horizons 4,8,16 \
    --max-train-rows 800000 \
    --max-val-rows 250000 \
    --max-test-rows 250000 \
    --models-dir models \
    --metrics-json analysis/evict_value_wulver_v1_train_metrics_heavy_r1.json \
    --comparison-csv analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv \
    --best-config-json analysis/evict_value_wulver_v1_best_config_heavy_r1.json
  cp models/evict_value_wulver_v1_best.pkl models/evict_value_wulver_v1_best_heavy_r1.pkl
  ```
  (Dataset summary step ran between build and train:
  `python scripts/summarize_wulver_evict_value_dataset.py --manifest .../manifest.json
  --out-json .../dataset_summary_extended_heavy_r1.json --out-md
  analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md`, exit `SUMMARY_EXIT=0`,
  log `logs/kbs_policy_comparison_heavy_r1/summary_run.log`.)
- **Training log:** `logs/kbs_policy_comparison_heavy_r1/train_run.log` (17807 bytes) —
  contains the full per-(horizon, model) metrics JSON as stdout, ending `TRAIN_EXIT=0`.
- **Training exit code:** `TRAIN_EXIT=0` (clean).
- **Selected best model/config:** horizon=4, `random_forest`,
  `val_mean_regret_vs_oracle ≈ 0.02078`.
- **Horizons/models compared:** horizons {4, 8, 16} × models {ridge, random_forest,
  hist_gb} = 9 configurations.
- **Model file:** `models/evict_value_wulver_v1_best_heavy_r1.pkl`, 13,301,662 bytes
  (13M), mtime 2026-06-18 10:41, gitignored (not tracked). sha256
  `0c9e8a48066f8bb80bfab31b023c9785ca5b955f408d50c18008dbcc314ea61b`.
- **Freshness:** **Yes, fresh from the current 96G dataset** — same session, same day,
  immediately downstream of the build documented in §4. Internally consistent with each
  other; **not** consistent with the git-committed (2026-04-10) versions of these same
  filenames (see §4 caveat) — the working tree currently has these 4 files modified
  (`git status --short`), which is expected and correct; they should be committed as a set,
  not piecemeal.
- **Canonical or intermediate:** **Canonical heavy_r1 training outputs** per
  `CANONICAL_KBS_SUBMISSION.md`'s `_heavy_r1` suffix convention — these directly feed
  Table 4 / Fig. 4 (offline ablation).

### Open methodological caveat carried over from `reports/kbs_revision_gap_tracker.md`

`scripts/train_evict_value_wulver_v1.py`'s row-loading shuffles shard paths with a fixed
seed and early-stops once it has buffered `max_val_rows * 2` raw rows, then samples down —
this is an early-stop over a shuffled shard *prefix*, not a proportional draw across all
shards. As a result the validation-set metrics used for model/horizon selection cover only
5 of 7 trace families (brightkite, cloudphysics, metacdn, twemcache, wiki2018 — citibike and
metakv entirely absent from the ~4,572-row validation sample, ~3.2% of available decisions).
Deterministic, not corruption — but the close margin between `random_forest` (0.02078) and
`hist_gb` (0.02100) at horizon=4 could plausibly flip under full-family validation coverage.
**Worth disclosing if this selection is cited in the manuscript.**

---

## 6. Validation eval provenance

**Status label: NON-CANONICAL VALIDATION / NOT FINAL MANUSCRIPT EVIDENCE.** Do not cite
as the end-to-end miss-ratio result. This is a pipeline-correctness + timing-calibration
check only (1 of the eventual 196 trace×capacity×policy rows).

- **CSV:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv` —
  743 bytes, mtime 2026-06-18 14:47. sha256
  `ef98d6f4809964905244d57157ab9810d0bc067027f6cd43cee76dea7cdd8b0b`. Untracked
  (`git status --short` shows `??`).
- **MD companion:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.md`
  — 1140 bytes, mtime 2026-06-18 14:47. sha256
  `1d367373788ce7c4b8e73f913c4d631761c38e222b27375126bfa5a5ea6548ae`. Untracked.
- **Exact command** (from `logs/kbs_policy_comparison_heavy_r1/validation_eval_run.log`,
  run under `/usr/bin/time -v`, wrapped in tmux session `kbs_policy_comparison_heavy_r1`):
  ```bash
  python scripts/run_policy_comparison_wulver_v1.py \
    --trace-manifest analysis/wulver_trace_manifest_full.csv \
    --max-traces 1 \
    --capacities 256 \
    --max-requests-per-trace 50000 \
    --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
    --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
    --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv \
    --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.md
  ```
- **Runtime:** **4:02:57** (wall clock, `h:mm:ss`). User time 14570.75s, sys 2.53s, 99% CPU
  (single-threaded/single-core bound). Max RSS ≈ 2.1GB.
- **Exit code:** `VALIDATION_EXIT=0` (clean, no stderr, no tracebacks).
- **Scope:** trace = `brightkite_50k` (1 of 7), capacity = 256 (the most expensive of
  32/64/128/256), policies = 7 (`lru, blind_oracle, predictive_marker,
  blind_oracle_lru_combiner, trust_and_doubt, rest_v1, evict_value_v1`; `ml_gate_v1`/
  `ml_gate_v2` excluded — their model files aren't present on this machine). Model:
  `models/evict_value_wulver_v1_best_heavy_r1.pkl`.
- **Output values** (misses / hit_rate, capacity 256, brightkite_50k, 50,000 requests):

  | policy | misses | hit_rate | vs LRU |
  |---|---|---|---|
  | lru | 14758 | 0.70484 | — |
  | rest_v1 | 14758 | 0.70484 | tie |
  | blind_oracle_lru_combiner | 14759 | 0.70482 | ≈tie |
  | predictive_marker | 14923 | 0.70154 | −1.12% |
  | trust_and_doubt | 15306 | 0.69388 | −3.71% |
  | **evict_value_v1** | **20533** | **0.58934** | **−39.13%** |
  | blind_oracle | 29554 | 0.40892 | −100.26% |

  ("vs LRU": negative = more misses than LRU, i.e. worse.)

- **⚠ Warning, carried forward verbatim:** `evict_value_v1` had **~39% more misses than
  LRU** on this one trace — the model is *underperforming* the simplest baseline here,
  not just trailing the best one. `blind_oracle` was the single worst performer (−100%
  vs LRU). Both require a sanity audit before the full run is trusted to be worth the
  multi-day compute cost (see §9). One piece of context gathered for that audit (not a
  full resolution): `src/lafc/policies/blind_oracle.py`'s own docstring states the policy
  "fully trusts the predictor" and "can be arbitrarily bad" when predictions are weak —
  so a bad `blind_oracle` score is *expected* if it's being fed a weak/synthetic
  next-arrival signal in this driver, not automatically evidence of a wiring bug. This
  does not resolve the `evict_value_v1` concern, which is the one that matters for the
  manuscript.
- **Does this confirm the pipeline works end-to-end?** Yes, mechanically: model loading
  (including the earlier joblib-fallback bugfix, commit `b9df6f1`), trace loading, all 7
  policies, and CSV/MD aggregation all worked with no exceptions. It does **not** confirm
  the *model's* miss-ratio result is good.
- **Should this be used in the manuscript?** **No.** 1-trace × 1-capacity sample (7 of the
  eventual 196 rows). Filename carries `_validation` specifically so it can't be confused
  with, or auto-consumed by, the canonical-artifact builder.

---

## 7. Canonical policy-comparison status

- **Canonical target path:** `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`
  (+ `.md` companion).
- **Exists?** **No.** Confirmed absent in the working tree
  (`ls analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` → not present).
- **Tracked/untracked/ignored?** N/A — does not exist, so it has no git status. It is not
  listed in `.gitignore` (no pattern matches this filename), so once produced it will be a
  normal trackable file, same as its sibling `_validation` files are (untracked, pending
  `git add`).
- **Exists in any other local clone or git history?** **No.** Checked:
  - `git log --all --diff-filter=A --name-only -- analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv` → zero commits, on any branch, ever added this file.
  - Looped `git cat-file -e <branch>:<path>` over every local and remote-tracking branch → no hits.
  - The second local worktree `/home/soroush/Augmented-caching-kbs-parallel`
    (branch `kbs-revision-parallel-cleanup`) does **not** have this file either — only
    the older non-canonical `analysis/evict_value_wulver_v1_policy_comparison.csv` and
    `..._heavy_smoke.csv`/`.md` (both explicitly non-canonical per
    `docs/wulver_heavy_evict_value_experiment.md` — the unsuffixed file includes extra
    policies like `atlas_v3`/`ml_gate_*` from a broader, different run; the smoke file is
    a 2-trace/2-capacity/5000-req wiring check, ~10x fewer requests/trace than canonical).
  - `tables/manuscript/table3_main_quantitative_comparison.csv` (the manuscript-builder
    stub) literally contains: `status=NOT_VERIFIED, detail="Canonical file
    analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv is absent."`
  - `reports/manuscript_artifacts/manuscript_artifact_manifest.json` records
    `"policy_comparison_present": false` (current, machine-checked, still accurate today).
- **Is the full run still needed?** **Yes** — this is the single file that would resolve
  reviewer concern (1) (end-to-end miss-ratio evaluation) and is a major contributor to
  concern (3). Nothing in this repo or either worktree substitutes for it.
- **Existing inventory report:** `reports/kbs_existing_results_file_inventory.md` does
  **not exist anywhere** (checked: `main`, `kbs-revision-parallel-cleanup`, both
  worktrees' filesystems). The search summarized above (this section) is the closest
  thing to that inventory and should be treated as satisfying that need — no separate
  file was created since this ledger already records the full search and result.

---

## 8. Full-run execution plan

Source: `reports/kbs_full_policy_comparison_execution_plan.md` (decision report;
**nothing in it has been executed**).

- **Recommendation:** chunked execution, split by capacity (cheapest first: 32 → 64 →
  128 → 256), **unless the sanity audit in §9 changes this** (e.g. if it finds the model
  needs rework before spending multi-day compute on it).
- **Estimated runtime** (anchored to the one validation data point, §6):
  - Optimistic (cost ∝ capacity, matches confirmed O(capacity) cost in
    `EvictValueV1Policy._choose_victim`): **~53h (~2.2 days)** across all 4 capacity chunks.
  - Conservative (every capacity costs like cap256): **~113h (~4.7 days)**.
  - Stated uncertainty band: **±2x**, since this is extrapolated from a single
    (trace, capacity) data point; the 7 trace families vary hugely in unique-page count
    (citibike: 1,917 unique pages vs. wiki2018: 50,000/50,000 — no repeats at all), which
    can swing per-trace cost in either direction for content-aware policies.
  - Historical corroboration: the unchunked version of this exact command previously
    **exceeded a 24h walltime limit on Wulver (job 908326, 16 dedicated CPUs) without
    completing**, and a 72h resubmission was never confirmed to finish either (blocked by
    a maintenance window before getting compute time).
- **No checkpointing in the current script:** `scripts/run_policy_comparison_wulver_v1.py`
  accumulates all `(trace, capacity, policy)` rows in one in-memory list and writes the
  CSV/MD exactly once, after the full triple-nested loop completes. No checkpoint file, no
  incremental write, no flag to skip already-computed combinations, no resume mechanism.
- **Chunk/resume limitations:** chunking (by `--capacities`, `--policies`, or
  `--max-traces`/custom manifest) does **not** add resume *within* a chunk — it only
  bounds how much work a single failure can cost (worst case ~28h for an unsplit cap256
  chunk, vs. ~113h for the whole unchunked run).
- **Exact commands prepared but not launched** (capacity-chunked, Option B, the
  recommended approach — see the full plan doc for the tmux wrapper pattern and the
  post-hoc CSV-merge script):
  ```bash
  # Chunk 1 — capacity 32
  python scripts/run_policy_comparison_wulver_v1.py \
    --trace-manifest analysis/wulver_trace_manifest_full.csv \
    --capacities 32 --max-requests-per-trace 50000 \
    --policies lru,blind_oracle,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
    --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
    --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv \
    --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.md
  # Repeat with --capacities 64 / 128 / 256 and matching _cap64/_cap128/_cap256 paths.
  # Then merge all 4 chunk CSVs into the canonical
  # analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv (merge script in the
  # full plan doc, §8) and separately regenerate the .md aggregate.
  ```
- **Risk of wasting days if a validation bug exists:** explicitly flagged in the plan —
  `evict_value_v1` underperforming LRU on the one trace tested (§6) is a *research* risk,
  not just a compute risk; running the full ~2.2–4.7 day sweep before resolving §9 risks
  spending that time only to confirm a result that needs the model reworked anyway.
- **Machine is idle and capable of running chunks concurrently** if/when launch is
  approved: 20 cores, load average 0.22–0.61, 59GB RAM available — the script is
  single-core-bound, so running all 4 capacity chunks in parallel tmux sessions would cut
  wall-clock to roughly the cost of the single slowest chunk (~28h) instead of summing
  them serially. This is optional and only relevant once a launch is explicitly approved.

---

## 9. Sanity-audit status

`reports/kbs_validation_sanity_audit.md` — **complete.** Conclusion: **A
— validation is trustworthy; poor performance appears real; proceed with
chunked full sweep.**

Summary of findings:

- **`evict_value_v1` worse than LRU is real, not a bug.** Full code-path
  audit of feature construction (`request.metadata`-derived, not
  `predicted_next`-derived), eviction-direction comparator (`min` over
  predicted loss — correct given the `y_loss` target semantics), and the
  training target (`_simulate_lru_misses` over a real, decision-aligned
  short horizon) found no logic defect. 58 targeted pytest tests
  (`lru`/`oracle`/`belady`/`evict_value`/`policy_comparison`/`miss`) all
  pass. A second, independent sanity-check data point
  (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation_sanity.csv`,
  capacity 32 / 5,000 requests, same `brightkite` trace) again shows
  `evict_value_v1` worse than LRU (+11.81% misses), at a smaller magnitude
  than the cap256/50k validation point (+39.13%) — directionally consistent
  with the open hypothesis that the selected model's short training horizon
  (h=4) becomes more of a liability as the live replay window/capacity
  grows.
- **`blind_oracle` worse than LRU is explained by a found configuration
  gap, fully resolved as "not suspicious, not blocking."**
  `scripts/run_policy_comparison_wulver_v1.py` only attaches a real
  prediction stream (`attach_predicted_caches` → `td_reqs`) to
  `trust_and_doubt`; every other policy (including `blind_oracle` and
  `evict_value_v1`) gets the plain `reqs`, and `.jsonl` trace loading
  (`build_requests_from_lists`) defaults every request's `predicted_next`
  to `math.inf` when no predictions are supplied. With all candidates tied
  at `math.inf`, `BlindOraclePolicy` degenerates to evicting the smallest
  page_id every time — unrelated to recency or reuse. This is **not** a
  defect in `BlindOraclePolicy` itself (its own tests, including
  perfect-prediction-matches-Belady, all pass) and, critically,
  **`blind_oracle` is not in `TABLE3_POLICIES`**
  (`scripts/paper/build_kbs_main_manuscript_artifacts.py`) — it is an
  internal-only baseline, so this gap does not taint or block any
  manuscript-citable result. Action item: never cite `blind_oracle`'s
  number from this comparison script as a meaningful oracle baseline
  unless/until it is given a real prediction source.
- Freshness chain (dataset → training → model pkl → validation run) is
  clean and monotonic; validation definitely used the fresh
  `models/evict_value_wulver_v1_best_heavy_r1.pkl` (horizon=4,
  random_forest), not a stale artifact.
- Miss-ratio/hit-rate accounting (`src/lafc/metrics/cost.py`) is computed
  identically for every policy via the shared `run_policy()` path — no
  divergent accounting found.
- Trace/capacity selection in the validation command is correctly aligned
  (`--max-traces 1` → manifest row 1 = `brightkite_50k`; `--capacities 256`
  = deliberate worst-case timing point, not a misconfiguration).

**Status: Unblocked.** The full canonical chunked sweep (§8) may now
proceed — see `reports/kbs_full_policy_comparison_execution_plan.md` §9 for
the (not-yet-launched) exact commands.

---

## 10. Related work / baseline evidence

- `reports/kbs_baseline_positioning_plan.md` — **exists only on branch
  `kbs-revision-parallel-cleanup`** (commit `64e39bb`), not on `main`. Separates internal
  (implemented, audited) baselines from external (positioning-only) systems and gives an
  explicit per-system recommended manuscript action.
- `reports/kbs_related_work_citation_verification.md` — exists only as an **untracked
  file** in the second worktree `/home/soroush/Augmented-caching-kbs-parallel` (not
  committed anywhere, not on `main`). Read in full for this ledger. Conclusion: all 11
  external/theory citations named in the revision-plan documents already have a BibTeX
  key in `refs/related_work_table6.bib` — none invented, none missing as a *key*. Four
  entries (`lykouris2018competitive`, `bansal2022weightedpaging`, `wei2020lacaching`,
  `chledowski2021robustlacaching`) carry incomplete volume/page metadata flagged
  `note = {Verify ...}` — fix before camera-ready, not a blocker now.
- `tables/manuscript/table6_related_work_learned_caching.tex` — present on `main`, 4663
  bytes, sha256 `2fcba3c8fe7d4e124d580b5fda4dcd1d9de1ebc7ede23075c02ede04770c00ca`.
- `refs/related_work_table6.bib` — present on `main`, 4797 bytes, sha256
  `42fa085fea6dce7eeb368d55a59ce7a05ccc00fc8e4f5c8894d481e9953e523e`. (No project-wide
  `refs.bib` exists at repo root — only this Table-6-scoped bibliography. The full
  manuscript's bibliography, if different, lives inside the zip-only manuscript source,
  not under version control here.)
- **HALP / MUSTACHE / `adaptive_query` resolution** (explicitly called out as
  potentially-unresolved in the task prompt — now resolved by direct inspection):
  - **HALP** (`song2023halp`): citation **present** and complete (NSDI'23). **Not** a
    missing-citation issue. Flagged as the **highest-priority primary-source
    verification target** before sharpening any HALP contrast — its label/supervision
    object needs to be checked against the actual paper.
  - **MUSTACHE** (`tolomei2022mustache`): citation **present** but as an **arXiv preprint
    only**, not a peer-reviewed venue entry. Flagged for primary-source verification and
    a decision on whether to keep citing the preprint or find a peer-reviewed version.
  - **`adaptive_query`**: this is **not** an external citation gap in `refs/related_work_table6.bib`
    — it's an **internal baseline name** (`docs/baselines.md` calls it "Baseline 5",
    aka `parsimonious_caching`, grounded in Im et al., *Parsimonious Learning-Augmented
    Caching*, ICML 2022). It does not appear in Table 6 or `TABLE3_POLICIES`, so it was
    out of scope for the Table-6 bib file by design. **Genuine, confirmed gap:** Im et
    al. 2022 has **no BibTeX entry anywhere in this repo** — only prose citation in
    `docs/baselines.md`. **Action needed only if `adaptive_query` results are ever
    discussed in the manuscript** — add a BibTeX entry then, not before.

### Baseline/external-system classification

| System | Classification |
|---|---|
| `lru` | Directly compared empirically (canonical `TABLE3_POLICIES`) |
| `predictive_marker` | Directly compared empirically (canonical `TABLE3_POLICIES`) |
| `trust_and_doubt` | Directly compared empirically (canonical `TABLE3_POLICIES`); cites Antoniadis et al. 2020, fully verified mapping in `docs/baselines.md` |
| `blind_oracle_lru_combiner` | Directly compared empirically (canonical `TABLE3_POLICIES`); cites Wei 2020 |
| `rest_v1` | Directly compared empirically (canonical `TABLE3_POLICIES`) |
| `evict_value_v1` | Directly compared empirically (canonical `TABLE3_POLICIES`) — this repo's own method |
| `blind_oracle` | Internal baseline only — implemented, audited, run in validation pass, but not in `TABLE3_POLICIES` by default |
| `advice_trusting`, `marker`, `weighted_lru` | Internal baseline only — implemented/audited, not in `TABLE3_POLICIES` |
| `la_det` / `la_det_approx` / `la_det_faithful` | Internal baseline only, **and** conceptually grounded in Bansal et al. 2022 (cited) — but no canonical `heavy_r1` numbers exist for this family; do not cite numbers for it |
| `robust_ftp_d_marker` / `robust_ftp` | Internal baseline only, grounded in Chłędowski et al. 2021 (cited) — same caveat as `la_det*`: no canonical numbers |
| `adaptive_query` (`parsimonious_caching`) | Internal baseline only; grounding citation (Im et al. 2022) **missing from BibTeX** — see above |
| Lykouris & Vassilvitskii 2018, Bansal et al. 2022 (theory) | Conceptual related work only |
| PARROT, LRB, Raven, Mockingjay | Conceptual related work only — cited, not primary-source-verified, not implemented |
| HALP, MUSTACHE | Conceptual related work only — cited, **flagged unsafe to sharpen** without primary-source verification first (highest overclaiming risk of the group) |
| GUARD-style robustification, Cold-RL/offline RL, decision-focused/SPO framing | Missing citation — genuinely uncited anywhere in the repo; not in the current formal revision plan; pin exact papers before using prominently |

---

## 11. Reproducibility evidence

- **Test suite, re-verified today (this session, on `main`, in `.venv_kbs_heavy_r1`):**
  ```
  283 passed, 12 warnings in 10.47s   (also independently 10.68s on a second run)
  ```
  Logged at `logs/kbs_revision_evidence/pytest_reverify.log`, `PYTEST_EXIT=0`. This
  matches, almost exactly, the figure asserted in `kbs-revision-parallel-cleanup` branch
  reports (`reports/kbs_manuscript_revision_change_checklist.md`,
  `reports/kbs_response_to_reviewers_skeleton.md`): **283 passed, 12 warnings, 11.01s**,
  said to be logged at `logs/kbs_overnight_revision/pytest_parallel.log`. That exact log
  file does **not** exist on disk anywhere checked (`logs/` is not committed, by design,
  in either worktree) — so the 11.01s figure itself is **unverifiable from a stored
  artifact**, but it is now **independently corroborated** by today's fresh run on `main`
  (283 passed / 12 warnings, ~10.5–10.7s — timing naturally varies run to run).
- **README quick-start fixes:** two bugs were fixed —
  1. `scripts/run_evict_value_v1_first_check.py` crashed with `FileNotFoundError` for
     `models/ml_gate_v2_random_forest.pkl` on a clean clone; fixed to skip
     `ml_gate_v1`/`ml_gate_v2` gracefully and report them as "skipped" when their model
     files are absent.
  2. `scripts/train_ml_gate_v1.py` crashed with
     `AttributeError: 'LinearProbabilityEstimator' object has no attribute 'named_steps'`.
  Both fixes exist **only on branch `kbs-revision-parallel-cleanup`, commit `64e39bb`**
  (`git merge-base --is-ancestor 64e39bb main` → **NO**, not an ancestor of `main`). They
  are **not present on `main`** as of HEAD `e9ac132`.
- **Verifier script:** `scripts/paper/verify_kbs_heavy_r1_csv.py` — structural
  completeness checker for the canonical policy-comparison CSV (existence, non-empty,
  parses as CSV, expected columns). Exists **only on `kbs-revision-parallel-cleanup`**
  (same commit `64e39bb`), **not on `main`**.
- **Artifact builder:** `scripts/paper/build_kbs_main_manuscript_artifacts.py` — present
  on `main`. Single source of truth for Table 1/2/3/4/5 and Fig. 1/2/3/4/5; degrades
  gracefully (does not error) when the canonical policy-comparison CSV is absent, emitting
  a stub Table 3 and offline-only Table 5/Fig. 5 supplements instead.
- **Cleanup commit `64e39bb`** ("Prepare KBS revision cleanup and verification
  utilities") — **on branch `kbs-revision-parallel-cleanup` only.** Not merged to `main`.
- **Reviewer-response package commit `9a1642a804e78eb3998e734c8d5ba2324432a01d`** ("Add
  KBS reviewer-response planning package") — **on branch `kbs-revision-parallel-cleanup`
  only.** Not merged to `main`. Adds `reports/kbs_complete_reviewer_comment_matrix.md`,
  `reports/kbs_manuscript_revision_change_checklist.md`, `reports/kbs_revision_gap_tracker.md`
  (an earlier/different version than the one now also present on `main`),
  `reports/kbs_submission_package_checklist.md`, `reports/need_full_kbs_reviewer_comments.md`,
  and expands `reports/kbs_response_to_reviewers_skeleton.md`.
- **Merged to `main`?** **No, neither commit is merged to `main`.** Verified via
  `git merge-base --is-ancestor <hash> main` for both — both report `NO`. They exist only
  on the local branch `kbs-revision-parallel-cleanup`, which itself maps to the second
  worktree `/home/soroush/Augmented-caching-kbs-parallel`. **This ledger does not merge
  them** — merging was explicitly out of scope for this documentation task and is a
  decision for the user to make deliberately (review the diff first; the parallel
  worktree's own `kbs_overnight_revision_summary.md` says the same).

---

## 12. Submission-package requirements

Per Elsevier/KBS requirements (recorded in `reports/kbs_submission_package_checklist.md`,
which **exists only on branch `kbs-revision-parallel-cleanup`**, not `main` — summarized
here so the requirement list is available from `main` too):

- Revised manuscript (DOCX) — **none in repo**, authored entirely outside this codebase.
- Response to reviewers / rebuttal (DOCX) — only a Markdown skeleton exists
  (`reports/kbs_response_to_reviewers_skeleton.md`, branch-only), not a DOCX.
- Revised cover letter (DOCX) — **none in repo.**
- Highlights (DOCX) — **none in repo.**
- CRediT author statement (DOCX) — **none in repo**, outside repo scope (author input).
- Author agreement (DOCX) — **none in repo**, administrative.
- Declaration of interest (DOCX) — **none in repo**, administrative.
- Editable tables/figures/equations — partial: `tables/manuscript/*.{csv,tex}` and
  `figures/manuscript/*.{pdf,png}` exist for Tables 1, 2, 4, 5 and Figs. 1, 4, 5; Table 3
  / Figs. 2–3 are stubs pending the canonical eval CSV (§7).
- All references/tables/figures cited in text — partial; `refs/related_work_table6.bib`
  has all 11 Table-6 citations; full-manuscript citation completeness can't be confirmed
  from this repo since the manuscript source itself isn't version-controlled here.
- Figures at 300 dpi, not PDF — **action needed, not yet confirmed.** Current figure
  outputs are `.pdf`; the builder also emits `.png` alongside, but the PNG DPI setting has
  not been confirmed to meet 300 dpi.
- LaTeX source support files — partial; per-table/figure `.tex` snippets exist under
  `reports/manuscript_artifacts/latex_snippets/` and `tables/manuscript/`; whether the
  *full* manuscript is LaTeX vs. Word is not determinable from this repo (the only
  manuscript source found anywhere is a `.zip` at repo root, not LaTeX source under
  version control).

---

## 13. Do-not-rerun list

| Artifact / job | Status | Why not rerun | When rerun is justified |
|---|---|---|---|
| 96G heavy_r1 dataset build (`build_evict_value_dataset_wulver_v1.py`) | Done, exit 0 (implicit), ~10h37m, today | Just rebuilt overnight with full 7-family coverage (fixed the prior twemcache undercoverage); rebuilding again wastes ~10h+ for no new information | Only if trace acquisition inputs change, or a bug is found in the build script itself |
| `evict_value_v1_wulver_dataset_summary_heavy_r1` (summary stage) | Done, `SUMMARY_EXIT=0` | Cheap (~seconds) but derived directly from the dataset above — no reason to rerun without rerunning the dataset | Only if the dataset above is rebuilt |
| heavy_r1 training (`train_evict_value_wulver_v1.py`, all 9 horizon×model configs) | Done, `TRAIN_EXIT=0`, ~8 min | Fresh, consistent with current dataset; selection caveat (§5) is a disclosure issue, not a rerun issue | Only if the dataset is rebuilt, or if full-family validation coverage is specifically required (would need a code change to the sampling logic, not just a rerun) |
| Validation eval (1 trace × cap256 × 7 policies) | Done, `VALIDATION_EXIT=0`, 4h02m57s | Already answered its only purpose (pipeline correctness + timing calibration); its job is done | Only if the model/dataset changes and a fresh timing/correctness spot-check is needed before committing to a full sweep |
| Full canonical 7×4×7 policy-comparison sweep | **Not yet run** — this is the one job still pending, not one to avoid | N/A — listed here only to be explicit that it is *not* in the do-not-rerun category | Launch only after §9's sanity audit, and only with explicit user go-ahead given the ~2.2–4.7 day cost |
| Reviewer-response planning package (commits `64e39bb`, `9a1642a`) | Done, on branch `kbs-revision-parallel-cleanup` | Substantial planning work already complete; redoing it from scratch on `main` would duplicate effort | Only if the branch is deliberately abandoned instead of merged — re-derive on `main` instead of redoing the analysis |
| Full repository audit (`reports/kbs_revision_repo_audit.md`) | Done, 2026-06-17, 334 lines | Comprehensive, dated, still substantially accurate (only the dataset/model/validation facts it predates have since changed locally — see §4) | Only if repo structure changes significantly, or enough time passes that a fresh audit is warranted before final submission |
| Related-work / citation audit (`reports/kbs_related_work_citation_verification.md`) | Done, second worktree, untracked | All 11 external citations checked against `refs/related_work_table6.bib`; conclusions stable until the bib file or Table 6 text changes | Only if new external systems are added to Table 6, or after HALP/MUSTACHE primary-source verification changes the citation entries |

---

## 14. Remaining blockers

1. ~~Full reviewer comments still missing.~~ **RESOLVED 2026-06-19** — the
   real AE / Reviewer #2 / Reviewer #3 text was supplied directly by the
   user and recorded in `reports/kbs_real_reviewer_comments.md`. No longer a
   blocker.
2. **Sanity audit needed** because the validation result is suspicious:
   `evict_value_v1` had ~39% more misses than LRU on the one trace tested (§6).
   **RESOLVED** — `reports/kbs_validation_sanity_audit.md`, conclusion A
   (real, not a wiring bug), and reconfirmed by the cap32 chunk (§16).
3. **Canonical full policy-comparison CSV missing**
   (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1.csv`) —
   **partially resolved**: cap32 (1 of 4 capacity chunks) is done (§16);
   cap64/cap128/cap256 remain not launched. This is now the single biggest
   remaining blocker, ranked #1 in `reports/kbs_before_cap64_baseline_decision_memo.md`.
4. **Final manuscript artifacts blocked** until the canonical CSV exists and is verified —
   Table 3 and Figs. 2–3 cannot be produced from real evidence until then (§7, §12, §16).
5. **Final response-to-reviewers blocked, but now point-by-point against real
   comments rather than a category-level paraphrase.** Still blocked on (a)
   the canonical results (blocker 3 above) and (b) the SIEVE/FIFO-Reinsertion/
   HALP baseline-scope decision (R3-Issue2/3, R3-Rec2 — see
   `reports/kbs_before_cap64_baseline_decision_memo.md`). The real-comment-based
   skeleton (`reports/kbs_response_to_reviewers_skeleton.md`) now exists on
   `main` with explicit per-comment status tags.
6. **Baseline-scope decision for SIEVE/FIFO-Reinsertion/HALP** — explicitly
   requested by Reviewer #3 (Issue 3, Rec 2). **SIEVE half resolved at the
   implementation level 2026-06-19** — implemented, tested, smoke-tested
   (§17); still has no canonical cap32/64/128/256 numbers. FIFO-Reinsertion's
   exact variant remains undecided; HALP remains out of scope for empirical
   reimplementation. See `reports/kbs_before_cap64_baseline_decision_memo.md`
   and `reports/kbs_sieve_implementation_report.md`.
7. **Fallback-mechanism validate-or-demote decision** — explicitly requested
   by Reviewer #3 (Issue 6, Rec 5: "Validate the fallback mechanism or
   remove it"). Not yet decided.
8. **Manuscript rewrite (hedging/length) and DOCX package** — Reviewer #3
   Rec 4/6 and AE item 13 respectively; both writing-only, not compute-blocked.

---

## 15. Completion audit (2026-06-19)

Full 14-row reviewer/AE concern matrix, Step-4 completion audit (experiments/data,
reports/planning, submission package), ranked remaining-work list, and ordered
next-actions list now live in `reports/kbs_revision_completion_audit.md` — produced
as a read-only audit, nothing launched/merged/pushed.

Headline updates from that audit, superseding stale assumptions elsewhere in this
ledger and in earlier planning docs:

- **cap32 chunk is complete, not running.** Finished Fri 2026-06-19 00:31:13 EDT,
  exit 0, 49 rows. Aggregate: `evict_value_v1` loses to LRU on 6/7 trace families,
  ties on `wiki2018` (universal 100% miss rate there), -4.84% vs. LRU overall —
  consistent in direction with the validation pass (§6) and the validation-sanity
  data point (§9). cap64/cap128/cap256 confirmed **not** launched (`tmux
  list-sessions` shows the cap32 session idle at a shell prompt; `pgrep` shows no
  matching process running).
- **A previously unknown manuscript submission package exists.**
  `Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
  (repo root, added in commit `e9ac132 "Add files via upload"`, 2026-06-17
  22:08:24 EDT) contains a real LaTeX manuscript (`main.tex`, ~8,919 words,
  `elsarticle` class), `cover-letter.tex`, `author-agreement.tex`, and two
  embedded figures independently confirmed 300dpi via PIL. This corrects
  `kbs_submission_package_checklist.md`'s claim (parallel branch) that no
  cover letter / author agreement / manuscript source exists anywhere — that
  claim is false and the checklist needs updating before being trusted again.
- **New, concrete gap found**: `main.tex`'s embedded offline-ablation table
  (`tab:evict-value-ablation`) does not match the current heavy_r1 retrain at
  all (e.g. H=4/hist_gb val regret 0.0091 in the manuscript vs. 0.02100 in
  `analysis/evict_value_wulver_v1_model_comparison_heavy_r1.csv`, with a
  different best-model ranking). The zip predates the heavy_r1 training run.
  This table needs a refresh before resubmission, independent of the
  online-replay sweep.
- **SIEVE / FIFO-Reinsertion confirmed absent everywhere**, including the real
  manuscript bibliography (`refs.bib` inside the zip, 37 entries) and text —
  not just the analysis repo's planning docs. Clean, total gap, no prior work
  to build on.
- **AI-tool/single-authorship disclosure already exists** in `main.tex`'s "AI
  Declaration" section (names ChatGPT, Cursor, Codex, Copilot, Gemini,
  Perplexity, with an explicit author-responsibility statement) — the
  disclosure itself is not a gap; what may still be needed is response-letter
  framing using existing rigor evidence (test suite, audits, provenance
  ledger).
- **Highlights and CRediT statement are genuinely absent** in any format
  (confirmed by grep across all `.tex` files in the zip); Declaration of
  Competing Interest exists but only inline inside `main.tex`, not as a
  standalone file.
- **DOCX submission-package question remains unresolved** — the existing
  package is entirely LaTeX; whether KNOSYS Editorial Manager requires the
  main manuscript itself in Word format, or only certain ancillary files,
  cannot be determined from local files and needs the actual portal
  instructions for KNOSYS-D-26-07461.

See `reports/kbs_revision_completion_audit.md` for the full matrix and ranked
action lists; this entry is a pointer/summary, not a replacement for it.

## 16. Zero-compute consolidation pass (2026-06-19, post-cap32)

Read-only inspection + writing only; nothing launched, merged, pushed, or
overwritten. Seven new/updated reports produced:

| Report | New or updated | Headline finding |
|---|---|---|
| `reports/kbs_cap32_policy_comparison_report.md` | New | Full verification of the completed cap32 chunk: exit 0, 5:13:02 runtime, 49/49 rows, `evict_value_v1` loses to LRU on 6/7 trace families (-4.84% mean misses), ties on wiki2018 (100%-miss floor). Schema/dedup-checked as safe to merge later. `blind_oracle` reconfirmed diagnostic-only. |
| `reports/kbs_overhead_and_scalability_evidence.md` | New | Measured: dataset build ~10.3h/96G/662 shards (shard-mtime triangulated); training ~7-8min for 9 model fits. Code-verified: `evict_value_v1` is O(capacity) per miss (`evict_value_v1.py:179-200`), `lru` is O(1) (`lru.py:47`). 3 chunk-level wall-clock anchors recorded but explicitly flagged as confounded, not a controlled scaling curve. Proposed (not run) lightweight timing benchmark included. |
| `reports/kbs_stale_artifact_refresh_plan.md` | New | Root-caused the stale offline-ablation table to an exact commit boundary: committed `53726ce` (H=4/hist_gb) vs. current uncommitted working-tree retrain (H=4/random_forest). `table4`, `table5`, `figure4`, `figure5`, and `main.tex`'s embedded table all match the old committed numbers. Fix identified (`build_kbs_main_manuscript_artifacts.py` + manual `main.tex` paste) but **not executed** per instruction. |
| `reports/kbs_baseline_gap_action_plan.md` | New | Strict per-baseline classification (HALP/SIEVE/FIFO-Reinsertion/PARROT/Mockingjay/LRU/Marker/Predictive Marker/Trust-and-Doubt/REST). SIEVE recommended first (genuinely absent, low engineering risk). New finding: `rest_v1` (canonically evaluated, manuscript short-label "REST" per `build_kbs_main_manuscript_artifacts.py`'s `SHORT_POLICY_LABEL`) has no Table 2 row in `main.tex`; the similarly-named "R-FTP+Marker" row actually describes a different, separate, unwired policy (`robust_ftp_marker_combiner.py`'s `RobustFtPDeterministicMarkerCombiner`). Also found plain `Marker` and `R-FTP+Marker` are implemented and documented (`docs/baselines.md`) but wired only into an older runner (`src/lafc/runner/run_policy.py`), not the canonical heavy_r1 script — no heavy_r1 numbers exist for either. |
| `reports/kbs_complete_reviewer_comment_matrix.md` | New on `main` (a differently-structured, unreconciled draft of the same filename already existed on the unmerged `kbs-revision-parallel-cleanup` branch) | Restates, with an explicit top-of-file caveat, that verbatim AE/Reviewer #2/Reviewer #3 numbered text has never been supplied in any session — only a 12-13 item category-level paraphrase exists. Matrix built around that paraphrase (`AE item N` labels), with an explicit "Reviewer #2/#3 mapping: not determinable" column rather than a fabricated split. |
| `reports/kbs_response_to_reviewers_skeleton.md` | New on `main` (same parallel-branch-duplicate caveat as above) | 13-section draft response organized by the AE concern list, each with draft response language, current evidence pointers, and explicit blocked/ready status. Summary: §5 (partial)/§6/§12 have real, non-placeholder text ready today; §1/§2/§4/§7/§9 blocked on compute or a scope decision; §13 blocked on external information. |
| `reports/kbs_docx_submission_package_report.md` | New | Reconfirmed zero `.docx` files exist anywhere and no `submission_kbs_revision_docx` directory exists. The only real package (`Decision_aligned_...zip`) is 10 files, entirely LaTeX/bib/cls/bst/figures — no Highlights, no CRediT, no standalone Declaration of Interest, `cover-letter.tex` still framed for an initial (not revision) submission. Whether KNOSYS requires the main manuscript itself in DOCX vs. only ancillary files is unresolved — same class of external-information blocker as the missing reviewer letter. |

No new compute was run to produce any of the above — every number is either
read directly from an existing log/file/timestamp/git diff, or computed from
such numbers, or a direct source-code read. `reports/kbs_revision_gap_tracker.md`
and `reports/kbs_next_actions_without_rerun.md` updated in parallel with this
session to reflect cap32 completion and these seven reports.

## 17. SIEVE implementation pass (2026-06-19, post-baseline-gap-classification)

Directly responds to R3-Issue3/R3-Rec2 and acts on §15's "SIEVE / FIFO-
Reinsertion confirmed absent everywhere" finding and
`reports/kbs_before_cap64_baseline_decision_memo.md` §1's recommendation.
Gated, verify-then-implement pass: zero-compute research first
(`reports/kbs_sieve_source_verification.md`), then implementation, then one
authorized tiny smoke test only.

- **Verification**: the official NSDI'24 paper's Algorithm 1 (downloaded
  directly from `usenix.org`, read in full) and the official reference
  implementation (`libCacheSim/cache/eviction/Sieve.c`, fetched
  independently) were cross-checked line-for-line — no discrepancy found,
  no ambiguity in adapting to this repo's unweighted-paging simulator
  (Algorithm 1 is already object-count-based). Recorded in
  `reports/kbs_sieve_source_verification.md`.
- **Implementation**: new file `src/lafc/policies/sieve.py` (`SievePolicy`,
  CLI/registry name `sieve`), faithful to Algorithm 1 (one FIFO-ordered
  resident structure, one `visited` bit per page, one `hand` cursor; hits
  set `visited=True` only, no reordering; full-cache misses scan from the
  hand toward the head clearing visited bits, wrap past the head, evict the
  first unvisited object, update the hand to the victim's predecessor;
  insertions go to the head with `visited=False`). Wired into
  `scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES` dict (the
  canonical heavy_r1 pipeline) and `src/lafc/runner/run_policy.py`'s
  `POLICY_REGISTRY` (the older runner shared by most unit tests).
- **Tests**: `tests/test_sieve.py`, 8 new tests covering hit-without-move,
  head-insertion, eviction-skip-and-clear, eviction-evicts-first-unvisited,
  capacity-1 behavior, the shared `run_policy()` driver, and both registries
  accepting `sieve`. `pytest tests/ -k "sieve or policy_comparison or lru or
  marker" -v` → 45 passed, 0 failed. Full suite `pytest tests/ -q` → **291
  passed** (up from 283 in `AGENTS.md`; +8 matches the new SIEVE tests
  exactly), 0 failed, 12 pre-existing PuLP deprecation warnings only.
- **Smoke test** (the one authorized tiny compute action in this pass): 1
  trace (`brightkite`), 1000 requests, capacity 32, policies `lru,sieve`,
  written only to two new non-canonical files
  (`analysis/evict_value_wulver_v1_policy_comparison_sieve_smoke.csv/.md`).
  Result: SIEVE 496 mean misses vs. LRU 503 (-1.39%) — plausible,
  non-degenerate, consistent in direction with SIEVE's published
  competitiveness with LRU. No canonical output file was read from or
  written to.
- **Not done in this pass** (explicitly out of scope per instruction):
  cap64/cap128/cap256 remain **not launched**; cap32 was **not rerun** (it
  still has no SIEVE column and would need a full chunk rerun to get one,
  per §16's no-checkpointing finding); no `.bib`/`main.tex`/Table 2 edit was
  made (SIEVE is not yet discussed in the manuscript text); FIFO-
  Reinsertion's definition remains unresolved and untouched.
- Full detail, including the exact verbatim Algorithm 1 pseudocode quoted
  from the paper and the independent code cross-check: see
  `reports/kbs_sieve_source_verification.md` and
  `reports/kbs_sieve_implementation_report.md`.

This resolves the **SIEVE half** of blocker 6 in §14 at the implementation
level (code exists, tested, smoke-verified) — but blocker 6 as a whole
remains open: FIFO-Reinsertion's definition is still undecided, and SIEVE
still has no canonical (cap32/64/128/256) empirical numbers, which is what
blocker 3 in §14 still tracks.

---

## 18. cap32-with-SIEVE rerun preparation pass (2026-06-19, post-SIEVE-implementation)

Zero-compute planning pass, gated on §17's completed SIEVE implementation.
Goal: prepare (not launch) a SIEVE-inclusive rerun of the cap32 chunk, since
§17 explicitly left cap32 unrerun and SIEVE-less. Five sub-steps, all
read-only/planning except the two report edits noted below; no policy code
changed, no job launched.

1. **Naming-strategy decision**: Option B chosen over Option A — write new,
   distinctly-suffixed output files
   (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve.csv/.md`)
   rather than renaming the existing canonical-named no-SIEVE cap32 files.
   Reasoning and full file-by-file cross-reference audit in
   `reports/kbs_cap32_rerun_with_sieve_plan.md` §1 — renaming would require
   auditing/fixing stale references in at least 5 other files
   (`kbs_cap32_policy_comparison_report.md`, `kbs_revision_gap_tracker.md`,
   this ledger, `evidence_manifest.json`,
   `kbs_full_policy_comparison_execution_plan.md`); writing new files
   touches none of them.
2. **Pre-run inventory** (read-only commands, see `kbs_cap32_rerun_with_sieve_plan.md`
   §2): `git status --short` clean of surprises; existing no-SIEVE cap32
   files confirmed present and untouched (5.1K/1.8K, dated the original
   `Jun 19 00:31` run); SIEVE code files present
   (`src/lafc/policies/sieve.py` 6.4K, `tests/test_sieve.py` 5.0K); `grep
   -Rni "sieve"` across `src/lafc scripts tests reports` → 264 matches
   across 19 files (consistent wiring, no stray duplicate implementation);
   all 4 required pipeline inputs (trace manifest, model checkpoint,
   dataset manifest, split-summary) confirmed present with exact sizes.
3. **SIEVE re-verification** (lightweight only, per instruction — full
   suite not re-run): `pytest tests/test_sieve.py -v` → 8 passed;
   `pytest tests/ -k "sieve or policy_comparison" -v` → 8 passed, 283
   deselected (the `policy_comparison` half of the filter matches 0
   additional tests — no test function/file is literally *named* with that
   substring; this is a naming artifact of `-k`, not a coverage gap, since
   the canonical pipeline is exercised indirectly elsewhere per §17).
4. **`blind_oracle` inclusion decision**: **exclude** from the new
   SIEVE-inclusive policy list. Two independent confirmations: (a)
   `scripts/paper/build_kbs_main_manuscript_artifacts.py`'s
   `TABLE3_POLICIES` tuple never included `blind_oracle`; (b)
   `reports/kbs_validation_sanity_audit.md` Q3 already root-caused
   `blind_oracle`'s number as diagnostic-only (consumes
   `request.predicted_next`, which the canonical pipeline never populates).
   The user's own proposed policy list already excluded it — no change
   needed, decision just confirms it in writing.
5. **Exact tmux command drafted, not launched** — session name
   `kbs_full_policy_comparison_cap32_with_sieve`, policies
   `lru,sieve,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`,
   same `--max-requests-per-trace 50000`/`--capacities 32`/model checkpoint
   as the original cap32 run, output to the new `*_with_sieve.csv/.md`
   paths, log to `logs/kbs_full_policy_comparison/cap32_with_sieve.log`.
   Full command text in `kbs_cap32_rerun_with_sieve_plan.md` §5. **Not
   executed** — requires explicit approval. Estimated runtime ~5h (4.5-6h
   band), extrapolated from the original cap32 run's measured 5:13:02 (same
   7-policy count, same O(1)-class swap of `blind_oracle`↔`sieve`).

**Files touched this pass**: `reports/kbs_cap32_rerun_with_sieve_plan.md`
(new), `reports/kbs_cap32_policy_comparison_report.md` (status banner
added — old chunk tagged `CANONICAL CHUNK WITHOUT SIEVE / SUPERSEDED FOR
FINAL TABLE IF SIEVE-INCLUSIVE CHUNKS ARE USED`),
`reports/kbs_before_cap64_baseline_decision_memo.md` (third update note),
`reports/kbs_baseline_gap_action_plan.md` (second update note),
`reports/kbs_revision_gap_tracker.md` (cap32 row re-tagged, new
cap32-with-SIEVE row added, tagged `PENDING / CANONICAL CHUNK WITH SIEVE`),
`reports/kbs_next_actions_without_rerun.md` (new checklist item, marked
done for planning/not-done for the actual launch), this ledger (§18,
`evidence_manifest.json` (entries appended, see manifest's own `notes`
field for the exact batch description).

**Not done in this pass** (explicitly out of scope, same as §17 and per
instruction): the cap32-with-SIEVE rerun itself was **not launched**;
cap64/cap128/cap256 remain **not launched**; no existing artifact was
renamed, overwritten, or deleted; no push/merge performed.

---

## 19. Zero-compute reviewer-response package while cap32_with_sieve runs (2026-06-19, later pass)

This later same-day pass was constrained to:

- monitor `kbs_full_policy_comparison_cap32_with_sieve` read-only only
- avoid launching `cap64`, `cap128`, `cap256`, or any other multi-hour job
- do reviewer-response/manuscript-planning work only

### 19.1 Read-only job status captured

- `tmux ls` shows `kbs_full_policy_comparison_cap32_with_sieve`
- `pgrep -af "cap32_with_sieve|run_policy_comparison_wulver_v1|python"`
  shows the active `run_policy_comparison_wulver_v1.py` process
- `logs/kbs_full_policy_comparison/cap32_with_sieve.log` is still empty at
  the time of the check
- no `CAP32_WITH_SIEVE_EXIT=` marker exists yet

Interpretation: the job is still running; this pass did not touch it.

### 19.2 New evidence/planning outputs

- `reports/kbs_parallel_reviewer_work/non_compute_reviewer_work_package.md`
  — pass-local summary
- `reports/kbs_halp_fifo_source_verification.md`
  — HALP primary-source recheck, FIFO-Reinsertion definition recheck,
  conservative recommendation text
- `reports/kbs_fallback_revision_strategy.md`
  — drafted demote-vs-validate fallback response
- `reports/kbs_horizon_h4_revision_strategy.md`
  — drafted H=4 justification using existing offline metrics only
- `reports/kbs_overhead_manuscript_text_draft.md`
  — drafted manuscript/rebuttal overhead text
- `reports/kbs_manuscript_shortening_and_reframing_plan.md`
  — drafted shortening and reframing plan
- `reports/kbs_docx_package_action_plan.md`
  — drafted DOCX/package plan

### 19.3 Tracker/skeleton updates

- `reports/kbs_response_to_reviewers_skeleton.md`
- `reports/kbs_complete_reviewer_comment_matrix.md`
- `reports/kbs_revision_gap_tracker.md`
- `reports/kbs_revision_completion_audit.md`
- `reports/kbs_next_actions_without_rerun.md`
- this ledger
- `reports/kbs_revision_evidence/evidence_manifest.json`

### 19.4 Main conclusions of this pass

1. HALP is now source-verified against the NSDI'23 paper and should still
   be treated as citation+differentiation, not a realistic empirical
   reimplementation before 2026-07-08.
2. FIFO-Reinsertion is now source-verified as the CLOCK / Second-Chance
   family baseline used as SIEVE's direct contrast in the NSDI'24/S3-FIFO
   literature; the remaining issue is canonical run budget, not algorithm
   definition.
3. The fallback mechanism should be demoted unless new validation is
   deliberately run later.
4. The H=4 explanation, overhead text, shortening plan, and DOCX action
   plan now all exist in drafted form and can feed the reviewer letter or
   manuscript rewrite without any new heavy compute.

**Not done in this pass**: no experiment code edited, no running job
interrupted, no further capacity chunk launched, no artifact overwritten or
deleted, no push/merge performed.

---

## 20. FIFO-Reinsertion readiness audit while cap32_with_sieve runs (2026-06-19, still later pass)

This pass was still constrained to leave `cap32_with_sieve` untouched. It
audited the already-present FIFO-Reinsertion baseline and did only
lightweight validation.

### 20.1 What was confirmed

- `src/lafc/policies/fifo_reinsertion.py` exists and implements a
  single-queue FIFO + visited-bit algorithm that:
  - sets `visited=True` on hit without reordering
  - on full-cache miss, repeatedly clears and reinserts visited tail items
    at the head
  - evicts the first unvisited tail item found
- this is wired into:
  - `scripts/run_policy_comparison_wulver_v1.py`
  - `src/lafc/runner/run_policy.py`
- dedicated tests already exist at `tests/test_fifo_reinsertion.py`
- the current manuscript-facing policy roster files still do **not** include
  either `sieve` or `fifo_reinsertion`

### 20.2 Source-verification judgment

Using `reports/kbs_halp_fifo_source_verification.md` plus direct source
inspection, the current implementation is scientifically defensible as
FIFO-Reinsertion in the CLOCK / Second-Chance family, not a mislabeled
proxy. The critical behavioral distinction from SIEVE is retained-object
reinsertion to the head versus in-place retention behind a moving hand.

### 20.3 Lightweight validation run in this pass

1. Requested pytest slice run as written with bare `pytest`:
   - failed at collection with `ModuleNotFoundError: No module named 'lafc'`
   - this is an environment/import-path issue, not a policy failure
2. Same requested slice run in the project venv:
   ```bash
   .venv_kbs_heavy_r1/bin/pytest tests/ -k "fifo or reinsertion or second or clock or sieve or policy" -v
   ```
   - **39 passed, 260 deselected**
3. Fresh tiny smoke run with separate output filenames only:
   ```bash
   .venv_kbs_heavy_r1/bin/python scripts/run_policy_comparison_wulver_v1.py \
     --trace-manifest analysis/wulver_trace_manifest_full.csv \
     --capacities 32 --max-traces 1 --max-requests-per-trace 1000 \
     --policies lru,sieve,fifo_reinsertion \
     --out-csv analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke_audit.csv \
     --out-md analysis/evict_value_wulver_v1_policy_comparison_fifo_reinsertion_smoke_audit.md
   ```
   Result: LRU 503, SIEVE 496, FIFO-Reinsertion 504 on the one brightkite
   sample.

### 20.4 Decision consequence

The new `reports/kbs_fifo_reinsertion_baseline_audit.md` and
`reports/kbs_final_baseline_set_decision_before_cap64.md` conclude:

- FIFO-Reinsertion should be included in the final canonical policy set
- if that happens, the currently running `cap32_with_sieve` chunk will not
  be sufficient by itself, because it omits FIFO-Reinsertion
- cap32 will need a later rerun with both SIEVE and FIFO-Reinsertion if the
  final manuscript-citable list includes both

## 21. Approved stop of obsolete cap32_with_sieve and launch of corrected cap32_with_sieve_fifo (2026-06-19)

This pass changed from read-only planning to an approved execution step.
The user explicitly authorized stopping the obsolete
`cap32_with_sieve` tmux job because it omitted `fifo_reinsertion`, then
launching only the corrected FIFO-inclusive cap32 chunk on the current
local/cloud machine via `tmux`.

### 21.1 Recorded pre-stop state

Status was recorded to:

- `logs/kbs_full_policy_comparison/cap32_with_sieve_aborted_status.txt`

Key observed facts:

- tmux session `kbs_full_policy_comparison_cap32_with_sieve` existed
- runner PID `191758` was active with policy list
  `lru,sieve,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`
- tee PID `191759` was active
- no `analysis/..._cap32_with_sieve.csv/.md` outputs existed yet
- `logs/kbs_full_policy_comparison/cap32_with_sieve.log` was still empty

### 21.2 Stop sequence and verification

Actions taken:

1. sent `C-c` to tmux session
   `kbs_full_policy_comparison_cap32_with_sieve`
2. verified that no real
   `python scripts/run_policy_comparison_wulver_v1.py` process remained
3. killed the now-idle obsolete tmux session shell
4. appended
   `CAP32_WITH_SIEVE_ABORTED_BECAUSE_FIFO_REINSERTION_NOW_REQUIRED=1`
   to the status log

Important note: the temporary cleanup shell's own command line contained the
substring `cap32_with_sieve`, so an intermediate fallback `pgrep | grep`
sequence matched the cleanup shell itself. This did **not** terminate any
unrelated Python job. Final verification showed the obsolete policy-
comparison runner was gone, and only the expected non-policy Python
processes remained.

### 21.3 Corrected preflight

Before launch, the pass confirmed:

- corrected outputs
  `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md`
  did not already exist
- required inputs existed:
  `analysis/wulver_trace_manifest_full.csv`,
  `models/evict_value_wulver_v1_best_heavy_r1.pkl`,
  `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json`
- `src/lafc/policies/sieve.py`,
  `src/lafc/policies/fifo_reinsertion.py`,
  `tests/test_sieve.py`,
  `tests/test_fifo_reinsertion.py`
  all existed
- targeted tests passed in the project venv:

```bash
pytest tests/test_sieve.py tests/test_fifo_reinsertion.py -v
```

Result: `16 passed in 0.11s`.

### 21.4 Corrected launch

New tmux session:

- `kbs_full_policy_comparison_cap32_with_sieve_fifo`

Launch command:

```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 32 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap32_with_sieve_fifo.log
```

Followed in tmux by:

- `echo "CAP32_WITH_SIEVE_FIFO_EXIT=${PIPESTATUS[0]}"`
- `date | tee logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_finished_at.txt`

### 21.5 Immediate post-launch verification

Verified immediately after launch:

- tmux session exists: `kbs_full_policy_comparison_cap32_with_sieve_fifo`
- runner PID `198488`
- tee PID `198489`
- start marker exists:
  `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_started_at.txt`
- recorded start time:
  `Fri Jun 19 02:42:27 PM EDT 2026`
- log exists but is still size 0 at the earliest verification point

### 21.6 Artifacts created/updated in this pass

New:

- `reports/kbs_cap32_with_sieve_aborted_superseded_report.md`
- `reports/kbs_cap32_with_sieve_fifo_launch_report.md`
- `logs/kbs_full_policy_comparison/cap32_with_sieve_aborted_status.txt`
- `logs/kbs_full_policy_comparison/cap32_with_sieve_fifo_started_at.txt`

Updated:

- `reports/kbs_final_baseline_set_decision_before_cap64.md`
- `reports/kbs_fifo_reinsertion_baseline_audit.md`
- `reports/kbs_cap32_rerun_with_sieve_plan.md`
- `reports/kbs_before_cap64_baseline_decision_memo.md`
- `reports/kbs_baseline_gap_action_plan.md`
- `reports/kbs_revision_gap_tracker.md`
- `reports/kbs_next_actions_without_rerun.md`
- this ledger
- manifest + reviewer skeleton updated in the same pass

## 22. Results-independent manuscript/rebuttal drafting pass (2026-06-19, while `cap32_with_sieve_fifo` runs)

### 22.1 Scope and safety

This pass was a read-only status check of the running
`cap32_with_sieve_fifo` job (tmux session
`kbs_full_policy_comparison_cap32_with_sieve_fifo`, runner PID `198488`,
tee PID `198489`, still running, no exit marker, no output CSV/MD yet at
the time of this pass) followed by drafting work only. It did not touch,
stop, restart, or interrupt that job, and did not launch
cap64/cap128/cap256 or any other canonical experiment.

### 22.2 New artifacts

All under `reports/kbs_manuscript_rebuttal_drafts/`:

- `started_at.txt`
- `response_paragraph_bank.md` — 11 rebuttal-letter paragraphs (AE;
  R2-MC1, R2-MC2, R2-MC3; R3-Issue2, R3-Issue3, R3-Issue5, R3-Issue6,
  R3-Issue7; R3-Minor8, R3-Minor9), each grounded in
  `reports/kbs_real_reviewer_comments.md` and the existing evidence
  reports cited throughout this ledger.
- `manuscript_insertions_draft.md` — manuscript-body draft text for 8
  named subsections.
- `title_and_contribution_reframing_options.md` — 3 title options keyed to
  3 possible canonical-sweep outcomes, plus 5 revised contribution
  bullets.
- `docx_package_text_bank.md` — cover letter, Highlights, CRediT
  statement, Declaration of Competing Interest, and AI-tool disclosure
  text, cross-checked against the real existing text inside
  `Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`'s
  `main.tex` (AI disclosure ~line 540, Declaration of Competing Interest
  ~lines 553-555) and `cover-letter.tex` (full text read via `unzip -p`,
  confirmed written for the *initial* submission, not this revision).

### 22.3 Hard constraints respected

- No paragraph or subsection in any of the four files asserts that
  `evict_value_v1` outperforms LRU, SIEVE, or FIFO-Reinsertion.
- Every claim depending on cap32-with-SIEVE-and-FIFO-Reinsertion,
  cap64, cap128, or cap256 is an explicit `[INSERT FINAL CANONICAL
  RESULT: ...]` or `[INSERT ...]` placeholder, never a fabricated number.
- The one completed canonical data point (cap32, no SIEVE/FIFO:
  `evict_value_v1` loses to LRU on 6/7 trace families, -4.84% mean misses)
  is stated directly and is treated as the current ground truth throughout.
- The fallback mechanism is treated as unvalidated (per
  `reports/kbs_fallback_revision_strategy.md`) in every file that mentions
  it; none of the four drafts present it as a validated contribution.

### 22.4 Tracking files updated in this pass

- `reports/kbs_response_to_reviewers_skeleton.md`
- `reports/kbs_complete_reviewer_comment_matrix.md`
- `reports/kbs_revision_gap_tracker.md`
- `reports/kbs_revision_completion_audit.md` (new §J)
- `reports/kbs_next_actions_without_rerun.md`
- this ledger (this section)
- `reports/kbs_revision_evidence/evidence_manifest.json`

All marked the four new drafting files as **prepared, not final** rather
than changing any underlying status tag (e.g. `[PENDING
CAP64/CAP128/CAP256]` items remain pending; only a drafting resource now
exists alongside them).

---

## 23. cap32_with_sieve_fifo completion, result analysis, and after-cap32 decision memo (2026-06-19)

### 23.1 Job completion

The corrected canonical cap32 chunk (`cap32_with_sieve_fifo`) finished on the
current local/cloud machine (not Wulver, no Slurm):

| Field | Value |
|-------|-------|
| Exit | `CAP32_WITH_SIEVE_FIFO_EXIT=0` |
| Started | Fri Jun 19 14:42:27 EDT 2026 |
| Finished | Fri Jun 19 19:58:48 EDT 2026 |
| Wall-clock | ~5h 16m |
| tmux session | `kbs_full_policy_comparison_cap32_with_sieve_fifo` (now idle) |
| CSV | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv` (5.8K, 57 lines) |
| MD | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.md` |
| Coverage | 7 traces × cap32 × 8 policies = 56 rows |

### 23.2 Result summary (honest)

Aggregate mean misses (7 families):

| Policy | Mean misses | vs LRU |
|--------|-------------|--------|
| lru / rest_v1 | 34,667 | baseline |
| fifo_reinsertion | 34,688 | −0.06% |
| sieve | 35,314 | −1.86% |
| evict_value_v1 | 36,344 | **−4.84%** |

- `evict_value_v1` loses on 6/7 families; ties wiki2018 (degenerate all-miss floor)
- Loses to SIEVE (−2.92%) and FIFO-Reinsertion (−4.77%) on aggregate
- Core common-policy rows (lru, rest_v1, evict_value_v1, etc.) match prior no-SIEVE cap32 exactly

Full analysis: `reports/kbs_cap32_with_sieve_fifo_result_analysis.md`

### 23.3 After-cap32 decision

Decision memo: `reports/kbs_after_cap32_decision_memo.md`

**Recommendation: Option C** — run cap64 only (~5–6h) before deciding on
cap128/256. Parallel reframing strongly indicated. **cap64 not launched** in
this pass.

### 23.4 Tracking files updated

- `reports/kbs_cap32_with_sieve_fifo_result_analysis.md` (new)
- `reports/kbs_after_cap32_decision_memo.md` (new)
- `reports/kbs_response_to_reviewers_skeleton.md`
- `reports/kbs_complete_reviewer_comment_matrix.md`
- `reports/kbs_revision_gap_tracker.md`
- `reports/kbs_revision_completion_audit.md` (§K)
- `reports/kbs_next_actions_without_rerun.md`
- this ledger (§23)
- `reports/kbs_revision_evidence/evidence_manifest.json`

---

## 24. cap64_with_sieve_fifo launch (2026-06-19, Option C approved)

### 24.1 Decision

The user approved Option C from §23.3's decision memo
(`reports/kbs_after_cap32_decision_memo.md`): launch cap64 only, using the
**same final 8-policy baseline set** as the completed `cap32_with_sieve_fifo`
chunk, and defer the cap128/cap256 decision until cap64 results are
reviewed. Explicit instruction: do not launch cap128 or cap256; do not push,
merge, delete, overwrite, or rename existing artifacts; run on the
local/cloud machine (not Wulver, no Slurm) in tmux so SSH/network
disconnects cannot interrupt the run.

### 24.2 Pre-flight checks (all passed)

| Check | Result |
|---|---|
| Branch | `main` |
| `git status --short` | reviewed, no destructive state |
| `analysis/wulver_trace_manifest_full.csv` | present |
| `models/evict_value_wulver_v1_best_heavy_r1.pkl` | present (13M) |
| `data/derived/evict_value_v1_wulver_heavy_r1/manifest.json` | present (104K) |
| `src/lafc/policies/sieve.py` | present |
| `src/lafc/policies/fifo_reinsertion.py` | present |
| `pytest tests/test_sieve.py tests/test_fifo_reinsertion.py -v` | 16/16 passed |
| Disk space | 499G available on `/` |
| Memory | 59Gi available |
| Load average | 0.18, 0.07, 0.02 (idle) |
| cap64 output paths confirmed absent | `..._cap64_with_sieve_fifo.csv` / `.md` — both absent |

### 24.3 Launch

| Field | Value |
|---|---|
| tmux session | `kbs_full_policy_comparison_cap64_with_sieve_fifo` |
| Runner PID | `261643` |
| tee PID | `261644` |
| Started | Fri Jun 19 22:29:23 EDT 2026 |
| Policy list | `lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1` |
| Capacity | 64 only |
| Outputs (pending) | `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv/.md` |
| Log | `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log` |
| Expected runtime | ~5-7h, by analogy to cap32_with_sieve_fifo's actual 5h16m |

Exact command — identical to the `cap32_with_sieve_fifo` command in §21
with only `--capacities 32` → `--capacities 64` and matching output
filenames changed:

```bash
python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv \
  --capacities 64 \
  --max-requests-per-trace 50000 \
  --policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1 \
  --evict-value-model models/evict_value_wulver_v1_best_heavy_r1.pkl \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.md \
  2>&1 | tee logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log
```

### 24.4 Verification

- `tmux ls` confirms session `kbs_full_policy_comparison_cap64_with_sieve_fifo` exists
- `pgrep -af` confirms runner PID `261643` and tee PID `261644` alive
- `ps -p 261643` confirmed alive ~30s after launch
- Log file exists (size 0 at verification time — same early-stage behavior
  observed for `cap32_with_sieve_fifo` at its own launch verification)
- `cap128`/`cap256`: confirmed **not** launched (no tmux session, no process)

### 24.5 New report

- `reports/kbs_cap64_with_sieve_fifo_launch_report.md` — exact command,
  tmux session name, start timestamp, expected runtime, policy list, output
  filenames, safety checks, and rationale for cap64-only (not
  cap128/cap256).

### 24.6 Tracking files updated

- `reports/kbs_revision_gap_tracker.md` (new cap64 row + cross-references)
- `reports/kbs_complete_reviewer_comment_matrix.md` (R2-MC3, R3-Issue1/3/4,
  R3-Minor9, R3-Rec1/2/8)
- `reports/kbs_response_to_reviewers_skeleton.md` (header + R2-MC3,
  R3-Issue1/3/4, R3-Minor9, Rec1/2, status-summary note)
- `reports/kbs_next_actions_without_rerun.md` (cap64 decision items marked
  done/approved)
- `reports/kbs_revision_completion_audit.md` (§L new; experiments/data table
  row updated)
- this ledger (§24)
- `reports/kbs_revision_evidence/evidence_manifest.json`

cap128/cap256 remain **not launched**, per explicit instruction. Nothing was
pushed, merged, deleted, overwritten, or renamed.

---

## 25. Manuscript consistency audit (2026-06-19, zero-compute, while `cap64_with_sieve_fifo` runs)

### 25.1 Scope and constraints

Read-only audit of the actual submission package against current
repository state. Explicit constraints honored throughout: not Wulver, no
Slurm/sbatch/squeue/sacct; `cap64_with_sieve_fifo` not touched, stopped, or
restarted; no cap128/cap256 launch; no push/merge/delete/overwrite of
existing artifacts; analysis and report-writing only.

### 25.2 Read-only job status check (Step 1)

| Check | Result |
|---|---|
| `date` | Fri Jun 19 10:51:12 PM EDT 2026 (audit time) |
| `tmux ls` | `kbs_full_policy_comparison_cap64_with_sieve_fifo` present, 1 window |
| `pgrep -af run_policy_comparison_wulver_v1` | PID `261643` alive, same command as §24.3 |
| Log size | `logs/kbs_full_policy_comparison/cap64_with_sieve_fifo.log` — 0 bytes |
| Exit-marker grep | no match (job not finished) |

No action taken on the job.

### 25.3 Locating manuscript/package sources (Step 2)

`find . -maxdepth 5` over `.tex/.bib/.cls/.docx/.zip` and cover/highlight/
credit/interest/author-agreement filenames confirmed: no loose manuscript
files exist anywhere in the repo outside
`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
(repo root). Zip contents (`unzip -l`): `main.tex` (70153B/559 lines),
`refs.bib` (14399B), `cover-letter.tex` (3903B), `author-agreement.tex`
(1651B), `elsarticle.cls` + 3 `.bst` files, `figures/method_overview.png`,
`figures/figure4_ablation.png`. No DOCX anywhere. Extracted read-only to
`/tmp/kbs_manuscript_audit_readonly` (outside the repo, scratch directory,
not committed, original zip untouched).

### 25.4 Claim/keyword audit (Step 3)

Full read of `main.tex` plus targeted greps for: `miss ratio`, `hit rate`,
`outperform`, `improve`, `robust`, `superior`, `beats`, `better than`,
`baseline`, `SIEVE`, `FIFO`, `HALP`, `H=4`, `horizon`, `fallback`, `guard`,
`overhead`, `O(`, `runtime`, `scalability`, `Top-1`, `regret`, `LRU`,
`Table`, `Figure`. Key results:

- `superior`/`superiority`: 4 occurrences, all self-disclaiming.
- `outperform`: 2 occurrences, neither claims `evict_value_v1` outperforms
  anything online; one is "we do not claim... universally outperforms,"
  the other refers to tree models beating ridge in the *offline* ablation.
- `SIEVE`/`FIFO`: **0 occurrences anywhere in `main.tex`**, despite both
  being implemented, tested, and now results-backed (cap32 done, cap64
  running). Same null result in `tables/manuscript/table6_related_work_learned_caching.tex`.
  `refs.bib` confirmed missing a `zhang2024sieve` entry.
- `overhead`/`O(`/`scalability`: **0 occurrences anywhere in `main.tex`.**
- `H=4`: 2 occurrences (lines 419, 462), both scoped to "the current
  artifact set" / "the evaluated settings" — not framed as general.
- `fallback`: 38 occurrences; `guard`: 22. Read in full: Introduction
  Contribution #3 (line 75) states the mechanism in confident,
  completed-capability language; Discussion (line 466) and Limitations
  (lines 507, 511) describe the same mechanism as unvalidated/heuristic.
  This asymmetry is the one real claim/hedge inconsistency found.
- Cross-checked `tables/manuscript/table2_policy_roster.csv` (6 entries,
  has REST, no Marker/R-FTP/SIEVE/FIFO) against `main.tex`'s embedded
  Table 2 (7 entries, has Marker/R-FTP+Marker, no REST/SIEVE/FIFO) against
  the actual canonical 8-policy `--policies` CLI list
  (`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`).
  All three disagree.
- Cross-checked `tables/manuscript/table4_main_ablation.csv` against
  `main.tex`'s embedded `tab:evict-value-ablation` — both reflect the same
  stale commit-`53726ce`-boundary numbers, per
  `reports/kbs_stale_artifact_refresh_plan.md` (already known, reconfirmed
  here against the actual manuscript table rather than just the repo
  artifact).
- Confirmed via `reports/manuscript_artifacts/end_to_end_evidence_gap_report.md`
  that the online-results Table 3 / Figs. 2-3 have never existed (gated on
  a canonical multi-capacity CSV that has never been built) — `main.tex`
  has no online-replay table or figure in any form, stale or otherwise.
- Confirmed Figures 5/6/7/8 in `figures/manuscript/` are not referenced by
  `\includegraphics` anywhere in `main.tex` and are not part of the
  submission zip (zip has only `method_overview.png` and
  `figure4_ablation.png`) — dormant, not a manuscript inconsistency.

### 25.5 Deliverable (Step 4)

`reports/kbs_manuscript_consistency_audit.md` — 8 sections per the
specified structure (executive summary; claims to remove/soften; claims
that can remain; tables/figures requiring regeneration or net-new
creation; per-section rewrite classification using exactly `SAFE NOW` /
`REWRITE NOW` / `WAIT FOR CAP64` / `WAIT FOR FINAL SWEEP` /
`REMOVE/DEMOTE`; reviewer-comment coverage mapped to real AE/R2/R3 labels;
recommended immediate zero-compute edits; edits that must wait on
cap64/cap128/256).

### 25.6 Tracking files updated (Step 5)

- `reports/kbs_revision_gap_tracker.md` (new row)
- `reports/kbs_revision_completion_audit.md` (§M new; one table-row note
  updated)
- `reports/kbs_next_actions_without_rerun.md` (new checked item)
- `reports/kbs_response_to_reviewers_skeleton.md` (cross-check addendum;
  no status tags changed — audit confirmed existing tags are accurate)
- this ledger (§25)
- `reports/kbs_revision_evidence/evidence_manifest.json`

The manuscript rewrite itself was **not** performed in this pass — only
the audit and tracker updates. `cap64_with_sieve_fifo` confirmed still
running/untouched (§25.2). cap128/cap256 remain not launched. Nothing was
pushed, merged, deleted, or overwritten.

---

## 26. Safe manuscript-source edits applied (2026-06-19, zero-compute, while `cap64_with_sieve_fifo` runs)

### 26.1 Scope and constraints

Direct continuation of §25's audit: applied the audit's recommended
zero-compute fixes to the actual manuscript source, not just internal
reports. Explicit constraints honored throughout: not Wulver, no
Slurm/sbatch/squeue/sacct; `cap64_with_sieve_fifo` not touched, stopped, or
restarted; no cap128/cap256 launch; no push/merge/delete/overwrite of
existing artifacts; edits restricted to content independent of final
numeric results (no online-results sections/tables/figures touched).

### 26.2 Read-only job status check (Step 1)

| Check | Result |
|---|---|
| `tmux ls` | `kbs_full_policy_comparison_cap64_with_sieve_fifo` present |
| `pgrep -af` | runner PID `261643` alive, same command as §24.3 |
| Exit-marker grep | no match (job not finished) |

No action taken on the job at the start of this pass.

### 26.3 Durable working copy

The submission zip
(`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`,
repo root) was extracted **read-write** into a new, durable, repo-tracked
directory, `manuscript_source/` (untracked `??` in git, not deleted, not
overwriting the original zip). All edits in this section were made to the
files inside `manuscript_source/`, not to the zip itself.

### 26.4 Edits applied

- **Related Work**: added a SIEVE/FIFO-Reinsertion paragraph (CLOCK/
  Second-Chance family discussion) and added a `zhang2024sieve` (NSDI'24)
  BibTeX entry to `manuscript_source/refs.bib`. Verified the existing
  `song2023halp` citation/discussion was already adequate; deliberately
  did **not** add an S3-FIFO (`yang2023s3fifo`) citation since it is not
  evaluated anywhere in this work.
- **Fallback/guard demotion**: revised Contribution #3 and method-section
  wording in 5 locations to describe the mechanism as an "implementation
  safeguard" / "optional guard," evaluated only as an ablation/future
  robustness mechanism — never as a validated miss-ratio improvement.
- **Policy roster reconciliation**: fixed three previously-disagreeing
  policy lists (`main.tex`'s embedded Table 2; the embedded Related-Work
  table; `tables/manuscript/table2_policy_roster.csv/.tex`) to the
  canonical 8-policy set
  (`lru, sieve, fifo_reinsertion, predictive_marker,
  blind_oracle_lru_combiner, trust_and_doubt, rest_v1, evict_value_v1`),
  excluding the diagnostic-only `blind_oracle`. This fixed a
  previously-undiscovered total absence of `rest_v1`/REST from
  `main.tex`'s prose (§25.4 had only flagged the table-level mismatch, not
  this deeper omission). Regenerated
  `tables/manuscript/table2_policy_roster.csv/.tex` and
  `reports/manuscript_artifacts/latex_snippets/table2_snippet.tex` via a
  matching fix to `_build_table2_policy_roster()` inside
  `scripts/paper/build_kbs_main_manuscript_artifacts.py`, so future
  regenerations from the generator script will not revert the fix.
- **Overhead/horizon placeholders**: inserted two new Discussion
  subsections — "Overhead and Scalability" (dataset-build cost ~10.3h/
  96GB/662 shards; training cost ~7-8 min; code-verified O(k)-per-miss vs.
  O(1) complexity claim; explicit `% TODO(revision)` noting no controlled
  timing benchmark exists yet) and "Replay Horizon Selection" (H=4 finding
  holds across 5 of 7 trace families — BrightKite, CloudPhysics, MetaCDN,
  Twemcache, Wikimedia pageviews; explicit `% TODO(revision)` disclosing
  that Citi Bike/MetaKV coverage and horizon×capacity interaction remain
  open). Neither subsection inserts a final measured claim beyond what was
  already source-backed.
- **Left untouched**: the Conclusions section; Table 3 (does not exist
  anywhere, per §25.4's confirmation); all cap64/128/256-dependent claims;
  Tables 3-6 and Figures (beyond the Table 2 roster fix); the cover letter
  and author agreement.

### 26.5 Verification (Step 5)

Two full `tectonic` compiles of `manuscript_source/main.tex`, both exit 0,
zero undefined references, scratch output under
`/tmp/kbs_manuscript_compile_check*` (outside the repo, not committed).
`git diff --stat` and targeted greps for SIEVE/FIFO/HALP/fallback/guard
confirmed the edits landed as described and nothing else changed.

### 26.6 Deliverable (Step 6)

`reports/kbs_safe_manuscript_source_edits_report.md` — full account of
where the manuscript source now lives, files edited, exact nature of each
edit, citations added/verified, fallback demotion wording, policy roster
reconciliation, sections deliberately left untouched, checks run, and
remaining risks/TODOs (S3-FIFO citation deferred, two `% TODO(revision)`
comments, the two-copies-of-manuscript risk — zip vs. `manuscript_source/`
— documented and mitigated by treating `manuscript_source/` as the single
durable working copy going forward).

### 26.7 Tracking files updated (Step 7)

- `reports/kbs_revision_gap_tracker.md` (new row)
- `reports/kbs_revision_completion_audit.md` (§N new)
- `reports/kbs_next_actions_without_rerun.md` (3 checklist items marked
  done/updated)
- `reports/kbs_response_to_reviewers_skeleton.md` (R2-MC1/R2-MC2,
  R3-Issue3/Rec2, R3-Issue6/Rec5, R3-Rec3/Rec7 notes updated — no status
  tag moved to `[DONE]`, since cap64/128/256 numbers and the fallback
  validate-or-remove decision both remain pending)
- this ledger (§26)
- `reports/kbs_revision_evidence/evidence_manifest.json`

The manuscript rewrite applied here is still results-independent —
nothing claims a final cap64/128/256 outcome, and no online-results
section/table/figure was touched. `cap64_with_sieve_fifo` confirmed still
running/untouched both at the start and end of this pass (re-checked via
`tmux ls`/`pgrep`/exit-marker grep). cap128/cap256 remain not launched.
Nothing was pushed, merged, deleted, or overwritten.

---

## 27. Submission-package skeletons created (2026-06-19, zero-compute, while `cap64_with_sieve_fifo` runs)

### 27.1 Scope and constraints

Direct continuation of §26: prepared draft submission-package skeleton
files for the 6 non-manuscript items required for a KBS revision
submission, while leaving the revised manuscript DOCX (item 1) untouched
and still blocked. Explicit constraints honored throughout: not Wulver, no
Slurm/sbatch/squeue/sacct; `cap64_with_sieve_fifo` not touched, stopped, or
restarted; no cap128/cap256 launch; no push/merge/delete/overwrite of
existing artifacts; draft/skeleton creation only.

### 27.2 Read-only job status check (Step 1)

| Check | Result |
|---|---|
| `tmux ls` | `kbs_full_policy_comparison_cap64_with_sieve_fifo` present |
| `pgrep -af` | runner PID `261643` alive, `ps -p 261643` shows state `R`, elapsed ~1h05m at check time |
| Exit-marker grep | no match (job not finished) |
| Output CSV/MD | absent |

No action taken on the job.

### 27.3 Pre-existing groundwork reused

Before creating anything new, read the existing groundwork to avoid
duplicating or contradicting it: `submission_kbs_revision_docx/README.md`
and `README_next_steps.md` (placeholder directory, already listed the 7
required files); `reports/kbs_docx_submission_package_report.md`
(read-only inspection — zero DOCX files existed anywhere, confirmed again);
`reports/kbs_docx_package_action_plan.md` (classified items 5/6/7 as
draftable today with no blocker, item 3 as nearly unblocked, items 2/4 as
partially blocked, item 1 as fully blocked; ran a `pandoc` fidelity probe
in `/tmp` that found equations/`algorithmicx`/cross-references/tables all
fail to convert cleanly for the *main manuscript*, but did not flag any
risk for the short prose-only ancillary documents);
`reports/kbs_manuscript_rebuttal_drafts/docx_package_text_bank.md`
(already-drafted text for the cover letter, Highlights, CRediT statement,
and Declaration of Interest, cross-checked against the real inline text in
`manuscript_source/main.tex`); `manuscript_source/cover-letter.tex` and
`author-agreement.tex` (the existing initial-submission-era LaTeX source);
and `reports/kbs_real_reviewer_comments.md` (the authoritative AE/R2/R3
verbatim text and labels).

### 27.4 New files created

All under `submission_kbs_revision_docx/`, each as a markdown source plus
a `pandoc`-rendered `.docx`:

| File | Placeholders | Notes |
|---|---|---|
| `response_to_reviewers_skeleton.md/.docx` | 16× `[INSERT FINAL CANONICAL RESULT AFTER CAP64/FINAL SWEEP]` | Letter-formatted draft using the real AE/R2-MC1-3/R3-Issue1-7/R3-Minor8-9/R3-Rec1-8 labels; explicitly defers to `reports/kbs_response_to_reviewers_skeleton.md` as the authoritative status tracker |
| `cover_letter_skeleton.md/.docx` | 2× same placeholder | Reframes `manuscript_source/cover-letter.tex` from an initial-submission pitch to a revision letter |
| `highlights_skeleton.md/.docx` | 1× conditional placeholder | 5 draft bullets, none overstating an unproven result |
| `credit_author_statement_skeleton.md/.docx` | none | Mechanical single-author statement |
| `declaration_of_interest_skeleton.md/.docx` | none | Verbatim text lifted from `manuscript_source/main.tex` lines 579-581 |
| `author_agreement_placeholder.md/.docx` | none | Verbatim text lifted from `manuscript_source/author-agreement.tex` |

`pandoc 3.9.0.2` (`/home/soroush/bin/pandoc`) used for all 6 conversions,
each exit 0. Fidelity check: round-tripped
`response_to_reviewers_skeleton.docx` and `cover_letter_skeleton.docx`
back to plain text and diffed the placeholder occurrences against the
markdown source — all 16 and all 2 placeholders, respectively, survive
intact (only line-wrapping differs, which is why a naive per-line `grep -c`
undercounts; `grep -o` confirms full content parity). None of these 6
documents contain equations, `algorithmicx` blocks, `\ref{...}`
cross-references, or tables, so none of the four fidelity failure modes
found for the main manuscript in §3 of `kbs_docx_package_action_plan.md`
apply here.

### 27.5 The revised manuscript DOCX (item 1) — deliberately not created

Confirmed still blocked, consistent with the existing action plan: the
underlying `main.tex` content, while improved by §26's safe edits, is not
yet revision-final (no cap64/128/256 numbers, no Table 3, fallback
validate-or-remove decision still open, manuscript-shortening pass not yet
applied). Converting to DOCX now would only need to be redone once those
land. No `.docx` conversion of `manuscript_source/main.tex` was attempted
in this pass.

### 27.6 Deliverable (Step 7)

`reports/kbs_docx_submission_package_report.md` — updated in place with a
new section recording the 6 files created, their placeholder counts,
the pandoc fidelity verification, and explicit confirmation that item 1
remains untouched.

### 27.7 Tracking files updated

- `reports/kbs_revision_gap_tracker.md` (new row)
- this ledger (§27)

The submission-package work in this pass is entirely results-independent
— every numeric/empirical claim in the new files is either factual
(measured overhead numbers, code-verified complexity, the completed cap32
result already cited elsewhere) or an explicit placeholder.
`cap64_with_sieve_fifo` confirmed still running/untouched both at the
start and end of this pass. cap128/cap256 remain not launched. Nothing
was pushed, merged, deleted, or overwritten.

---

## 28. cap64_with_sieve_fifo completion, result analysis, after-cap64 decision memo, and cap128_with_sieve_fifo launch (2026-06-20)

### 28.1 Read-only status check (Step before this pass)

A strictly read-only status check confirmed `cap64_with_sieve_fifo` had
exited cleanly (`CAP64_WITH_SIEVE_FIFO_EXIT=0`), with both output files
present (`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv`
and `.md`, 56/56 rows each). No other KBS jobs were active; cap128/cap256
confirmed not launched at that point. No files were modified during this
check.

### 28.2 cap64 result analysis (read-only)

`reports/kbs_cap64_result_analysis.md` was created, comparing cap32 and
cap64 aggregate and per-trace-family numbers for all 8 policies. Headline
findings:

- Aggregate mean misses fell for every policy from cap32 to cap64 (lru
  −2.93%, sieve −2.69%, fifo_reinsertion −2.93%, evict_value_v1 −5.32%) —
  `evict_value_v1` improved faster than the baselines.
- `evict_value_v1`'s gap vs LRU narrowed from −4.84% (cap32) to −2.26%
  (cap64); vs SIEVE from −2.92% to −0.13% (nearly tied); vs
  FIFO-Reinsertion from −4.77% to −2.19%.
- This narrowing is not a generic "everyone converges at larger capacity"
  artifact: SIEVE's and FIFO-Reinsertion's own gaps vs LRU stayed flat to
  slightly wider over the same step, so the relative improvement is
  specific to `evict_value_v1`.
- The improvement is concentrated in 2 of 7 trace families (metacdn
  22.4%→2.7% gap vs LRU; metakv 1.7%→~0%, likely a policy-insensitive
  workload rather than a real signal); brightkite and cloudphysics stayed
  flat-to-worse.
- `evict_value_v1` still loses to LRU, SIEVE (barely), and
  FIFO-Reinsertion on aggregate at cap64 — no reversal, but a real and
  family-attributable trend toward parity.

### 28.3 Decision memo and approval

`reports/kbs_after_cap64_decision_memo.md` was created, presenting three
options (A: stop/reframe now; B: cap128 only, then reassess; C:
cap128+cap256 together) with a compute-cost table (cap32 ~5h16m done,
cap64 ~10h04m done, cap128 est. ~15-20h, cap256 est. ~30-40h under a
2x-per-doubling assumption) and recommending **Option B** — get one more
capacity data point before committing to the final, most expensive chunk,
since `run_policy_comparison_wulver_v1.py` has no mid-run checkpointing
and a combined cap128+cap256 run risks losing both chunks to one failure.

The user approved Option B explicitly in this session.

### 28.4 cap128_with_sieve_fifo launch

Pre-flight checks (Step 1) all passed: `main` branch confirmed, all
required input files present (trace manifest, model `.pkl`, dataset
manifest), `pytest tests/test_sieve.py tests/test_fifo_reinsertion.py -v`
→ 16/16 passed, disk/memory/load healthy. Output files confirmed absent
via `test ! -e` (Step 2) before launch.

Launched in a new tmux session `kbs_full_policy_comparison_cap128_with_sieve_fifo`
(created Sat Jun 20 2026 09:31:02), running the same 8-policy canonical
command (`lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1`)
at `--capacities 128`, piped through `tee` into
`logs/kbs_full_policy_comparison/cap128_with_sieve_fifo.log`, with an
`CAP128_WITH_SIEVE_FIFO_EXIT=` marker and start/finish timestamp files.
Driver PID 276106, tee PID 276107, start timestamp `Sat Jun 20 09:31:03
AM EDT 2026`. Launched via tmux only — no `sbatch`/`squeue`/`sacct` or any
Slurm/Wulver tooling.

Full detail: `reports/kbs_cap128_with_sieve_fifo_launch_report.md`.

### 28.5 Tracking files updated

- `reports/kbs_revision_gap_tracker.md` — cap64 row marked done, new
  cap128 row marked running.
- `reports/kbs_revision_completion_audit.md` — status table row updated;
  new §O appended.
- `reports/kbs_next_actions_without_rerun.md` — two checklist items
  checked off (cap64 sanity-check, Option B decision).
- `reports/kbs_response_to_reviewers_skeleton.md` — all `cap64 running`
  status mentions updated to reflect cap64 done / cap128 running.
- `reports/kbs_complete_reviewer_comment_matrix.md` — R2-MC3, R3-Issue1,
  R3-Issue3, R3-Issue4, R3-Minor9, R3-Rec1, R3-Rec2, R3-Rec8 rows updated.
- this ledger (§28).
- `reports/kbs_revision_evidence/evidence_manifest.json` — pending update
  in the same pass.

cap256 remains **not launched**. Nothing was pushed, merged, deleted,
renamed, or overwritten in this pass.

## 29. Chunk verification + draft trend tooling prepared while cap128_with_sieve_fifo runs (2026-06-20)

### 29.1 Read-only cap128/cap256 status check

Re-verified before any other work in this pass: `kbs_full_policy_comparison_cap128_with_sieve_fifo`
tmux session live, driver `python scripts/run_policy_comparison_wulver_v1.py
--capacities 128 ...` at PID `276106` in state `R`, 99.9% CPU, ~2.4% memory,
elapsed climbing from ~6h12m to ~6h41m across the checks in this pass. No
`CAP128_WITH_SIEVE_FIFO_EXIT=` marker in
`logs/kbs_full_policy_comparison/cap128_with_sieve_fifo.log` (still 0 bytes
at the first check). No `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.{csv,md}`.
No cap256 tmux session or process. Job not touched at any point in this pass.

### 29.2 Verification script

`scripts/paper/verify_kbs_policy_chunks.py` — reads one or more chunk CSVs
(schema: `trace_name,trace_family,path,capacity,policy,misses,hit_rate`) and
checks: exact column match; every `--expected-capacities` value present;
every `--expected-policies` value present overall *and* per (trace,
capacity) pair; no duplicate (trace_name, capacity, policy) rows within or
across the given inputs; `misses` is a nonnegative integer; `hit_rate` (when
non-empty) lies in [0, 1]; the diagnostic-only `blind_oracle` policy
(distinct from the fair `blind_oracle_lru_combiner` used in the real
8-policy roster) is absent unless `--allow-diagnostic-blind-oracle` is
passed; and warns (does not fail) on capacities/policies/traces present but
not requested. Verified against the existing repo data first: the old,
superseded `evict_value_wulver_v1_policy_comparison_heavy_r1_cap32.csv`
(no `_with_sieve_fifo` suffix) does contain bare `blind_oracle` and is
missing `sieve`/`fifo_reinsertion` — exactly the two conditions this script
is designed to catch — confirming the diagnostic-only-policy check and the
per-(trace, capacity) policy-completeness check both fire correctly on a
known-bad input before being trusted on the real cap32/cap64 chunks.

Run command and result:

```bash
python scripts/paper/verify_kbs_policy_chunks.py \
  --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
           analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
  --expected-capacities 32,64 \
  --expected-policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1
# -> PASSED: chunks are structurally and numerically sound for merge. (exit 0, zero errors, zero warnings)
```

### 29.3 Draft trend-artifact script

`scripts/paper/build_kbs_policy_trend_artifacts.py` — reads the same
verified chunk CSVs and computes, per available capacity: mean misses by
policy; relative gap vs LRU (%); per-trace-family policy ranking (best to
worst by mean misses); `evict_value_v1`'s gap vs LRU/SIEVE/FIFO-Reinsertion
specifically (%); and the multi-capacity trend direction
(improving/worsening/flat) for every policy across whichever of
cap32/64/128/256 are present in the inputs. Every output row/file is
stamped with a `--draft-label` (default `DRAFT / AVAILABLE CAPACITIES
ONLY`), and the script writes only to new `*_available_capacities`
filenames — never to `tables/manuscript/table3_main_quantitative_comparison.csv`
or any other existing canonical artifact — regardless of whether all
requested `--capacities` happen to be present.

Run command and result (cap32+cap64 only, cap128 deliberately excluded
since it has not finished):

```bash
python scripts/paper/build_kbs_policy_trend_artifacts.py \
  --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
           analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
  --capacities 32,64 \
  --draft-label "DRAFT / AVAILABLE CAPACITIES ONLY"
```

Outputs (all new files, confirmed absent before this pass):

- `analysis/kbs_policy_trend_available_capacities.csv`
- `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`
- `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`

Draft headline (cap32→cap64, consistent with `reports/kbs_cap64_result_analysis.md`):
`evict_value_v1` mean-misses gap vs LRU narrows +4.84%→+2.26%; vs SIEVE
+2.92%→+0.13%; vs FIFO-Reinsertion +4.77%→+2.19%. Still a loss vs all three
at both capacities; `wiki2018` remains degenerate (50000/50000 misses, every
policy, both capacities) and is uninformative for the trend.

### 29.4 Templates prepared (fill-in only)

- `reports/kbs_post_cap128_decision_template.md` — 7 fill-in sections
  (completion status, headline result, cap32→64→128 trend, gap-vs-baselines
  table, improvement/stall/reversal classification, cap256 go/no-go,
  manuscript framing). No section filled in — explicitly gated on cap128
  actually completing and passing verification.
- `reports/kbs_cap256_launch_template.md` — exact cap256 command, mirroring
  the cap128 launch convention in `reports/kbs_cap128_with_sieve_fifo_launch_report.md`,
  with the same 8-policy roster and tmux/no-Slurm constraints. Marked **DO
  NOT RUN UNTIL USER APPROVES AFTER CAP128 ANALYSIS** at the top. Not
  executed; no cap256 tmux session, process, or output file created.

### 29.5 Tracking files updated

- `reports/kbs_revision_gap_tracker.md` — new row added documenting this
  pass's tooling.
- `reports/kbs_revision_completion_audit.md` — status-table row updated
  (final canonical merged CSV row); new §P appended.
- `reports/kbs_next_actions_without_rerun.md` — one "can do immediately"
  item checked off; two new unchecked items added under "needs a human
  decision" (wait-for-cap128-then-verify, approve-cap256-launch).
- this ledger (§29).
- `reports/kbs_revision_evidence/evidence_manifest.json` — pending update
  in the same pass.

cap128 confirmed untouched throughout (no stop/start/restart). cap256
confirmed not launched. No canonical merged CSV created. No existing
manuscript artifact overwritten. Nothing pushed, merged, deleted, or
committed in this pass.

## 30. cap128_with_sieve_fifo completion, anomaly discovery, and sanity/root-cause audit (2026-06-20/21)

### 30.1 cap128 completion and result analysis (read-only)

`cap128_with_sieve_fifo` (tmux session
`kbs_full_policy_comparison_cap128_with_sieve_fifo`, PID `276106`)
completed cleanly: `CAP128_WITH_SIEVE_FIFO_EXIT=0`, runtime ~22h08m, 56/56
rows in
`analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv/.md`.

A read-only comparison against cap32/cap64 found `evict_value_v1`'s gap vs
LRU — which had narrowed from +4.84% (cap32) to +2.26% (cap64) — **reversed
and widened sharply to +11.76% at cap128**, breaking the improving trend
that motivated running cap128 in the first place (per the Option B
decision memo, `reports/kbs_after_cap64_decision_memo.md`). The reversal
is concentrated in two families: brightkite (+50.9% gap vs LRU at cap128)
and citibike (+45.5%); the other five families (cloudphysics, metacdn,
metakv, twemcache, and the degenerate wiki2018) did not show a comparably
sharp reversal.

### 30.2 Sanity/root-cause audit (2026-06-21)

Given the size of the reversal, a dedicated low-cost audit was performed
before any cap256 decision — explicitly scoped to read-mostly checks under
a ~1h compute budget, no cap256, no full sweep, no overwrite of canonical
cap32/cap64/cap128 outputs. Full report:
`reports/kbs_cap128_anomaly_sanity_audit.md`.

1. **Raw row verification**: re-read
   `..._cap128_with_sieve_fifo.csv/.md` directly for
   brightkite/citibike/metacdn/twemcache (plus the remaining 3 families) —
   56/56 rows, correct trace names, `capacity=128` uniformly, all 8
   policies present exactly once per trace, `misses`/`hit_rate` internally
   consistent with 50,000 requests/trace, no duplicate rows. Clean.
2. **Cross-capacity structural verification**: ran
   `scripts/paper/verify_kbs_policy_chunks.py` against
   cap32_with_sieve_fifo + cap64_with_sieve_fifo + cap128_with_sieve_fifo
   together, `--expected-capacities 32,64,128` and the canonical 8-policy
   roster — **PASSED**, zero errors, zero warnings.
3. **Targeted reproducibility check**: extrapolating from cap128's total
   wall-clock (22.14h for 7 traces, dominated by `evict_value_v1`'s
   O(capacity) candidate scan), a brightkite+citibike-only full-scale
   (50,000-request) recheck was estimated at **~6.3h** — over the ~1h
   budget for this pass, so it was **not run**. The exact staged command is
   recorded in `reports/kbs_cap128_anomaly_sanity_audit.md` §3, pending
   separate approval. As a substitute, six small-scale (1,000-request)
   throwaway probes were run in `/tmp/cap128_probe/` (outside the repo, not
   tracked by git, not part of any deliverable): confirmed the run is
   deterministic (`probe_run2.csv`/`probe_run3.csv`, identical params,
   byte-identical via `diff`), and confirmed the same reversal direction
   reproduces even at 1/50th scale — brightkite's gap vs LRU widens from
   −11.24% (cap64, 1k requests) to −34.88% (cap128, 1k requests), matching
   the canonical 50k-request run's qualitative pattern.
4. **Code-path audit**: read `src/lafc/policies/evict_value_v1.py`,
   `src/lafc/evict_value_features_v1.py`, `src/lafc/evict_value_model_v1.py`,
   `src/lafc/evict_value_dataset_v1.py`, `src/lafc/learned_gate/features.py`,
   `src/lafc/policies/base.py`, `src/lafc/simulator/cache_state.py`; grepped
   `src scripts tests` for any capacity==128/>=128/<=128 branching — zero
   matches. Victim selection is a generic, capacity-agnostic
   minimum-predicted-loss argmin over all cache-resident candidates.
   Features are rank/percentile-normalized (capacity-scale-invariant)
   except one raw count (`cache_unique_bucket_count`), flagged as a
   plausible but unconfirmed contributor. Model loading uses
   persisted-by-name feature columns (no train/serve order mismatch).
   The offline label-builder reuses the identical feature function as the
   online policy (no feature-code divergence). Confirmed the documented
   single-step LRU-continuation counterfactual labeling methodology
   (per-candidate labels assume LRU continues after evicting that one
   candidate; the real labeling-time trajectory always evicts the true LRU
   victim regardless) — an existing, known limitation
   (`docs/evict_value_v1_method_spec.md`), and the leading (unconfirmed)
   hypothesis for capacity-dependent compounding error in the deployed,
   self-referential policy trajectory.
5. **Model/training-distribution check**: confirmed cap128 was included in
   training (87,004,032 rows, independently re-derived by summing actual
   shard-file row counts, matching the dataset summary exactly); confirmed
   all three canonical runs (cap32/64/128) used the identical, single,
   multi-capacity-trained model file
   (`models/evict_value_wulver_v1_best_heavy_r1.pkl`, also byte-identical
   to `models/evict_value_wulver_v1_best.pkl` via md5sum — a naming
   duplicate, not a provenance issue); confirmed brightkite (6.77% of
   cap128 training rows) and citibike (7.50%) are the two
   least-represented families in the shared training set, but this
   under-representation is essentially constant across cap32/64/128 (not
   uniquely thin at cap128) — a credible aggravating factor for these two
   families' general performance, but not a sufficient standalone
   explanation for why the gap specifically widens between cap64 and
   cap128.

### 30.3 Conclusion and recommendation

**Likely real model/policy behavior, not a bug or data artifact** — every
one of the above checks came back clean (no corruption, no schema drift,
no capacity-specific code branching, no train/serve mismatch, deterministic
and reproducible at small scale). The precise causal mechanism for the
cap128-specific severity remains **unconfirmed**; the leading hypothesis is
compounding distribution shift from the single-step LRU-continuation
training-label methodology, plausibly aggravated by brightkite/citibike's
thinner (but capacity-constant) training representation.

**Recommendation: do not launch cap256 yet.** Report cap32→cap64→cap128 as
a genuine non-monotonic capacity-sensitivity finding tied to the documented
labeling limitation, rather than treating it as something to compute past.
The staged ~6.3h full-scale 2-trace recheck (§30.2 item 3) remains the
next-cheapest confirmation step if higher confidence is wanted before any
cap256 decision, but requires separate approval — it was not run in this
pass.

### 30.4 Tracking files updated

- `reports/kbs_revision_gap_tracker.md` — cap128 row marked done with the
  anomaly finding; new audit row added; final-canonical-CSV row updated to
  reflect 3 of 4 chunks done with cap256 on hold; SIEVE/FIFO-Reinsertion and
  response-draft rows updated to cap128-done status.
- `reports/kbs_revision_completion_audit.md` — status-table rows updated
  (cap64/cap128/cap256 row, final canonical merged CSV row); new §Q
  appended.
- `reports/kbs_next_actions_without_rerun.md` — the wait-for-cap128 item
  checked off with the anomaly result; a new checked item added for the
  sanity audit itself; the cap256-decision item updated to reflect that the
  audit is now the analysis it was waiting on, recommending hold.
- this ledger (§30).
- `reports/kbs_revision_evidence/evidence_manifest.json` — pending update in
  the same pass.

cap128 was not re-run, stopped, or modified in this pass (it had already
completed before this pass began). cap256 confirmed not launched. No
canonical cap32/cap64/cap128 output overwritten. Nothing pushed, merged,
deleted, renamed, or committed in this pass. New artifacts: this section,
`reports/kbs_cap128_anomaly_sanity_audit.md`, and throwaway probes in
`/tmp/cap128_probe/` (outside the repo, not tracked by git).

## 31. Revision-writing pass — manuscript and response-to-reviewers updated using cap32/64/128 evidence (2026-06-21)

### 31.1 Instruction and constraints

Proceed with revision-writing: convert the completed cap32/cap64/cap128
evidence and the cap128 anomaly audit into manuscript and
response-to-reviewers updates. Hard constraints: do not launch cap256; do
not run heavy experiments; do not overwrite raw CSV/MD outputs; do not
push, commit, merge, delete, or rename anything; manuscript/report editing
is allowed; keep all claims honest (`evict_value_v1` does not beat
LRU/SIEVE/FIFO-Reinsertion end-to-end).

### 31.2 Capacity-explicit tables/figures (zero new compute)

Re-ran `scripts/paper/build_kbs_policy_trend_artifacts.py` with
`--inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap{32,64,128}_with_sieve_fifo.csv --capacities 32,64,128 --draft-label "DRAFT / AVAILABLE CAPACITIES ONLY (cap32, cap64, cap128 — cap256 NOT run)"`.
Refreshed (all explicitly draft-labeled, none overwriting the canonical
capacity-blind `table3_main_quantitative_comparison.csv`, which was left
untouched in its `NOT_VERIFIED` stub state):
- `analysis/kbs_policy_trend_available_capacities.csv`
- `tables/manuscript/table3_policy_miss_ratio_available_capacities.csv`
- `reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md`

Created a new script, `scripts/paper/build_kbs_available_capacities_figure.py`
(pure plotting from the trend CSV above, no new simulation/training compute),
producing:
- `figures/manuscript/figure_available_capacities_trend_DRAFT.pdf` / `.png`
- `reports/manuscript_artifacts/latex_snippets/figure_available_capacities_trend_DRAFT_snippet.tex`

Two-panel figure: (a) mean replay misses by capacity per policy (8 policies,
3 capacities), `evict_value_v1` highlighted; (b) `evict_value_v1`'s relative
miss gap vs LRU/SIEVE/FIFO-Reinsertion by capacity, computed directly from
the mean-misses values (the trend CSV's `rel_gap_vs_lru_pct` column only
covers the LRU baseline, so SIEVE/FIFO-Reinsertion gaps were derived in the
figure script itself). The PNG was copied to
`manuscript_source/figures/figure_available_capacities_trend_DRAFT.png`
because `manuscript_source/figures/` is a separate, flat directory from the
repo-root `figures/manuscript/` path and contains only the two pre-existing
figures (`method_overview.png`, `figure4_ablation.png`) with no
`manuscript/` subdirectory — confirmed via `grep -n "graphicspath\|includegraphics" main.tex`
and `ls manuscript_source/figures/`.

### 31.3 Manuscript edits (`manuscript_source/main.tex`)

Seven edits, in document order:
1. **Abstract** — appended two sentences stating the end-to-end evaluation
   was added across cap32/64/128 and all 7 families, that `evict_value_v1`
   does not outperform LRU/SIEVE/FIFO-Reinsertion in it, that the gap widens
   non-monotonically at cap128, and reframing the overall claim as
   supervision-target design rather than demonstrated online superiority.
2. **New subsection "End-to-End Online Replay Evaluation Across Available
   Capacities"** (`\label{subsec:end_to_end_capacities}`) — setup
   paragraphs, a table (`tab:available-capacities-trend`) with all 8
   policies × 3 capacities of mean misses, the new figure
   (`fig:available-capacities-trend`), and discussion paragraphs with the
   exact gap percentages vs LRU/SIEVE/FIFO-Reinsertion at each capacity.
3. **New subsection "Workload-Specific Breakdown"**
   (`\label{subsec:workload_breakdown}`) — a table
   (`tab:family-capacity-gap`) of per-family gap vs LRU at cap32/64/128 for
   all 7 families (computed directly from the three chunk CSVs via a
   Python one-liner per capacity, not taken solely from the audit's
   narrower two-family framing), discussion of the wiki2018 degeneracy,
   CloudPhysics/MetaKV stability, and the MetaCDN (U-shaped) / Twemcache
   (steadily increasing) patterns.
4. **Discussion and Analysis** — added explicit acknowledgment that the
   end-to-end evidence directly answers the superiority question
   negatively; added the LRU-continuation compounding-shift causal
   hypothesis; revised the closing paragraph to the decision-aligned-
   supervision-study framing.
5. **Limitations point 3** — replaced the pre-evaluation hedge with the
   actual finding, four supporting diagnostic checks from the anomaly
   audit, the leading hypothesis, what would confirm it, and the explicit
   cap256-on-hold statement.
6. **Summary of Findings, paragraph 1** — softened "practically useful
   basis" to "productive object of study," distinguishing the offline
   finding from end-to-end online competitiveness.
7. **Summary of Findings, closing paragraph** — replaced the
   pre-evaluation hedge with a "focused but mixed conclusion" restating the
   negative end-to-end finding.

Exact before/after text and rationale for each edit:
`reports/kbs_manuscript_change_log_2026-06-21.md`.

**Discovery made while writing edit 3**: computing exact per-family gaps
directly from the chunk CSVs (rather than relying on the anomaly audit's
original brightkite/citibike-only framing) showed the cap128 degradation is
broader — MetaCDN (+17.62%) and Twemcache (+14.06%) also show double-digit
gaps, with MetaCDN U-shaped across all three capacities and Twemcache
steadily increasing. This four-family characterization, not the audit's
two-family one, is what was written into the manuscript.

### 31.4 Response-to-reviewers edits

`reports/kbs_response_to_reviewers_skeleton.md` (authoritative tracker) —
12 edits: new banner; AE; R2-MC3; R3-Issue1; R3-Issue3 (update
sub-paragraph); R3-Issue4; R3-Issue6 (update sub-paragraph); R3-Issue7;
R3-Minor8; R3-Minor9; Recommended-Revisions items 1/2/4/6/8; and a full
recomputation of the bottom status-summary table and its prose paragraph.
Status-count delta: `[PENDING CAP128/CAP256]` 6→0; `[IN PROGRESS]` 8→13;
`[PENDING BASELINE DECISION]` 5→4; `[PENDING MANUSCRIPT REWRITE]` 6→3;
`[DONE]` unchanged at 0.

`submission_kbs_revision_docx/response_to_reviewers_skeleton.md`
(externally-facing letter rendering) — mirrored the same substantive
content into letter prose (banner, AE, R2-MC2/MC3, R3-Summary,
R3-Issue1/3/4/6, R3-Minor8/9, Recommended Revisions 1-8, closing "Do not
submit as-is" caveat), per the file's own stated convention. Its `.docx`
render was **not** regenerated in this pass (no `pandoc` re-run) and is now
stale relative to its own `.md` source.

### 31.5 New reports created

- `reports/kbs_manuscript_change_log_2026-06-21.md` — exhaustive
  change-by-change manuscript diff rationale.
- `reports/kbs_revision_writing_progress_report.md` — overall progress
  summary, status-count recomputation, remaining-work list.

### 31.6 Tracking files updated in this pass

- `reports/kbs_revision_gap_tracker.md` — response-to-reviewers row
  updated; new row added for the manuscript revision-writing pass itself.
- `reports/kbs_revision_completion_audit.md` — new §R appended.
- `reports/kbs_next_actions_without_rerun.md` — new checked item for this
  pass; new pending items for the `tectonic` re-compile, `.docx`
  regeneration, fallback/HALP scope decisions, and the manuscript-
  shortening pass.
- this ledger (§31).
- `reports/kbs_revision_evidence/evidence_manifest.json` — updated in the
  same pass (see entries appended immediately after this section's
  reproduction commands, in the Appendix below).

### 31.7 Confirmed constraints

cap256 was not launched and is not claimed or implied anywhere in any
edited or new file. No heavy experiment was run — only an existing
trend-builder script re-run on already-complete chunk inputs, plus one new
zero-compute plotting script. No raw CSV/MD chunk output was modified
(chunk CSVs were read only, including via direct Python computation for
exact per-family percentages). Nothing was pushed, committed, merged,
deleted, or renamed.

### 31.8 Post-edit verification (same pass)

Re-ran `scripts/paper/verify_kbs_policy_chunks.py` on the cap32+cap64+cap128
chunks together (`--expected-capacities 32,64,128`, full 8-policy roster):
**PASSED**, zero errors/warnings. Re-ran `tectonic main.tex` from a clean
state (removed `main.pdf/.aux/.log/.bbl/.blg/.out` first, then rebuilt):
**exit 0**, `main.pdf` written (42 pages, 890,312 bytes), zero `undefined`
reference/citation warnings in the final pass (only pre-existing cosmetic
overfull/underfull `\hbox` warnings remain, none introduced by this pass).
Independently confirmed all five new labels
(`subsec:end_to_end_capacities`, `subsec:workload_breakdown`,
`tab:available-capacities-trend`, `fig:available-capacities-trend`,
`tab:family-capacity-gap`) are each defined exactly once via
`grep -n "\label{...}" main.tex`.

---

## 32. Final revision audit and cleanup pass (2026-06-21)

**Scope of this pass**: a reviewer-focused final revision audit, explicitly
constrained to **no cap256 launch, no new heavy experiments, no
modification of raw CSV/MD experiment outputs, no push/commit/merge/
delete/rename**. Manuscript editing, reporting, and packaging work were
in scope, but no manuscript edits were actually applied in this pass —
every deliverable below is analysis/decision/planning, with concrete
ready-to-apply text where applicable.

### 32.1 Manuscript shortening audit

`reports/kbs_manuscript_shortening_execution_plan.md` — re-measured the
current `manuscript_source/main.tex` at **11,682 words** (up from the
prior 8,919-word baseline, +31%, due to the cap32/64/128 revision-writing
pass), against R3-Rec6's 30-40% reduction target. Identified 9 itemized
cuts (duplicate eviction-loss equation; §1.1/1.2 overlap; 5x-repeated
"mixed conclusion" paragraph; Algorithmic Workflow vs. Algorithm 1
redundancy; Datasets/Baselines vs. Tables redundancy; Related Work
SIEVE/FIFO mechanism-detail redundancy; Discussion/Summary/Implications
3-way merge; fallback caveat repetition; general hedging tightening),
each with location, estimated words removed, and risk level. Honest
finding: full execution of all 9 cuts reaches only ~19-25%, short of the
30-40% target — closing the rest would require removing evidentiary
content other reviewers explicitly asked to see added.

### 32.2 Fallback contribution decision

`reports/kbs_fallback_final_decision_report.md` — re-verified the three
required facts from `kbs_fallback_revision_strategy.md` (2026-06-19) still
hold: guard not wired into the canonical pipeline, never run on real
traces; only related ablation (`sentinel_budgeted_guard_v2`) shows
unguarded baseline outperforming guarded variants. Found the mechanism
still occupies Contributions item 3 of 5 (`main.tex` line 75) and is
mentioned in 19 distinct locations across 13 subsections, against zero
empirical evidence anywhere in the repo. **Decision: KEEP AS OPTIONAL
IMPLEMENTATION DETAIL** — remove from the numbered Contributions list;
retain but shrink the Robust Decision Mechanism subsection/Algorithm 1;
fix Table 6's guarded-layer clause; leave the unused schematic figure
out. This resolves the previously `[PENDING BASELINE DECISION]` status
for R3-Issue6/R3-Rec5 — **decision made, manuscript edit not yet
applied.**

### 32.3 HALP positioning audit

`reports/kbs_halp_positioning_final_audit.md` — confirmed HALP is
mentioned in exactly 2 sentences, both in Related Work (`main.tex` lines
89, 91), with no mention in Introduction, Discussion, or Limitations.
Found the existing differentiation thin (omits pairwise-vs-pointwise,
online-vs-offline, and especially factual-vs-counterfactual structural
distinctions) and found Limitations contains no disclosure that an
empirical HALP comparison was not attempted (unlike cap256 and fallback,
which both get explicit Limitations disclosure). Produced exact
ready-to-paste text for (a) a sharper Related Work paragraph, (b) a new
Limitations sentence, and (c) the response-letter's still-unfilled
placeholder bracket at `kbs_response_to_reviewers_skeleton.md` lines
250-265 (replacement text already existed in
`kbs_halp_fifo_source_verification.md` §1.4 since 2026-06-19 but was
never copied in). **None of these three edits applied yet.**

### 32.4 Reviewer-response completeness audit

`reports/kbs_final_reviewer_coverage_audit.md` — strict 22-row Comment |
Status | Evidence | Remaining Gap table using the exact AE/R2-MC1-3/
R3-Summary/R3-Issue1-7/R3-Minor8-9/R3-Rec1-8 labels. Independently
re-derives the response skeleton's own "zero `[DONE]` items" finding.
Tally: 0 FULLY ADDRESSED, 11 SUBSTANTIALLY ADDRESSED, 7 PARTIALLY
ADDRESSED, 4 NOT ADDRESSED (R3-Issue6, R3-Minor8, R3-Rec5, R3-Rec6).
New finding from direct re-check of `main.tex` lines 640-642: the AI
Declaration section is a generic 3-sentence tool-disclosure paragraph
that does **not** contain the validation-methodology argument
(`scripts/paper/verify_kbs_policy_chunks.py`, the cap128 anomaly audit)
that the drafted R3-Issue7 response paragraph relies on — confirming the
response skeleton's own self-flagged caveat was correct, not just a
hedge.

### 32.5 Submission-readiness assessment

`reports/kbs_submission_readiness_assessment.md` — weighted estimates:
reviewer concerns completed ~50%, manuscript readiness ~65%,
response-letter readiness ~35%, submission-package readiness ~20%,
overall readiness ~48%. Direct re-check of `submission_kbs_revision_docx/`
found `response_to_reviewers_skeleton.docx` is stale: its markdown source
differs from the current, actively-maintained
`reports/kbs_response_to_reviewers_skeleton.md` (confirmed via `diff`),
and the `.docx` was never regenerated after the 2026-06-19 23:39 build —
it predates the entire cap128 update and all three decisions made in this
pass. Top-10 remaining tasks ranked by importance, all writing/editing
tasks against already-completed analysis — none require new compute.

### 32.6 Confirmed constraints

cap256 was not launched and is not claimed anywhere. No new heavy
experiment was run — every action in this pass was reading existing
files or writing new report/analysis markdown files. No raw CSV/MD
experiment output was modified. Nothing was pushed, committed, merged,
deleted, or renamed. `main.tex` was read but not edited in this pass —
the fallback, HALP, and shortening decisions above are recorded as
ready-to-apply text, not yet applied to the manuscript source.

---

## Appendix: exact reproduction commands for everything in this ledger

```bash
cd /home/soroush/Augmented-caching
git branch --show-current && git status --short
du -sh data/derived/evict_value_v1_wulver_heavy_r1
ls -lh data/derived/evict_value_v1_wulver_heavy_r1/manifest.json
ls -lh data/derived/evict_value_v1_wulver_heavy_r1/split_summary.csv
sha256sum data/derived/evict_value_v1_wulver_heavy_r1/manifest.json
sha256sum data/derived/evict_value_v1_wulver_heavy_r1/split_summary.csv
tail -120 logs/kbs_heavy_r1/build_dataset.log
ls -lh analysis/evict_value_wulver_v1_train_metrics_heavy_r1.*
ls -lh analysis/evict_value_wulver_v1_model_comparison_heavy_r1.*
ls -lh analysis/evict_value_wulver_v1_best_config_heavy_r1.*
cat analysis/evict_value_wulver_v1_best_config_heavy_r1.*
ls -lh analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv
cat analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_validation.csv
tail -200 logs/kbs_policy_comparison_heavy_r1/validation_eval_run.log
pytest tests/ -q   # 283 passed, 12 warnings, ~10.5s, re-verified 2026-06-18
git log --all --oneline | grep -E "64e39bb|9a1642a"
git merge-base --is-ancestor 64e39bb main || echo "not merged"
git merge-base --is-ancestor 9a1642a main || echo "not merged"

# SIEVE implementation pass (2026-06-19, §17) — uses the existing
# .venv_kbs_heavy_r1 venv (editable lafc install pointing at this checkout).
.venv_kbs_heavy_r1/bin/pytest tests/ -k "sieve or policy_comparison or lru or marker" -v
# -> 45 passed, 246 deselected
.venv_kbs_heavy_r1/bin/pytest tests/ -q
# -> 291 passed, 12 warnings (+8 vs. 283 above, exactly the new SIEVE tests)
.venv_kbs_heavy_r1/bin/python scripts/run_policy_comparison_wulver_v1.py \
  --trace-manifest analysis/wulver_trace_manifest_full.csv --capacities 32 \
  --max-traces 1 --max-requests-per-trace 1000 --policies lru,sieve \
  --out-csv analysis/evict_value_wulver_v1_policy_comparison_sieve_smoke.csv \
  --out-md analysis/evict_value_wulver_v1_policy_comparison_sieve_smoke.md
# -> sieve: 496.0 mean misses, lru: 503.0 mean misses (1 trace, brightkite)

# Chunk verification + draft trend tooling (2026-06-20, §29) — read-only
# cap128 check, then verify+trend on cap32/cap64 only (cap128 excluded,
# still running).
tmux ls
pgrep -af "cap128_with_sieve_fifo|run_policy_comparison_wulver_v1|python"
grep -Rni "CAP128_WITH_SIEVE_FIFO_EXIT=" logs/kbs_full_policy_comparison
python scripts/paper/verify_kbs_policy_chunks.py \
  --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
           analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
  --expected-capacities 32,64 \
  --expected-policies lru,sieve,fifo_reinsertion,predictive_marker,blind_oracle_lru_combiner,trust_and_doubt,rest_v1,evict_value_v1
# -> PASSED: chunks are structurally and numerically sound for merge.
python scripts/paper/build_kbs_policy_trend_artifacts.py \
  --inputs analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv \
           analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv \
  --capacities 32,64 \
  --draft-label "DRAFT / AVAILABLE CAPACITIES ONLY"
# -> writes analysis/kbs_policy_trend_available_capacities.csv,
#    tables/manuscript/table3_policy_miss_ratio_available_capacities.csv,
#    reports/manuscript_artifacts/kbs_policy_trend_available_capacities.md
```
