# KBS revision — platform-consistency audit (local/cloud + tmux vs. Wulver/Slurm)

Status: decision report only. **Nothing executed, nothing pushed, nothing
merged, nothing deleted or overwritten.** All actions in this audit were
read-only (grep, `unzip -l`/`unzip` to a `/tmp` scratch dir, `git
rev-parse`/`status`, `hostname`/`uname`/`lscpu`/`free`/`df`/`nvidia-smi`,
`python --version`, `pip list`, `numpy.show_config()`).

Question asked: is it scientifically and manuscript-wise okay to run the KBS
revision experiments on this local/cloud machine using tmux, instead of
Wulver/Slurm where some original experiments were run?

**Bottom line: A.** Safe to proceed locally with tmux; update provenance
only. See §5 for the full reasoning and §6 for the final summary.

---

## 1. Manuscript/reports inspection (Step 1)

The actual Elsevier manuscript source is **not** a loose `.tex` file in the
repo — it's bundled in
`Decision_aligned_eviction_value_prediction_for_robust_learning_augmented_caching.zip`
at repo root (pre-existing, added in the current HEAD commit `e9ac132`, "Add
files via upload"). It was extracted read-only to
`/tmp/kbs_manuscript_audit_extract/` for inspection; the zip in the repo was
never touched. It contains `main.tex` (the actual submission, `elsarticle`
class), `cover-letter.tex`, `author-agreement.tex`, `refs.bib`, two figures.

**The exact string "Wolverine" does not appear anywhere in the repo or the
manuscript.** The repo and manuscript consistently use "**Wulver**" — NJIT's
actual HPC cluster name. Treating "Wolverine" in the user's question as an
informal name for the same system, not a separate/conflicting one.

Searching `main.tex`, `cover-letter.tex`, `author-agreement.tex`,
`CANONICAL_KBS_SUBMISSION.md`, and `reports/` for
`Wolverine|Wulver|Slurm|cluster|node|GPU|CPU|hardware|machine|server|runtime|environment|experiments were run|we ran`
found:

- **Manuscript body (`main.tex`)**: exactly **one** sentence mentions
  hardware, in the Acknowledgements section (line 532):

  > "The author thanks Professor Ioannis Koutis for his guidance and support
  > throughout this work. The author also acknowledges the use of the
  > Wulver high-performance computing system at the New Jersey Institute of
  > Technology **for part of** the experimental evaluation."

  This is **hedged, non-exclusive** wording ("for part of"), not a claim
  that all (or even most) experiments ran on Wulver. No CPU model, GPU, RAM
  amount, core count, or wall-clock benchmark appears anywhere else in the
  manuscript body, cover letter, or author agreement. The Funding section
  (line 536) explicitly states "no specific funding," and Data/Code
  Availability (lines 542–551) point only to the GitHub repository
  (`https://github.com/SoroushVahidi/Augmented-caching`) — no hardware
  claim there either.
- The manuscript's "Experimental Setup" discussion (lines 355–410) is
  explicit that reported results are scoped to "canonical manuscript
  artifacts generated from the designated evaluation pipeline" and
  distinguishes "repository-supported" workload breadth from the
  "artifact-backed subset" actually reported — i.e., the manuscript already
  frames its empirical claims around *which pipeline/artifacts* produced a
  number, not *which physical machine* ran it. Nothing ties a reported
  number to a specific compute platform.
- **`CANONICAL_KBS_SUBMISSION.md`** (repo root, internal doc, not
  manuscript-facing): uses "Wulver"/"Slurm"/`EXP_TAG=heavy_r1` purely as an
  internal script/artifact-naming convention for repo navigation. It is not
  text that goes into the manuscript or response to reviewers.
- **`reports/`**: many internal audit/planning docs reference Wulver/Slurm
  (job IDs, sbatch commands, squeue/sacct output) — these document
  engineering history and are evidence *for* this audit, not manuscript
  text themselves.

**Historical evidence Wulver usage was real, not aspirational:** prior Slurm
job IDs **909870, 910352, 910353** (`evictv1-heavy-smoke`) with captured
`squeue`/`sacct` output exist in
`reports/manuscript_artifacts/heavy_smoke_audit.md`,
`heavy_r1_readiness_note.md`, `heavy_r1_execution_audit.md`,
`wulver_scheduler_feasibility_check.md`. So the Acknowledgements sentence is
grounded in genuine (if partial) Wulver compute, and the current local/cloud
heavy_r1 work is best understood as a **continuation/supplement** of that
work on a different machine for this revision cycle — not a substitution
that falsifies an existing claim.

## 2. Current environment (Step 2)

```
=== git ===
HEAD: e9ac132c0aba12c640b0663af8c8bb6c946e81e1
git status --short: only pre-existing modified/untracked revision-work files
  (analysis/evict_value_v1_wulver_dataset_summary_heavy_r1.md, three other
  analysis/ outputs, .venv_kbs_heavy_r1/, logs/, several reports/*.md) —
  nothing new/destructive from this audit.

=== machine ===
hostname: al-khwarizmi
uname -a: Linux al-khwarizmi 6.17.0-35-generic #35~24.04.1-Ubuntu SMP, x86_64
CPU: Intel Core i7-12700K, 20 threads (12 cores, hyperthreaded), up to 5.0GHz,
     25MiB L3 cache — a desktop/workstation-class CPU, architecturally
     different from typical HPC cluster compute nodes (e.g. Wulver's Xeon
     nodes), but this matters only insofar as it affects determinism (§3),
     not correctness.
RAM: 62GiB total, 59GiB available
Disk: 700G filesystem, 165G used, 500G available
GPU: NVIDIA GPU present (driver 580.159.03, CUDA 13.0) but confirmed UNUSED
     by this pipeline — grepped scripts/train_evict_value_wulver_v1.py,
     scripts/build_evict_value_dataset_wulver_v1.py,
     scripts/run_policy_comparison_wulver_v1.py, src/lafc/evict_value_wulver_v1.py
     for CUDA/GPU imports: zero matches. Pure CPU scikit-learn pipeline.

=== python (.venv_kbs_heavy_r1, the venv actually used for heavy_r1) ===
Python 3.12.3 (GCC 13.3.0 build)
platform: Linux-6.17.0-35-generic-x86_64-with-glibc2.39
numpy: 2.4.6
scikit-learn: 1.9.0
scipy: 1.17.1
joblib: 1.5.3
pandas: NOT INSTALLED — confirmed harmless: grepped all four pipeline
  scripts above for "import pandas" and found zero matches. This pipeline
  uses csv/json/numpy directly; pandas is not a dependency at all.

numpy.show_config(): BLAS/LAPACK backend = scipy-openblas 0.3.31.188.0
  (OpenBLAS, USE64BITINT, DYNAMIC_ARCH, NO_AFFINITY, SkylakeX kernel
  selected at runtime for this CPU, MAX_THREADS=64), built with gcc 14.2.1.
  SIMD baseline X86_V2, X86_V3 (AVX2-class) found; X86_V4/AVX512 not found.
  This is the standard PyPI numpy wheel build — DYNAMIC_ARCH means OpenBLAS
  auto-selects a CPU-appropriate kernel at runtime, so a different physical
  CPU (e.g. a Wulver compute node) would likely select a different kernel
  (e.g. Haswell instead of SkylakeX) for the same numpy version.
```

## 3. Determinism/reproducibility controls (Step 3)

Code audit (consistent with, and extending, the determinism findings already
recorded in `reports/kbs_validation_sanity_audit.md`):

- **`scripts/train_evict_value_wulver_v1.py`**: `--seed` (default 7) controls
  train/val/test row sampling (`args.seed`, `args.seed+1`, `args.seed+2`).
  `RandomForestRegressor(..., random_state=args.seed, n_jobs=1)` — fixed
  seed **and** forced single-threaded (eliminates thread-scheduling-order
  nondeterminism). `HistGradientBoostingRegressor(..., random_state=args.seed)`
  — fixed seed. `Ridge(alpha=1.0)` — no randomness at all (closed-form/LAPACK
  solve).
- **`scripts/build_evict_value_dataset_wulver_v1.py`**: `--split-seed`
  (default 0) controls shard shuffling deterministically.
- **Policy evaluation (`src/lafc/policies/*.py`)**: exactly 4 files use
  `random.Random(self._seed)` (Python stdlib, deterministic given a fixed
  seed, platform-independent): `adaptive_query.py`,
  `blind_oracle_randomized_combiner.py`, `equitable.py`,
  `trust_and_doubt.py`. Of these, only `trust_and_doubt` is in
  `TABLE3_POLICIES`, and it is invoked as `TrustAndDoubtPolicy(seed=7)` (per
  the sanity audit). All other `TABLE3_POLICIES` members (`lru`,
  `predictive_marker`, `blind_oracle_lru_combiner`, `rest_v1`,
  `evict_value_v1`) have **no randomness at all** — fully deterministic
  comparison/branching logic.

**Answers:**

1. **Are model training and policy evaluation deterministic enough across
   machines? Yes**, for the purposes this manuscript needs (reproducible
   miss-counts/hit-rates), with one minor caveat noted in Q3.
2. **Are random seeds fixed? Yes** — every stochastic component (row
   sampling, RandomForest, HistGB, the one seeded policy in scope) takes an
   explicit, fixed seed (`7` for training, `0` for shard shuffling, `7` for
   `trust_and_doubt`). Nothing relies on an unseeded global RNG state.
3. **Could CPU/OS/library differences change results materially? Practically,
   no — with one caveat worth naming.** OpenBLAS's `DYNAMIC_ARCH` means a
   different physical CPU (e.g. a Wulver node) would select a different
   BLAS kernel than this machine's SkylakeX kernel, which could in principle
   produce sub-bit-level floating-point differences in BLAS/LAPACK-heavy
   linear algebra. This affects `Ridge` (which uses a LAPACK solve) more
   than `RandomForestRegressor`/`HistGradientBoostingRegressor` (tree-based,
   dominated by comparisons/branching, not matrix multiply). In practice
   this is immaterial here: (a) `Ridge` was not the selected best model for
   this revision (random_forest at horizon=4 was), (b) final eviction
   decisions are `argmin()`-style comparisons among predicted-loss values
   with deterministic index/page_id tie-breaking, not exact-equality tests,
   so tiny float perturbations essentially never flip a discrete eviction
   choice or a miss count. Library **version** differences (numpy/sklearn
   major-version bumps) are a more realistic risk than CPU/BLAS-kernel
   differences, but this revision's entire heavy_r1 chain (dataset build →
   training → eval) is being produced from one venv on one machine, so no
   version-mixing occurs within this revision's own numbers (see §5).
4. **Is policy evaluation itself deterministic once model artifacts are
   fixed? Yes.** All 7 `TABLE3_POLICIES` are either unseeded-deterministic
   or seeded-deterministic (`trust_and_doubt`, seed=7); `run_policy()`'s
   miss-accounting logic (confirmed in the prior sanity audit) is a
   straightforward deterministic simulation loop with no randomized
   tie-breaking, no wall-clock dependence, and no parallelism
   (`evict_value_v1` calls the model with `n_jobs=1`-trained estimators in a
   single-threaded scan over candidates). Given a fixed model `.pkl` and a
   fixed trace file, two runs on two different machines should produce
   bit-identical miss-counts in the overwhelming majority of cases, and at
   worst would differ only via the negligible floating-point effect in Q3
   (which would have to land exactly on a decision boundary to change a
   single eviction — not observed in any of this session's repeated runs).

## 4. Honest-reporting check (Step 4)

1. **Does the manuscript currently say experiments were run on
   Wolverine/Wulver/Slurm?** Yes, but only the single hedged Acknowledgements
   sentence quoted in §1 ("...for part of the experimental evaluation").
   There is no Methods/Experimental-Setup sentence tying any specific
   reported number to Wulver, and no exclusivity claim ("all experiments
   were run on...").
2. **If yes, which files/sections must be edited if we run the revision
   experiments locally?** **None are factually required to change.** The
   existing wording already accommodates non-exclusive compute sourcing —
   running the new canonical Table 3 sweep locally does not make "for part
   of the experimental evaluation" false; it remains true that Wulver was
   used for part of the project's experimental history (job IDs 909870,
   910352, 910353 and earlier dataset/training work). Optionally (not
   required), the Acknowledgements sentence could be sharpened — e.g.
   "...for part of the experimental evaluation; final canonical results
   reported in this revision were computed on local compute infrastructure"
   — purely for extra precision/transparency, not because the current text
   is wrong.
3. **If no, can we simply report the code/data/commit/environment without
   naming Wolverine?** N/A (answer to Q1 is yes), but the same mechanism
   applies: the manuscript's Code/Data Availability sections (lines 542–551)
   already only point to the GitHub repo and never name a specific machine,
   so no edit is needed there regardless.
4. **Is tmux acceptable as execution management?** Yes, unambiguously. tmux
   is a terminal session multiplexer — it has zero bearing on scientific
   validity, numerical results, or reproducibility. It is not a claim made
   anywhere in the manuscript (the word "Slurm"/"tmux" never appears in
   manuscript text, only in internal repo docs), and this entire revision's
   work (Phases 1–2 of this audit cycle) has already been run this way
   without issue. Slurm vs. tmux is purely a job-scheduling/orchestration
   choice, orthogonal to whether the underlying computation is correct.
5. **Should the response-to-reviewers mention the rerun environment?**
   Recommended but not strictly required: a brief sentence (e.g., "the
   canonical policy-comparison results in the revised manuscript were
   regenerated on [venv/commit] using fixed random seeds; see Methods/
   Acknowledgements for compute provenance") adds transparency and
   preempts a reviewer question about reproducibility, at near-zero cost.
   This is consistent with the manuscript's own stated emphasis on
   reproducibility (lines 355, 360 stress "reproducible at the level of
   scripts, replay configuration, and generated outputs").
6. **Is a small cross-check against old Wolverine outputs necessary?**
   Not strictly necessary for this revision's validity, for two reasons:
   (a) the entire heavy_r1 chain for this revision (dataset build →
   training → validation eval, all from a single freshness-confirmed
   timestamp chain on 2026-06-18) is being produced consistently on this
   one local machine — there is no "patchwork" where part of one reported
   metric came from Wulver and part from local; (b) the manuscript's
   existing Table 3 is currently an unpopulated stub (per
   `tables/manuscript/table3_main_quantitative_comparison.tex`: "Canonical
   policy comparison ... was not found") — there are no prior Wulver-era
   Table-3 numbers to cross-check against in the first place, since the
   24h-timeout job (908326) never produced output (per
   `reports/kbs_full_policy_comparison_execution_plan.md` §2). A
   cross-check would be valuable as an extra confidence-builder if old
   partial Wulver outputs existed and overlapped on the same
   (trace, capacity, policy) cells, but no such overlapping artifact was
   found in this audit. Treat as optional, not blocking.

## 5. Recommendation (Step 5)

**A. Safe to proceed locally with tmux; update provenance only.**

Reasoning:

- The manuscript makes no exclusive or results-tied hardware claim — only a
  hedged Acknowledgements sentence that remains true regardless of where
  this revision's new numbers come from (§1, §4.1–4.2).
- tmux vs. Slurm is an orchestration choice with no bearing on scientific
  validity (§4.4).
- The pipeline is deterministic by design — fixed seeds throughout, no
  GPU dependency, `n_jobs=1` — so results are not "this machine's results,"
  they're reproducible outputs of a fixed seed/config/model artifact, which
  is the property that actually matters for a KBS reviewer (§3).
- This revision's entire heavy_r1 chain (dataset, training, validation,
  and the about-to-be-decided full sweep) is being generated consistently
  on **one machine, one venv** — there is no cross-machine mixing within a
  single reported metric, which is the cleanest possible provenance story
  (§4.6).
- Genuine prior Wulver usage is documented and real (job IDs in §1), so the
  Acknowledgements sentence is not retroactively falsified by doing this
  revision's work elsewhere — it is simply describing an earlier phase of
  the project's history.
- B (manuscript edits required) is not triggered because no manuscript text
  makes a now-false claim. C (prefer Wulver) is not warranted because there
  is no material scientific-validity gap — only a cosmetic precision
  opportunity in the Acknowledgements wording, which is optional (§4.2).

No code, manuscript, or report changes are required by this recommendation
itself. The only **optional** follow-ups (not executed, left for the
author's discretion): (a) sharpen the Acknowledgements sentence per §4.2,
and (b) add a one-line provenance note to the eventual response-to-reviewers
per §4.5. Neither blocks proceeding.

## 6. Final summary

1. **Recommendation: A** — safe to proceed locally with tmux; update
   provenance only (no manuscript edits required).
2. **Manuscript text mentioning Wolverine/Wulver/Slurm**: exactly one
   sentence, in the Acknowledgements section of `main.tex` (line 532):
   "...acknowledges the use of the Wulver high-performance computing system
   at the New Jersey Institute of Technology for part of the experimental
   evaluation." Hedged ("for part of"), not exclusive, not tied to any
   specific reported number. No other manuscript section, the cover letter,
   or the author agreement mentions hardware/cluster/runtime at all.
3. **Can local tmux results be used for the KBS revision? Yes.** The
   pipeline is fully seeded/deterministic (fixed seeds for sampling,
   RandomForest, HistGB; `Ridge` has no randomness; only one in-scope policy,
   `trust_and_doubt`, is seeded and uses a fixed seed=7); tmux is purely
   session management with no effect on the computation; and this revision's
   results will come consistently from one machine/venv, avoiding any
   cross-machine mixing within a single metric.
4. **Provenance to record**: git commit (`e9ac132c0aba12c640b0663af8c8bb6c946e81e1`),
   venv (`.venv_kbs_heavy_r1`: Python 3.12.3, numpy 2.4.6, scikit-learn
   1.9.0, scipy 1.17.1, joblib 1.5.3), machine class (local/cloud
   workstation, Intel i7-12700K, 20 threads, 62GiB RAM, CPU-only — no GPU
   used), and the fixed seeds already in use (training seed=7, shard
   split-seed=0, `trust_and_doubt` seed=7). All of this is now captured in
   this file and cross-referenced in the evidence manifest.
5. **Whether to proceed with the cap32 chunk locally**: this audit finds no
   scientific or manuscript-consistency objection to doing so — the cap32
   command is unchanged and ready
   (`reports/kbs_full_policy_comparison_execution_plan.md` §6, recommendation
   Option B, already approved by the prior sanity audit). This audit clears
   the platform-consistency concern specifically; it does not by itself
   constitute the launch go-ahead.
6. **Per your instruction, the cap32 chunk has NOT been launched.** Nothing
   was pushed, merged, deleted, or overwritten during this audit. Awaiting
   your explicit approval before launching it.
