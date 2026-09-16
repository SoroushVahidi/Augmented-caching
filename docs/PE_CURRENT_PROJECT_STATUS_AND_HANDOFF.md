# LAFC-Evict / Performance Evaluation (PE) — Current Project Status and Handoff

Snapshot time: 2026-09-16 ~02:20 UTC (2026-09-15 ~22:20 EDT). This is a
point-in-time snapshot, not a live document — re-verify branch HEADs, job
states, and process states before relying on anything time-sensitive below.

This repository (`augmented-caching`) hosts the code and evidence for **two
separate, unrelated publication lines**. This document covers only the **PE**
line. For the **KBS** (Knowledge-Based Systems) line, see
`CANONICAL_KBS_SUBMISSION.md` at the repo root — do not conflate the two;
branches, docs, and worktrees prefixed `kbs`, `feat/*-baseline`,
`feat/reviewer-*`, and `feat/supervision-objective-ablation` belong to KBS,
not PE, even though they live in this same repo.

## A. Project purpose

LAFC-Evict is a counterfactual benchmark/supervision dataset for learned
cache eviction — not a new eviction policy. For each full-cache miss
(an eviction decision), it scores every resident candidate object under a
finite-horizon counterfactual: "if we evicted this specific candidate instead,
how many subsequent misses occur over the next H requests under a fixed
continuation policy?" This produces candidate-level, decision-level, and
pairwise-level supervision that a researcher can use to train and evaluate
learned eviction/scoring models, study offline-vs-closed-loop agreement, and
study continuation-policy sensitivity — all without needing a live cache
simulator in the loop for the supervised-learning parts. The PE (Elsevier
*Performance Evaluation*) manuscript is the paper describing this benchmark
and its validation.

## B. Two-repository layout

- **Publication/manuscript repository**: `lafc-evict-dataset`
  (`/home/soroush/projects/lafc-evict-dataset/repo`) — the PE manuscript
  LaTeX source, the public dataset-release packaging/documentation, and
  release-facing provenance docs. Multiple worktrees exist under both
  `.claude/worktrees/<name>` (agent-created) and `worktrees/<name>`
  (human-named).
- **Scientific/experiment repository**: `augmented-caching`
  (`/home/soroush/projects/augmented-caching/repo`) — all experiment code,
  policy implementations, dataset generators, and evidence directories for
  both the PE and KBS lines. Worktrees live under
  `augmented-caching/worktrees/<short-name>`.

## C. Canonical manuscript

- Branch: `polish/pe-results-independent-cleanup-20260915`
- HEAD as of this snapshot: `aa535ae` — **pushed to origin** this session
  (`git push -u origin polish/pe-results-independent-cleanup-20260915`,
  verified local SHA == remote SHA).
- Worktree: `/home/soroush/projects/lafc-evict-dataset/worktrees/pe-results-polish`
- PDF: `paper/performance_evaluation/latex/main.pdf`, 47 pages (as built this
  session).
- Latest QA status: clean `latexmk` build, 0 undefined references/citations,
  equation-numbering gate PASS (3/3 displayed formulas numbered and labeled),
  citation-density gate PASS (≤4 distinct references per sentence), no
  scientific numbers changed in the latest polish pass. See
  `paper/performance_evaluation/PE_VENUE_COMPLIANCE_AUDIT.md` in that repo
  for the full Elsevier *Performance Evaluation* Guide-for-Authors
  compliance check (no formal page/word limit found).
- The Wulver acknowledgment is **intentionally absent** from the manuscript:
  the provenance record for every result currently reported
  (`analysis/closed_loop_tier1_evidence_20260914/.../provenance.json`)
  records hostname `al-khwarizmi`, a local workstation, not Wulver. Add the
  acknowledgment only once Wulver-derived long-horizon results actually enter
  the manuscript (see section J).

## D. Current PE scientific evidence (canonical / completed)

| Artifact | Path (repo-relative unless noted) | Status | Manuscript-used |
|---|---|---|---|
| Tier-1 closed-loop production (LRU/MRU/random/SIEVE, 230 runs) | `analysis/closed_loop_tier1_evidence_20260914/` | COMPLETE, VALIDATED | Yes |
| Full-population MRU continuation census (60/60 chunks, 2,363,286 decision-horizon records) | raw 1.6G output in `lafc-evict-dataset` worktree `.claude/worktrees/continuation-mru-population-census-20260914`; validation in `.claude/worktrees/continuation-mru-census-validation-20260915` | COMPLETE, VALIDATED | Yes (compact summaries only; raw output preserved separately) |
| Mechanistic/linkage workload analysis | `analysis/closed_loop_mechanistic_analysis_20260914/` | COMPLETE, VALIDATED | Yes |
| Continuation-sensitivity pilot (5,000+500 pre-registered decisions) | branch `experiment/continuation-sensitivity-pilot-20260914` | COMPLETE, VALIDATED | Yes |
| Cap32 long-horizon preflight + cap256 timing probe | branch `experiment/pe-long-horizon-preflight-20260915` (repo root, currently dirty — see section F/K) | COMPLETE (timing probe is timing-only, not a scientific result) | No (preflight only; confirms H∈{4,8,16} was the full prior horizon coverage before the current DAG) |
| Canonical H∈{4,8,16} candidate-label dataset | `paper/sigmod2027/results/candidate_label_stats/y_loss_summary.csv` and the `evict_value_v1_wulver_heavy_r1` derived dataset | COMPLETE, VALIDATED | Yes (this is the existing basis the long-horizon DAG extends) |

## E. Active / pending experiments (truthful as of snapshot time — do not treat as complete)

- **Long-horizon Wulver DAG** (H=32/64/128): production job `1288869`
  (array, 17/20 tasks COMPLETED exit 0:0, 3 tasks still RUNNING at ~3h
  elapsed each), validation job `1288880` (PENDING, dependency-blocked on
  production), aggregation job `1288881` (PENDING, dependency-blocked on
  validation). **ACTIVE. No cells validated yet. No results exist to
  integrate.** Provenance/launch code: branch
  `experiment/pe-long-horizon-production-prep-20260915` (pushed to origin
  this session).
- **Publication-grade learned model, attempt 2**: branch
  `experiment/pe-publication-learned-retrain-attempt2-20260915`, worktree
  `augmented-caching/worktrees/pe-publication-learned-retrain-attempt2-20260915`.
  Training complete and model selection frozen (`hist_gb`, sha256-verified).
  Test evaluation started 2026-09-16T01:44:24Z and was **still running**
  (PID confirmed alive, CPU-active) as of this snapshot, with
  `EXECUTION_STATE.json` reporting `stage: "test_starting"` for over 40
  minutes with no new log lines past the Python-version header — this may
  simply be a long silent evaluation pass with no progress checkpoints
  wired into that stage, but has not been independently confirmed either
  way. **This worktree was not touched in this pass because its process is
  active** — do not commit, edit, or otherwise modify it while the test is
  running. **Publication gate: PENDING.**
- **Tier-2 LFU closed-loop campaign**: branch
  `experiment/pe-tier2-closed-loop-integration-20260915` (pushed to origin
  this session). LFU policy implementation and unit tests are committed and
  passing; campaign manifest, results aggregator, and launch guard are
  prepared. **The campaign itself has not been launched.** This is a
  deliberate next step, not an oversight — see section K.

## F. Failed / historical / superseded experiments

- **Attempt 1 of the publication-grade learned model**
  (`experiment/pe-publication-learned-retrain-20260915`): interrupted, not
  failed outright — training completed (model sha256
  `c5bf9840f6be4b82903c82bbf1e3ace87c5b4656e4bed163eba84f2ee805611e`) but no
  model-selection freeze marker and no test-evaluation step was ever run.
  Preserved as historical evidence this session (commit `6c17c0b`, pushed to
  origin) with an explicit "NOT PUBLICATION VALID" statement in the commit
  message. **Do not use this model or any number derived from it for final
  manuscript claims** — attempt 2 above is the current, in-progress
  publication attempt.
- **Earlier long-horizon production attempt, job 1288536**: failed on HOME
  filesystem quota (per the commit message on
  `experiment/pe-long-horizon-production-prep-20260915`,
  "replaces failed HOME campaign 1288536"); replaced by the current
  scratch-backed DAG (1288869/1288880/1288881). Not independently
  re-verified beyond that commit message.
- **Cap256 timing probe**: explicitly a single-family timing-only probe, not
  a scientific result — do not cite as evidence of anything beyond runtime
  feasibility.
- **`manuscript/performance-evaluation-template-20260913`** (in the
  `lafc-evict-dataset` repo): superseded by the current polish branch. Not
  removed; flagged as a Query-3 candidate for archival/removal with human
  approval.

## G. Do not recompute

See `docs/PE_DO_NOT_RECOMPUTE.md` (this repo) for the full list with reasons.

## H. Reader-facing naming

Internal code/manifests/evidence use the identifier `cloudphysics` for one of
the five trace families. Reader-facing manuscript prose calls this family
**`alibaba-block`** — it is the Alibaba Cloud Elastic Block Storage
production trace, **not** the VMware CloudPhysics dataset. This mapping is
documented once, clearly, in a footnote in
`paper/performance_evaluation/latex/sections/03_benchmark_design.tex` (in
the `lafc-evict-dataset` repo). The other four reader-facing family names
(`metacdn`, `metakv`, `twemcache`, `wiki2018`) match their internal
identifiers directly.

## I. Public release scope (as documented in-repo; not independently verified against live hosting)

- The full scientific evaluation corpus (five families, ~277,995,072
  candidate rows, 2,363,286 decision-horizon records at H∈{4,8,16}) is
  **not** fully publicly redistributed — `alibaba-block`, `metacdn`,
  `metakv`, and `twemcache` remain uncleared for public derived-data
  redistribution under their own upstream source licensing.
- The current public release (v0.3) is **wiki2018-only**, distributed via
  Hugging Face (`SoroushVahidi/lafc-evict`) and AWS Open Data (sponsored
  hosting, up to 600GB for two years per the AWS Open Data Sponsorship
  Program).
- A Zenodo v0.2 record and related publication-state docs exist in the
  `lafc-evict-dataset` repo (`ZENODO_V0_2_*.md`, `V0_3_*.md`,
  `lafc_evict_v0_2_publication_status.md`) — this snapshot did not
  independently re-verify their live-platform state (no external network
  fetches performed).

## J. Pending manuscript integration

None of the following are integrated yet, and none should be until the
underlying evidence is COMPLETE_VALID:

1. H=32/64/128 long-horizon results (blocked on the active DAG above).
2. Publication-grade learned-model results (blocked on attempt 2's test
   evaluation, then its publication gate).
3. LFU closed-loop results (blocked on running the prepared-but-not-launched
   Tier-2 campaign).
4. The Wulver acknowledgment sentence — add only once a Wulver-run result
   actually appears in the manuscript's reported evidence. Recommended
   wording (previously drafted, not yet inserted):
   *"Computationally intensive experiments used the Wulver high-performance
   computing system at the New Jersey Institute of Technology."*
5. Any resulting final-pass updates to the abstract, discussion, limitations,
   and conclusion sections that follow from 1–3 above.

## K. Immediate next steps (in order)

1. Let the long-horizon DAG finish (production → validation → aggregation);
   do not intervene unless a task fails.
2. Let learned-model attempt 2's test evaluation finish; then apply the
   Task-8-style completion gate (COMPLETE_VALID / COMPLETE_BUT_GATE_FAILED /
   PARTIAL / FAILED) before treating its results as usable.
3. Review and resolve the ~14 dirty tracked-file changes in the
   `augmented-caching` repo root worktree (branch
   `experiment/pe-long-horizon-preflight-20260915`) — this pass determined
   it's a mix of stale checkout residue from the unrelated
   `chore/repository-polish` branch and possibly-unique changes to
   `scripts/validation/revision_readiness.py` / `revision_status.py` and
   their tests; needs a human/agent pass to separate and either commit or
   discard the right pieces (nothing was touched this session).
4. Once the DAG and attempt 2 both resolve, launch the Tier-2 LFU closed-loop
   campaign (manifest/aggregator/launch-guard already prepared on
   `experiment/pe-tier2-closed-loop-integration-20260915`).
5. Only after 1–4: integrate results into the manuscript (section J) and run
   a final results-dependent QA pass.
6. Separately, reconcile local `main` being 12 commits behind `origin/main`
   before basing genuinely new (non-PE) work off local `main` — the 12
   origin-only commits are pure KBS submission/publishing work and, based on
   ancestry alone, do not appear to conflict with the PE branches (all PE
   branches and origin/main's extra commits both descend cleanly from the
   same commit, `a01a60d`), but this was not verified via an actual merge
   dry run in this pass and reconciliation was deliberately deferred to a
   future pass, not attempted here.

## L. Persistence rule (standing operational rule for this project)

- Any long-running **local** computation must run in a detached `tmux`
  session, not tied to an interactive terminal/agent/chat lifetime.
- Any long-running **Wulver** computation must be submitted via a saved
  `.sbatch` file with `sbatch`, not run interactively over SSH.
- No long-running experiment should depend on an SSH connection, a Cursor/
  agent session, or a chat session staying open.
