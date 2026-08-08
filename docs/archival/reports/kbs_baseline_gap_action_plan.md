# Baseline gap action plan (2026-06-19)

**Third update, same day (2026-06-19), final cap32 correction pass**: the
FIFO row's "include before cap64?" question is now resolved in practice
for cap32. The obsolete `cap32_with_sieve` run was stopped and replaced by
the corrected `cap32_with_sieve_fifo` canonical chunk. cap64/cap128/cap256
remain not launched.

Read-only inspection only. No code was written, no policy was implemented, no
new run was launched in this pass — this is a classification + plan.

**Update, same day (2026-06-19), later pass**: the SIEVE row below is now
**superseded by implementation**. SIEVE has been implemented
(`src/lafc/policies/sieve.py`), verified against the official NSDI'24 paper
and reference code (`reports/kbs_sieve_source_verification.md`), wired into
the canonical `POLICIES` dict, tested (8 new unit tests, full suite still
291/291 passing), and smoke-tested. Full detail:
`reports/kbs_sieve_implementation_report.md`. The row below is left as
written for provenance (it correctly predicted SIEVE as the lowest-risk,
highest-priority addition); treat its "Needed action" column as **done**
except for the run-it-through-cap64+ part, which is unaffected by this
update (cap64/128/256 still not launched). FIFO-Reinsertion's row is
**unaffected** — its definition is still unresolved.

**Second update, same day (2026-06-19), still later pass**: the "optionally
re-run cap32 retroactively" half of the SIEVE row's "Needed action" column
is now a concrete plan, not just an option — see
`reports/kbs_cap32_rerun_with_sieve_plan.md`. Existing no-SIEVE cap32
outputs stay untouched (`CANONICAL CHUNK WITHOUT SIEVE / SUPERSEDED FOR
FINAL TABLE IF SIEVE-INCLUSIVE CHUNKS ARE USED`); a new SIEVE-inclusive
`..._cap32_with_sieve.csv/.md` rerun is drafted and ready to launch
(`PENDING / CANONICAL CHUNK WITH SIEVE`) but **not yet launched**, pending
explicit approval. "Run it through cap64/128/256" itself remains undone —
cap64/128/256 are still not launched.

**Third update, same day (2026-06-19), FIFO-Reinsertion audit pass**:
FIFO-Reinsertion is now also **superseded by implementation + audit**. The
repo contains `src/lafc/policies/fifo_reinsertion.py`, both runner
registrations, dedicated tests, and tiny smoke outputs; see
`reports/kbs_fifo_reinsertion_implementation_report.md` and
`reports/kbs_fifo_reinsertion_baseline_audit.md`. The original row below
is left intact for provenance, but its "exact intended definition is
unresolved" text is now stale; use
`reports/kbs_halp_fifo_source_verification.md` plus the new audit report
for current status.

## Method

For each baseline named in the reviewer concern list, checked three things
directly against the repo, not from memory or prior reports:

1. **Implemented?** — `grep`/`ls` across `src/lafc/policies/*.py` for a class,
   plus whether that class is wired into a runnable registry.
2. **Citation exists?** — `grep -i` against the real manuscript's
   `refs.bib` (extracted zip at `/tmp/kbs_manuscript_inspect/refs.bib`) and
   whether it's discussed in `main.tex` (Related Work or Table 2's
   `tab:main_policy_families`).
3. **Wired into the canonical heavy_r1 pipeline?** — whether the policy name
   appears in `scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES` dict
   (the only registry that feeds `TABLE3_POLICIES` / the heavy_r1 cap32+
   results actually being produced right now). This is a stricter bar than
   "implemented somewhere in the repo" — several baselines below are
   implemented but only wired into *older, different* runner scripts
   (`src/lafc/runner/run_policy.py` and one-off `scripts/run_*.py` files),
   not the current canonical pipeline.

## Summary table

| Baseline | Requested by reviewer? | Already implemented? | Citation exists? | Feasible before July 8? | Needed action | Risk |
|---|---|---|---|---|---|---|
| **HALP** | Yes (named explicitly) | No | Yes — `song2023halp` in `refs.bib`, discussed in Related Work | **No**, not as a faithful empirical reimplementation — HALP is a preference-learning system over re-access outcomes, materially more complex than the O(1)/O(capacity) classical baselines here | Decide: (a) keep citation+differentiation prose only [already exists, lowest risk], or (b) build a simplified proxy [schedule risk, may not satisfy "the real HALP"], or (c) add an explicit, honest limitation sentence. Recommend (a)+(c) combined. | Low if (a)+(c); high if a reviewer insists on (b) under this timeline |
| **SIEVE** | Yes (named explicitly) | **No** — zero hits in `src/lafc/policies/`, in any registry, or in `refs.bib`/`main.tex`/`docs/baselines.md` | No | **Yes** — algorithm is simple (FIFO + one "hand" pointer + visited bit), `BasePolicy` interface is ~45-80 lines for comparable baselines (see `lru.py`) | (1) add `src/lafc/policies/sieve.py`; (2) add one line to `POLICIES` dict in `run_policy_comparison_wulver_v1.py`; (3) add a handful of unit tests; (4) add bib entry; (5) run it through whichever capacity chunks remain (cap64/128/256), and optionally re-run cap32 retroactively since it's cheap relative to the chunks already done | **Low.** This is the strongest "add it" candidate. |
| **FIFO-Reinsertion** | Yes (named explicitly) | No | No | Likely yes, **but the exact intended definition is unresolved** — could mean a textbook "FIFO + requeue-to-tail on hit" (near-identical behavior to LRU, low informational value as a distinct baseline) or a specific mechanism like S3-FIFO's small→main reinsertion. No verbatim reviewer text pinning this down was available in this session. | **First** confirm/clarify the exact variant intended (re-check the original reviewer comment text, or pick the most commonly-cited "FIFO-Reinsertion" definition and state the choice explicitly in the response letter); only then implement (similar effort to SIEVE once defined) | Low implementation risk once defined; moderate risk of silently implementing the "wrong" variant if not checked |
| **PARROT** | **No** — not in the reviewer concern list used for this audit | No | Yes — `liu2020parrot` in `refs.bib`, discussed in Related Work | N/A | None beyond existing citation/discussion | Low |
| **Mockingjay** | **No** — not in the reviewer concern list used for this audit | No | Yes — `shah2022mockingjay` in `refs.bib`, discussed in Related Work | N/A | None beyond existing citation/discussion | Low |
| **LRU** | Implicit (baseline-sufficiency concern) | Yes — `lru.py`, in `TABLE3_POLICIES`, in cap32 results | Yes — `fiat1991competitive_paging` | Already done | None | None |
| **Marker** | Implicit | **Partially** — `MarkerPolicy` exists (`src/lafc/policies/marker.py`) and is registered in the *older* `src/lafc/runner/run_policy.py` registry and documented in `docs/baselines.md`, but it is **not** wired into the canonical heavy_r1 script's `POLICIES` dict, so it has **no heavy_r1 empirical numbers** and is not in `TABLE3_POLICIES` | Yes — described in Table 2 (`tab:main_policy_families`) of `main.tex` as "Classical phase-based paging reference" | Trivial if wanted — one registry line to add it to `run_policy_comparison_wulver_v1.py`, then it could ride along with cap64+ | If the manuscript's own Table 2 names Marker as an evaluated baseline family, double-check whether the team wants real heavy_r1 Marker numbers to back that row, or whether Table 2 is meant as related-work/context only (no run needed) | Low-moderate — currently a roster/results mismatch: a baseline *named* in the manuscript's own policy table has no corresponding empirical row |
| **Predictive Marker** | Implicit | Yes — `predictive_marker.py`, in `TABLE3_POLICIES`, in cap32 results | Yes — `lykouris2021competitive_caching` | Already done | None | None |
| **Trust-and-Doubt** | Implicit | Yes — `trust_and_doubt.py`, in `TABLE3_POLICIES`, in cap32 results | Yes — `antoniadis2020online_metric_untrusted` | Already done | None | None |
| **REST (`rest_v1`)** | Implicit (it's one of the 6 `TABLE3_POLICIES` actually run) | Yes — `rest_v1.py`, in `TABLE3_POLICIES`, in cap32 results, build script assigns it the manuscript short-label `"REST"` (`SHORT_POLICY_LABEL["rest_v1"] = "REST"` in `scripts/paper/build_kbs_main_manuscript_artifacts.py`) | **No — confirmed gap, see below** | N/A (already implemented/run) | Add a manuscript-consistency fix: either a new Table 2 row labeled "REST" with `rest_v1`'s actual citation/description, or fold its description into an existing row if it is genuinely meant to extend one. **Do not assume it's the same row as "R-FTP+Marker" — see evidence below that those are two different code policies.** | Low engineering risk, but real: Table 3 (once finalized) will show a results column for a policy ("REST") that Table 2 never defines or cites |

## Detail: the REST / Marker / R-FTP+Marker mapping (the one genuinely confusing case)

This needed extra digging because the names are similar. Resolved with direct
evidence, not guesswork:

- `scripts/paper/build_kbs_main_manuscript_artifacts.py` defines
  `SHORT_POLICY_LABEL["rest_v1"] = "REST"` — i.e., the build script's own
  intended manuscript label for the canonically-evaluated `rest_v1` policy is
  **"REST"**, not "R-FTP+Marker".
- `main.tex`'s Table 2 (`tab:main_policy_families`) has **no row literally
  labeled "REST" or "ReST"**. Its closest-sounding row, **"R-FTP+Marker"**
  ("Robust follow-the-prediction with marker fallback",
  `\cite{wei2020better_simpler_lac,chledowski2021robust_lac_experimental}`),
  turns out to describe a **different, separate** code policy:
  `src/lafc/policies/robust_ftp_marker_combiner.py`'s
  `RobustFtPDeterministicMarkerCombiner` class — confirmed because
  `docs/baselines.md` §"Baseline 4b: RobustFtP-D with MARKER fallback
  (Chłędowski et al., 2021)" cites the *same* Chłędowski 2021 reference and
  documents `robust_ftp_d_marker` / `robust_ftp` as its registry names in
  `src/lafc/runner/run_policy.py` — matching Table 2's citation exactly, and
  matching neither `rest_v1.py`'s docstring/citations nor its registry name.
- `rest_v1.py`'s own docstring (read directly) describes it as "ReST v1:
  Regret-Driven Selective Trust for unweighted paging" — a TRUST/ABSTAIN
  gating policy that falls back to **LRU**, not to a marker algorithm. This
  is conceptually distinct from "R-FTP+Marker"'s marker-fallback design, not
  just a renamed duplicate.
- **Conclusion**: "R-FTP+Marker" and "REST" are two different baselines that
  happen to sound similar. `R-FTP+Marker`'s actual code
  (`RobustFtPDeterministicMarkerCombiner`) is implemented, documented in
  `docs/baselines.md`, and registered in `src/lafc/runner/run_policy.py` —
  but **not** wired into the canonical heavy_r1 script, so it has no
  heavy_r1 numbers either (same situation as plain `Marker`, see table row
  above). `rest_v1`/"REST" **is** canonically run and has heavy_r1 numbers
  (it's in the cap32 results), but has **no Table 2 row of its own** in the
  current manuscript.

This means there are, in effect, **two distinct loose ends**, not one:

1. Table 2 names two baselines (`Marker`, `R-FTP+Marker`) whose code exists
   and is documented but was never run through the canonical heavy_r1
   pipeline that produced cap32/cap64+ — so neither has heavy_r1 numbers to
   back its row.
2. One canonically-run, heavy_r1-evaluated policy (`rest_v1`/"REST") has no
   row in Table 2 at all.

## Recommendation

1. **Implement SIEVE first.** Lowest engineering risk, explicitly requested,
   genuinely absent (no citation, no code, anywhere). Can ride along with
   whichever capacity chunk is run next rather than requiring a separate
   multi-day pass.
2. **Resolve "FIFO-Reinsertion"'s exact definition before implementing it.**
   Don't guess at a specific algorithm and present it as "the" reviewer
   request — confirm the intended variant first (or state the chosen
   interpretation explicitly in the response-to-reviewers letter), then
   implement; the effort is comparable to SIEVE once defined.
3. **For HALP**, do not attempt a full faithful reimplementation under this
   timeline. The lowest-risk path is the existing citation + Related-Work
   differentiation, paired with an explicit, honest limitation statement in
   the response letter (e.g., "a full empirical HALP comparison requires
   reproducing its preference-learning training pipeline, which we consider
   out of scope for this revision cycle; we instead differentiate
   analytically in Section X"). Do not claim an empirical HALP number that
   doesn't exist.
4. **PARROT and Mockingjay need no action** — neither was in the reviewer
   concern list used for this audit; both already have adequate
   citation+discussion.
5. **Fix the REST / Marker / R-FTP+Marker manuscript-consistency gap**
   regardless of the SIEVE/FIFO-Reinsertion decision — this is a pure
   documentation fix (no new compute), and is the kind of internal
   inconsistency (a results-table policy with no roster-table row) that a
   careful reviewer or AE could flag on its own. Two options, in increasing
   order of effort:
   - **Minimal**: add one Table 2 row for "REST" with `rest_v1`'s actual
     description/citation (none currently exists in `refs.bib` for ReST's
     own method — would need to confirm whether `rest_v1` is meant to cite
     prior work or is itself an internal/exploratory baseline with no
     external citation; this needs a human decision, not assumed here).
   - **Fuller**: also decide whether `Marker` and `R-FTP+Marker` (Table 2
     rows with no heavy_r1 numbers) should be (a) run through the canonical
     pipeline so Table 2's rows have real backing numbers, or (b) explicitly
     marked in prose as "discussed for context, not included in the
     quantitative comparison" to avoid implying they were run when they
     weren't.
6. None of the above blocks cap64. SIEVE (and FIFO-Reinsertion, once
   defined) are the only items with new-compute implications, and per the
   cap32 report's own recommendation, finalizing their scope before cap64
   avoids a second multi-day sweep later — but this is a scheduling
   suggestion, not a hard dependency the user needs to resolve before any
   further read-only work.
