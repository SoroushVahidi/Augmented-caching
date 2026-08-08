# Pre-cap64 baseline decision memo (2026-06-19)

**Fourth update, same day (2026-06-19), final cap32 correction pass**: the
baseline decision is now operationalized. The obsolete
`cap32_with_sieve` job was stopped because it omitted
`fifo_reinsertion`, and the corrected `cap32_with_sieve_fifo` chunk is now
running in tmux with both SIEVE and FIFO-Reinsertion included. cap64/
cap128/cap256 remain not launched.

Zero-compute decision memo. No code was written, no policy was implemented,
no job was launched in producing this memo. cap64/cap128/cap256 remain **not
launched**, per instruction.

**Update, same day (2026-06-19), later pass**: §1's recommendation has now
been **acted on**. SIEVE was independently verified against the official
NSDI'24 paper and reference implementation
(`reports/kbs_sieve_source_verification.md`), implemented as
`src/lafc/policies/sieve.py`, wired into both the canonical heavy_r1
pipeline (`scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES` dict)
and the older runner's `POLICY_REGISTRY`, covered by 8 new unit tests (all
passing, plus the full 291-test suite with no regressions), and
smoke-tested on one tiny sample (1 trace, 1000 requests, capacity 32: SIEVE
496 misses vs. LRU 503). Full detail in
`reports/kbs_sieve_implementation_report.md`. **This does not change §5's
conclusion**: cap32 still has no SIEVE column and would need a rerun to get
one, and cap64/cap128/cap256 are still **not launched**. §2 (FIFO-
Reinsertion's exact variant) remains undecided and is unaffected by this
update.

**Third update, same day (2026-06-19), still later pass**: the "rerun to get
one" clause above is now a concrete, approved-pending plan —
`reports/kbs_cap32_rerun_with_sieve_plan.md`. Decision: leave the existing
no-SIEVE cap32 outputs untouched (status `CANONICAL CHUNK WITHOUT SIEVE /
SUPERSEDED FOR FINAL TABLE IF SIEVE-INCLUSIVE CHUNKS ARE USED`) and write a
SIEVE-inclusive rerun to new, distinctly-named
`..._cap32_with_sieve.csv/.md` files (status `PENDING / CANONICAL CHUNK
WITH SIEVE`), excluding `blind_oracle` (already outside `TABLE3_POLICIES`,
already known diagnostic-only per `kbs_validation_sanity_audit.md` Q3). The
tmux command is fully drafted in that plan's §5 but **not launched**,
pending explicit approval. cap64/128/256 remain not launched regardless.

**Fourth update, same day (2026-06-19), FIFO-Reinsertion audit pass**:
FIFO-Reinsertion is no longer just a pending definition question. The repo
already contains `src/lafc/policies/fifo_reinsertion.py`, both runner
registrations, dedicated tests, and a tiny smoke run; the new strict audit
`reports/kbs_fifo_reinsertion_baseline_audit.md` concludes that the current
implementation is scientifically defensible as a FIFO-Reinsertion / CLOCK /
Second-Chance-family baseline and recommends including it in the final
canonical policy set before cap64. This changes the practical question from
"what does FIFO-Reinsertion mean?" to "do we commit to a final baseline set
that includes it before launching cap64?"

**Why this memo exists**: Reviewer #3 explicitly requested SIEVE and
FIFO-Reinsertion (R3-Issue3, R3-Rec2) and flagged insufficient
differentiation from HALP (R3-Issue2, R3-Rec2) — see
`reports/kbs_real_reviewer_comments.md`. Launching the remaining
cap64/cap128/cap256 capacity chunks (each a multi-hour, possibly multi-day,
unrepeatable sweep per `reports/kbs_full_policy_comparison_execution_plan.md`)
without first deciding whether SIEVE and/or FIFO-Reinsertion ride along risks
having to repeat all three chunks later just to add two more policy columns.
This memo answers the six questions needed to make that call, building
directly on the classification already done in
`reports/kbs_baseline_gap_action_plan.md` (prior session), now reframed
against the exact real reviewer text rather than a category-level paraphrase.

---

## 1. Should SIEVE be implemented before cap64/128/256?

**Yes, recommended.**

- **Requested explicitly**: R3-Issue3 names SIEVE (NSDI '24) by name;
  R3-Rec2 repeats the ask. Not implementing it leaves an exact, easily
  re-checked gap a reviewer could flag again on a second round.
- **Engineering risk is low.** Confirmed via direct repo inspection
  (`reports/kbs_baseline_gap_action_plan.md` §Summary table): zero hits for
  "sieve" anywhere in `src/lafc/policies/`, any registry, `refs.bib`,
  `main.tex`, or `docs/baselines.md` — a clean implementation, not a
  conflicting one. The published SIEVE design (FIFO + one "hand" pointer +
  a per-object visited bit) is comparable in complexity to the existing
  ~45-80 line `BasePolicy` implementations already in the repo (e.g.
  `src/lafc/policies/lru.py`).
- **Integration path is mechanical**: add `src/lafc/policies/sieve.py`, add
  one line to the `POLICIES` dict in
  `scripts/run_policy_comparison_wulver_v1.py`, add unit tests, add a bib
  entry, add a Table 2 row. None of this requires retraining
  `evict_value_v1` or touching the dataset/training pipeline.
- **Risk of not doing it now**: if cap64/128/256 are launched without SIEVE
  and a reviewer in a later round still wants it, the entire remaining sweep
  would need to be re-run to get SIEVE numbers at the same capacities/traces
  — the expensive scenario this memo exists to avoid.

## 2. Should FIFO-Reinsertion be implemented or deferred?

**Defer the implementation decision until the definition is pinned down —
do not implement blind.**

- **Requested explicitly** (R3-Issue3, R3-Rec2), but the real comment text
  names the baseline without defining the mechanism. "FIFO-Reinsertion"
  is genuinely ambiguous between at least two readings:
  - A textbook "FIFO queue, re-enqueued to the tail on a hit" — behaviorally
    close to LRU/Second-Chance, of limited informational value as a
    *distinct* baseline.
  - An S3-FIFO-style small-queue/main-queue reinsertion mechanism — a
    materially different, more interesting algorithm closer to the spirit
    of "modern lightweight algorithms" the comment is grouping it with.
- Implementing the wrong variant and presenting it as "the" FIFO-Reinsertion
  the reviewer asked for is a worse outcome than asking for clarification or
  explicitly stating the chosen interpretation in the response letter — a
  reviewer who meant the other variant would not consider this resolved.
- **Recommended path**: in the response-to-reviewers letter, state the
  specific variant being implemented (recommend the S3-FIFO-style
  reinsertion reading, since it is the one that matches "recently proposed
  lightweight algorithms that achieve strong performance with minimal
  overhead" — the textbook FIFO+requeue variant does not fit that
  description well). Implement that variant once stated. Engineering effort
  is comparable to SIEVE once the definition is fixed.
- This does **not** need to block SIEVE — the two are independent
  decisions, and SIEVE's lack of ambiguity makes it the cleaner, faster win
  of the two.

## 3. Is HALP implementation feasible before July 8?

**No, not as a faithful empirical reimplementation.**

- HALP (R3-Issue2's `song2023halp`) is a preference-learning system that
  learns candidate-level preferences from realized future re-access
  outcomes — a materially more complex training pipeline than the
  classical O(1)/O(capacity) baselines already implemented (LRU,
  Predictive Marker, Trust-and-Doubt, BO/LRU combiner, REST), all of which
  require no learned model at decision time except `evict_value_v1` itself.
  Faithfully reproducing HALP would require its own training pipeline,
  hyperparameter selection, and validation — comparable in scope to the
  `evict_value_v1` pipeline itself, which took the bulk of this project's
  engineering effort to date.
- Due date is 2026-07-08 (per `reports/kbs_real_reviewer_comments.md`).
  Today is 2026-06-19. That leaves roughly 19 days, against which the
  remaining canonical sweep (cap64/128/256, ~1.5-3 days of compute per
  `kbs_full_policy_comparison_execution_plan.md`) and a from-scratch
  preference-learning reimplementation would compete directly for the same
  calendar time. Attempting both raises real risk of finishing neither
  cleanly.
- HALP is already cited (`song2023halp` in `refs.bib`) and differentiated
  in prose in Related Work (per `kbs_baseline_gap_action_plan.md`'s prior
  finding) — this lowers the cost of declining the empirical comparison,
  since the manuscript isn't starting from zero on this point.

## 4. If not implementing HALP, what exact response should we give?

Recommended response text for R3-Issue2/R3-Rec2 (HALP half), to be used as
the basis for the placeholder already in
`reports/kbs_response_to_reviewers_skeleton.md`:

> We thank the reviewer for highlighting HALP as a closely related method.
> We agree that HALP's candidate-level preference learning from realized
> future re-access outcomes is operationally close to our approach. The key
> distinction is that HALP learns a relative *preference* signal between
> candidates from observed outcomes, whereas our finite-horizon replay
> target explicitly estimates the *magnitude* of downstream miss-count harm
> under counterfactual eviction at each decision point — a direct, replay-
> grounded cost estimate rather than a learned ranking. We have expanded
> [Section X] to state this distinction explicitly. A faithful empirical
> reproduction of HALP requires reimplementing its preference-learning
> training pipeline, which we consider out of scope for the timeline of
> this revision; we flag this as a concrete direction for future empirical
> comparison in our Limitations section.

This is an **honest limitation statement, not an empirical claim** — it
must not be paired with any invented HALP number. This mirrors the
treatment `kbs_baseline_gap_action_plan.md` already recommended for
PARROT/Mockingjay (citation-only, no fabricated empirical row).

## 5. Would running cap64/128/256 now without SIEVE/FIFO-Reinsertion force a rerun later?

**Yes — this is the central risk this memo is meant to head off.**

- `scripts/run_policy_comparison_wulver_v1.py` (per
  `kbs_full_policy_comparison_execution_plan.md`) has no checkpointing
  across policies within a chunk and writes its CSV/MD exactly once at the
  end of a full run (confirmed directly from the cap32 run log, which is
  only 2 lines: a "Wrote ..." line and an exit-code line). Adding a new
  policy after a chunk has already run means **re-running that entire
  chunk** (all 7 traces × all existing policies, not just the new one) to
  get a single CSV with consistent columns/rows for that capacity — the
  script does not support adding a policy column to an existing CSV.
- Each chunk (cap64/128/256) costs multiple hours of wall-clock (cap32 took
  5:13:02 for 7 traces × 7 policies at one capacity; the execution plan's
  own ±2x estimate band suggests a similar or longer time per remaining
  chunk). Re-running any of them solely to add SIEVE/FIFO-Reinsertion
  columns would waste that wall-clock a second time.
- Conversely, **adding SIEVE now costs nothing extra in re-run risk**: if
  it's wired into `POLICIES` before cap64 is launched, cap64/128/256 (and
  optionally a cheap retroactive cap32 re-run, since cap32 alone took
  ~5 hours, far less than a full chunk at higher capacities) would produce
  SIEVE numbers as part of the same run, at no marginal wall-clock cost
  beyond what one extra policy adds per request (small, since SIEVE is
  O(1) by design).

## 6. Recommendation: launch cap64 now, implement SIEVE first, or make a smaller benchmark first?

**Implement SIEVE first, then launch cap64 — in that order, before
cap128/cap256.**

Reasoning, combining the answers above:

1. SIEVE is low-risk, explicitly requested, and cheap to add (§1). Doing it
   first costs at most a day of engineering time and avoids the rerun risk
   in §5.
2. FIFO-Reinsertion's definition should be pinned down in parallel (§2) —
   ideally before cap64, but if the definition decision takes longer than
   expected, it should not block SIEVE from riding along with cap64. If
   FIFO-Reinsertion's variant is settled in time, implement it alongside
   SIEVE; if not, document the chosen interpretation and accept that it
   will ride along with cap128/cap256 instead (a much smaller rerun cost
   than waiting until after cap256).
3. HALP is out of scope for empirical reimplementation regardless of
   sequencing (§3) — it does not factor into the cap64 launch decision at
   all, only into the response-letter text (§4).
4. A smaller dedicated benchmark (e.g., a single-trace, single-capacity
   smoke test of the new SIEVE policy) before committing it to the full
   cap64 chunk is **recommended as a cheap sanity check**, not a
   replacement for cap64 itself — this mirrors how `evict_value_v1` itself
   was validated (`reports/kbs_validation_sanity_audit.md`) before trusting
   its cap32 numbers. A SIEVE smoke test (1 trace × 2-3 policies × 1
   capacity, well under cap32's own 5-hour runtime) is low-cost insurance
   against discovering a SIEVE bug only after a multi-hour cap64 run
   completes.
5. **Do not launch cap64/128/256 in this pass** — this remains an explicit
   instruction in force for this session. This memo is a decision aid for
   when the user is ready to authorize that launch, not an authorization
   itself.

### Sequencing summary

| Step | Compute cost | Blocks cap64? |
|---|---|---|
| Implement SIEVE (`src/lafc/policies/sieve.py` + registry + tests + bib) | None (engineering only) | No, but recommended to finish first |
| Decide FIFO-Reinsertion's exact variant | None (decision only) | No — can ride along with cap128/256 if it slips past cap64 |
| SIEVE smoke test (1 trace, small capacity, few policies) | Minutes, far less than a full chunk | No — cheap insurance, recommended |
| Launch cap64 (with SIEVE wired in; FIFO-Reinsertion if ready) | Multi-hour, per `kbs_full_policy_comparison_execution_plan.md` | N/A — this is the launch itself |
| Launch cap128, cap256 | Multi-hour each | N/A |

This memo makes no compute calls and authorizes none. The decision to
actually implement SIEVE, decide FIFO-Reinsertion's definition, or launch
any capacity chunk remains the user's to make.
