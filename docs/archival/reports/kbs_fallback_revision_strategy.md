# Fallback-mechanism revision strategy (2026-06-19)

Directly answers **R3-Issue6** ("Fallback mechanism is unvalidated and
oversold as a contribution") and **R3-Rec5** ("Validate the fallback
mechanism or remove it"), plus the AE summary's "unvalidated fallback
design" and **R2-MC** language calling out the "guarded fallback
mechanism" as insufficiently demonstrated. Zero-heavy-compute pass — no
new run was launched; this is an audit of what evidence already exists in
the repo, done in parallel with the running
`kbs_full_policy_comparison_cap32_with_sieve` tmux job (not touched).

## 1. Is the fallback mechanism currently empirically validated?

**No.** Three separate pieces of evidence, checked directly against the
repo rather than assumed:

1. **The manuscript's actual mechanism has no canonical numbers.** The
   guarded fallback described in `main.tex` (contribution #3, §"Optional
   fallback control", Algorithm 2/`alg:evict_value_guarded`) is implemented
   as `EvictValueV1GuardedPolicy` in `src/lafc/policies/guard_wrapper.py`
   (a generic `GuardWrapperPolicy` wrapping a base + fallback policy with
   an early-return detector). **It is not present in
   `scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES` dict** — the
   only registry that feeds the canonical heavy_r1 pipeline
   (cap32/cap32_with_sieve/cap64/128/256). It has never been run on any of
   the 7 real trace families this manuscript reports results for.
2. **The one dedicated "first check" script produced no output.**
   `scripts/run_guarded_evict_value_first_check.py` exists and is designed
   exactly for this purpose (compares base `evict_value_v1` vs.
   `EvictValueV1GuardedPolicy`, logs `guard_triggers`/`guard_time_steps`
   diagnostics to `analysis/guarded_evict_value_first_check.csv/.md/.json`)
   — but **those three output files do not exist on disk** (confirmed via
   `ls`). The script was either never successfully run, or its output was
   never retained. Either way, there is no artifact backing this mechanism
   today.
3. **The only empirical ablation that does exist is for a different,
   related mechanism, on toy data, and the result is negative.**
   `analysis/sentinel_budgeted_guard_v2/v2_ablation_report.md` ablates a
   separate "sentinel_budgeted_guard_v2" guard design (override budget,
   temporary guard, reentry gating — not the same code path as
   `GuardWrapperPolicy`/`EvictValueV1GuardedPolicy`) on **3 synthetic
   traces at capacities 2 and 3** — nowhere near the canonical heavy_r1
   real-trace evaluation. Its own stated conclusions: *"Is any single v2
   component actually useful? No: none of the single-component variants
   beat v1 on mean misses."* and the recommended "main empirical candidate
   after this ablation" is `v1_baseline`, i.e., the plain (non-guarded)
   policy outperforms every guarded variant tested. This is not evidence
   *for* the manuscript's guarded fallback — if anything, it is a
   cautionary data point from an adjacent design in the same family.

## 2. What evidence exists, precisely?

- **Positive evidence**: none, on real traces, for the manuscript's actual
  mechanism.
- **Negative/cautionary evidence**: the sentinel_budgeted_guard_v2
  ablation above (different code, synthetic traces, capacity 2-3) found
  every guarded variant tested performed equal-to-or-worse than the
  unguarded baseline on mean misses.
- **Indirect context**: cap32's own canonical result
  (`reports/kbs_cap32_policy_comparison_report.md`) already shows the
  *unguarded* `evict_value_v1` losing to plain LRU on 6/7 trace families.
  This does not directly speak to the guarded variant, but it removes any
  presumption that wrapping a currently-underperforming base policy in an
  unvalidated guard would reliably help — if the base scorer's candidate
  rankings are not yet winning against LRU, a fallback layer triggered by
  the base policy's own suspicious behavior is, at best, an open empirical
  question, not a demonstrated improvement.

## 3. What would be needed to validate it properly?

To make an honest "validated" claim, the following would be required —
all of which are heavier-than-this-pass and explicitly **not** undertaken
here:

1. Wire `EvictValueV1GuardedPolicy` (or a finalized guard configuration)
   into `scripts/run_policy_comparison_wulver_v1.py`'s `POLICIES` dict.
2. Run it through at least one full canonical capacity chunk (ideally all
   four: 32/64/128/256) across all 7 trace families, alongside the
   unguarded `evict_value_v1` and LRU, with `guard_triggers`/
   `guard_time_steps` diagnostics logged per the existing (currently
   unused) schema in `run_guarded_evict_value_first_check.py`.
3. Report whether the guarded variant reduces misses relative to unguarded
   `evict_value_v1` on a *family-by-family* basis (not just in aggregate),
   since the guard is explicitly meant to help in *locally* unsafe
   regimes — an aggregate-only number could hide family-level harm or
   benefit.
4. Sweep at least the trigger threshold and guard duration
   (`(W, T, M, D)` in the manuscript's Algorithm 2 notation) to show the
   result isn't an artifact of one arbitrary hyperparameter choice — this
   is the same concern the sentinel_budgeted_guard_v2 ablation was
   designed to address for its own (different) mechanism.

This is a multi-hour-class experiment (new canonical sweep dimension on
top of cap64/128/256, which are themselves not yet launched) — explicitly
out of scope under this pass's "do not launch any multi-hour experiment"
constraint.

## 4. Decision: validate, demote, or remove?

**Demote, do not remove, do not claim validation.** Reasoning:

- **Validate** is not achievable without new heavy compute, which this
  pass is explicitly barred from launching, and which is not yet
  scheduled even after cap32_with_sieve/cap64+ — it would be a fifth
  experimental dimension layered on top of an already-large remaining
  sweep. Claiming validation now would be asserting something the evidence
  (§1-2) directly contradicts.
- **Remove entirely** (deleting `guard_wrapper.py`,
  `EvictValueV1GuardedPolicy`, and all manuscript text) is more drastic
  than necessary and destroys a working, reusable piece of code and a
  legitimate methodological idea (candidate-level scoring plus a separate
  control layer is a reasonable design point, just not yet an empirically
  demonstrated one). R3-Rec5 itself frames this as "validate **or**
  remove" — but the manuscript's actual problem, per R3-Issue6's own
  wording, is that it is "**oversold as a contribution**," not that the
  idea itself is invalid. The proportionate fix is to stop claiming it as
  a validated empirical contribution while keeping the design as an
  explicitly-flagged, not-yet-validated extension — this satisfies the
  reviewer's concern (no overselling) without discarding work that may be
  validated in a future revision cycle.
- **Demote** (reframe from "main contribution" to "design extension /
  future work, validation explicitly deferred") is the only option
  consistent with both constraints: it removes the overselling the
  reviewer flagged, and it doesn't require new compute.

## 5. Draft manuscript edits

### 5.1 Abstract (currently `main.tex` line 29)

Current text (relevant clause): *"...we also examine lightweight guarded
variants that can temporarily revert to a conservative fallback policy
when recent behavior indicates locally unsafe learned decisions."*

**Revised**: replace "we also examine" with explicit non-validation
framing:

> We also describe, but do not yet empirically validate, a lightweight
> guarded extension that can temporarily revert to a conservative fallback
> policy when recent behavior indicates locally unsafe learned decisions;
> we present this as a design extension for future empirical study rather
> than as a validated component of the present results.

### 5.2 Contributions list (currently `main.tex` lines 75-76, contribution #3)

Current text: *"We develop a practical robust extension in which the
learned candidate scorer can be combined with lightweight fallback
behavior when recent online outcomes suggest locally unsafe decisions.
This preserves the candidate-level structure of the method while providing
a conservative control layer for imperfect predictive regimes."*

**Revised** (demoted out of the numbered contributions list, or kept but
explicitly marked non-empirical — recommend full removal from the
numbered list and folding into a single sentence at the end of the
contributions paragraph, or into Limitations/Future Work instead):

> We additionally describe a lightweight guard-style extension, compatible
> with the candidate-level scoring framework, that can temporarily defer to
> a conservative fallback policy when recent online behavior suggests
> locally unsafe decisions; we present this as a design extension and flag
> explicitly that we have not yet empirically validated its effect on
> end-to-end miss ratio (see Limitations).

If the editorial preference is to keep five contributions for parallelism
with the original structure, replace contribution #3 with the candidate-
scoring framework's reproducibility/diagnostics angle (already covered by
the existing contribution #4) merged into one item, and move the
fallback-extension sentence above into Limitations only — recommend this
fuller version if word count is also being cut per R3-Rec6 (see
`reports/kbs_manuscript_shortening_and_reframing_plan.md`).

### 5.3 Method section (currently `main.tex` lines 191, 229-231)

Line 191 currently: *"We therefore study a lightweight fallback mechanism
that preserves a conservative decision path when recent online behavior
suggests that the learned policy is making locally unsafe choices."*

**Revised**: *"We therefore describe a lightweight fallback mechanism,
intended to preserve a conservative decision path when recent online
behavior suggests that the learned policy is making locally unsafe
choices; we treat this as a candidate design for future empirical
validation rather than a demonstrated component of the present results
(see Limitations)."*

Line 229-231 already contains useful hedging ("The guard should therefore
be interpreted as a practical empirical mechanism, not as a theorem-backed
robustness guarantee.") — **keep this sentence**, it is consistent with
the demoted framing and does not need to change. Strengthen the following
sentence ("Overall, the decision process has two layers...") by adding an
explicit empirical-status flag:

> Overall, the decision process has two layers. The first layer is the
> learned eviction-value scorer ... The second layer is an optional guard
> ... **We note that the second layer's empirical effect on end-to-end
> outcomes has not yet been measured in the present study; we discuss this
> gap directly in Limitations.**

### 5.4 Limitations section (currently `main.tex` lines 507-522)

Line 511 currently: *"Finally, the fallback variants introduce additional
design choices, including the early-return window, trigger threshold,
monitoring interval, fallback duration, and fallback policy. These choices
are operationally interpretable and easy to implement, but they remain
heuristic parameters rather than theoretically derived constants. Their
value may therefore depend substantially on workload regime and deployment
conditions."*

**Add immediately after this sentence** (new, explicit non-validation
disclosure — this is the single most important sentence to add anywhere in
the manuscript for this issue):

> We emphasize that the guarded fallback extension described in this paper
> has not yet been evaluated end-to-end against the unguarded scorer or
> against the classical and predictive baselines on the canonical trace
> suite used for our main results; we therefore do not claim it as an
> empirically validated contribution of the present study, and we present
> it solely as a design extension motivated by the candidate-level
> formulation, with empirical validation left to future work.

### 5.5 Response-to-reviewers paragraph (R3-Issue6/R3-Rec5)

> We thank the reviewer for this important correction. We agree that the
> guarded fallback mechanism, as presented in the original submission, was
> listed alongside our empirically supported contributions without
> sufficient evidence to justify that placement. In this revision, we have
> (1) removed the fallback mechanism from the numbered list of main
> contributions, (2) added an explicit statement in the Abstract, Method
> section, and Limitations explaining that the mechanism is a design
> extension whose end-to-end effect on miss ratio has not yet been
> empirically measured, and (3) retained the mechanism's description and
> implementation in the paper and code repository as a candidate direction
> for future empirical validation, since we believe the underlying idea
> (a separate control layer triggered by locally observed unsafe
> decisions) remains a reasonable design point even though it is not yet
> validated. We did not remove the mechanism outright because doing so
> would discard a concrete, reusable design that may be validated in a
> future revision cycle; instead we have reframed its role precisely to
> match the evidence we actually have.

## 6. What this does NOT resolve

- This strategy does not itself add any empirical fallback numbers — it
  is a framing/honesty fix, consistent with the "be conservative, do not
  claim validation that does not exist" instruction for this pass.
- If a future pass does run `EvictValueV1GuardedPolicy` through a
  canonical chunk and the result is positive, §5's demoted language should
  be promoted back to a validated-contribution framing with real numbers
  cited — this report's recommendation is conditioned on the current
  zero-compute evidence state, not a permanent judgment on the mechanism's
  merit.
- Does not touch `kbs_full_policy_comparison_cap32_with_sieve` or launch
  any new run; `EvictValueV1GuardedPolicy` remains unwired in the
  canonical `POLICIES` dict (no code change made in this report).
