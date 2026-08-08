# After-cap64 decision memo (2026-06-20)

**Purpose:** Decide whether to launch cap128 and/or cap256 after analyzing
the completed `cap64_with_sieve_fifo` canonical chunk.

**Environment:** current local/cloud machine; tmux; not Wulver/Slurm.

**Evidence base:**
- `reports/kbs_cap64_result_analysis.md` (this session)
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv/.md`
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md` (prior chunk)
- `reports/kbs_after_cap32_decision_memo.md` (prior decision: recommended cap64-only, then reassess — this memo is that reassessment)
- `reports/kbs_real_reviewer_comments.md` (R3-Issue1/Rec1, R2-MC3: multi-capacity completeness; R3-Issue3/Rec2: SIEVE/FIFO-Reinsertion; R3-Issue4/Rec4: narrow claims; deadline 2026-07-08)

**No compute was performed in writing this memo.** cap128/cap256 remain
**not launched**, per instruction.

---

## What changed since the prior memo

The prior memo (`kbs_after_cap32_decision_memo.md`) recommended cap64-only
as the next step specifically *because* cap32 alone showed a uniform
negative pattern and a second capacity point would test whether that
pattern was capacity-sensitive. It now is:

- evict_value_v1's aggregate gap vs LRU roughly halved (−4.84% → −2.26%),
  vs SIEVE nearly closed (−2.92% → −0.13%), vs FIFO-Reinsertion roughly
  halved (−4.78% → −2.19%).
- This narrowing is **not** a generic "everything converges at higher
  capacity" artifact — SIEVE's and FIFO-Reinsertion's own gaps vs LRU
  stayed flat over the same span (§3 of the result analysis).
- The narrowing is concentrated in 2/7 families (metacdn, metakv); 2
  others are flat-to-slightly-worse (brightkite, cloudphysics); twemcache
  shows an outright (and widening) win vs SIEVE specifically.
- evict_value_v1 still has **zero outright wins vs LRU or FIFO-Reinsertion**
  on any non-degenerate family, at either capacity.

This is a materially different evidentiary position than after cap32:
not a reversal into a positive result, but a real, partially-explained,
family-dependent trend that two data points (32, 64) cannot yet
distinguish from noise, a one-off, or a genuine capacity-interaction
effect that would continue at cap128.

---

## Options

| Option | Description |
|--------|-------------|
| **A** | Stop experiments now; reframe the paper as a methodology/sensitivity study using cap32+cap64 only |
| **B** | Run **cap128 only** (~3rd capacity point), then decide on cap256 |
| **C** | Run **both cap128 and cap256** in this next push |

---

## Recommendation: **B — run cap128 only, then reassess before cap256**

This continues the same staged-decision pattern already used for cap64
(propose one chunk, evaluate, decide the next). Do **not** launch cap128
in this memo's authoring pass — this is a recommendation for human
approval, consistent with the standing instruction.

### Scientific rationale

- Two points (32, 64) establish a *direction* but not a *trend with
  confidence* — halving could continue, plateau, or reverse at cap128.
  cap32→cap64 already showed family-level disagreement (metacdn/metakv
  improving sharply, brightkite/cloudphysics flat-to-worse), so a third
  point is needed to tell whether the aggregate narrowing is a stable
  multi-family pattern or dominated by 1-2 traces that happen to be
  capacity-sensitive in this particular range.
- A prior single-trace cap256 validation pass (referenced in
  `kbs_after_cap32_decision_memo.md` §"Risk of mostly negative results")
  already showed evict_value_v1 reversing in the wrong direction on at
  least one trace at the largest capacity tested historically — this is a
  concrete, non-hypothetical reason to expect cap256 specifically carries
  elevated risk of disrupting an otherwise-improving narrative, and a
  reason to get cap128 results in hand before committing to cap256.
- cap128 is the lowest-risk way to either (a) confirm the narrowing trend
  with a 3-point curve, strengthening the capacity-sensitivity story, or
  (b) catch an early reversal before sinking compute into cap256 as well.

### Reviewer-response rationale

- R3-Issue1/Rec1 and R2-MC3 ask for multi-capacity completeness. A 3-point
  curve (32/64/128, a 4x range) is a materially stronger answer than the
  current 2-point curve, and is generally accepted as sufficient evidence
  of a capacity-sensitivity trend in cache-replacement literature without
  requiring every power-of-two up to 256.
- R3-Issue4/Rec4 (narrow claims) is *already* well served by the current
  framing — cap64's result supports "relative competitiveness improves
  with capacity, parity not yet reached" rather than a superior-policy
  claim, and cap128 only sharpens that statement, it does not require
  walking it back.
- A 2-point-only submission (cap32+cap64) invites an obvious reviewer
  question: "you show an improving trend with capacity — did you test
  whether it continues?" Having declined to look is a weaker position than
  having looked and reporting whatever was found, even if it's mixed.

### Compute cost

| Step | Actual/est. wall-clock | Cumulative |
|------|------------------------|------------|
| cap32_with_sieve_fifo (done) | ~5h 16m | 5h 16m |
| cap64_with_sieve_fifo (done) | ~10h 04m | ~15h 20m |
| cap128 (proposed next) | ~15–20h (if ~2x cap64, consistent with the 32→64 doubling observed) | ~30–35h |
| cap256 (if run after) | ~30–40h (same doubling assumption) | ~60–75h |

Today is 2026-06-20; the reviewer-response deadline is 2026-07-08 (18 days
out). cap128 alone (~1 day unattended wall-clock) fits comfortably within
that window alongside manuscript/response-letter work already in
progress (`kbs_response_to_reviewers_skeleton.md`,
`kbs_revision_completion_audit.md`, `kbs_docx_submission_package_report.md`).
cap256 added on top (~1.5–2 more days) is also calendar-feasible in
isolation, but the **script has no checkpointing** — it writes its CSV/MD
exactly once at the end of a full run (confirmed in
`kbs_before_cap64_baseline_decision_memo.md` §5) — so committing to both
chunks back-to-back means a single failure partway through forfeits both,
right as the deadline window narrows.

### Risk comparison: B vs C

- **B (cap128 only)** isolates risk to one ~1-day run. If it confirms the
  narrowing trend, cap256 becomes a high-confidence follow-up with time to
  spare. If it reverses or plateaus, the decision to skip cap256 and
  reframe is made with a 3-point curve instead of a 2-point curct — a
  stronger evidentiary basis either way.
- **C (both now)** is the optimistic path: it assumes the narrowing trend
  continues, that no run fails partway, and that no intermediate decision
  point is needed. Given the explicit instruction to base this
  recommendation on reviewer satisfaction rather than optimism, and given
  that a single-trace cap256 data point has *already* reversed once
  before, C is not recommended as the next step — it is a reasonable
  *follow-up* to B if cap128 confirms the trend and calendar time remains
  comfortable.

### Option A's case (for completeness)

Stopping now and reframing on cap32+cap64 alone remains *scientifically
defensible* — the prior memo already established that an honest negative
result with rigorous baselines is publishable in a KBS-style venue. But
choosing A right after cap64 revealed an improving, partially-explained
trend (not attributable to generic capacity convergence, per §3 of the
result analysis) means stopping at the most ambiguous point in the curve:
two points show a direction, not a trend. A is the right call only if
calendar risk turns out to be tighter than currently estimated, or if
cap128 is later found to reverse the trend.

---

## Should cap128 be launched now?

**No — not without explicit human approval after reading this memo.**

If approved, launch with:
- Same 8-policy set as cap32_with_sieve_fifo / cap64_with_sieve_fifo
- Output: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap128_with_sieve_fifo.csv/.md`
- Do **not** launch cap256 in the same step — reassess after cap128 completes

---

## Decision matrix

| Criterion | A (stop/reframe now) | **B (cap128 only)** | C (cap128 + cap256) |
|-----------|------------------------|------------------------|------------------------|
| Closes R3-Issue1/Rec1 fastest with low risk | partial (2-point curve) | ✓✓ | ✓✓✓ (if both succeed) |
| Minimizes compute/calendar risk | ✓✓ | ✓ | ✗ (no checkpointing, longest combined run) |
| Tests whether narrowing trend continues | ✗ | ✓✓ | ✓✓ (but commits before knowing) |
| Preserves option to stop early if trend reverses | n/a | ✓✓ (decision point preserved) | ✗ (already committed) |
| Supports honest, narrowed-claim framing (R3-Issue4/Rec4) | ✓ | ✓✓ | ✓✓ |

---

## Final report

1. **cap64 headline result**: evict_value_v1 remains the worst-ranked
   policy in aggregate mean misses at cap64, but its gap vs LRU (−4.84% →
   −2.26%), SIEVE (−2.92% → −0.13%, nearly closed), and FIFO-Reinsertion
   (−4.78% → −2.19%) all roughly halved relative to cap32. It already wins
   outright against SIEVE on twemcache, and that win widened with
   capacity. No outright win vs LRU or FIFO-Reinsertion on any
   non-degenerate family.

2. **cap32→cap64 trend**: Improving, and specifically attributable to
   evict_value_v1 — SIEVE and FIFO-Reinsertion's own distance from LRU
   stayed flat over the same capacity step, so this is not simply "the
   whole field converges at higher capacity." The improvement is
   concentrated in 2/7 families (metacdn: 22.4%→2.7% gap vs LRU; metakv:
   1.7%→0.01%, a near-exact tie), while brightkite and cloudphysics are
   flat-to-slightly-worse, and wiki2018 remains fully degenerate
   (0% hit rate, all policies) at both capacities.

3. **Strongest baseline**: LRU and rest_v1, tied for best in aggregate
   mean misses at both cap32 and cap64. blind_oracle_lru_combiner is a
   near-exact tie with LRU (by construction); FIFO-Reinsertion is the
   closest *distinct* baseline to LRU (−0.06% at cap32, −0.07% at cap64).

4. **evict_value_v1 status**: Still losing in aggregate to all three
   target baselines (LRU, SIEVE, FIFO-Reinsertion) at cap64, but the
   margin has narrowed substantially and unevenly across workloads. Not
   yet a positive result; more nuanced than cap32's uniform negative
   result.

5. **Recommendation**: **Option B — run cap128 only, then reassess before
   committing to cap256.** This is not authorized by this memo; it
   requires explicit human approval to launch.

6. **Expected reviewer reaction under each option**:
   - **A (stop/reframe on cap32+cap64)**: Defensible and honest, but
     invites an obvious follow-up question — "you observed an improving
     trend with capacity; did you check whether it continues?" Likely to
     be read as adequately scoped but thinner than it could be, given the
     trend was already visible.
   - **B (cap128 only)**: A 3-point capacity curve (32/64/128) is
     generally accepted as sufficient evidence for a capacity-sensitivity
     claim without requiring exhaustive coverage to 256. Best balance of
     reviewer completeness (R3-Issue1/Rec1, R2-MC3) against compute/
     calendar risk, and preserves the option to extend or stop based on
     what cap128 shows.
   - **C (cap128 + cap256 together)**: Most complete answer if both
     chunks succeed and continue the trend (a 4-point curve spanning
     32–256 would be hard for a reviewer to ask for more on), but carries
     the highest compute/calendar risk (no mid-run checkpointing, longest
     combined wall-clock, and a precedent of cap256 reversing on at least
     one trace previously). If either chunk fails or reverses the trend,
     this option converts the cap64 narrative back toward ambiguity right
     before the deadline, with less buffer to recover than B leaves.
