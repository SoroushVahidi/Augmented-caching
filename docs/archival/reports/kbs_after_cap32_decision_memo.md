# After-cap32 decision memo (2026-06-19)

**Purpose:** Decide whether to launch cap64/cap128/cap256 after analyzing the
completed `cap32_with_sieve_fifo` canonical chunk.

**Environment:** current local/cloud machine; tmux; not Wulver/Slurm.

**Evidence base:**
- `reports/kbs_cap32_with_sieve_fifo_result_analysis.md` (this session)
- `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap32_with_sieve_fifo.csv/.md`
- Prior validation sanity audit (conclusion A: negative result is real)

---

## Options

| Option | Description |
|--------|-------------|
| **A** | Continue full sweep: cap64 + cap128 + cap256 (same 8-policy set) |
| **B** | Pause compute; reframe paper around honest negative/audit findings |
| **C** | Run **cap64 only** (~5h), then decide on cap128/256 |
| **D** | Add overhead/timing benchmarks before any more policy-comparison compute |

---

## Recommendation: **C — run cap64 only before deciding**

Do **not** launch cap64 now in this memo's authoring pass (per instruction).
This is a recommendation for human approval.

### Scientific rationale

- cap32 shows a **consistent negative pattern**: evict_value_v1 loses to LRU on 6/7 families (−4.84%), to SIEVE (−2.92%), and to FIFO-Reinsertion (−4.77%).
- Three families are within ~2% of best (cloudphysics, metakv, twemcache) but never win; metacdn is a large loss.
- Capacity scaling is a legitimate open question: learning might help more at larger caches, but nothing in cap32 suggests that — and the cap256 validation point on one trace already went the wrong direction.
- One additional capacity (64) tests sensitivity without committing ~10–15 more hours if the pattern holds.

### Reviewer-response rationale

- Reviewers (R3-Issue1, R2-MC3, R3-Rec1) require **multi-capacity** end-to-end results. cap32 alone cannot close Table 3.
- R3-Issue3/Rec2 is **partially closed for cap32**: SIEVE and FIFO-Reinsertion now have canonical numbers at one capacity.
- A single cap64 chunk gives a **two-point capacity curve** (32 + 64) — enough to draft an interim response paragraph and to judge whether cap128/256 are worth the cost.
- Parallel reframing (Option B elements) should proceed **regardless** — cap32 already supports softening performance claims.

### Compute cost

| Step | Est. wall-clock | Cumulative |
|------|-----------------|------------|
| cap32_with_sieve_fifo (done) | ~5h 16m | 5h 16m |
| cap64 (proposed next) | ~5–6h | ~10–11h |
| cap128 + cap256 (if both run) | ~10–12h | ~20–23h |
| Merge + reports | ~1h | — |

Full sweep (A) ≈ **15–18h additional** beyond cap32. Option C adds **one** ~5–6h chunk.

### Risk of mostly negative results

**High.** cap32, validation (cap256, 1 trace), and validation sanity (cap32, 5k) all point the same way. cap64 may show:
- continued losses (most likely);
- capacity-dependent improvement on 1–2 families (possible but not indicated by cap32);
- no change on wiki2018 (certain — still degenerate at cap64).

Spending 15+ hours to confirm a pattern already visible at cap32 has **diminishing scientific return**, though it strengthens reviewer-facing completeness.

### Publishability of honest negative results

**Yes, if reframed.** A paper that:
- introduces a decision-aligned replay target;
- reports rigorous end-to-end evaluation including modern baselines (SIEVE, FIFO-Reinsertion);
- documents when/where learning-augmented eviction **does not** beat O(1) baselines;
- separates offline metric quality from online miss ratio;

…is publishable in KNOSYS-style venues **if** claims are narrowed (R3-Issue4/Rec4). cap32 is already sufficient to begin that reframing; more capacities strengthen but do not create the finding.

---

## Should cap64 be launched now?

**No — not without explicit human approval after reading this memo.**

If approved, launch with:
- Same 8-policy set as cap32_with_sieve_fifo
- Output: `analysis/evict_value_wulver_v1_policy_comparison_heavy_r1_cap64_with_sieve_fifo.csv/.md`
- Do **not** launch cap128/cap256 in the same step

---

## Decision matrix

| Criterion | A (full sweep) | B (pause/reframe) | **C (cap64 only)** | D (overhead first) |
|-----------|----------------|-------------------|--------------------|--------------------|
| Closes R3-Issue1 fastest | ✓✓ | ✗ | ✓ | ✗ |
| Minimizes compute risk | ✗ | ✓✓ | ✓ | ✓ |
| Tests capacity sensitivity | ✓✓ | ✗ | ✓ | ✗ |
| Supports honest reframing | ✓ | ✓✓ | ✓ | partial |
| Addresses R3-Issue5 overhead | partial | partial | partial | ✓✓ |

---

## Immediate non-compute actions (can proceed now)

1. Update reviewer-response drafts with cap32 SIEVE/FIFO numbers (placeholders for cap64+).
2. Begin title/contribution reframing (`reports/kbs_manuscript_rebuttal_drafts/title_and_contribution_reframing_options.md`, Option 1 or 2).
3. Draft cap32-only Table 3 partial / supplementary table.
4. Investigate trust_and_doubt 1–73-miss drift between cap32 runs (low priority).
5. Human decision: approve cap64 launch or choose Option B.

---

## Summary verdict

| Question | Answer |
|----------|--------|
| Best policy on cap32 | LRU ≡ rest_v1; fifo_reinsertion ≈ tied |
| SIEVE vs LRU | SIEVE worse (−1.86% aggregate) |
| FIFO vs LRU | FIFO marginally worse (−0.06% aggregate) |
| evict_value_v1 vs LRU/SIEVE/FIFO | Worse on all three (−4.84% / −2.92% / −4.77%) |
| Common policies match old cap32? | Yes for lru, rest_v1, evict_value_v1, predictive_marker, blind_oracle_lru_combiner (exact); trust_and_doubt minor drift |
| Recommended path | **C** (cap64 only, then reassess) |
| Launch cap64 now? | **No** — await explicit approval |
