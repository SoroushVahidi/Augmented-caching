# Internal note: current project decisions (working record)

> **Internal research note only.**
> This file is a working record for the team.
> It is **not manuscript text**, and it is **not canonical `heavy_r1` evidence**.

## 1) Purpose of this note

This note records the current agreed conceptual decisions from recent discussions so future manuscript writing can reference a stable internal record.

Current phase assumptions:

- We are **not** writing the manuscript yet.
- We are **not** running heavy experiments right now.
- The immediate goal is to preserve conceptual decisions and lightweight exploratory findings.

## 2) Current main claim direction

The main contribution should be framed primarily as a **better training target**.

Most defensible current novelty hypothesis:

- **explicit finite-horizon counterfactual supervision for candidate-level eviction-value prediction**.

Interpretation for future drafting:

- Candidate-level eviction is part of the setup, but the differentiating point is the supervision target design and what it asks the model to predict.
- Online decision rules may still matter operationally, but they are not the primary novelty framing.

## 3) What we should not claim

At this stage, avoid claiming novelty for the following by themselves:

- candidate-level eviction,
- fallback/guard mechanisms,
- continuation-policy choice,
- exploratory ablation outcomes.

Also keep boundaries explicit:

- exploratory notes/results must remain clearly separate from canonical manuscript evidence,
- canonical `heavy_r1` artifacts remain the manuscript evidence source of truth.

## 4) Why the training-target framing is primary

The supervision target should be described as more decision-aligned than common alternatives because it aims to model **downstream eviction harm** directly.

In internal terms, this is the key contrast to:

- next-arrival prediction,
- Belady imitation targets,
- binary labels,
- preference labels.

The rationale is alignment: the target is intended to better match the actual eviction objective than proxy labels that do not explicitly model harm from a specific eviction choice over a finite continuation.

## 5) Horizon choice logic

Horizon should be presented as a **learnability / data-availability tradeoff**:

- use shorter horizons when usable data is limited,
- use longer horizons when enough data is available.

This is a practical bias-variance and sample-efficiency decision, not a one-size-fits-all theorem claim.

## 6) Continuation-policy decision

Continuation-policy choice matters for label construction, but it is **not** part of the main novelty claim.

Agreed default practical continuation policy: **LRU**.

Reasons to present for this default:

- simple,
- deterministic,
- cheap,
- stable,
- competitive enough in lightweight exploratory checks.

Framing requirement:

- LRU continuation is a practical approximation choice, **not** a theoretically preferred continuation policy.

Scope guard:

- self-rollout, hybrid rollout, and other continuation rules are currently exploratory only.

## 7) Summary of recent exploratory findings

Record these as exploratory status, not canonical evidence:

1. **History-context ablation**: benefit was small/inconsistent; not strong evidence.
2. **Hybrid fallback experiment**: no material gain in the tested setup.
3. **Continuation-policy light ablation**: continuation choice appeared to affect label proxies more than replay outcomes; LRU remained competitive enough as practical default.

## 8) Immediate implications for future work

Near-term implications:

- Keep method language centered on training-target design and decision alignment.
- Keep novelty claims conservative and prior-work aware.
- Treat continuation and fallback mechanisms as implementation/exploration axes unless stronger evidence changes this.
- Continue lightweight checks that reduce framing uncertainty (horizon and continuation sensitivity).
- Preserve strict separation between exploratory records and canonical `heavy_r1` manuscript evidence.

## 9) Status and update policy

As of **2026-04-12**, this note reflects current team agreement from recent discussions.

If decisions change, update this note directly and retain conservative wording so downstream manuscript drafting can track the evolution of agreed positioning.
