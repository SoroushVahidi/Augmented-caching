from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


INF = 10**9


def fmt_gap(g: int) -> str:
    return "inf" if g >= INF else str(g)


@dataclass
class DecisionEvent:
    t: int
    phase: int
    true_order: Tuple[str, ...]
    pred_order: Tuple[str, ...]
    inversions: int
    belady_victim: str
    chosen_victim: str
    gap: int


@dataclass
class RunResult:
    misses: int
    miss_mask: List[bool]
    phases: List[int]
    events: List[DecisionEvent]


@dataclass
class Case:
    seq: str
    k: int
    alphabet: int
    perturbation: str
    event_t: int
    event_phase: int
    inversions: int
    gap: int
    belady_misses: int
    test_misses: int
    extra_misses: int
    extra_same_phase: int


def next_use_distance(seq: Sequence[str], t: int, page: str) -> int:
    for j in range(t + 1, len(seq)):
        if seq[j] == page:
            return j - t
    return INF


def belady_order(cache: Sequence[str], seq: Sequence[str], t: int) -> List[str]:
    return sorted(cache, key=lambda p: (next_use_distance(seq, t, p), p), reverse=True)


def inversion_count(order_true: Sequence[str], order_pred: Sequence[str]) -> int:
    pos = {p: i for i, p in enumerate(order_pred)}
    inv = 0
    for i in range(len(order_true)):
        for j in range(i + 1, len(order_true)):
            a, b = order_true[i], order_true[j]
            if pos[a] > pos[b]:
                inv += 1
    return inv


def phase_ids(seq: Sequence[str], k: int) -> List[int]:
    out: List[int] = []
    seen: set[str] = set()
    phase = 0
    for p in seq:
        if p not in seen and len(seen) == k:
            phase += 1
            seen = set()
        seen.add(p)
        out.append(phase)
    return out


def perturb_order(true_order: List[str], kind: str) -> List[str]:
    pred = list(true_order)
    if kind == "top_swap" and len(pred) >= 2:
        pred[0], pred[1] = pred[1], pred[0]
    elif kind == "lower_swap" and len(pred) >= 3:
        pred[1], pred[2] = pred[2], pred[1]
    return pred


def simulate(seq: Sequence[str], k: int, perturb_at: int | None, perturbation: str) -> RunResult:
    cache: List[str] = []
    misses = 0
    miss_mask = [False] * len(seq)
    phases = phase_ids(seq, k)
    events: List[DecisionEvent] = []

    decision_idx = 0
    for t, req in enumerate(seq):
        if req in cache:
            continue

        misses += 1
        miss_mask[t] = True
        if len(cache) < k:
            cache.append(req)
            continue

        true_order = belady_order(cache, seq, t)
        if perturb_at is not None and decision_idx == perturb_at:
            pred_order = perturb_order(true_order, perturbation)
        else:
            pred_order = list(true_order)

        inv = inversion_count(true_order, pred_order)
        bel_victim = true_order[0]
        chosen = pred_order[0]
        gap = abs(next_use_distance(seq, t, bel_victim) - next_use_distance(seq, t, chosen))

        cache.remove(chosen)
        cache.append(req)

        events.append(
            DecisionEvent(
                t=t,
                phase=phases[t],
                true_order=tuple(true_order),
                pred_order=tuple(pred_order),
                inversions=inv,
                belady_victim=bel_victim,
                chosen_victim=chosen,
                gap=gap,
            )
        )
        decision_idx += 1

    return RunResult(misses=misses, miss_mask=miss_mask, phases=phases, events=events)


def format_case(c: Case) -> str:
    return (
        f"|{c.seq}|{c.k}|{c.alphabet}|{c.perturbation}|{c.event_t}|{c.event_phase}|{c.inversions}|{fmt_gap(c.gap)}|"
        f"{c.belady_misses}|{c.test_misses}|{c.extra_misses}|{c.extra_same_phase}|"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Search counterexamples to Step-4 order-only local charge claims")
    ap.add_argument("--output-md", default="analysis/pairwise_step4_attack_examples.md")
    ap.add_argument("--max-len", type=int, default=9)
    args = ap.parse_args()

    pages = ["A", "B", "C", "D"]
    configs = [(2, 3), (2, 4), (3, 4)]

    all_cases: List[Case] = []
    strongest_violations: List[Case] = []

    for k, alpha in configs:
        alphabet = pages[:alpha]
        for n in range(k + 2, args.max_len + 1):
            for seq_tup in itertools.product(alphabet, repeat=n):
                seq = "".join(seq_tup)
                bel = simulate(seq_tup, k, perturb_at=None, perturbation="top_swap")
                decision_count = len(bel.events)
                if decision_count == 0:
                    continue

                for perturbation in ("top_swap", "lower_swap"):
                    for d_idx in range(decision_count):
                        run = simulate(seq_tup, k, perturb_at=d_idx, perturbation=perturbation)
                        event = run.events[d_idx]
                        if event.inversions == 0:
                            continue
                        extra_positions = [
                            t
                            for t, (m_b, m_r) in enumerate(zip(bel.miss_mask, run.miss_mask))
                            if m_r and not m_b
                        ]
                        extra_same_phase = sum(1 for t in extra_positions if run.phases[t] == event.phase)
                        case = Case(
                            seq=seq,
                            k=k,
                            alphabet=alpha,
                            perturbation=perturbation,
                            event_t=event.t,
                            event_phase=event.phase,
                            inversions=event.inversions,
                            gap=event.gap,
                            belady_misses=bel.misses,
                            test_misses=run.misses,
                            extra_misses=run.misses - bel.misses,
                            extra_same_phase=extra_same_phase,
                        )
                        all_cases.append(case)
                        if case.extra_misses > case.inversions:
                            strongest_violations.append(case)

    def pick_min(xs: List[Case], key):
        return sorted(xs, key=key)[0] if xs else None

    def pick_max(xs: List[Case], key):
        return sorted(xs, key=key)[-1] if xs else None

    zero_damage = [c for c in all_cases if c.inversions == 1 and c.extra_misses == 0 and c.perturbation == "top_swap"]
    high_damage = [c for c in all_cases if c.inversions == 1 and c.extra_misses >= 3 and c.perturbation == "top_swap"]
    big_gap = [c for c in all_cases if c.inversions == 1 and c.gap >= 3 and c.perturbation == "top_swap"]
    shortlist_diff = [c for c in all_cases if c.perturbation == "lower_swap" and c.extra_misses == 0]
    one_local = [c for c in all_cases if c.inversions == 1 and c.extra_misses == 1 and c.extra_same_phase == 1 and c.perturbation == "top_swap"]
    one_cascade = [c for c in all_cases if c.inversions == 1 and c.extra_misses >= 3 and c.extra_same_phase < c.extra_misses and c.perturbation == "top_swap"]

    restricted_reuse = [c for c in all_cases if c.perturbation == "top_swap" and c.gap >= 3]
    restricted_phase = [c for c in all_cases if c.perturbation == "top_swap" and c.extra_misses == c.extra_same_phase]
    restricted_one_inv = [c for c in all_cases if c.inversions == 1 and c.perturbation == "top_swap"]

    def violation_rate(xs: List[Case]) -> str:
        if not xs:
            return "n/a"
        v = sum(1 for c in xs if c.extra_misses > c.inversions)
        return f"{v}/{len(xs)}"

    lines: List[str] = ["# Step-4 targeted attack examples", "", "This note is exploratory only and does not claim a proof.", ""]
    lines.append("## Search setup")
    lines.append(f"- Configs (k, alphabet): {configs}")
    lines.append(f"- Max sequence length: {args.max_len}")
    lines.append(f"- Evaluated perturbation cases: {len(all_cases)}")
    lines.append(f"- Strongest-claim violations found: {len(strongest_violations)}")
    lines.append("")
    lines.append("Strongest tested Step-4 form: `extra_misses <= local_inversions` for a single perturbed decision.")
    lines.append("")

    lines.append("## Restricted-form violation rates")
    lines.append("- Reuse-gap-separated subset (gap >= 3): " + violation_rate(restricted_reuse))
    lines.append("- Phase-local subset (all extra misses stay in inversion phase): " + violation_rate(restricted_phase))
    lines.append("- One-inversion-per-decision subset: " + violation_rate(restricted_one_inv))
    lines.append("")

    lines.append("## Family table")
    lines.append("|family|seq|k|alphabet|perturb|event_t|phase|inversions|gap|belady|test|extra|extra_same_phase|")
    lines.append("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    family_rows: List[Tuple[str, Case | None]] = [
        ("same inversion, near-zero damage", pick_min(zero_damage, key=lambda c: (len(c.seq), c.seq, c.k))),
        ("same inversion, high damage", pick_max(high_damage, key=lambda c: (c.extra_misses, -len(c.seq), c.seq))),
        ("large reuse-gap separation", pick_max(big_gap, key=lambda c: (c.gap, c.extra_misses, c.seq))),
        ("shortlist/full ranking difference", pick_min(shortlist_diff, key=lambda c: (len(c.seq), c.seq, c.k))),
        ("one inversion stays local", pick_min(one_local, key=lambda c: (len(c.seq), c.seq, c.k))),
        ("one inversion triggers cascade", pick_max(one_cascade, key=lambda c: (c.extra_misses, c.gap, c.seq))),
    ]

    for label, case in family_rows:
        if case is None:
            lines.append(f"|{label}|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|")
        else:
            lines.append(f"|{label}{format_case(case)}")

    lines.append("")
    lines.append("## Family interpretation")
    max_extra = max((c.extra_misses for c in all_cases if c.perturbation == "top_swap"), default=0)
    lines.append(f"1. Same inversion count can produce different damage in this grid (observed range for top-swap one-inversion: 0 to {max_extra} extra misses).")
    if strongest_violations:
        lines.append("2. The strongest order-only local-charge inequality is violated on this grid.")
    else:
        lines.append("2. No strongest-form violation was found on this grid; this is evidence of plausibility, not a proof.")
    lines.append("3. Lower-rank inversions can be visible to full-ranking metrics while being invisible to shortlist eviction behavior.")
    if one_cascade:
        lines.append("4. At least one one-inversion case triggers non-local cascade, motivating a residual term.")
    else:
        lines.append("4. No one-inversion non-local cascade was found at this search depth.")
    lines.append("5. Restricted forms remain candidate theorem targets; broadened search may still overturn them.")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"cases={len(all_cases)} strongest_violations={len(strongest_violations)}")


if __name__ == "__main__":
    main()
