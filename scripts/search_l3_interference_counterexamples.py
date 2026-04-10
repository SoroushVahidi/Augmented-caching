from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple

INF = 10**9


@dataclass
class DecisionEvent:
    decision_idx: int
    t: int
    phase: int
    belady_victim: str
    chosen_victim: str
    inversions: int


@dataclass
class RunResult:
    misses: int
    miss_mask: List[bool]
    phases: List[int]
    events: List[DecisionEvent]


@dataclass
class PairCase:
    seq: str
    k: int
    alphabet: int
    i: int
    j: int
    t_i: int
    t_j: int
    phase_i: int
    phase_j: int
    victim_i: str
    victim_j: str
    extra_i: int
    extra_j: int
    extra_ij: int
    extra_ij_same_phase: int


def next_use_distance(seq: Sequence[str], t: int, page: str) -> int:
    for u in range(t + 1, len(seq)):
        if seq[u] == page:
            return u - t
    return INF


def belady_order(cache: Sequence[str], seq: Sequence[str], t: int) -> List[str]:
    return sorted(cache, key=lambda p: (next_use_distance(seq, t, p), p), reverse=True)


def inversion_count(true_order: Sequence[str], pred_order: Sequence[str]) -> int:
    pos = {p: i for i, p in enumerate(pred_order)}
    inv = 0
    for a in range(len(true_order)):
        for b in range(a + 1, len(true_order)):
            if pos[true_order[a]] > pos[true_order[b]]:
                inv += 1
    return inv


def phase_ids(seq: Sequence[str], k: int) -> List[int]:
    out: List[int] = []
    seen: set[str] = set()
    phase = 0
    for req in seq:
        if req not in seen and len(seen) == k:
            phase += 1
            seen = set()
        seen.add(req)
        out.append(phase)
    return out


def top_swap(order: List[str]) -> List[str]:
    pred = list(order)
    if len(pred) >= 2:
        pred[0], pred[1] = pred[1], pred[0]
    return pred


def simulate(seq: Sequence[str], k: int, perturb_decisions: Set[int]) -> RunResult:
    cache: List[str] = []
    phases = phase_ids(seq, k)
    misses = 0
    miss_mask = [False] * len(seq)
    events: List[DecisionEvent] = []

    d_idx = 0
    for t, req in enumerate(seq):
        if req in cache:
            continue
        misses += 1
        miss_mask[t] = True
        if len(cache) < k:
            cache.append(req)
            continue

        true_order = belady_order(cache, seq, t)
        pred_order = top_swap(true_order) if d_idx in perturb_decisions else list(true_order)
        inv = inversion_count(true_order, pred_order)

        bel = true_order[0]
        chosen = pred_order[0]
        cache.remove(chosen)
        cache.append(req)

        events.append(
            DecisionEvent(
                decision_idx=d_idx,
                t=t,
                phase=phases[t],
                belady_victim=bel,
                chosen_victim=chosen,
                inversions=inv,
            )
        )
        d_idx += 1

    return RunResult(misses=misses, miss_mask=miss_mask, phases=phases, events=events)


def gather_cases(max_len: int) -> List[PairCase]:
    pages = ["A", "B", "C", "D"]
    configs = [(2, 3), (2, 4), (3, 4)]
    cases: List[PairCase] = []

    for k, alpha in configs:
        alphabet = pages[:alpha]
        for n in range(k + 3, max_len + 1):
            for seq_t in itertools.product(alphabet, repeat=n):
                seq = "".join(seq_t)
                base = simulate(seq_t, k, perturb_decisions=set())
                if len(base.events) < 2:
                    continue

                for i in range(len(base.events) - 1):
                    for j in range(i + 1, len(base.events)):
                        run_i = simulate(seq_t, k, perturb_decisions={i})
                        run_j = simulate(seq_t, k, perturb_decisions={j})
                        run_ij = simulate(seq_t, k, perturb_decisions={i, j})

                        e_i = run_i.misses - base.misses
                        e_j = run_j.misses - base.misses
                        e_ij = run_ij.misses - base.misses

                        # Keep only pairs where both local events are true inversions.
                        if run_i.events[i].inversions == 0 or run_j.events[j].inversions == 0:
                            continue

                        extra_ij_same_phase = sum(
                            1
                            for t, (m0, m2) in enumerate(zip(base.miss_mask, run_ij.miss_mask))
                            if m2 and not m0 and run_ij.phases[t] in {run_ij.events[i].phase, run_ij.events[j].phase}
                        )

                        cases.append(
                            PairCase(
                                seq=seq,
                                k=k,
                                alphabet=alpha,
                                i=i,
                                j=j,
                                t_i=base.events[i].t,
                                t_j=base.events[j].t,
                                phase_i=base.events[i].phase,
                                phase_j=base.events[j].phase,
                                victim_i=run_i.events[i].chosen_victim,
                                victim_j=run_j.events[j].chosen_victim,
                                extra_i=e_i,
                                extra_j=e_j,
                                extra_ij=e_ij,
                                extra_ij_same_phase=extra_ij_same_phase,
                            )
                        )
    return cases


def pick_first(xs: Iterable[PairCase], key) -> PairCase | None:
    ys = sorted(xs, key=key)
    return ys[0] if ys else None


def pick_last(xs: Iterable[PairCase], key) -> PairCase | None:
    ys = sorted(xs, key=key)
    return ys[-1] if ys else None


def row(label: str, c: PairCase | None) -> str:
    if c is None:
        return f"|{label}|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|"
    return (
        f"|{label}|{c.seq}|{c.k}|{c.alphabet}|({c.i},{c.j})|({c.t_i},{c.t_j})|({c.phase_i},{c.phase_j})|"
        f"({c.victim_i},{c.victim_j})|{c.extra_i}|{c.extra_j}|{c.extra_ij}|{c.extra_i + c.extra_j}|"
        f"{c.extra_ij_same_phase}|{'Y' if c.extra_ij > c.extra_i + c.extra_j else 'N'}|"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Search L3 interference/super-additivity counterexamples")
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--output-md", default="analysis/pairwise_l3_attack_examples.md")
    args = ap.parse_args()

    cases = gather_cases(args.max_len)

    # Strong L3 attack target: both single events are bounded (+0/+1) but combined is super-additive.
    strong_failures = [
        c for c in cases if c.extra_i <= 1 and c.extra_j <= 1 and c.extra_ij > (c.extra_i + c.extra_j)
    ]

    bounded_overlap_subset = [c for c in cases if c.victim_i == c.victim_j]
    separated_time_subset = [c for c in cases if (c.t_j - c.t_i) >= 3]
    one_shot_subset = [c for c in cases if c.extra_i <= 1 and c.extra_j <= 1]

    def violation_rate(xs: List[PairCase]) -> str:
        if not xs:
            return "n/a"
        v = sum(1 for c in xs if c.extra_ij > (c.extra_i + c.extra_j))
        return f"{v}/{len(xs)}"

    independent = [c for c in cases if c.extra_ij == (c.extra_i + c.extra_j) and c.extra_ij > 0]
    overlap_bounded = [c for c in bounded_overlap_subset if c.extra_ij <= (c.extra_i + c.extra_j)]
    merged_cascade = [c for c in cases if c.extra_ij > (c.extra_i + c.extra_j)]
    separated_good = [c for c in separated_time_subset if c.extra_ij <= (c.extra_i + c.extra_j)]

    lines: List[str] = ["# L3 interference/additivity attack examples", "", "This note is exploratory and not a proof.", ""]
    lines.append("## Search setup")
    lines.append("- Perturbation type: top-swap at chosen eviction decisions.")
    lines.append("- Configs (k, alphabet): [(2,3), (2,4), (3,4)]")
    lines.append(f"- Max sequence length: {args.max_len}")
    lines.append(f"- Pair-event cases evaluated: {len(cases)}")
    lines.append(f"- Strong-L3 failure cases found: {len(strong_failures)}")
    lines.append("")
    lines.append("Strong L3 attack condition tested: single events individually bounded (+0/+1), but joint run is super-additive.")
    lines.append("")

    lines.append("## Restricted-form super-additivity rates")
    lines.append("- Bounded-overlap subset (same perturbed victim page): " + violation_rate(bounded_overlap_subset))
    lines.append("- Separated-return-times subset (event time gap >= 3): " + violation_rate(separated_time_subset))
    lines.append("- One-shot local subset (both singles <= +1): " + violation_rate(one_shot_subset))
    lines.append("")

    lines.append("## Targeted example families")
    lines.append("|family|seq|k|alpha|events(i,j)|times|phases|victims|extra_i|extra_j|extra_ij|sum_singles|joint_same_phase_extra|super_additive?|")
    lines.append("|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|")
    lines.append(row("two events add independently", pick_first(independent, key=lambda c: (len(c.seq), c.seq))))
    lines.append(row("overlap but bounded", pick_first(overlap_bounded, key=lambda c: (len(c.seq), c.seq))))
    lines.append(row("merge into cascade", pick_last(merged_cascade, key=lambda c: (c.extra_ij - (c.extra_i + c.extra_j), len(c.seq), c.seq))))
    lines.append(row("separated returns prevent interference", pick_first(separated_good, key=lambda c: (len(c.seq), c.seq))))
    lines.append("")

    max_gap = max((c.extra_ij - (c.extra_i + c.extra_j) for c in cases), default=0)
    lines.append("## Interpretation")
    if strong_failures:
        lines.append(f"1. Strong L3 fails on this grid; max super-additivity gap observed is {max_gap}.")
    else:
        lines.append("1. No strong-L3 super-additivity case was found on this grid; evidence is supportive but not a proof.")
    lines.append("2. Interaction is sensitive to event footprint overlap and timing, but this grid does not show a clean monotone protection from simple time separation alone.")
    lines.append("3. Phase boundary proximity remains a risk variable: near-boundary events should be audited separately in theorem wording.")
    lines.append("4. Among tested restrictions, bounded-overlap with an explicit residual currently looks more plausible than strict separation-only additivity.")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"cases={len(cases)} strong_failures={len(strong_failures)} max_super_gap={max_gap}")


if __name__ == "__main__":
    main()
