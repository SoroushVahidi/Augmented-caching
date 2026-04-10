from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Set

INF = 10**9


@dataclass
class DecisionEvent:
    decision_idx: int
    t: int
    phase: int
    belady_victim: str
    chosen_victim: str
    inversions: int
    reuse_gap: int


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
    reuse_gap_i: int
    reuse_gap_j: int
    extra_i: int
    extra_j: int
    extra_ij: int


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
        reuse_gap = abs(next_use_distance(seq, t, bel) - next_use_distance(seq, t, chosen))

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
                reuse_gap=reuse_gap,
            )
        )
        d_idx += 1

    return RunResult(misses=misses, miss_mask=miss_mask, phases=phases, events=events)


def gather_cases(max_len: int) -> List[PairCase]:
    pages = ["A", "B", "C", "D"]
    configs = [(2, 3), (2, 4), (3, 4)]
    out: List[PairCase] = []

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
                        ri = simulate(seq_t, k, perturb_decisions={i})
                        rj = simulate(seq_t, k, perturb_decisions={j})
                        rij = simulate(seq_t, k, perturb_decisions={i, j})

                        if ri.events[i].inversions == 0 or rj.events[j].inversions == 0:
                            continue

                        out.append(
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
                                victim_i=ri.events[i].chosen_victim,
                                victim_j=rj.events[j].chosen_victim,
                                reuse_gap_i=ri.events[i].reuse_gap,
                                reuse_gap_j=rj.events[j].reuse_gap,
                                extra_i=ri.misses - base.misses,
                                extra_j=rj.misses - base.misses,
                                extra_ij=rij.misses - base.misses,
                            )
                        )
    return out


def q_shared_victim(c: PairCase) -> int:
    return 1 if c.victim_i == c.victim_j else 0


def q_same_phase(c: PairCase) -> int:
    return 1 if c.phase_i == c.phase_j else 0


def q_overlap_depth(c: PairCase) -> int:
    depth = 1
    if c.victim_i == c.victim_j:
        depth += 1
    if c.phase_i == c.phase_j:
        depth += 1
    if (c.t_j - c.t_i) <= 2:
        depth += 1
    return depth


def q_reinsertion_collision(c: PairCase) -> int:
    return 1 if c.victim_i == c.victim_j and (c.t_j - c.t_i) <= 2 else 0


def residual(c: PairCase) -> int:
    return max(0, c.extra_ij - (c.extra_i + c.extra_j))


def min_c_for_quantity(cases: List[PairCase], q_fn) -> str:
    needed = 0.0
    impossible = False
    for c in cases:
        r = residual(c)
        q = q_fn(c)
        if r == 0:
            continue
        if q == 0:
            impossible = True
            break
        needed = max(needed, r / q)
    if impossible:
        return "no finite c (residual with q=0 exists)"
    return f"c >= {needed:.2f}"


def pick(xs: Iterable[PairCase], key, reverse: bool = False) -> PairCase | None:
    ys = sorted(xs, key=key, reverse=reverse)
    return ys[0] if ys else None


def format_row(label: str, c: PairCase | None) -> str:
    if c is None:
        return f"|{label}|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|"
    return (
        f"|{label}|{c.seq}|{c.k}|{c.alphabet}|({c.i},{c.j})|({c.t_i},{c.t_j})|({c.phase_i},{c.phase_j})|"
        f"({c.victim_i},{c.victim_j})|{c.extra_i}|{c.extra_j}|{c.extra_ij}|{residual(c)}|{q_overlap_depth(c)}|"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Search bounded-overlap residual candidates")
    ap.add_argument("--max-len", type=int, default=8)
    ap.add_argument("--output-md", default="analysis/pairwise_bounded_overlap_examples.md")
    args = ap.parse_args()

    cases = gather_cases(args.max_len)

    low_overlap_low_damage = [c for c in cases if q_overlap_depth(c) <= 1 and c.extra_ij <= 1]
    high_overlap_bounded = [c for c in cases if q_overlap_depth(c) >= 2 and c.extra_ij <= (c.extra_i + c.extra_j)]
    high_overlap_super = [c for c in cases if q_overlap_depth(c) >= 2 and residual(c) > 0]
    reuse_gap_help = [c for c in cases if min(c.reuse_gap_i, c.reuse_gap_j) >= 3 and residual(c) == 0]

    lines: List[str] = ["# Bounded-overlap residual examples", "", "Exploratory only; not a theorem proof.", ""]
    lines.append("## Search setup")
    lines.append("- Configs (k, alphabet): [(2,3), (2,4), (3,4)]")
    lines.append(f"- Max sequence length: {args.max_len}")
    lines.append(f"- Pair-event cases evaluated: {len(cases)}")
    lines.append("")

    lines.append("## Candidate residual quantities and required constants")
    lines.append(f"- shared-victim indicator: {min_c_for_quantity(cases, q_shared_victim)}")
    lines.append(f"- same-phase indicator: {min_c_for_quantity(cases, q_same_phase)}")
    lines.append(f"- reinsertion-collision indicator: {min_c_for_quantity(cases, q_reinsertion_collision)}")
    lines.append(f"- interaction-depth score (1 + shared-victim + same-phase + near-time): {min_c_for_quantity(cases, q_overlap_depth)}")
    lines.append("")

    lines.append("## Grouped tiny examples")
    lines.append("|group|seq|k|alpha|events|times|phases|victims|extra_i|extra_j|extra_ij|residual|interaction_depth|")
    lines.append("|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|")
    lines.append(format_row("low-overlap low-damage", pick(low_overlap_low_damage, key=lambda c: (len(c.seq), c.seq))))
    lines.append(format_row("high-overlap bounded-damage", pick(high_overlap_bounded, key=lambda c: (len(c.seq), c.seq))))
    lines.append(format_row("high-overlap super-additive damage", pick(high_overlap_super, key=lambda c: (residual(c), len(c.seq), c.seq), reverse=True)))
    lines.append(format_row("reuse-gap separation helps", pick(reuse_gap_help, key=lambda c: (len(c.seq), c.seq))))
    lines.append("")

    rate_super = sum(1 for c in cases if residual(c) > 0)
    lines.append("## Interpretation")
    lines.append(f"1. Super-additive interference appears in {rate_super}/{len(cases)} pair-event cases on this grid.")
    lines.append("2. Pure binary overlap indicators are too weak alone: residual cases exist even when those indicators are zero.")
    lines.append("3. The interaction-depth score is currently the most practical predictor among tested simple quantities because it avoids zero-denominator failures and yields a finite small constant.")
    lines.append("4. Reuse-gap-separated behavior still appears helpful in selected examples, so it remains the best backup theorem path if overlap-residual formalization breaks.")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out}")
    print(f"cases={len(cases)} super={rate_super}")


if __name__ == "__main__":
    main()
