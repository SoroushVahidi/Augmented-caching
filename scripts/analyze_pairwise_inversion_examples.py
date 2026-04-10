from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence, Tuple


@dataclass
class DecisionRecord:
    t: int
    request: str
    cache_before: Tuple[str, ...]
    belady_victim: str | None
    chosen_victim: str | None
    inversion_count: int
    phase: int


@dataclass
class RunResult:
    misses: int
    decisions: List[DecisionRecord]


def next_use_distance(seq: Sequence[str], t: int, page: str) -> int:
    for j in range(t + 1, len(seq)):
        if seq[j] == page:
            return j - t
    return 10**9


def belady_order(cache: Sequence[str], seq: Sequence[str], t: int) -> List[str]:
    # Highest rank = best eviction target (furthest next use)
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


def apply_inversions(order: List[str], n_inversions: int) -> List[str]:
    out = list(order)
    i = 0
    applied = 0
    while applied < n_inversions and len(out) >= 2:
        a = i % (len(out) - 1)
        out[a], out[a + 1] = out[a + 1], out[a]
        applied += 1
        i += 1
    return out


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


def simulate(seq: Sequence[str], k: int, inversion_budget_per_decision: int) -> RunResult:
    cache: List[str] = []
    misses = 0
    recs: List[DecisionRecord] = []
    phases = phase_ids(seq, k)
    for t, req in enumerate(seq):
        if req in cache:
            recs.append(DecisionRecord(t=t, request=req, cache_before=tuple(sorted(cache)), belady_victim=None, chosen_victim=None, inversion_count=0, phase=phases[t]))
            continue
        misses += 1
        if len(cache) < k:
            cache.append(req)
            recs.append(DecisionRecord(t=t, request=req, cache_before=tuple(sorted([c for c in cache if c != req])), belady_victim=None, chosen_victim=None, inversion_count=0, phase=phases[t]))
            continue

        cache_before = tuple(sorted(cache))
        true_order = belady_order(cache, seq, t)
        pred_order = apply_inversions(true_order, inversion_budget_per_decision)
        inv = inversion_count(true_order, pred_order)
        bel_victim = true_order[0]
        chosen = pred_order[0]
        cache.remove(chosen)
        cache.append(req)
        recs.append(
            DecisionRecord(
                t=t,
                request=req,
                cache_before=cache_before,
                belady_victim=bel_victim,
                chosen_victim=chosen,
                inversion_count=inv,
                phase=phases[t],
            )
        )
    return RunResult(misses=misses, decisions=recs)


def stringify_seq(seq: Sequence[str]) -> str:
    return "".join(seq)


def main() -> None:
    ap = argparse.ArgumentParser(description="Bruteforce tiny pairwise inversion examples against Belady")
    ap.add_argument("--output-md", default="analysis/pairwise_inversion_examples.md")
    ap.add_argument("--max-len", type=int, default=8)
    args = ap.parse_args()

    pages = ["A", "B", "C", "D"]
    configs = [(2, 3), (2, 4), (3, 4)]  # (k, alphabet size)

    rows: List[Dict[str, object]] = []
    for k, alpha in configs:
        alphabet = pages[:alpha]
        for n in range(k + 2, args.max_len + 1):
            for seq in itertools.product(alphabet, repeat=n):
                bel = simulate(seq, k, inversion_budget_per_decision=0)
                inv1 = simulate(seq, k, inversion_budget_per_decision=1)
                decision_steps = [d for d in bel.decisions if d.belady_victim is not None]
                mean_inv = mean([d.inversion_count for d in inv1.decisions if d.belady_victim is not None]) if decision_steps else 0.0
                rows.append(
                    {
                        "seq": stringify_seq(seq),
                        "k": k,
                        "alphabet": alpha,
                        "length": n,
                        "belady_misses": bel.misses,
                        "inv1_misses": inv1.misses,
                        "extra_misses_inv1": inv1.misses - bel.misses,
                        "decision_count": len(decision_steps),
                        "mean_local_inversions": mean_inv,
                        "phase_count": max(d.phase for d in bel.decisions) + 1 if bel.decisions else 0,
                    }
                )

    zero_agree = [r for r in rows if int(r["extra_misses_inv1"]) == 0]
    small_damage = [r for r in rows if int(r["extra_misses_inv1"]) == 1]
    large_damage = [r for r in rows if int(r["extra_misses_inv1"]) >= 2]

    def top_examples(xs: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
        return sorted(xs, key=lambda r: (int(r["extra_misses_inv1"]), int(r["length"]), str(r["seq"])))[:limit]

    def top_large(xs: List[Dict[str, object]], limit: int = 5) -> List[Dict[str, object]]:
        return sorted(xs, key=lambda r: (-int(r["extra_misses_inv1"]), int(r["length"]), str(r["seq"])))[:limit]

    lines: List[str] = ["# Pairwise inversion brute-force examples", ""]
    lines.append("This note is exploratory only: it does not prove a theorem.")
    lines.append("")
    lines.append("## Search setup")
    lines.append(f"- Configs (k, alphabet): {configs}")
    lines.append(f"- Max sequence length: {args.max_len}")
    lines.append(f"- Total enumerated sequences: {len(rows)}")
    lines.append("")
    lines.append("## Aggregate counts")
    lines.append(f"- zero-damage under one inversion per eviction step: {len(zero_agree)}")
    lines.append(f"- small-damage (+1 miss): {len(small_damage)}")
    lines.append(f"- large-damage (>= +2 misses): {len(large_damage)}")
    lines.append("")
    lines.append("## Example table: zero inversions / agreement-like behavior")
    lines.append("|seq|k|alphabet|len|belady|inv1|extra|decisions|mean_local_inversions|phases|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top_examples(zero_agree):
        lines.append(
            f"|{r['seq']}|{r['k']}|{r['alphabet']}|{r['length']}|{r['belady_misses']}|{r['inv1_misses']}|{r['extra_misses_inv1']}|{r['decision_count']}|{r['mean_local_inversions']:.2f}|{r['phase_count']}|"
        )

    lines.append("")
    lines.append("## Example table: one inversion with small damage")
    lines.append("|seq|k|alphabet|len|belady|inv1|extra|decisions|mean_local_inversions|phases|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top_examples(small_damage):
        lines.append(
            f"|{r['seq']}|{r['k']}|{r['alphabet']}|{r['length']}|{r['belady_misses']}|{r['inv1_misses']}|{r['extra_misses_inv1']}|{r['decision_count']}|{r['mean_local_inversions']:.2f}|{r['phase_count']}|"
        )

    lines.append("")
    lines.append("## Example table: one inversion with large damage")
    lines.append("|seq|k|alphabet|len|belady|inv1|extra|decisions|mean_local_inversions|phases|")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in top_large(large_damage):
        lines.append(
            f"|{r['seq']}|{r['k']}|{r['alphabet']}|{r['length']}|{r['belady_misses']}|{r['inv1_misses']}|{r['extra_misses_inv1']}|{r['decision_count']}|{r['mean_local_inversions']:.2f}|{r['phase_count']}|"
        )

    lines.append("")
    lines.append("## Observed patterns")
    lines.append("- A single local inversion can be harmless on some traces (especially with short horizons / low revisit pressure).")
    lines.append("- A single local inversion can also amplify into multiple extra misses when it evicts a page reused quickly while preserving a far-future page.")
    lines.append("- Damage appears sensitive to request spacing and phase transitions, not just raw inversion count.")
    lines.append("- Tempting conjecture that 'one inversion => at most +1 miss' is falsified by examples in the large-damage table.")

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} with {len(rows)} sequences")


if __name__ == "__main__":
    main()
