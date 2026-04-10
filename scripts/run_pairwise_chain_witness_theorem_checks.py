from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from lafc.evict_value_wulver_v1 import load_trace_from_any

INF = 10**12


@dataclass
class EvictionEvent:
    t: int
    phase: int
    req: str
    stale_candidates: Tuple[str, ...]
    stale: bool
    victim: str
    chosen_rank: int
    next_arrivals: Dict[str, int]
    stale_next_arrivals: Dict[str, int]
    pairwise_error_count: int


@dataclass
class ChainRecord:
    chain_id: str
    phase: int
    victim_events: List[EvictionEvent]
    head_stale_set: Tuple[str, ...]
    head_next_arrivals: Dict[str, int]


def _next_use(reqs: Sequence[str], t: int, page: str) -> int:
    for j in range(t + 1, len(reqs)):
        if reqs[j] == page:
            return j - t
    return INF


def _phase_ids(reqs: Sequence[str], k: int) -> List[int]:
    phase = 0
    seen: set[str] = set()
    out: List[int] = []
    for r in reqs:
        if r not in seen and len(seen) == k:
            phase += 1
            seen = set()
        seen.add(r)
        out.append(phase)
    return out


def _rank_candidates(
    cache: Sequence[str],
    reqs: Sequence[str],
    t: int,
    mode: str,
    rng: random.Random,
) -> Tuple[List[str], Dict[str, int]]:
    next_arr = {p: _next_use(reqs, t, p) for p in cache}
    truth = sorted(cache, key=lambda p: (next_arr[p], p), reverse=True)
    if mode == "oracle":
        return truth, next_arr
    if mode == "noisy":
        scored = [(next_arr[p] + rng.gauss(0.0, 2.0), p) for p in cache]
        ranked = [p for _s, p in sorted(scored, reverse=True)]
        return ranked, next_arr
    if mode == "adv_bias":
        # Push pages with near-future reuse up (wrong direction for eviction).
        scored = [(-next_arr[p] + rng.uniform(-1.5, 1.5), p) for p in cache]
        ranked = [p for _s, p in sorted(scored, reverse=True)]
        return ranked, next_arr
    if mode == "near_tie":
        # Compress distances into buckets then perturb lightly.
        scored = [((next_arr[p] // 3) + rng.uniform(-0.25, 0.25), p) for p in cache]
        ranked = [p for _s, p in sorted(scored, reverse=True)]
        return ranked, next_arr
    raise ValueError(f"unknown ranking mode: {mode}")


def _generate_random_trace(
    k: int,
    length: int,
    alphabet: int,
    clean_prob: float,
    repeat_bias: float,
    rng: random.Random,
) -> List[str]:
    pages = [f"p{i}" for i in range(alphabet)]
    trace: List[str] = []
    hot: List[str] = pages[: max(k + 1, min(alphabet, k + 4))]
    for _ in range(length):
        if trace and rng.random() < repeat_bias:
            # Reappear from recent history to force stale-vs-clean tensions.
            cand = trace[max(0, len(trace) - 2 * k) :]
            trace.append(rng.choice(cand))
            continue
        if rng.random() < clean_prob:
            trace.append(rng.choice(pages))
        else:
            trace.append(rng.choice(hot))
    return trace


def _generate_structured_trace(k: int, blocks: int, rng: random.Random) -> List[str]:
    # Hard-case generator:
    # alternate between k+1 set and repeated head returns to trigger stale chains.
    base = [f"s{i}" for i in range(k + 1)]
    out: List[str] = []
    for _ in range(blocks):
        cyc = list(base)
        rng.shuffle(cyc)
        out.extend(cyc)
        # Repeated stale-page reappearances with near ties.
        out.extend([cyc[0], cyc[1], cyc[0], cyc[2], cyc[0]])
    return out


def _simulate(reqs: Sequence[str], k: int, ranking_mode: str, seed: int) -> Tuple[List[EvictionEvent], int]:
    cache: List[str] = []
    marks: set[str] = set()
    phase_ids = _phase_ids(reqs, k)
    rng = random.Random(seed)
    events: List[EvictionEvent] = []
    eta_pair = 0

    for t, req in enumerate(reqs):
        if req in cache:
            marks.add(req)
            continue
        # miss
        if len(cache) < k:
            cache.append(req)
            marks.add(req)
            continue
        stale = [p for p in cache if p not in marks]
        ranked, next_arr = _rank_candidates(cache, reqs, t, ranking_mode, rng)
        victim = ranked[0]
        stale_next = {p: next_arr[p] for p in stale}
        chosen_rank = ranked.index(victim)
        # Decision-local pairwise budget: how many candidates should outrank chosen by true order.
        pairwise_err = sum(1 for p in cache if next_arr[p] > next_arr[victim])
        eta_pair += pairwise_err

        events.append(
            EvictionEvent(
                t=t,
                phase=phase_ids[t],
                req=req,
                stale_candidates=tuple(sorted(stale)),
                stale=(victim in stale),
                victim=victim,
                chosen_rank=chosen_rank,
                next_arrivals=next_arr,
                stale_next_arrivals=stale_next,
                pairwise_error_count=pairwise_err,
            )
        )

        cache.remove(victim)
        cache.append(req)
        marks.add(req)
        if len(marks) == k + 1:
            marks = {req}
    return events, eta_pair


def _build_chains(events: Sequence[EvictionEvent]) -> List[ChainRecord]:
    # Conservative chain extractor:
    # each phase has stale-victim runs; each run is a chain.
    by_phase: Dict[int, List[EvictionEvent]] = {}
    for e in events:
        by_phase.setdefault(e.phase, []).append(e)
    chains: List[ChainRecord] = []
    for phase, evs in sorted(by_phase.items()):
        current: List[EvictionEvent] = []
        chain_idx = 0
        for e in evs:
            if e.stale:
                current.append(e)
            else:
                if current:
                    head = current[0]
                    chains.append(
                        ChainRecord(
                            chain_id=f"ph{phase}_c{chain_idx}",
                            phase=phase,
                            victim_events=list(current),
                            head_stale_set=head.stale_candidates,
                            head_next_arrivals=head.stale_next_arrivals,
                        )
                    )
                    chain_idx += 1
                    current = []
        if current:
            head = current[0]
            chains.append(
                ChainRecord(
                    chain_id=f"ph{phase}_c{chain_idx}",
                    phase=phase,
                    victim_events=list(current),
                    head_stale_set=head.stale_candidates,
                    head_next_arrivals=head.stale_next_arrivals,
                )
            )
    return chains


def _check_chain_lemmas(chains: Sequence[ChainRecord], eta_pair: int) -> Dict[str, object]:
    chain_rows: List[Dict[str, object]] = []
    violations: List[Dict[str, object]] = []
    total_I = 0
    for c in chains:
        if not c.victim_events:
            continue
        a_vals = [ev.next_arrivals[ev.victim] for ev in c.victim_events]
        monotone_ok = all(a_vals[i] < a_vals[i + 1] for i in range(len(a_vals) - 1))
        D = len(c.victim_events)
        e1 = c.victim_events[0].victim
        head_map = c.head_next_arrivals
        if e1 not in head_map:
            # If head victim not in stale map, this chain definition is inconsistent; treat as violation.
            I_set: List[str] = []
            head_bound_ok = False
        else:
            I_set = sorted([s for s in c.head_stale_set if head_map.get(s, -INF) > head_map[e1]])
            head_bound_ok = D <= (len(I_set) + 1)
        total_I += len(I_set)
        row = {
            "chain_id": c.chain_id,
            "phase": c.phase,
            "damage_D": D,
            "I_size": len(I_set),
            "lemma1_chain_monotone": monotone_ok,
            "lemma2_head_bound": head_bound_ok,
            "head_victim_e1": e1,
            "head_stale_set": ",".join(c.head_stale_set),
            "I_set": ",".join(I_set),
            "a_values": ",".join(str(v) for v in a_vals),
        }
        chain_rows.append(row)
        if not monotone_ok:
            violations.append({"lemma": "lemma1_chain_monotone", **row})
        if not head_bound_ok:
            violations.append({"lemma": "lemma2_head_bound", **row})
    lemma3_ok = total_I <= eta_pair
    if not lemma3_ok:
        violations.append(
            {
                "lemma": "lemma3_phase_global_budget",
                "chain_id": "GLOBAL",
                "phase": -1,
                "damage_D": -1,
                "I_size": total_I,
                "lemma1_chain_monotone": True,
                "lemma2_head_bound": True,
                "head_victim_e1": "",
                "head_stale_set": "",
                "I_set": "",
                "a_values": f"sum_I={total_I};eta_pair={eta_pair}",
            }
        )
    return {
        "chain_rows": chain_rows,
        "violations": violations,
        "sum_I": total_I,
        "eta_pair": eta_pair,
        "lemma3_ok": lemma3_ok,
    }


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _minimize_counterexample(
    reqs: List[str],
    k: int,
    ranking_mode: str,
    seed: int,
    lemma: str,
) -> List[str]:
    # Greedy deletion minimizer preserving at least one violation for target lemma.
    cur = list(reqs)
    changed = True
    while changed and len(cur) > k + 2:
        changed = False
        for i in range(len(cur)):
            cand = cur[:i] + cur[i + 1 :]
            evs, eta = _simulate(cand, k, ranking_mode, seed)
            chk = _check_chain_lemmas(_build_chains(evs), eta)
            if any(v["lemma"] == lemma for v in chk["violations"]):
                cur = cand
                changed = True
                break
    return cur


def _instance_iter_exhaustive(
    seed: int,
    max_len: int,
    k_values: Sequence[int],
    alphabet_sizes: Sequence[int],
) -> Iterable[Tuple[str, List[str], int, int]]:
    rng = random.Random(seed)
    modes = ["oracle", "near_tie", "noisy", "adv_bias"]
    for k in k_values:
        for alpha in alphabet_sizes:
            alpha = max(alpha, k + 1)
            pages = [f"p{i}" for i in range(alpha)]
            # Semi-exhaustive bounded enumeration (keep tractable).
            for n in range(k + 2, max_len + 1):
                budget = min(2000, alpha**n)
                seen = set()
                for _ in range(budget):
                    reqs = [rng.choice(pages) for _ in range(n)]
                    key = tuple(reqs)
                    if key in seen:
                        continue
                    seen.add(key)
                    mode = modes[(len(seen) + n + k + alpha) % len(modes)]
                    yield mode, reqs, k, alpha


def _instance_iter_random(seed: int, n_instances: int) -> Iterable[Tuple[str, List[str], int, int]]:
    rng = random.Random(seed)
    modes = ["noisy", "adv_bias", "near_tie"]
    for _ in range(n_instances):
        k = rng.choice([2, 3, 4, 5, 6, 8])
        alpha = rng.choice([k + 1, k + 2, k + 4, 2 * k + 2])
        length = rng.randint(5 * k, 20 * k)
        clean_prob = rng.uniform(0.2, 0.75)
        repeat_bias = rng.uniform(0.2, 0.9)
        mode = rng.choice(modes)
        reqs = _generate_random_trace(k=k, length=length, alphabet=alpha, clean_prob=clean_prob, repeat_bias=repeat_bias, rng=rng)
        yield mode, reqs, k, alpha


def _instance_iter_structured(seed: int, n_instances: int) -> Iterable[Tuple[str, List[str], int, int]]:
    rng = random.Random(seed)
    modes = ["near_tie", "adv_bias", "noisy"]
    for _ in range(n_instances):
        k = rng.choice([2, 3, 4, 5, 6])
        blocks = rng.randint(8, 24)
        reqs = _generate_structured_trace(k=k, blocks=blocks, rng=rng)
        alpha = k + 1
        mode = rng.choice(modes)
        yield mode, reqs, k, alpha


def _instance_iter_real(
    manifest: Path,
    max_traces: int,
    prefix: int,
) -> Iterable[Tuple[str, List[str], int, int]]:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    for i, row in enumerate(rows[:max_traces]):
        path = row["path"].strip()
        req_objs, pages, _src = load_trace_from_any(path)
        reqs = [r.page_id for r in req_objs[:prefix]]
        if len(reqs) < 20:
            continue
        uniq = len(set(reqs))
        # Multiple capacities per real prefix to probe sensitivity.
        for k in sorted({max(2, min(uniq - 1, x)) for x in [2, 3, 4, 5, 8, 16, 32]}):
            if k >= uniq:
                continue
            mode = ["near_tie", "noisy", "adv_bias"][((i + k) % 3)]
            yield mode, reqs, k, uniq


def main() -> None:
    ap = argparse.ArgumentParser(description="High-compute chain-witness theorem validation campaign.")
    ap.add_argument("--mode", choices=["exhaustive", "random", "structured", "real"], required=True)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--job-label", default="job0")
    ap.add_argument("--out-dir", type=Path, default=Path("analysis/pairwise_chain_witness_campaign/jobs"))
    ap.add_argument("--max-instances", type=int, default=20000)
    ap.add_argument("--max-len", type=int, default=10)
    ap.add_argument("--exhaustive-k", default="2,3,4,5")
    ap.add_argument("--exhaustive-alphabet", default="3,4,5,6")
    ap.add_argument("--manifest", type=Path, default=Path("analysis/wulver_trace_manifest_full.csv"))
    ap.add_argument("--real-max-traces", type=int, default=7)
    ap.add_argument("--real-prefix", type=int, default=20000)
    args = ap.parse_args()

    out_root = args.out_dir / args.job_label
    out_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "exhaustive":
        k_values = [int(x) for x in args.exhaustive_k.split(",") if x.strip()]
        alpha_values = [int(x) for x in args.exhaustive_alphabet.split(",") if x.strip()]
        inst_iter = _instance_iter_exhaustive(seed=args.seed, max_len=args.max_len, k_values=k_values, alphabet_sizes=alpha_values)
    elif args.mode == "random":
        inst_iter = _instance_iter_random(seed=args.seed, n_instances=args.max_instances)
    elif args.mode == "structured":
        inst_iter = _instance_iter_structured(seed=args.seed, n_instances=args.max_instances)
    else:
        inst_iter = _instance_iter_real(manifest=args.manifest, max_traces=args.real_max_traces, prefix=args.real_prefix)

    instance_rows: List[Dict[str, object]] = []
    chain_rows_all: List[Dict[str, object]] = []
    violations: List[Dict[str, object]] = []
    minimals: List[Dict[str, object]] = []

    for idx, (ranking_mode, reqs, k, alpha) in enumerate(inst_iter):
        if idx >= args.max_instances:
            break
        events, eta_pair = _simulate(reqs=reqs, k=k, ranking_mode=ranking_mode, seed=args.seed + idx)
        chains = _build_chains(events)
        chk = _check_chain_lemmas(chains, eta_pair)
        chain_rows = chk["chain_rows"]
        vios = chk["violations"]
        l1_ok = all(r["lemma1_chain_monotone"] for r in chain_rows) if chain_rows else True
        l2_ok = all(r["lemma2_head_bound"] for r in chain_rows) if chain_rows else True
        l3_ok = bool(chk["lemma3_ok"])
        passed_all = l1_ok and l2_ok and l3_ok
        trace_str = ",".join(reqs)
        instance_rows.append(
            {
                "job_label": args.job_label,
                "instance_id": idx,
                "mode": args.mode,
                "ranking_mode": ranking_mode,
                "k": k,
                "alphabet": alpha,
                "trace_len": len(reqs),
                "chain_count": len(chain_rows),
                "sum_I": chk["sum_I"],
                "eta_pair": chk["eta_pair"],
                "lemma1_pass": l1_ok,
                "lemma2_pass": l2_ok,
                "lemma3_pass": l3_ok,
                "all_pass": passed_all,
                "violation_count": len(vios),
                "trace": trace_str,
            }
        )
        for r in chain_rows:
            chain_rows_all.append({"job_label": args.job_label, "instance_id": idx, **r})
        for v in vios:
            vio = {
                "job_label": args.job_label,
                "instance_id": idx,
                "mode": args.mode,
                "ranking_mode": ranking_mode,
                "k": k,
                "trace_len": len(reqs),
                "trace": trace_str,
                **v,
            }
            violations.append(vio)
            if len(minimals) < 200:
                min_trace = _minimize_counterexample(
                    reqs=list(reqs),
                    k=k,
                    ranking_mode=ranking_mode,
                    seed=args.seed + idx,
                    lemma=str(v["lemma"]),
                )
                minimals.append(
                    {
                        "job_label": args.job_label,
                        "instance_id": idx,
                        "lemma": v["lemma"],
                        "k": k,
                        "ranking_mode": ranking_mode,
                        "orig_trace_len": len(reqs),
                        "min_trace_len": len(min_trace),
                        "orig_trace": trace_str,
                        "min_trace": ",".join(min_trace),
                    }
                )

    _write_csv(out_root / "instances.csv", instance_rows)
    _write_csv(out_root / "chains.csv", chain_rows_all)
    _write_csv(out_root / "violations.csv", violations)
    _write_csv(out_root / "minimized_counterexamples.csv", minimals)

    summary = {
        "job_label": args.job_label,
        "mode": args.mode,
        "seed": args.seed,
        "instances_checked": len(instance_rows),
        "chains_checked": len(chain_rows_all),
        "violations_found": len(violations),
        "lemma1_failures": sum(1 for v in violations if v["lemma"] == "lemma1_chain_monotone"),
        "lemma2_failures": sum(1 for v in violations if v["lemma"] == "lemma2_head_bound"),
        "lemma3_failures": sum(1 for v in violations if v["lemma"] == "lemma3_phase_global_budget"),
        "any_counterexample": len(violations) > 0,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
