from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("task_out_dir")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--task-id", required=True)
    args = ap.parse_args()

    out_dir = Path(args.task_out_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tasks = [t for t in manifest["tasks"] if t["task_id"] == args.task_id]
    if len(tasks) != 1:
        print(f"FAIL: expected exactly one task in manifest for {args.task_id}, found {len(tasks)}")
        return 1
    task = tasks[0]
    expected = task["canonical_h16"]
    expected_horizons = set(manifest["production_horizons"])

    seen_by_horizon: dict[int, set[str]] = {h: set() for h in expected_horizons}
    candidate_keys: set[tuple[int, str, str]] = set()
    duplicate_candidate_keys = 0
    for shard in sorted((out_dir / "shards").glob("*.csv")):
        with shard.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                h = int(row["horizon"])
                if h not in expected_horizons:
                    print(f"FAIL: unexpected horizon {h} in {shard}")
                    return 1
                decision_key = (
                    f"{row['trace_name']}|{row['trace_family']}|cap={int(row['capacity'])}|"
                    f"t={int(row['decision_t'])}|split={row['split']}"
                )
                seen_by_horizon[h].add(decision_key)
                candidate_key = (h, decision_key, row["candidate_page_id"])
                if candidate_key in candidate_keys:
                    duplicate_candidate_keys += 1
                candidate_keys.add(candidate_key)

    failures = []
    summaries = {}
    for h in sorted(expected_horizons):
        keys = sorted(seen_by_horizon[h])
        digest = hashlib.sha256()
        for key in keys:
            digest.update(key.encode("utf-8"))
            digest.update(b"\n")
        summary = {
            "decision_count": len(keys),
            "key_sha256": digest.hexdigest(),
            "matches_canonical_h16": len(keys) == expected["decision_count"] and digest.hexdigest() == expected["key_sha256"],
        }
        summaries[str(h)] = summary
        if not summary["matches_canonical_h16"]:
            failures.append(f"H{h} does not match canonical H16 keys: {summary} expected={expected}")

    if duplicate_candidate_keys:
        failures.append(f"duplicate candidate keys: {duplicate_candidate_keys}")

    out = {
        "task_id": args.task_id,
        "family": task["family"],
        "capacity": task["capacity"],
        "expected_canonical_h16": expected,
        "per_horizon": summaries,
        "duplicate_candidate_keys": duplicate_candidate_keys,
        "validation_result": "PASS" if not failures else "FAIL",
        "failures": failures,
    }
    out_path = out_dir / "key_identity_validation.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print("KEY_IDENTITY_VALIDATION_RESULT:", out["validation_result"])
    if failures:
        for failure in failures:
            print("FAIL:", failure)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
