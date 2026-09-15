"""Freeze a quota-audit snapshot immediately before production launch.

This is a THIN WRAPPER documenting the exact read-only commands to run on
Wulver (via SSH, outside this script -- it does not itself open an SSH
connection) and how to record their result. It exists because:

  - `quota_info $LOGNAME` (NJIT's documented tool) was not found on this
    system even after `module load wulver` (checked 2026-09-15).
  - `quota -s` only reports an unrelated local filesystem
    (/dev/mapper/vg_main-lv_local), not /mmfs1.
  - `df -h` on a HOME path reports the full /mmfs1 pool size (809TB free),
    NOT the actual per-user HOME fileset quota -- this is exactly what
    caused job 1288536's failure to go undetected by the original launch
    gate.
  - `df -T` on a PROJECT or SCRATCH path DOES correctly report the
    per-fileset quota (confirmed: PROJECT reports a clean 2TiB, SCRATCH a
    clean 10TiB, matching NJIT's documented ~2TB/~10TB per-PI-group
    allocations exactly) -- this is the one reliable read-only signal
    found for this cluster.

Usage (run the SSH command yourself, then pass its output here):

    ssh njit-wulver 'df -T --block-size=1 /mmfs1/scratch/ikoutis/sv96'
    python scripts/pe_long_horizon/freeze_quota_audit.py \
        --scratch-path /mmfs1/scratch/ikoutis/sv96 \
        --scratch-total-bytes <TOTAL> --scratch-used-bytes <USED> \
        --scratch-available-bytes <AVAIL>

Writes configs/pe_long_horizon_production/provenance/quota_audit.json,
which launch_guard.py requires to be present and fresh (<24h old) before
it will pass.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = REPO_ROOT / "configs" / "pe_long_horizon_production" / "provenance" / "quota_audit.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch-path", required=True)
    ap.add_argument("--scratch-total-bytes", type=int, required=True)
    ap.add_argument("--scratch-used-bytes", type=int, required=True)
    ap.add_argument("--scratch-available-bytes", type=int, required=True)
    ap.add_argument("--method", default="df -T (fileset-level; quota_info unavailable on this system)")
    ap.add_argument("--quota-domain", default="SCRATCH")
    args = ap.parse_args()

    payload = {
        "captured_at_epoch": time.time(),
        "captured_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": args.method,
        "quota_domain": args.quota_domain,
        "scratch_path": args.scratch_path,
        "scratch_total_bytes": args.scratch_total_bytes,
        "scratch_used_bytes": args.scratch_used_bytes,
        "scratch_available_bytes": args.scratch_available_bytes,
        "note": (
            "quota_info (NJIT's documented tool) was unavailable on this "
            "system even after module load wulver; df -T on the fileset "
            "path is used instead, which was confirmed to correctly report "
            "PROJECT (2TiB) and SCRATCH (10TiB) fileset quotas exactly "
            "matching NJIT's documented per-PI-group allocations."
        ),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
