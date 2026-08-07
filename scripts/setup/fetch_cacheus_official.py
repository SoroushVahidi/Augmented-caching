"""Fetch the official CACHEUS source (sylab/cacheus) as a pinned, external,
NOT-vendored dependency.

Why external rather than vendored: github.com/sylab/cacheus has no LICENSE
file (`GET /repos/sylab/cacheus/license` returns 404, `license` field in the
repo API response is `null`). With no explicit license, default copyright
applies and no redistribution permission is granted. This script therefore
clones the repository into a location outside this repository's version
control (see `.gitignore`: `/external/`) and the CACHEUS policy adapter
(`src/lafc/policies/cacheus.py`) imports the official `Cacheus` class
directly from that external clone at runtime -- the official code is
executed unmodified, never copied into this repository's own license.

Usage:
    python scripts/setup/fetch_cacheus_official.py
    python scripts/setup/fetch_cacheus_official.py --commit <sha>

Records the exact commit and a sha256 of every consulted file to
external/cacheus_official/PROVENANCE.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

OFFICIAL_REPO_URL = "https://github.com/sylab/cacheus.git"
# Pinned commit: sylab/cacheus@main as of 2026-08-06 (verified as the
# authors' own repository -- linked from the FIU Systems Research
# Laboratory CACHEUS project page and from the paper's own README/BibTeX).
DEFAULT_COMMIT = "1eec63ce166502be33ddd1f35bc041ed73a24f4d"

# Only the files this repository's adapter actually imports/executes are
# tracked for provenance hashing (the official Cacheus class and the small
# stdlib-only data-structure/utility modules it depends on). run.py,
# run_alg.py, visual.py, get_algorithm.py, and algs/lib/traces.py are part
# of the official repo but are NOT used here: this repository drives
# `Cacheus.request(oblock, ts)` directly with its own already-parsed
# request stream instead of the official blkparse-format trace reader (see
# docs/cacheus_method_spec.md, "Semantic compatibility").
TRACKED_FILES = [
    "code/algs/cacheus.py",
    "code/algs/lru.py",
    "code/algs/lib/dequedict.py",
    "code/algs/lib/heapdict.py",
    "code/algs/lib/cacheop.py",
    "code/algs/lib/optional_args.py",
    "code/algs/lib/pollutionator.py",
    "code/algs/lib/visualizinator.py",
    "code/algs/lib/__init__.py",
    "code/algs/__init__.py",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DIR = REPO_ROOT / "external" / "cacheus_official"
CLONE_DIR = EXTERNAL_DIR / "src"


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--repo-url", default=OFFICIAL_REPO_URL)
    ap.add_argument("--force", action="store_true", help="Re-clone even if already present.")
    args = ap.parse_args()

    if CLONE_DIR.exists() and not args.force:
        head = subprocess.run(
            ["git", "-C", str(CLONE_DIR), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        if head == args.commit:
            print(f"Already present at pinned commit {args.commit}: {CLONE_DIR}")
            return
        print(
            f"WARNING: existing clone at {CLONE_DIR} is at {head}, "
            f"not the pinned commit {args.commit}. Re-run with --force to reset.",
            file=sys.stderr,
        )
        sys.exit(1)

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    if CLONE_DIR.exists():
        subprocess.run(["rm", "-rf", str(CLONE_DIR)], check=True)

    print(f"Cloning {args.repo_url} into {CLONE_DIR} ...")
    subprocess.run(["git", "clone", args.repo_url, str(CLONE_DIR)], check=True)
    subprocess.run(["git", "-C", str(CLONE_DIR), "checkout", args.commit], check=True)

    # `code/algs` has no __init__.py files in the official repo (it relies
    # on being run as the working directory with `algs` as a top-level
    # package via `sys.path`, not as an installed/importable package). This
    # repository imports it as `cacheus_official.algs.cacheus`, so minimal,
    # empty __init__.py files are added here -- a portability-only change,
    # zero algorithmic content, applied only in the external clone
    # directory (never touching what git tracks at the pinned commit).
    for pkg_init in ["code/algs/__init__.py", "code/algs/lib/__init__.py"]:
        init_path = CLONE_DIR / pkg_init
        if not init_path.exists():
            init_path.write_text("")

    file_hashes = {}
    missing = []
    for rel in TRACKED_FILES:
        p = CLONE_DIR / rel
        if p.exists():
            file_hashes[rel] = sha256_of_file(p)
        else:
            missing.append(rel)

    resolved_commit = subprocess.run(
        ["git", "-C", str(CLONE_DIR), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    provenance = {
        "repo_url": args.repo_url,
        "pinned_commit_arg": args.commit,
        "resolved_commit": resolved_commit,
        "license": None,
        "license_note": (
            "No LICENSE file in the official repository as of the pinned "
            "commit (GitHub license API returns 404). Treated as all-rights"
            "-reserved by default copyright; code is executed from this "
            "external, non-vendored clone and never copied into this "
            "repository. See docs/cacheus_provenance.md."
        ),
        "tracked_file_sha256": file_hashes,
        "tracked_files_missing": missing,
        "added_portability_init_files": [
            "code/algs/__init__.py", "code/algs/lib/__init__.py",
        ],
    }
    (EXTERNAL_DIR / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    if missing:
        print(f"WARNING: expected files missing from clone: {missing}", file=sys.stderr)
    print(f"Fetched sylab/cacheus @ {resolved_commit}")
    print(f"Provenance written to {EXTERNAL_DIR / 'PROVENANCE.json'}")


if __name__ == "__main__":
    main()
