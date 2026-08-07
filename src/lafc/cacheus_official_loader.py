"""Dynamic loader for the official CACHEUS source (sylab/cacheus), fetched
externally (never vendored) by `scripts/setup/fetch_cacheus_official.py`.

See `docs/cacheus_provenance.md` for why this is external rather than
vendored (the official repository has no LICENSE file).

Imports the official code exactly the way its own harness (`run.py`) does:
`code/` on `sys.path`, `algs` as a plain top-level package (the fetch
script adds the two missing, empty `__init__.py` files the official repo
itself omits -- a portability-only change, no algorithmic content).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_CLONE_ROOT = REPO_ROOT / "external" / "cacheus_official" / "src"
EXTERNAL_CODE_DIR = EXTERNAL_CLONE_ROOT / "code"
PROVENANCE_PATH = REPO_ROOT / "external" / "cacheus_official" / "PROVENANCE.json"

EXPECTED_COMMIT = "1eec63ce166502be33ddd1f35bc041ed73a24f4d"

# Files this repository's adapter actually imports/executes at runtime.
# Shared between the fetch script (which hashes them right after cloning)
# and verify_official_source_integrity() (which re-hashes them before every
# reviewer-facing run) so the two can never drift out of sync with each
# other.
TRACKED_FILES = [
    "code/algs/cacheus.py",
    "code/algs/lru.py",
    "code/algs/lib/dequedict.py",
    "code/algs/lib/heapdict.py",
    "code/algs/lib/cacheop.py",
    "code/algs/lib/optional_args.py",
    "code/algs/lib/pollutionator.py",
    "code/algs/lib/visualizinator.py",
]
# Files this repository adds to the external clone itself (never committed
# to this repository) purely for import portability -- see
# docs/cacheus_provenance.md section 4. Expected to show as untracked (or
# gitignored by the upstream repo's own .gitignore) in `git status`.
PORTABILITY_ADDITIONS = ["code/algs/__init__.py", "code/algs/lib/__init__.py"]


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CacheusIntegrityError(RuntimeError):
    """Raised when the external CACHEUS clone fails an integrity check:
    wrong commit, dirty tree (beyond the known portability additions), a
    tracked algorithm file's hash doesn't match what was recorded at fetch
    time, or PROVENANCE.json is missing/unreadable. Reviewer-facing runs
    must refuse to start rather than silently execute a different, unknown
    checkout.
    """


def verify_official_source_integrity() -> Dict[str, object]:
    """Re-verify the external CACHEUS clone against the pinned commit and
    the hashes recorded at fetch time. Raises CacheusIntegrityError with a
    precise description of the first problem found; returns a report dict
    on success.
    """
    if not EXTERNAL_CLONE_ROOT.exists():
        raise CacheusIntegrityError(
            f"External CACHEUS clone not found at {EXTERNAL_CLONE_ROOT}. Run "
            "python scripts/setup/fetch_cacheus_official.py first."
        )

    resolved = subprocess.run(
        ["git", "-C", str(EXTERNAL_CLONE_ROOT), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    if resolved.returncode != 0:
        raise CacheusIntegrityError(
            f"Could not read git HEAD of {EXTERNAL_CLONE_ROOT}: {resolved.stderr.strip()}"
        )
    resolved_commit = resolved.stdout.strip()
    if resolved_commit != EXPECTED_COMMIT:
        raise CacheusIntegrityError(
            f"External CACHEUS clone is at commit {resolved_commit}, expected "
            f"the pinned commit {EXPECTED_COMMIT}. Re-run "
            "python scripts/setup/fetch_cacheus_official.py --force to reset it."
        )

    status = subprocess.run(
        ["git", "-C", str(EXTERNAL_CLONE_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    ).stdout
    allowed_untracked = {f"?? {p}" for p in PORTABILITY_ADDITIONS}
    unexpected_status_lines = [
        line for line in status.splitlines()
        if line and line not in allowed_untracked and "__pycache__" not in line
    ]
    if unexpected_status_lines:
        raise CacheusIntegrityError(
            "External CACHEUS clone has unexpected local changes beyond the "
            f"known portability additions ({PORTABILITY_ADDITIONS}): "
            f"{unexpected_status_lines}. Tracked upstream algorithm files "
            "must not be modified; refusing to run."
        )

    if not PROVENANCE_PATH.exists():
        raise CacheusIntegrityError(
            f"{PROVENANCE_PATH} not found -- the fetch script's own hash "
            "record is missing. Re-run python scripts/setup/fetch_cacheus_official.py."
        )
    recorded = json.loads(PROVENANCE_PATH.read_text())
    recorded_hashes = recorded.get("tracked_file_sha256", {})

    current_hashes: Dict[str, str] = {}
    mismatches: List[str] = []
    for rel in TRACKED_FILES:
        p = EXTERNAL_CLONE_ROOT / rel
        if not p.exists():
            mismatches.append(f"{rel}: file missing")
            continue
        current_hashes[rel] = sha256_of_file(p)
        recorded_hash = recorded_hashes.get(rel)
        if recorded_hash is None:
            mismatches.append(f"{rel}: no recorded hash in PROVENANCE.json")
        elif recorded_hash != current_hashes[rel]:
            mismatches.append(
                f"{rel}: hash mismatch (recorded {recorded_hash[:12]}..., "
                f"current {current_hashes[rel][:12]}...)"
            )

    if mismatches:
        raise CacheusIntegrityError(
            "External CACHEUS clone failed hash verification against "
            f"PROVENANCE.json: {mismatches}. Refusing to run -- this would "
            "silently execute a different checkout than the one recorded as "
            "the pinned, verified source."
        )

    return {
        "repo_url": recorded.get("repo_url"),
        "resolved_commit": resolved_commit,
        "clean": True,
        "tracked_file_sha256": current_hashes,
        "license": recorded.get("license"),
    }

_NOT_FETCHED_MSG = (
    "Official CACHEUS source not found at {path}. This policy requires the "
    "authors' original implementation (github.com/sylab/cacheus) as an "
    "external, non-vendored dependency -- it is not reimplemented in this "
    "repository. Run:\n\n"
    "    python scripts/setup/fetch_cacheus_official.py\n\n"
    "before using the `cacheus` policy or its tests. See "
    "docs/cacheus_provenance.md for why the official code is fetched "
    "externally rather than copied into this repository."
)


class CacheusOfficialSourceMissing(RuntimeError):
    """Raised when the external CACHEUS clone has not been fetched.

    Deliberately not caught anywhere to silently fall back to a different
    policy: a missing official source is a hard error for this baseline,
    per this task's "no silent fallback to unrelated policies" requirement.
    """


def load_official_classes() -> Tuple[type, type]:
    """Import and return (Cacheus, LRU) from the external clone, unmodified."""
    algs_init = EXTERNAL_CODE_DIR / "algs" / "__init__.py"
    cacheus_path = EXTERNAL_CODE_DIR / "algs" / "cacheus.py"
    if not algs_init.exists() or not cacheus_path.exists():
        raise CacheusOfficialSourceMissing(_NOT_FETCHED_MSG.format(path=cacheus_path))

    code_dir_str = str(EXTERNAL_CODE_DIR)
    inserted = code_dir_str not in sys.path
    if inserted:
        sys.path.insert(0, code_dir_str)
    try:
        import algs.cacheus as cacheus_module  # type: ignore[import-not-found]
        import algs.lru as lru_module  # type: ignore[import-not-found]
    finally:
        if inserted:
            sys.path.remove(code_dir_str)

    return cacheus_module.Cacheus, lru_module.LRU
