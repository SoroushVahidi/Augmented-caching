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

import sys
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_CODE_DIR = REPO_ROOT / "external" / "cacheus_official" / "src" / "code"

EXPECTED_COMMIT = "1eec63ce166502be33ddd1f35bc041ed73a24f4d"

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
