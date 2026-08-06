"""Generic infrastructure shared by external-learned-baseline comparison
experiments (LRB, 3L-Cache, and future additions).

Deliberately policy-agnostic: nothing here knows about LRB's or 3L-Cache's
learning logic. It only provides the generic plumbing every external
baseline comparison needs -- trace-manifest reading, file hashing,
git-commit/version provenance, and an incremental, resumable CSV row
writer -- so a fresh comparison script doesn't need to reinvent it, and so
"nothing gets written until the whole campaign finishes" (the gap in the
first LRB experiment runner) doesn't recur.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def read_trace_manifest(manifest_csv: Path) -> List[Tuple[str, str, str]]:
    """Read a (path, trace_name, trace_family) manifest CSV, matching the
    schema of ``analysis/wulver_trace_manifest_full.csv``."""
    rows = list(csv.DictReader(manifest_csv.open(encoding="utf-8")))
    return [
        (
            r["path"].strip(),
            r.get("trace_name", "").strip() or r["path"],
            r.get("trace_family", "").strip() or "unknown",
        )
        for r in rows
    ]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def package_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return "not-installed"


def base_provenance() -> Dict[str, object]:
    return {
        "repository_commit": git_commit(),
        "repository_branch": git_branch(),
        "python_version": sys.version,
        "platform": platform.platform(),
    }


class IncrementalCsvWriter:
    """Append-as-you-go CSV writer with resume support.

    Every completed row is flushed to disk immediately (not buffered until
    the whole campaign finishes), so an interrupted run leaves a valid,
    readable partial CSV rather than nothing at all. On construction, any
    existing file at ``path`` is read back into :attr:`existing_keys` so a
    resumed run can skip combinations already completed -- callers decide
    what counts as "the same combination" via ``key_fields``.
    """

    def __init__(self, path: Path, fieldnames: List[str], key_fields: List[str]):
        self.path = path
        self.fieldnames = fieldnames
        self.key_fields = key_fields
        self.existing_keys: set = set()
        self._file = None
        self._writer = None

        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    self.existing_keys.add(self._key(row))

        write_header = not path.exists() or path.stat().st_size == 0
        self._file = path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if write_header:
            self._writer.writeheader()
            self._file.flush()

    def _key(self, row: Dict[str, object]) -> Tuple:
        return tuple(str(row.get(f, "")) for f in self.key_fields)

    def already_done(self, row_key_values: Dict[str, object]) -> bool:
        return self._key(row_key_values) in self.existing_keys

    def write_row(self, row: Dict[str, object]) -> None:
        self._writer.writerow(row)
        self._file.flush()
        self.existing_keys.add(self._key(row))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


def write_provenance_json(path: Path, provenance: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance, indent=2, default=str) + "\n")


def baseline_config_matches(
    stored: Optional[Dict[str, object]], current: Dict[str, object], *, fields: List[str]
) -> bool:
    """Strict-equality provenance check used to decide whether a previously
    computed baseline result (e.g. evict_value_v1's canonical numbers) may
    be reused rather than recomputed: every field in ``fields`` must match
    exactly (trace hash, capacity, request budget, code version, ...).
    Missing/partial provenance never counts as a match -- scientific
    fairness takes priority over saved runtime.
    """
    if stored is None:
        return False
    return all(stored.get(f) == current.get(f) for f in fields)
