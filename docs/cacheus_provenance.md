# CACHEUS Provenance and Licensing Document

## 1. Implementation origin

This baseline executes the **authors' own, unmodified** `Cacheus` class
from `github.com/sylab/cacheus`. Nothing in this repository reimplements
SR-LRU, CR-LFU, the expert-weighting scheme, or any other CACHEUS
algorithmic logic. `src/lafc/policies/cacheus.py` is a thin adapter that
translates between this repository's `BasePolicy` interface and the
official class's own interface, plus a small dynamic loader
(`src/lafc/cacheus_official_loader.py`).

## 2. Authoritative sources consulted

- **Paper** (official paper, primary source): Rodriguez, Yusuf, Lyons,
  Paz, Rangaswami, Liu, Zhao, Narasimhan, "Learning Cache Replacement with
  CACHEUS," FAST '21.
  https://www.usenix.org/conference/fast21/presentation/rodriguez
- **FIU Systems Research Laboratory project page**:
  https://sylab-srv.cs.fiu.edu/doku.php?id=projects:cacheus
- **Official source** (author-released — the repository's own description
  reads "The design and algorithms used in Cacheus are described in this
  USENIX FAST'21 paper and talk video," directly linking it to the paper;
  owner `sylab` matches the lab's identity):
  https://github.com/sylab/cacheus, pinned commit
  `1eec63ce166502be33ddd1f35bc041ed73a24f4d` (HEAD of `main` as of
  2026-08-06; a "Merge pull request #14 ... Update alecar6.py" merge
  commit — note this touches `alecar6.py`, a *different* algorithm file
  from the one used here, `cacheus.py`, which was not modified by that PR).

## 3. License status: none

`GET /repos/sylab/cacheus/license` → HTTP 404. The repository's own API
metadata reports `"license": null`. No `LICENSE`, `LICENSE.md`,
`COPYING`, or equivalent file exists in the repository at the pinned
commit (confirmed via the GitHub Contents API: the repository root
contains only `.gitignore`, `README.md`, and `code/`).

**Consequence**: with no explicit license, default copyright applies —
no redistribution or modification permission is granted by the repository
itself. This repository therefore:

- **does not copy** any official source file into its own tree;
- **does not vendor** the official source under this repository's own
  license;
- fetches the official source **externally**, via
  `scripts/setup/fetch_cacheus_official.py`, into `external/` (added to
  `.gitignore` — see the `/external/` entry), which is **never committed**;
- executes the official code **in place**, from that external clone, via
  `src/lafc/cacheus_official_loader.py`.

This mirrors this task's own preferred licensing-safe structure
("keep the official code as an external pinned dependency; download/clone
it through a setup script ... invoke its binary; record commit and source
hash") and this repository's own precedent structure for a similarly
unclear-license situation (see `docs/halp_provenance.md` for HALP, a
different situation — no code at all rather than code with no license —
but the same "never assume permission silently" posture).

No definitive legal conclusion is asserted beyond the above; this is not
legal advice. If the manuscript or its response-to-reviewers needs a
stronger position (e.g. explicit permission from the authors), that is a
follow-up action outside the scope of this implementation task.

## 4. Portability-only patch applied

The official repository's `code/algs/` directory has no `__init__.py`
files; it is designed to be imported as `algs.*` with `code/` as the
working directory (matching how the authors' own `run.py`/`run_alg.py`
invoke it), not as a proper installable/importable Python package from an
arbitrary working directory. `scripts/setup/fetch_cacheus_official.py`
adds two **empty** `__init__.py` files (`code/algs/__init__.py`,
`code/algs/lib/__init__.py`) to the **external clone only** (never to
anything git-tracked in this repository) so `algs.cacheus`/`algs.lru` can
be imported after inserting `code/` onto `sys.path`. This is classified as
a **portability-only patch**: zero algorithmic content, two empty files,
applied outside version control, needed only because this repository
imports the official code as a library rather than invoking it as a
`__main__` script the way the official `run.py` does.

No other patch of any kind is applied to the official source.

## 5. Files consulted vs. files used at runtime

| File | Consulted | Used at runtime by this baseline |
|---|---|---|
| `code/algs/cacheus.py` | Yes | **Yes — the baseline itself** |
| `code/algs/lru.py` | Yes | Yes — cross-simulator parity test only (`tests/test_cacheus.py`) |
| `code/algs/lib/dequedict.py`, `heapdict.py`, `cacheop.py`, `optional_args.py`, `pollutionator.py`, `visualizinator.py` | Yes | Yes — transitive dependencies of `cacheus.py`/`lru.py` |
| `code/run.py`, `code/run_alg.py`, `code/get_algorithm.py` | Yes (to confirm the official `.request(oblock, ts)` calling convention and the `window_size` default) | No — this repository drives `Cacheus.request()` directly with its own already-parsed request stream instead of the official blkparse-trace-reading harness |
| `code/algs/lib/traces.py` | No | No — not needed; see `docs/cacheus_method_spec.md`, "Trace/object identity" |
| `code/algs/{alecar6,arc,arcalecar,dlirs,lecar,lfu,lirs,lirsalecar,min,mru}.py` | Consulted only via `README.md`/directory listing, not read in full | No — out of scope; only the paper's primary SR-LRU+CR-LFU configuration (`cacheus.py`) is used, predeclared per `docs/cacheus_method_spec.md` |
| `README.md` | Yes | N/A (documentation only) |

## 6. Exact provenance record

`scripts/setup/fetch_cacheus_official.py` writes
`external/cacheus_official/PROVENANCE.json` on every fetch, recording the
resolved commit and a SHA-256 of every file in the table above marked
"used at runtime." That file is itself outside version control (inside
`/external/`), but `scripts/experiments/run_cacheus_comparison.py` copies
the commit identifier (`EXPECTED_COMMIT` in
`src/lafc/cacheus_official_loader.py`) into every run's own
`analysis/external_learned_baselines/cacheus/provenance.json`, which *is*
committed-repository-adjacent generated output (per this repository's
existing convention for LRB/3L-Cache/HALP).
