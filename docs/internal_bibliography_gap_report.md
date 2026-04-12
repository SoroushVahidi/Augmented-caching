# Internal bibliography gap report (eviction-value direction)

> Internal repo-maintenance note only.  
> Not manuscript prose and not a canonical `heavy_r1` artifact.

## 1) Scope and bibliography file used

- Repository scan found no pre-existing `.bib` files.
- For the current manuscript path, we added a single root bibliography file: `refs.bib` (consistent with prior repo note recommending a root-level or `paper/refs.bib` location).

## 2) Status of the six priority papers

| Paper | Already present in repo bibliography before this task? | Added now? | Notes |
|---|---:|---:|---|
| PARROT | No | Yes | Added as ICML/PMLR entry (`liu2020parrot`). |
| HALP | No | Yes | Added as NSDI 2023 entry (`song2023halp`). |
| LRB | No | Yes | Added as NSDI 2020 entry (`song2020lrb`). |
| Raven | No | Yes | Added as CoNEXT 2022 entry (`hu2022raven`, DOI included). |
| Mockingjay | No | Yes | Added as HPCA 2022 entry (`shah2022mockingjay`, DOI included). |
| MUSTACHE | No | Yes | Added conservatively as arXiv preprint entry (`tolomei2022mustache`). |

## 3) Metadata still needing verification (if any)

- HALP and LRB were added with standard USENIX NSDI metadata (title/authors/pages/year/url).  
  If a future manuscript requires publisher-specific BibTeX style normalization, re-export from authoritative USENIX citation blocks at manuscript-finalization time.
- MUSTACHE is currently represented as an arXiv preprint; update if/when a peer-reviewed venue version is explicitly selected for manuscript references.

## 4) Why these references matter for the current novelty framing

- They cover the closest learned-eviction neighborhood for the present claim boundary:
  - **PARROT**: main oracle-imitation contrast.
  - **HALP + Mockingjay**: strongest pressure against broad candidate-scoring novelty claims.
  - **LRB + Raven**: Belady-guided / learned-eviction predecessors that narrow broad “first learned eviction” language.
  - **MUSTACHE**: closest finite-horizon-style comparator (future request/page prediction framing), important for precise differentiation from finite-horizon downstream miss-harm supervision claims.
