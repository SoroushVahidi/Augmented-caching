# 3L-Cache Implementation Provenance and Licensing

## Implementation Origin
The 3L-Cache implementation in this repository (`src/lafc/policies/three_l_cache.py`, `src/lafc/three_l_cache_features.py`, and `src/lafc/three_l_cache_model.py`) is an **independent reimplementation** from the documented algorithm and published behavior. No source code was copied verbatim or translated line-by-line from the official repository.

## Official Sources Consulted
- **Paper:** Zhou, W., Niu, Z., Xiong, Y., Fang, J., & Wang, Q. (2025). 3L-Cache: Low Overhead and Precise Learning-based Eviction Policy for Caches. FAST 2025.
- **Official Code:** `optiq-lab/3L-Cache` (Pinned commit `134cd159b635cdab75419a4281bed1a330fef31f`)
- **Files Consulted:**
  - `3LCache/TLCache.h`
  - `3LCache/TLCache.cpp`
  - `3LCache/TLCache_Interface.cpp`

## Algorithmic Independence
- **Algorithmic Structure:** Extracted from the paper text and verified against the official source to resolve ambiguities (e.g. initialization conditions, label window freeze timing).
- **Data Structures:** Adapted to our repository simulator contract (e.g., using explicit Python lists and `dict`-based `_LRUList` instead of C++ array index queues).
- **License Note:** The official code is GPL-3.0. Because our codebase uses a different structure and was developed as an independent clean-room implementation of the abstract algorithm tailored for our unit-size simulator interface, this code does not incorporate GPL-3.0 materials. However, appropriate academic attribution is fully maintained.
