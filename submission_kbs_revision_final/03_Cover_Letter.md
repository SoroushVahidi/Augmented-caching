# Revised Cover Letter

**To:** Senior Editor, Knowledge-Based Systems Editorial Office  
**Manuscript number:** KNOSYS-D-26-07461R1  
**Title:** Decision-aligned eviction-value prediction for learning-augmented caching  
**Author:** Soroush Vahidi (sole author)

---

Dear Editor,

Please find enclosed the second revision of manuscript KNOSYS-D-26-07461R1,
entitled *Decision-aligned eviction-value prediction for learning-augmented
caching*. I thank the Associate Editor and both reviewers for comments that
shaped this revision. The enclosed PDF is 21 pages.

The revision reports a leakage-free leave-one-family-out matched evaluation
of `evict_value_v1` against LRB, 3L-Cache, CACHEUS, HALP, LRU, SIEVE, and
FIFO-Reinsertion. The overall result is negative: the learned policy does
not outperform any of these matched baselines. Primary capacities are
32/64/128; capacity 256 remains unevaluated and is disclosed.

A full-pipeline objective comparison and a matched Common-Model V2 control
are now reported. The pipeline ranking does not favor eviction-loss; the
matched control does not support blaming the eviction-loss training
objective itself.

Mechanistic diagnostics, including a tie-aware exact-target analysis,
indicate that the horizon-4 target is highly action-underdetermined. High
set-aware agreement is weak evidence of candidate discrimination because
the exact optimal set is extremely large. A C0/C1/C2 continuation test is
partially supported and regime-dependent. A one-step DAgger-style shift
correction is negative for miss ratio.

A controlled 420-run timing campaign is reported. The present
implementation is not claimed to be deployment-competitive, and no
validated deployment scenario is asserted. Fallback is not part of the
evaluated workflow or any quantitative claim.

A point-by-point response to both review rounds is enclosed. Code and
reviewer-verification materials are available at
https://github.com/SoroushVahidi/Augmented-caching.

This manuscript is original, has not been published previously, and is not
under consideration elsewhere. The corresponding author is Soroush Vahidi
(sv96@njit.edu), Ying Wu College of Computing, New Jersey Institute of
Technology, Newark, NJ, USA.

Thank you for considering this revision.

Sincerely,

Soroush Vahidi  
Ying Wu College of Computing  
New Jersey Institute of Technology  
Newark, NJ, USA  
sv96@njit.edu
