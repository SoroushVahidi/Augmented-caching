# Revised Cover Letter

**To:** Senior Editor, Knowledge-Based Systems Editorial Office  
**Manuscript number:** KNOSYS-D-26-07461R1
**Revised title:** Decision-aligned eviction-value prediction for learning-augmented caching  
**Author:** Soroush Vahidi (sole author)

---

Dear Editor,

Please find enclosed my second-revision manuscript, entitled *Decision-aligned eviction-value prediction for learning-augmented caching*, submitted in response to the Associate Editor's and both reviewers' comments on manuscript KNOSYS-D-26-07461R1. I thank the Associate Editor and reviewers for their detailed and constructive feedback, which has substantially improved this revision.

In summary, the revision includes the following main changes:

1. **Matched end-to-end online replay evaluation** across three cache capacities (32, 64, and 128 slots) and all seven trace families, including direct comparisons with LRB, 3L-Cache, CACHEUS, and HALP. The proposed `evict_value_v1` policy does not outperform any of the seven baselines; capacity 256 was not evaluated and is not claimed in this revision.

2. **SIEVE and FIFO-Reinsertion** added as strong lightweight baselines, implemented, tested, and evaluated at the same three capacities under the matched protocol.

3. **Computational overhead and scalability** discussion added, including offline dataset/training costs and a controlled repeated-measurement wall-clock campaign for LRU, FIFO-Reinsertion, SIEVE, and a HALP reimplementation. The separate single-run runtime of `evict_value_v1` is identified as such.

4. **HALP and prior-work positioning** clarified analytically and empirically, with the independent-reimplementation fidelity caveat disclosed.

5. **Guarded fallback mechanism** reframed as an unvalidated implementation safeguard rather than a demonstrated robustness contribution; no fallback ablation was added.

6. **Scope and hedging revised** so the manuscript describes a decision-aligned supervision-target study rather than a demonstrated online improvement over strong baselines.

7. **Single-author validation practices** discussed candidly, including AI-tool use and verification steps taken.

A detailed point-by-point response to each editor and reviewer comment is enclosed separately (`02_Response_to_Reviewers`).

This manuscript is original, has not been published previously, and is not under consideration for publication elsewhere. As a single-author paper, there is one corresponding author: Soroush Vahidi (sv96@njit.edu), Ying Wu College of Computing, New Jersey Institute of Technology, Newark, NJ, USA.

Thank you for your time and consideration in reviewing this revision.

Sincerely,

Soroush Vahidi  
Ying Wu College of Computing  
New Jersey Institute of Technology  
Newark, NJ, USA  
sv96@njit.edu
