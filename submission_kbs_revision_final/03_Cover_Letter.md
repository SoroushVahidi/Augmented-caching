# Revised Cover Letter

**To:** Senior Editor, Knowledge-Based Systems Editorial Office  
**Manuscript number:** KNOSYS-D-26-07461  
**Revised title:** Decision-aligned eviction-value prediction for learning-augmented caching  
**Author:** Soroush Vahidi (sole author)

---

Dear Editor,

Please find enclosed my revised manuscript, entitled *Decision-aligned eviction-value prediction for learning-augmented caching*, submitted in response to the Associate Editor's and both reviewers' comments on manuscript KNOSYS-D-26-07461. I thank the Associate Editor and reviewers for their detailed and constructive feedback, which has substantially improved this revision.

In summary, the revision includes the following main changes:

1. **End-to-end online replay evaluation** across three cache capacities (32, 64, and 128 slots) and all seven trace families, reporting results honestly. The proposed `evict_value_v1` policy does not outperform LRU, SIEVE, or FIFO-Reinsertion at any of these capacities; capacity 256 was not evaluated and is not claimed in this revision.

2. **SIEVE and FIFO-Reinsertion** added as strong lightweight baselines, implemented, tested, and evaluated at the same three capacities.

3. **Computational overhead and scalability** discussion added, including offline dataset/training costs and a controlled local wall-clock latency benchmark (tmux on the author's development machine, not Wulver/Slurm).

4. **HALP and prior-work positioning** clarified analytically; a faithful empirical HALP reimplementation is outside the scope of this revision.

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
