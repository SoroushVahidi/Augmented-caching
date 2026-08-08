# HALP and FIFO-Reinsertion source verification (2026-06-19)

Zero-heavy-compute source-verification pass, done in parallel with the
currently running `kbs_full_policy_comparison_cap32_with_sieve` job on
this local/cloud machine. The job was monitored read-only only: tmux
session present, `pgrep` shows the policy-comparison process still
running, `cap32_with_sieve.log` is still empty, and no
`CAP32_WITH_SIEVE_EXIT=` marker exists yet. Nothing in this report stops,
restarts, or modifies that run.

This report directly answers the HALP / FIFO-Reinsertion half of
R3-Issue2, R3-Issue3, and R3-Rec2. Web lookup was used for primary-source
verification:

- HALP USENIX page: <https://www.usenix.org/conference/nsdi23/presentation/song-zhenyu>
- HALP NSDI'23 PDF: <https://www.usenix.org/system/files/nsdi23-song-zhenyu.pdf>
- Google research summary: <https://research.google/blog/preference-learning-with-automated-feedback-for-cache-eviction/>
- SIEVE NSDI'24 PDF: <https://www.usenix.org/system/files/nsdi24spring_prepub_zhang-yazhuo.pdf>
- SIEVE explainer by an author: <https://blog.jasony.me/system/cache/2024/06/12/sieve>
- S3-FIFO SOSP'23 PDF: <https://yazhuozhang.com/assets/publication/sosp23-s3fifo.pdf>

## 1. HALP

### 1.1 Exact paper identity

- Title: `HALP: Heuristic Aided Learned Preference Eviction Policy for YouTube Content Delivery Network`
- Authors: Zhenyu Song, Kevin Chen, Nikhil Sarda, Deniz Altinbuken,
  Eugene Brevdo, Jimmy Coleman, Xiao Ju, Pawel Jurczyk, Richard Schooler,
  Ramki Gummadi
- Venue/year: `20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23)`, 2023
- Pages: `1149--1163`

The exact BibTeX block is printed on the USENIX page itself. Locally, the
HALP citation is already present in both:

- `refs/related_work_table6.bib` as `song2023halp`
- `/tmp/kbs_manuscript_inspect/refs.bib` as `song2023halp`

No new HALP BibTeX entry is needed.

### 1.2 What HALP actually learns and compares

HALP is not just a heuristic and not oracle-action imitation in the
PARROT sense. The HALP paper states that it converts eviction ranking
into pairwise preference queries, uses pairwise comparisons during
decision making, and continuously generates online training data from
those same pairwise comparisons. Concretely:

- it merges a fast heuristic baseline with a learned reward model
- it selects a small heuristic candidate set, then re-ranks via pairwise
  comparisons
- the training label is pairwise: which of two candidates is accessed
  further in the future
- training is online/continuous using automated feedback from future
  accesses

That makes HALP the closest cited prior work to this repo's
candidate-level framing, but the supervision target is still different:

- HALP learns a relative pairwise preference / reward signal
- `evict_value_v1` targets an explicit finite-horizon downstream
  miss-harm quantity under counterfactual eviction

That distinction is real enough for Related Work and reviewer response
text, but it is not strong enough to justify broad novelty claims like
"future-aware candidate scoring is new."

### 1.3 Feasibility before July 8

Conservative answer: no, not as a faithful implementation.

Reason:

- HALP requires a distinct online preference-learning loop, not just a
  drop-in eviction rule.
- Its training labels arise from deferred future feedback on pairwise
  comparisons, which is a materially different data-generation pipeline
  from this repo's current offline replay target construction.
- Its production evaluation context is YouTube CDN DRAM caching, which is
  not reproducible from the trace/data assets already in this repo.

Implementing a toy surrogate would risk producing a reviewer-facing
"HALP" row that is not actually HALP.

### 1.4 Recommended reviewer-response position if HALP is not run

Recommended position: cite and differentiate carefully; state directly
that empirical HALP reproduction is out of scope for this revision.

Draft:

> We thank the reviewer for highlighting HALP as a closely related
> baseline. We agree that HALP is operationally close in that it also
> makes candidate-level decisions using future-derived training signals.
> The key distinction is that HALP learns a pairwise preference/reward
> signal from automated future feedback, whereas our method assigns each
> candidate an explicit finite-horizon downstream miss-harm target under
> counterfactual eviction. We have revised the related-work discussion to
> make this distinction explicit. A faithful empirical HALP
> reimplementation would require its own online preference-learning
> pipeline and production-motivated feedback loop, which we judged out of
> scope for the present revision timeline; we therefore treat HALP as a
> closely related cited system rather than report an unfaithful surrogate
> comparison.

## 2. FIFO-Reinsertion

### 2.1 Intended algorithm

The strongest source-supported reading is that Reviewer #3's
"FIFO-Reinsertion" means the CLOCK / Second-Chance style FIFO queue with
reinsertion of visited survivors at the head.

Why this is the best reading:

- the SIEVE NSDI'24 paper explicitly says `Second Chance, CLOCK, and
  FIFO-Reinsertion are different implementations of the same eviction
  algorithm`
- the same paper contrasts SIEVE against FIFO-Reinsertion by one exact
  semantic difference: SIEVE keeps a retained object in place, while
  FIFO-Reinsertion moves it to the head
- the author blog post says the same thing in plainer language
- the S3-FIFO paper treats FIFO-Reinsertion as a known single-queue
  FIFO-family algorithm and says S3-FIFO's main queue is "similar to
  FIFO-Reinsertion"

So for reviewer-response purposes, the intended definition is no longer
best described as ambiguous in the earlier "textbook FIFO+requeue vs.
S3-FIFO-style mechanism" sense.

### 2.2 Exact behavior to cite

FIFO-Reinsertion / CLOCK / Second-Chance in this context:

1. Hit: mark the resident as visited.
2. Full-cache miss: inspect the oldest/tail object.
3. If it was visited, clear the bit and move it to the head.
4. Continue until an unvisited object is found, then evict it.
5. Insert the new object at the head.

The SIEVE paper provides a code-style comparison against SIEVE on page 1
and a prose explanation in Section 3.

### 2.3 Implement / defer / cite-only recommendation

Recommendation in this pass: **source verification done; no new code
action here**.

Current repo state already contains:

- `src/lafc/policies/fifo_reinsertion.py`
- `tests/test_fifo_reinsertion.py`
- canonical-pipeline wiring in `scripts/run_policy_comparison_wulver_v1.py`
  and `src/lafc/runner/run_policy.py`
- a tiny smoke output and implementation report from an earlier same-day
  pass

This pass did not modify any of that code. Given the verified definition,
the algorithm is well-defined enough that the existing implementation can
be discussed as the repo's chosen FIFO-Reinsertion baseline. The remaining
open question is not definition anymore; it is whether to spend canonical
rerun budget to include it in final results.

Conservative recommendation:

- `HALP`: cite-only / differentiate, do not implement before July 8
- `FIFO-Reinsertion`: definition verified, implementation already exists,
  but defer any canonical rerun decision to the same scope discussion as
  SIEVE and `cap32_with_sieve`

### 2.4 BibTeX needed

HALP already exists locally. FIFO-Reinsertion's modern supporting sources
do not appear in the repo's tracked auxiliary bibliography
`refs/related_work_table6.bib`, and a grep of the extracted manuscript
package also showed no `zhang2024sieve` or `yang2023s3fifo` entries in
`/tmp/kbs_manuscript_inspect/refs.bib`.

Recommended additions if the manuscript or response letter discusses the
definition explicitly:

```bibtex
@inproceedings{zhang2024sieve,
  author    = {Yazhuo Zhang and Juncheng Yang and Yao Yue and Ymir Vigfusson and K. V. Rashmi},
  title     = {SIEVE is Simpler than LRU: an Efficient Turn-Key Eviction Algorithm for Web Caches},
  booktitle = {21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24)},
  year      = {2024},
  pages     = {1229--1246},
  publisher = {USENIX Association},
  url       = {https://www.usenix.org/conference/nsdi24/presentation/zhang-yazhuo}
}

@inproceedings{yang2023s3fifo,
  author    = {Juncheng Yang and Yazhuo Zhang and Ziyue Qiu and Yao Yue and K. V. Rashmi},
  title     = {FIFO Queues Are All You Need for Cache Eviction},
  booktitle = {ACM SIGOPS 29th Symposium on Operating Systems Principles (SOSP '23)},
  year      = {2023},
  doi       = {10.1145/3600006.3613147}
}
```

Optional historical citation if desired: Corbato's early Second-Chance /
CLOCK line. Not necessary for the present reviewer response.

## 3. Bottom-line recommendation for this revision cycle

- HALP source verification: done
- FIFO-Reinsertion source verification: done
- HALP empirical implementation before July 8: not feasible
- FIFO-Reinsertion algorithm definition: sufficiently pinned down
- FIFO-Reinsertion canonical numbers: not available yet
- `cap32_with_sieve`: currently running, untouched by this pass

The safest reviewer-response posture is:

1. sharpen the HALP differentiation text,
2. state honestly that HALP reproduction is out of scope,
3. treat FIFO-Reinsertion as the CLOCK/Second-Chance family baseline,
4. avoid claiming any FIFO-Reinsertion or SIEVE canonical result until the
   relevant run artifacts actually exist.
