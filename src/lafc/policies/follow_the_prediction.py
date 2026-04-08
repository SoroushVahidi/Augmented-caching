"""
Follow-The-Prediction (FTP) policy for caching.

FTP is the lightweight prediction-following abstraction introduced (implicitly)
in:

    Antoniadis, Coester, Eliáš, Polak, Simon.
    "Online Metric Algorithms with Untrusted Predictions."
    ICML 2020.

For the caching specialization, FTP reduces to the Blind Oracle: on a miss,
evict the cached page whose predicted next arrival is farthest in the future.

FTP is the "trust" sub-routine used inside TRUST&DOUBT.  It is also exposed
as a standalone policy so that experiments can compare it directly against
TRUST&DOUBT and the Marker algorithm.

**FTP vs Blind Oracle** in this codebase:
- :class:`~lafc.policies.blind_oracle.BlindOraclePolicy` is documented in
  the context of the ICML 2020 paper as the algorithm that *completely trusts*
  the predictor.
- :class:`FollowThePredictionPolicy` (this class) is explicitly named after
  the FTP abstraction from the paper and is the sub-routine called in the
  TRUST phase of TRUST&DOUBT.
- Both are functionally identical.

See :class:`~lafc.policies.blind_oracle.BlindOraclePolicy` for full
algorithm documentation.
"""

from __future__ import annotations

from lafc.policies.blind_oracle import BlindOraclePolicy


class FollowThePredictionPolicy(BlindOraclePolicy):
    """FTP: Follow-The-Prediction for caching.

    Functionally identical to :class:`~lafc.policies.blind_oracle.BlindOraclePolicy`;
    exposes the FTP name used in the ICML 2020 paper.

    Consistent (optimal under perfect predictions); not robust.

    Used as the TRUST sub-routine inside
    :class:`~lafc.policies.trust_and_doubt.TrustAndDoubtPolicy`.
    """

    name: str = "follow_the_prediction"
