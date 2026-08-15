# Learned/Exact H4 Target Agreement v1

This diagnostic evaluates the already frozen held-out `objective_eviction_loss`
models. It does not train models and does not consume the exact-oracle
replication campaign.

The learned policy is replayed on its own deployment trajectory. At each scored
decision, the existing H4 eviction-loss target is recomputed for the current
candidate set. Set-aware agreement means the learned-selected victim belongs
to the exact target-optimal set. Lexicographic agreement means it equals the
existing deterministic lexicographic exact-oracle choice. These are distinct
metrics because H4 target ties are common.

Target regret is the learned-selected target value minus the minimum target
value. Online misses are reported separately and must not be conflated with
target agreement or regret.

Partial units are not citable. Aggregate output is generated only after all 21
family-capacity units pass protocol, trace-hash, model-provenance, finiteness,
and Brightkite/capacity-64 regression checks.
