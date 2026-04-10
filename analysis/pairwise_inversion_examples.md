# Pairwise inversion brute-force examples

This note is exploratory only: it does not prove a theorem.

## Search setup
- Configs (k, alphabet): [(2, 3), (2, 4), (3, 4)]
- Max sequence length: 8
- Total enumerated sequences: 184137

## Aggregate counts
- zero-damage under one inversion per eviction step: 87069
- small-damage (+1 miss): 51240
- large-damage (>= +2 misses): 45828

## Example table: zero inversions / agreement-like behavior
|seq|k|alphabet|len|belady|inv1|extra|decisions|mean_local_inversions|phases|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|AAAA|2|3|4|1|1|0|0|0.00|1|
|AAAA|2|4|4|1|1|0|0|0.00|1|
|AAAB|2|3|4|2|2|0|0|0.00|1|
|AAAB|2|4|4|2|2|0|0|0.00|1|
|AAAC|2|3|4|2|2|0|0|0.00|1|

## Example table: one inversion with small damage
|seq|k|alphabet|len|belady|inv1|extra|decisions|mean_local_inversions|phases|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ABCA|2|3|4|3|4|1|1|1.00|2|
|ABCA|2|4|4|3|4|1|1|1.00|2|
|ABCB|2|3|4|3|4|1|1|1.00|2|
|ABCB|2|4|4|3|4|1|1|1.00|2|
|ABDA|2|4|4|3|4|1|1|1.00|2|

## Example table: one inversion with large damage
|seq|k|alphabet|len|belady|inv1|extra|decisions|mean_local_inversions|phases|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|ABCACACA|2|3|8|3|8|5|1|1.00|2|
|ABCACACA|2|4|8|3|8|5|1|1.00|2|
|ABCBCBCB|2|3|8|3|8|5|1|1.00|2|
|ABCBCBCB|2|4|8|3|8|5|1|1.00|2|
|ABDADADA|2|4|8|3|8|5|1|1.00|2|

## Observed patterns
- A single local inversion can be harmless on some traces (especially with short horizons / low revisit pressure).
- A single local inversion can also amplify into multiple extra misses when it evicts a page reused quickly while preserving a far-future page.
- Damage appears sensitive to request spacing and phase transitions, not just raw inversion count.
- Tempting conjecture that 'one inversion => at most +1 miss' is falsified by examples in the large-damage table.
