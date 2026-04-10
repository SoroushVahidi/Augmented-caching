# evict_value_v1 Wulver dataset summary

## Totals
- **Rows:** 289304256
- **Decisions (unique decision_id):** 2524527

## Rows by split
- test: 28852992
- train: 213173952
- val: 47277312

## Rows by trace_family
- brightkite: 21928704
- citibike: 23119296
- cloudphysics: 68193696
- metacdn: 39463680
- metakv: 54268608
- twemcache: 10591392
- wiki2018: 71738880

## Rows by capacity
- 32: 21865728
- 64: 42383040
- 128: 76765440
- 256: 148290048

## Rows by horizon
- 4: 96434752
- 8: 96434752
- 16: 96434752

## Family × split (rows)
- brightkite / test: 3778848
- brightkite / train: 14413824
- brightkite / val: 3736032
- citibike / test: 2303808
- citibike / train: 18882816
- citibike / val: 1932672
- cloudphysics / test: 11199744
- cloudphysics / train: 51621024
- cloudphysics / val: 5372928
- metacdn / train: 23649120
- metacdn / val: 15814560
- metakv / test: 3866592
- metakv / train: 36803040
- metakv / val: 13598976
- twemcache / test: 1805760
- twemcache / train: 7861728
- twemcache / val: 923904
- wiki2018 / test: 5898240
- wiki2018 / train: 59942400
- wiki2018 / val: 5898240

## Manifest meta
- {'split_mode': 'trace_chunk', 'chunk_size': 4096, 'capacities': [32, 64, 128, 256], 'horizons': [4, 8, 16], 'trace_count': 7, 'shard_count': 594}
