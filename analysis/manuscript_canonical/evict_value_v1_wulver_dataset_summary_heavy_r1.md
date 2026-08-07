# evict_value_v1 Wulver dataset summary

## Totals
- **Rows:** 323043072
- **Decisions (unique decision_id):** 2767149

## Rows by split
- test: 34288608
- train: 239023008
- val: 49731456

## Rows by trace_family
- brightkite: 21928704
- citibike: 23119296
- cloudphysics: 68193696
- metacdn: 39463680
- metakv: 54268608
- twemcache: 44330208
- wiki2018: 71738880

## Rows by capacity
- 32: 23274816
- 64: 45139776
- 128: 87004032
- 256: 167624448

## Rows by horizon
- 4: 107681024
- 8: 107681024
- 16: 107681024

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
- twemcache / test: 7241376
- twemcache / train: 33710784
- twemcache / val: 3378048
- wiki2018 / test: 5898240
- wiki2018 / train: 59942400
- wiki2018 / val: 5898240

## Manifest meta
- {'split_mode': 'trace_chunk', 'chunk_size': 4096, 'capacities': [32, 64, 128, 256], 'horizons': [4, 8, 16], 'trace_count': 7, 'shard_count': 662}
