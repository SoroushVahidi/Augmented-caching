# evict_value_v1 Wulver dataset summary

## Totals
- **Rows:** 141014208
- **Decisions (unique decision_id):** 1945269

## Rows by split
- test: 19902720
- train: 96146976
- val: 24964512

## Rows by trace_family
- brightkite: 10791168
- citibike: 12071616
- cloudphysics: 32327328
- metacdn: 18929664
- metakv: 25479360
- twemcache: 7879584
- wiki2018: 33535488

## Rows by capacity
- 32: 21865728
- 64: 42383040
- 128: 76765440

## Rows by horizon
- 4: 47004736
- 8: 47004736
- 16: 47004736

## Family × split (rows)
- brightkite / test: 3086304
- brightkite / train: 5845152
- brightkite / val: 1859712
- citibike / test: 800928
- citibike / train: 5769984
- citibike / val: 5500704
- cloudphysics / test: 2651520
- cloudphysics / train: 24247296
- cloudphysics / val: 5428512
- metacdn / test: 3004320
- metacdn / train: 12926400
- metacdn / val: 2998944
- metakv / test: 6284544
- metakv / train: 18758400
- metakv / val: 436416
- twemcache / test: 1322592
- twemcache / train: 3257280
- twemcache / val: 3299712
- wiki2018 / test: 2752512
- wiki2018 / train: 25342464
- wiki2018 / val: 5440512

## Manifest meta
- {'split_mode': 'trace_chunk', 'chunk_size': 4096, 'capacities': [32, 64, 128], 'horizons': [4, 8, 16], 'trace_count': 7, 'shard_count': 293}
