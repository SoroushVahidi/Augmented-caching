# evict_value_v1 Wulver dataset summary

## Totals
- **Rows:** 849024
- **Decisions (unique decision_id):** 13266

## Rows by split
- train: 285312
- val: 563712

## Rows by trace_family
- brightkite: 155904
- citibike: 129408
- wiki2018: 563712

## Rows by capacity
- 64: 849024

## Rows by horizon
- 4: 283008
- 8: 283008
- 16: 283008

## Family × split (rows)
- brightkite / train: 155904
- citibike / train: 129408
- wiki2018 / val: 563712

## Manifest meta
- {'split_mode': 'trace_chunk', 'chunk_size': 4096, 'capacities': [64], 'horizons': [4, 8, 16], 'trace_count': 3, 'shard_count': 4}
