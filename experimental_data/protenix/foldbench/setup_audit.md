# FoldBench setup audit

poses `/home/ebrahim/tts_abag/poses/predictions_template`, seeds 101-150, scored with `eval/ost_dockq.py` into `ost_json/`

- kept: 109 complexes, 166 interfaces, 41498 pool rows
- excluded 8evd: no poses on disk
- excluded 8f60: interface A-C, B-C scored on no pose
- excluded 8im0: interface A-B-2 names a multi-letter chain
- excluded 8ts0: interface A-C, B-C scored on no pose
- free/: 0 seed links made, 9956 already present
- OST errors: 250, e.g. ['8ts0 106 1: OST produced no result', '8ts0 106 2: OST produced no result', '8ts0 106 3: OST produced no result']

## Audit

- poses scored: 27250 of 27250 with a chain map; 0 whose map lacks a role chain
- ipTM: 41498 of 41498 rows equal the pose's chain_pair_iptm read by pool_index.pair_iptm_at (tolerance 0.0001)
- against the run's table (`FOLDBENCH_OST_CSV`), rows present in both:

| seeds | same DockQ and ipTM | different |
|:---|---:|---:|
| seeds 101-127 | 18517 | 3893 |
| seeds 128-150 | 19088 | 0 |
