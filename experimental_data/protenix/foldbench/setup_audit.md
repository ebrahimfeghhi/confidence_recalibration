# FoldBench setup audit

poses `/home/ebrahim/tts_abag/poses/predictions_template`, seeds 101-150, scored with `eval/ost_dockq.py` into `ost_json/`

- kept: 104 complexes, 160 interfaces, 39998 pool rows
- excluded 7yv1: a protein entity has 2 standard residues; the template builder needs >= 4 for every entity
- excluded 8a67: labelled chain A lands on different molecules across poses
- excluded 8b1a: a protein entity has 2 standard residues; the template builder needs >= 4 for every entity
- excluded 8dtn: interface A-B: antibody and antigen chains are copies of one molecule
- excluded 8evd: no poses on disk
- excluded 8f60: interface A-C, B-C scored on no pose
- excluded 8im0: interface A-B-2 names a multi-letter chain
- excluded 8sfz: labelled chain D lands on different molecules across poses
- excluded 8ts0: interface A-C, B-C scored on no pose
- free/: 0 seed links made, 9607 already present
- OST errors: 250, e.g. ['8ts0 103 0: OST produced no result', '8ts0 103 1: OST produced no result', '8ts0 103 2: OST produced no result']

## Audit

- poses scored: 26000 of 26000 with a chain map; 0 whose map lacks a role chain
- ipTM: 39998 of 39998 rows equal the pose's chain_pair_iptm read by pool_index.pair_iptm_at (tolerance 0.0001)
- against the run's table (`FOLDBENCH_OST_CSV`), rows present in both:

| seeds | same DockQ and ipTM | different |
|:---|---:|---:|
| seeds 101-127 | 17977 | 3623 |
| seeds 128-150 | 18398 | 0 |

## Extra free seeds

- every free seed on disk outside 101-150, scored the same way: 27845 rows over 3413 folds in `free/results.csv`; 0 OST errors
