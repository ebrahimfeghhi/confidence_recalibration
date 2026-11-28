# arm_l_per_pose.csv — every pose, its score and its truth

42,150 rows: 21,000 training poses (280 interfaces x 75) and 21,150 held-out poses
(141 interfaces x 150). One row per (interface, seed, sample).

## Columns

| column | meaning |
|---|---|
| `split` | `train` = the hard-280 pool Arm L trained on; `val` = val83, held out |
| `iface` | `<pdb>_<chain1><chain2>`, the scored interface |
| `pdb`, `seed`, `sample` | the pose's identity; (seed, sample) names the generated candidate |
| `dockq` | ground truth for THIS interface |
| `acceptable` | 1 if `dockq >= 0.23`, the project's acceptance threshold |
| `shipped_iptm` | the shipped head's recorded ipTM -- the baseline ranker |
| `armL_iptm_max` | Arm L's ipTM, max-pooled over anchor rows |
| `armL_iptm_mean` | Arm L's ipTM, mean-pooled. The two poolings disagree often; both are reported because every conclusion here has been pooling-sensitive |
| `iface_frac_acceptable` | fraction of that interface's poses that are acceptable -- its intrinsic difficulty |
| `iface_difficulty` | val: `hard` = the 21 interfaces whose best seed has 1-4 of 5 acceptable, `easy` = the 106 at 5/5. train: `unwinnable` / `hard` (<25% acceptable) / `easy` |
| `ab_cluster`, `ag_cluster`, `ab_ag_cluster` | SAbDab2 cluster ids, matched by INSTANCE through `chain_roles` -> `abag_split.csv`. Use these for leakage checks |

## Which checkpoints

Arm L, the plain PAE cross-entropy arm on the hard 280. Training scores are its epoch-12 pass,
held-out its epoch-13 pass -- the only val pass it has, so the two are one epoch apart.

## Things to know before drawing conclusions

RANKING IS WITHIN AN INTERFACE. Scores are not comparable across interfaces; every metric here
sorts candidates inside one `iface` and asks where the acceptable ones land.

`8z7j` LEAKS. Its antibody (cluster 1302) is shared with five training structures, and SAbDab2's
own `ab_split` labels that instance `train`. It is the only one of the 83 val structures that
overlaps training under antibody, antigen OR antibody+antigen clustering. Most of the apparent
held-out top-5 gain is this one structure: excluding it drops the effect from +3 interfaces to +1.

SHIPPED TIES A LOT. 97 of 141 val interfaces have shipped's top two scores within 1e-3, and 48
within 1e-4, so a "rank 1 vs rank 2" difference is often a tie broken by row order. Arm L's gaps
are larger (median 0.00137 vs 0.00030), but when ITS top two are within 1e-3 the DockQ difference
reaches 0.810, against shipped's 0.089.

THE TRAINING POOL WAS SELECTED ON SHIPPED'S FAILURES. The hard 280 are interfaces with few
acceptable poses, where shipped's within-interface Spearman is +0.102 against +0.355 on val83.
Fitting them means fitting the complement of shipped's errors on 180 specific structures.

## Coordinates

NOT INCLUDED, and not includable: `training/fold` is 1.8 TB and `validation/fold` is 365 GB of
pose coordinates. If geometry is needed, say which features and they can be computed here and
exported as columns -- or name a handful of interfaces and those poses can be exported directly.
