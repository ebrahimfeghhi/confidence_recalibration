# RUN SPEC -- pose sites only, loss = within-seed centred TM error

Requested by the owner 2026-11-24 ~11:55 PM EST: "Train the confidence head on one GPU overnight.
Only the 6 pose injection sites everything else frozen. Only loss is to predict mean centered itm
across the 5 poses, mse. Do 5 poses at a time, exclude structures whose crop is too big for that.
No PAE loss." Then: "B" (centre within a seed, not within a structure) and "no PAE loss ok".

## Why this arm exists

Measured today on 947 training interfaces and 139 leak-free val83 interfaces:

* ipTM puts **2.6%** (train) / 1.7% (val) of its within-interface variance among the 5 diffusion
  samples of ONE seed. DockQ puts **49%** there, iTM `g_max` **52%** -- so this is not an artefact
  of DockQ's thresholds.
* Inside a seed ipTM picks the best of 5 **25.2%** of the time against a 20% coin flip, and its
  pick averages only +0.0098 DockQ (train) / +0.0030 (val) above picking at random.
* 22% of training seeds span more than 0.2 DockQ across their 5 samples, and in 35% of those ipTM
  spans less than 0.02 -- it is effectively blind where a whole CAPRI class is at stake.
* Consequence at fixed compute: going 1 -> 5 samples per seed raises the ORACLE by +5.7 (train)
  and +7.4 (val) points while Success@1 moves -0.2 to +0.6. The quality is generated and discarded.

So the objective is pointed at exactly that comparison.

## The loss

`--tm-centered` with the NEW `--tm-centered-per-seed`. The existing flag keys its per-pair
accumulator by `pdb` alone, so the mean is over every pose of a structure across all 15 seeds --
which is why it demanded `--rank-cross-seed`. That leaves the blind spot in the objective: the
between-seed differences it rewards are the ones ipTM already resolves. The new flag keys by
`(pdb, seed)`, making the loss the within-seed ordering and nothing else.

A unit IS (structure, seed) carrying its 5 samples, so pass A sweeps all 5 to build the per-pair
mean and pass B subtracts it pose by pose -- the centring group is the 5 poses whether or not they
share a forward.

NO PAE GRADIENT. `--tm-centered` replaces the cross-entropy; the CE is still computed as a readout
so trajectory.csv stays comparable with every prior arm, but it takes no gradient.
`--rank-loss` is OFF, so the centred TM MSE is the only term with a gradient.

## Trainable surface

`--train-only 'linear_no_bias_d,pose_perblock_,pose_readout_,frame_dir'` -- the six injection
sites and nothing else:

| site | tensors |
|---|---|
| block 0, pre-stack (pretrained) | `linear_no_bias_d`, `linear_no_bias_d_wo_onehot` |
| pairformer blocks 1, 2, 3 | `pose_perblock_*` (onehot, dist, orient) |
| readout `pre` | `pose_readout_*` |
| readout `post` | `pose_readout_post_*` |

Architecture flags are the reference arm's: `--pose-block-terms onehot,dist,orient`,
`--pose-block-weights untied`, `--pose-block-init pretrained`, `--pose-readout-sites pre,post`,
`--pose-readout-terms onehot,dist,orient`, `--pose-readout-init pretrained`,
`--frame-direction`.

## Deviations from the request, and why

1. **UNCROPPED, one pose per forward.** `--tm-centered` refuses a crop pool: its d0 would come
   from the cropped chain-pair token count while selection reads ipTM over the whole structure,
   and pass A's label is full size against a cropped prediction. Uncropped means 5 poses cannot
   share a forward (5 x 768 tokens is ~87 GB of a 98 GB card), so `--pose-batch 1`. The loss is
   unchanged -- see above. "Exclude structures whose crop is too big" therefore does not apply;
   nothing is excluded and nothing is cropped.
2. **The reference's second eval pass (`--extra-eval` over the whole training pool) is dropped.**
   It is GPU cost this arm did not ask for.

## Provenance

Every other flag is copied from `training/runs/7mo52r-readout-prepost/args.json`, not retyped:
lr 0.001, alpha_pae 1.0, pair_dropout 0.25, nsample 5, skip_dead_ops, pae_scored_pairs
training/combined/pool.csv. The legacy `--pose-perblock*` / `--pose-readout*` switches recorded
beside the axis flags are dropped, because the trainer refuses both surfaces at once.

## Environment

One card (4). Cards 5 and 7 carry other users' jobs. A 500 MB holder (pid 2172461) keeps card 4
visibly ours; `run_logging/cards_busy.py` returning 1 before launch is correct.
