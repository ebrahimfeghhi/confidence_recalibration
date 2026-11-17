# 5-site pose injection + rel, lr 1e-4, alpha 50 — results through epoch 30

Run `jyz3g2`, launched 2026-11-16 10:34 PM EST, 60 epochs, ~15.5 min/epoch.
`jyz3g2_val_slice.csv` has one row per completed eval pass.

## The arm

Five pose-injection sites: one before each of the four pairformer blocks (block 0 its own
pretrained injection, blocks 1-3 copy-initialised from it) and a fifth before the PAE readout's
layer norm, orientation at all five. **Only those sites train** — 15 of 261 head tensors, 55,040
of 19,797,070 parameters (0.278%). Loss = PAE (alpha 1.0) + mean-centred regression on the true
iTM (`--rank-form rel`, alpha 50, the p25 of a 153-step gradient-parity measurement on this exact
architecture — see ../alpha_parity/). lr 1e-4, pair-dropout 0.25. Val every 2 epochs on 141
held-out interfaces x 30 seeds x 5 poses; train slice every 10 on 280 in-sample interfaces.

Shipped baseline: q 0.5819, top-1 72.3%, within-interface rho 0.1882, margin 0.0347.

## Columns

`q` = mean DockQ of the selected pose. `top1` = fraction of interfaces where the selected pose is
acceptable (DockQ >= 0.23). `rho` = mean within-interface Spearman between the head's score and
DockQ — the metric that matches the task, since the job is ordering one interface's 150 candidates.
`_max` selects by max-pooled ipTM (the shipped convention, and the comparable headline); `_mean`
selects by mean-pooled ipTM, which is what this arm actually optimises. `margin` = score(top pick)
- score(best acceptable candidate), averaged over winnable items, **lower is better**.
`score_spread` = median within-interface max-min of the head's score. `margin_over_spread`
normalises the margin by the scale it is measured in.

## What it shows through epoch 30

- **rho rises monotonically and is the clearest signal**: max-pooled 0.1820 -> 0.2091, mean-pooled
  0.1719 -> 0.2005, against shipped's 0.1882. Every one of the 15 passes is above shipped from
  epoch 4 on.
- **mean-pooled q is above shipped and climbing**: 0.5710 -> 0.5875 against 0.5819, and mean-pooled
  top-1 reached parity (72.3%) at epochs 28 and 30. Max-pooled q sits essentially at shipped
  (0.5809-0.5819) with top-1 at parity throughout.
- **No memorisation through epoch 30.** The slice went 28.6% -> 27.9% -> 28.6% at epochs 10, 20
  and 30 — flat, and below shipped's 30.7% at every point. The three previous arms all climbed
  in-sample over the same span: the 1e-3 PAE-only arm went 28.6% -> 35.4% -> 34.6% -> 49.6% by
  epoch 50 while its held-out q fell from 0.5873 to 0.5591.
- **In-sample rho DOES rise while in-sample top-1 does not**: 0.0476 -> 0.0587 -> 0.0831 at epochs
  10, 20, 30 against shipped's 0.0819. So the head is learning the training set's ordering without
  converting it into more acceptable in-sample picks — the same disconnect the held-out side shows,
  which argues the ceiling is in the conversion and not in how much the head has learned.
- **MARGIN RISES, AND THAT IS SCALE, NOT RANKING.** Raw margin goes 0.0143 -> 0.0244 then flattens
  at 0.0239 — worse than its own start, but better than shipped's 0.0347 at every epoch. Over the
  same span the score spread nearly doubles, 0.1142 -> 0.2119, and margin is in ipTM units, so it
  inflates with the scale. Normalised, margin/spread goes 0.126 -> 0.113, i.e. slightly improving,
  which agrees with rho. `rel` regresses on the true iTM's magnitudes, not just its order, and true
  iTM is spread wider than the shipped head's compressed ipTM — so a widening spread is what this
  objective is supposed to produce. The lr 1e-4 PAE-only arm showed the mirror case: margin fell
  29% while spread fell 39%, so normalising reversed that reading too.

## SELECTION AT BUDGETS ABOVE ONE -- EXPLORATORY, NOT ESTABLISHED

**Held-out, epoch 46, max-pooled** (epoch 46 exactly; an earlier version of this table was
labelled 46 and carried epoch 44's numbers):

| metric | ours | shipped | gain |
|---|---|---|---|
| frac acceptable in top 1 | 72.3% | 72.3% | +0.0 |
| frac acceptable in top 5 | 74.5% (105/141) | 72.3% (102/141) | **+3 interfaces** |
| frac acceptable in top 10 | 74.5% | 74.5% | +0.0 |
| best-of-5 DockQ | 0.6106 | 0.6014 | +0.0092 |
| best-of-10 DockQ | 0.6210 | 0.6170 | +0.0040 |
| mean rank of first acceptable pose | 13.91 | 14.9 | -1.0 (lower better) |

**THIS IS THREE NET INTERFACES AND IT IS NOT SIGNIFICANT.** At epoch 50 the paired breakdown is
3 gained, 0 lost -- the most favourable possible split -- and exact two-sided McNemar on three
discordant pairs gives **p = 0.25**, before any correction. The selection surface is larger than
the six metrics: epochs, two poolings and two budgets were all examined. Interfaces are also not
independent, since several share a structure, so even that p is optimistic. Persistence across
later epochs makes it more interesting than a spike, but repeated scoring of the SAME 141
interfaces is not replication.

What it needs before it is a result: structure-clustered paired uncertainty on best-of-5 DockQ
from the per-interface rows, then one preselected metric and checkpoint confirmed on data not
used to choose them.

**"MOVES THE TAIL WITHOUT CHANGING THE WINNER" WAS WRONG AND IS WITHDRAWN.** Measured at epoch 50:
the head picks a DIFFERENT top-1 pose from shipped on **74 of 141 interfaces**. Top-1 success is
flat because the gains and losses cancel, not because the winner is unchanged. Note also that
best-of-5 is shortlist quality scored with true DockQ -- it says a good pose is present in the
top five, not that an automatic selector would find it.

## RARE-GOOD PAE: THE IMPROVEMENT IS REAL, THE "NO COLLAPSE" HEADLINE IS WITHDRAWN

Recomputed across ALL THREE ranks and paired on the exact (pdb, seed, sample) key between two
complete epochs. The earlier version read `trajectory.csv`, which is byte-identical to
`trajectory_rank0.csv` because the trainer writes it from rank 0 alone, and averaged each epoch
over whatever poses it contained, so a composition change could have passed as a loss change.
Paired: 34,245 of 34,425 poses, 180 dropped on each side.

| good-pose share | good poses | **structs** | ep1 | ep50 | delta | **relative** | bad delta | bad rel |
|---|---|---|---|---|---|---|---|---|
| 0-10% | 182 | **55** | 3.6807 | 3.4652 | -0.2155 | **-5.9%** | -0.0525 | -1.9% |
| 10-25% | 574 | 46 | 3.7445 | 3.2738 | -0.4707 | -12.6% | -0.1883 | -6.0% |
| 25-50% | 1453 | 52 | 3.8029 | 3.3070 | -0.4959 | -13.0% | -0.0837 | -2.6% |
| 50-75% | 2747 | 59 | 3.8207 | 3.3561 | -0.4646 | -12.2% | -0.1953 | -6.2% |
| 75-100% | 14145 | 380 | 3.6801 | 3.0838 | -0.5963 | **-16.2%** | -0.1482 | -4.6% |

**What this supports: rare-good PAE loss does improve.** Good poses beat bad poses in every
bucket, including where good poses are under 10% of a structure's poses.

**What it does NOT support, and the earlier README claimed:** that balancing is unnecessary, or
that the imbalance is mild. In RELATIVE terms the rare bucket improves **2.7x less** than the
most-common bucket, -5.9% against -16.2%, and that is the comparison that matters across buckets
with different absolute levels. The independent unit in the rare bucket is **55 structures**, not
182 poses. Excluding mixed-interface poses narrows the claim further. And none of this is an A/B
of the balancing term, which has never been run.

## What to challenge

1. Is `margin_over_spread` the right normalisation, or does dividing by a median max-min just trade
   one scale artefact for another? Name a better scale-free margin if there is one.
2. rho is up ~0.02 absolute over shipped and rising, but top-1 has not exceeded 72.3% on either
   pooling in 15 passes. Is a ranking gain that does not convert to top-1 worth anything here, and
   what would have to be true for it to convert?
3. Which pooling should be the headline for this arm? It trains on mean-pooled ipTM, mean-pooled is
   the better reading on q and top-1, and max-pooled is the shipped convention and the better
   reading on rho.
4. In-sample is BELOW shipped and falling at epoch 20 while held-out rises. With 0.278% of the head
   trainable, is that underfitting rather than good generalisation — and if so, does it argue for a
   larger alpha, a higher lr, or more trainable sites?
