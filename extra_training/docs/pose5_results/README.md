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
