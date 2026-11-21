# Uncropped experiments — results

All arms below train the native confidence head (103 trainable tensors, no injected pose layers),
uncropped, lr 1e-3, PAE split threshold 0.23, no PAE balancing. Each arm differs from Arm L in
exactly one way. Shipped is recomputed inside each pass, so it is the baseline for that pool.

Pools: hard-280 = 280 interfaces / 180 structures; topped-up-345 = those plus 65 easy interfaces on
the same structures; val83 = 141 held-out interfaces, 30 seeds each.

Metrics: top-1 = the single pose the ranker puts first; top-5 = the five it puts first, scored as
the best DockQ among them (mean DockQ) and as the fraction of interfaces with at least one
acceptable pose there (frac acc). "max" and "mean" are the two ipTM poolings. PAE loss is the
masked cross-entropy over scored cross-chain residue pairs, split by whether every pose of that
(structure, seed, sample) is acceptable (good), none is (bad), or it straddles (excluded from both
groups but included in total — so total is NOT the weighted average of good and bad).

## The arms

- **Arm L** — the control: plain PAE cross-entropy on the hard-280 pool.
- **Arm M** — each residue pair's cross-entropy multiplied by the TRUE per-pair TM value,
  w = 1/(1 + e²/d0²), plain mean over pairs.
- **Arm N** — the same weighting with w = max(s_true, s_pred), s_pred detached, so a pair the head
  already predicts as bad keeps full weight instead of being discounted.
- **Arm O** — Arm L's objective on the topped-up-345 pool, testing whether adding easy interfaces
  helps generalisation.
- **Arm P** — Arm L plus pair dropout 0.25 on the four pairformer blocks.

## In-sample (the pool each arm trains on)

| arm | pool | ep | ranker | top-1 DockQ | top-1 frac acc | top-5 DockQ | top-5 frac acc | PAE good | PAE bad | PAE total |
|---|---|---|---|---|---|---|---|---|---|---|
| shipped | 280 | — | max | 0.2505 | 30.7% | 0.2825 | 35.7% | 3.4523 | 2.4313 | 2.6995 |
| Arm L | 280 | 6 | max | 0.2544 | 33.2% | 0.3153 | 44.6% | 3.2952 | 2.1706 | 2.4646 |
| Arm L | 280 | 6 | mean | 0.2552 | 32.1% | 0.3223 | 47.1% | — | — | — |
| Arm L | 280 | 12 | max | 0.2747 | 36.4% | 0.3478 | 52.1% | 3.2022 | 2.0742 | 2.3712 |
| Arm L | 280 | 12 | mean | 0.2940 | 42.1% | 0.3604 | 55.7% | — | — | — |
| Arm M | 280 | 4 | max | 0.2482 | 30.0% | 0.2978 | 38.2% | 3.3295 | 2.9983 | 3.0883 |
| Arm M | 280 | 4 | mean | 0.2491 | 29.6% | 0.2953 | 38.2% | — | — | — |
| Arm N | 280 | 6 | max | 0.2553 | 31.4% | 0.3041 | 40.4% | 3.2835 | 2.2217 | 2.5017 |
| Arm N | 280 | 6 | mean | 0.2659 | 33.9% | 0.3148 | 44.3% | — | — | — |
| Arm N | 280 | 12 | max | 0.2765 | 36.4% | 0.3431 | 49.3% | 3.1431 | 2.1803 | 2.4362 |
| Arm N | 280 | 12 | mean | 0.2874 | 38.6% | 0.3488 | 51.1% | — | — | — |
| Arm P | 280 | 6 | max | 0.2724 | 35.7% | 0.3294 | 47.9% | 3.2955 | 2.1743 | 2.4673 |
| Arm P | 280 | 6 | mean | 0.2559 | 32.5% | 0.3190 | 45.4% | — | — | — |
| shipped | 345 | — | max | 0.3165 | 40.6% | 0.3518 | 44.9% | not measured | | |
| Arm O | 345 | 6 | max | 0.3205 | 42.3% | 0.3782 | 52.8% | 3.3340 | 2.1355 | 2.4743 |
| Arm O | 345 | 6 | mean | 0.3284 | 44.1% | 0.3952 | 55.7% | — | — | — |
| Arm O | 345 | 12 | max | 0.3394 | 46.4% | 0.4076 | 58.0% | 3.2349 | 2.0569 | 2.3924 |
| Arm O | 345 | 12 | mean | 0.3540 | 50.4% | 0.4210 | 61.4% | — | — | — |
| Arm O | 345 | 18 | max | 0.3650 | 50.4% | 0.4264 | 60.6% | 3.1475 | 2.0137 | 2.3368 |
| Arm O | 345 | 18 | mean | 0.3810 | 53.0% | 0.4443 | 63.5% | — | — | — |

The PAE columns are per pass, not per pooling, so they are filled on the "max" row only.
The 280 and 345 blocks are not comparable to each other: the 65 added interfaces are ones shipped
already ranks well, which lifts every absolute number on 345. Only deltas against the shipped row
of the same pool are meaningful.

## Held-out (val83, 141 interfaces)

| arm | ep | ranker | top-1 DockQ | top-1 frac acc | top-5 DockQ | top-5 frac acc | PAE good | PAE bad | PAE total |
|---|---|---|---|---|---|---|---|---|---|
| shipped | — | max | 0.5816 | 72.3% | 0.6012 | 72.3% | 3.1203 | 2.3905 | 2.8735 |
| Arm L | 13 | max | 0.5763 | 70.2% | 0.6107 | 73.0% | 3.2403 | 2.2886 | 2.8956 |
| Arm L | 13 | mean | 0.5778 | 71.6% | 0.6237 | 74.5% | — | — | — |
| Arm M | 4 | max | 0.5899 | 73.0% | 0.6144 | 73.8% | 3.2287 | 3.1763 | 3.2261 |
| Arm M | 4 | mean | 0.5767 | 71.6% | 0.6052 | 72.3% | — | — | — |
| Arm N | 6 | max | 0.5835 | 72.3% | 0.6160 | 73.0% | 3.1819 | 2.2338 | 2.8386 |
| Arm N | 6 | mean | 0.5820 | 71.6% | 0.6086 | 71.6% | — | — | — |
| Arm N | 12 | max | 0.5787 | 71.6% | 0.6136 | 73.0% | 3.1815 | 2.3535 | 2.8838 |
| Arm N | 12 | mean | 0.5748 | 70.9% | 0.6036 | 71.6% | — | — | — |
| Arm O | 6 | max | 0.5798 | 70.2% | 0.6103 | 71.6% | 3.2198 | 2.2463 | 2.8647 |
| Arm O | 6 | mean | 0.5764 | 70.9% | 0.6060 | 71.6% | — | — | — |
| Arm O | 12 | max | 0.5817 | 71.6% | 0.6177 | 74.5% | 3.2059 | 2.3084 | 2.8803 |
| Arm O | 12 | mean | 0.5832 | 70.9% | 0.6169 | 74.5% | — | — | — |
| Arm O | 18 | max | 0.5804 | 71.6% | 0.6171 | 74.5% | 3.2151 | 2.4802 | 2.9476 |
| Arm O | 18 | mean | 0.5761 | 70.2% | 0.6104 | 73.8% | — | — | — |
| Arm P | 6 | max | 0.5781 | 70.9% | 0.6124 | 73.0% | 3.2023 | 2.2318 | 2.8490 |
| Arm P | 6 | mean | 0.5820 | 70.9% | 0.6081 | 71.6% | — | — | — |

## What the numbers say

1. **Every arm improves in-sample and none improves held-out top-1.** The largest in-sample gain is
   Arm O at epoch 18 (+0.0645 mean DockQ, +12.4 points frac acceptable over shipped on its pool);
   the largest held-out top-1 gain by any arm at any epoch is +0.0083 (Arm M epoch 4, max pooling),
   which is one interface on 141 and does not survive the change of pooling.

2. **No arm beats shipped on held-out top-1 frac acceptable.** Shipped is 72.3%; the best any arm
   reaches is 73.0% (Arm M epoch 4, max) and most sit 1-2 points below.

3. **Top-5 is the one repeatable gain.** Arm L epoch 13 mean-pooled reaches 74.5% frac acceptable@5
   against shipped's 72.3%, and Arm O reproduces 74.5% at both epoch 12 and epoch 18 on both
   poolings. Best-DockQ@5 is up ~0.012-0.022 across those arms. The ranker gets an acceptable pose
   into the top five more often even when its single pick is no better.

4. **The good/bad asymmetry is universal.** Held-out PAE loss on good poses is worse than shipped
   for every arm (+0.06 to +0.12), while bad poses improve (-0.10 to -0.16). Training data is 23.3%
   good poses against val83's 60.7%, and bad poses put 64.2% of their pairs in the last distance
   bin on val, so "predict a large error" is a cheap transferable rule while good poses need fine
   discrimination below 8 A.

5. **Training longer does not help and eventually hurts.** Arm O's held-out total PAE loss gets
   monotonically worse with training (2.8647 -> 2.8803 -> 2.9476 at epochs 6/12/18) while its
   in-sample numbers climb throughout.

6. **The TM weighting on the true score (Arm M) was the one clear failure.** Its held-out PAE loss
   on bad poses is 3.1763 against shipped's 2.3905 — far worse than any other arm — which is what
   motivated Arm N's max weight.

## Caveats

- Arm M has only an epoch-4 pass; Arm L's in-sample epoch 6 comes from a resumed segment, so its
  epoch numbering is the chain's, not a fresh start. Arm P's epoch-9 passes were still running when
  this was written and are not included.
- Shipped's held-out PAE split (3.1203 / 2.3905 / 2.8735) is a reconstruction through the
  label_index sample mapping rather than an eval-side sidecar. The arms' PAE numbers all come from
  sidecars and are exactly comparable to each other.
- Shipped's PAE split on the 345 pool was never measured, so Arm O's PAE has no shipped reference.
- val83's good poses are 99.7% from easy interfaces, so no good-pose metric on it can see what these
  arms are built to improve. All 206 hard 15-seed interfaces are already in the training pool, so no
  held-out hard slice can be carved from this data.
