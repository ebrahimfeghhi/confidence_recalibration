# Gap-matched pairwise ranking: within-seed vs cross-seed pose pairs

Reproduce with:

```
python -I run_logging/gap_matched_within_vs_across.py      # interface level, both matchings
python -I run_logging/gap_matched_structure_level.py       # the four-cell split
```

Data: the 50-seed FoldBench/PXM fold set, 947 interfaces in 459 structures, 71,025 poses
(5 samples x 15 seeds for the interfaces that carry the full set). `iptm` is max-pooled
chain-pair ipTM; DockQ ties are dropped; an ipTM tie counts as half a hit.

## The question

ipTM ranks poses across seeds well and within a seed barely at all. But the two comparisons
are not equally hard: two samples off one trunk usually differ in DockQ by much less than two
poses from different seeds do (median gap 0.0250 vs 0.0510). An unconditional accuracy gap
could be entirely that. So condition on the gap.

Both populations are built INSIDE one interface, so interface identity is never the difference:

- **within** -- the C(5,2) = 10 pairs of samples that share a seed
- **across** -- every pair of poses from two different seeds of that interface

## Interface level, binned

| \|dDockQ\| bin | within acc | n | across acc | n | within-across | med\|dipTM\| w | a | ratio a/w |
|---|---|---|---|---|---|---|---|---|
| [0.00,0.02) | 0.5084 | 58,077 | 0.5116 | 753,693 | -0.0032 | 0.0023 | 0.0219 | 9.52x |
| [0.02,0.05) | 0.5336 | 29,642 | 0.5343 | 427,266 | -0.0007 | 0.0020 | 0.0218 | 10.90x |
| [0.05,0.10) | 0.5476 | 18,655 | 0.5526 | 327,158 | -0.0049 | 0.0034 | 0.0280 | 8.24x |
| [0.10,0.20) | 0.5669 | 11,947 | 0.6042 | 272,687 | **-0.0373** | 0.0066 | 0.0334 | 5.06x |
| [0.20,0.40) | 0.5885 | 7,304 | 0.6804 | 207,613 | **-0.0919** | 0.0125 | 0.0530 | 4.24x |
| [0.40,1.01) | 0.6648 | 5,468 | 0.8695 | 390,755 | **-0.2048** | 0.0411 | 0.2855 | 6.95x |
| ALL | 0.5360 | 131,093 | 0.6054 | 2,379,172 | -0.0694 | 0.0032 | 0.0362 | 11.31x |
| median gap | 0.0250 | | 0.0510 | | | | | |

Below a 0.1 gap the two are within 0.005 of each other. The cross-seed advantage appears only
above 0.1 and is large in the top bin.

## Interface level, paired

For each within-seed pair, the cross-seed pairs of the SAME interface whose gap is within
+-0.01. Both accuracies are then means over the same gap distribution by construction.
130,931 of the 131,093 within-seed pairs find a match.

| quantity | value |
|---|---|
| within-seed accuracy | 0.5359 |
| matched cross-seed accuracy | 0.5393 |
| difference | **-0.0034**  (t -2.52, p 1.18e-02) |
| median \|dipTM\| within-seed | 0.0032 |
| median \|dipTM\| matched cross-seed | 0.0412  (**12.89x**) |
| mean \|dipTM\| within / cross | 0.0181 / 0.0620  (3.42x) |

So the unconditional -0.0694 is almost entirely the gap, not the comparison type. What does
NOT shrink is the ipTM movement: at a matched gap, cross-seed pairs swing 12.89x further in
ipTM for the same ordering accuracy. That extra swing is orthogonal to pose quality
everywhere except the top bin -- below a 0.1 gap, 8-11x the movement buys +0.000 to +0.005
accuracy; in the top bin 7x the movement does buy +0.205.

## Structure level: the four-cell split

Pairs formed within a structure, split by whether the two poses share a seed and whether they
share an interface. Keeps only structures whose every interface carries the full seed set.
459 structures, 380 of them with more than one interface.

Accuracy by cell (med \|dipTM\| in parentheses):

| \|dDockQ\| bin | same seed, same iface | diff seed, same iface | same seed, diff iface | diff seed, diff iface |
|---|---|---|---|---|
| [0.00,0.02) | 0.5084 (0.0023) | 0.5116 (0.0219) | 0.5272 (0.0280) | 0.5132 (0.0476) |
| [0.02,0.05) | 0.5336 (0.0020) | 0.5343 (0.0218) | 0.5378 (0.0234) | 0.5387 (0.0387) |
| [0.05,0.10) | 0.5476 (0.0034) | 0.5526 (0.0280) | 0.5571 (0.0245) | 0.5472 (0.0407) |
| [0.10,0.20) | **0.5669** (0.0066) | 0.6042 (0.0334) | 0.5857 (0.0269) | 0.5822 (0.0427) |
| [0.20,0.40) | **0.5885** (0.0125) | 0.6804 (0.0530) | 0.6561 (0.0327) | 0.6691 (0.0577) |
| [0.40,1.01) | **0.6648** (0.0411) | 0.8695 (0.2855) | 0.7739 (0.0612) | 0.8668 (0.2263) |
| ALL | 0.5360 (0.0032) | 0.6054 (0.0362) | 0.5743 (0.0278) | 0.6156 (0.0558) |

Two readings:

1. The seed advantage is not an artifact of pooling interfaces -- it survives inside a single
   structure.
2. **The within-seed pose pair is the worst cell in every bin above a 0.1 gap**, beaten even by
   pairs drawn from two DIFFERENT interfaces of the same trunk. Sharing a trunk is not what
   hurts; sharing a trunk AND being the thing the trunk never saw is.

These are shipped-weights inference folds, so there is no crop anywhere in them -- cropping is
training-side only. What a seed fixes is the trunk: one trunk per (structure, seed), shared by
all of that structure's interfaces and all five of each pose set's samples.

## Open check: residual gap skew inside a bin

The bins are wide, and the top one is very wide (0.40 to 1.01). If cross-seed pairs sit
systematically nearer the upper edge of a bin than within-seed pairs do, their higher accuracy
inside that bin is partly still the gap. Three things bear on it:

- The paired +-0.01 matching is the instrument built for exactly this, and its tolerance is 40x
  tighter than the top bin's width. It takes the overall difference from -0.0694 to -0.0034.
- But the paired number is reported POOLED only. **The paired difference has never been
  computed restricted to large gaps**, which is where the binned tables put the whole effect.
  Until it is, the top-bin -0.2048 is not separated from within-bin skew.
- The within-seed population is bounded in a way the cross-seed one is not: a within-seed pair
  can only reach a 0.9 gap if one trunk produced both a near-perfect and a near-zero pose, so
  the top bin's within-seed members may be a genuinely different kind of pair.

The measurement that would settle it: run the paired matching and report the difference
stratified by the within-pair's own gap, so each stratum is a +-0.01 comparison rather than a
bin-wide one. Also worth doing cluster-robust on the interface (131,093 within pairs nest in
14,205 (interface, seed) groups, 947 interfaces), since the t -2.52 above treats pairs as
independent.

## Replication

Re-measured on the regenerated poses (10,779 units, same trunks, fresh diffusion noise):
top bin 0.6277 within / 0.8766 across, overall 0.5285 / 0.5996, matched \|dipTM\| ratio 13.80x,
paired difference -0.0056. Every figure within a couple of points of the table above.
