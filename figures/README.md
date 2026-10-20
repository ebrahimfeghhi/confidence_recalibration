# Figures

One script per figure, committed beside its output, so a figure in a draft can be traced to the
code and the tables that produced it. Each script reads the benchmark result tables directly and
writes its PNG atomically; none of them caches an intermediate, so re-running after a table changes
re-derives rather than re-uses.

Run from the repository root.

| figure | script | what it shows |
|---|---|---|
| `supp_selection_vs_budget_bands.png` | `selection_vs_budget.py --bands` | Supplementary: the same curves with the full 95% cluster-bootstrap ribbons. |
| `supp_dockq_vs_budget_bands.png` | `dockq_vs_budget.py --bands` | Supplementary: continuous version with ribbons. |
| `dockq_vs_budget.png` | `dockq_vs_budget.py` | The same three benchmarks, continuous: **mean** oracle DockQ against **mean** ipTM-selected DockQ per budget. Shows whether the ranking deficit persists above the 0.23 cutoff. |
| `selection_vs_budget.png` | `selection_vs_budget.py` | For the SAbDab2 held-out partition, FoldBench and PXM-2025: the % of interfaces reaching DockQ ≥ 0.23 by ipTM selection and by an oracle over the same pool, against candidate budget 1–100 seeds. The distance between them is the selection gap. |

## What `selection_vs_budget` is for

The first panel is the **held-out partition of SAbDab2** this project trains against — 53
structures, 92 interfaces — not a generic validation split. The other two are published
benchmarks.

Each panel has two curves over the same pool of N seeds:

- **oracle** — an acceptable pose (DockQ >= 0.23) is somewhere in the pool, whoever finds it.
- **ipTM-selected** — the pose ipTM actually picks is acceptable.

**The distance between them is the selection gap**: interfaces the folding solved and the
ranking then lost. It is the entire headroom available to a confidence head. Everything below the
oracle curve is a generation failure that no reranking can reach, so the oracle is the ceiling and
the gap is the prize.

Internally the sweep still partitions into three categories — selected, decoy, none — and oracle is
selected + decoy. The printed table carries selected, oracle and their **paired** gap, each with a
95% interval; the figure plots the two curves that frame the gap and annotates the gap at 50 seeds.

Read at 50 seeds, the free budget, with **95% cluster-bootstrap CIs**. The gap is bootstrapped
**paired** — the same interfaces carry both curves and move together, so differencing two
independently-banded curves would misstate its uncertainty:

| benchmark | interfaces / complexes | selected | oracle | paired selection gap |
|---|---|---|---|---|
| SAbDab2 partition | 92 / 53 | 72.0 [60.9, 82.6] | 85.7 [76.1, 94.6] | **13.7 [8.7, 19.6]** |
| FoldBench | 160 / 104 | 71.5 [63.5, 79.4] | 88.9 [83.3, 93.8] | **17.4 [11.4, 23.8]** |
| PXM-2025 | 292 / 123 | 50.4 [39.9, 61.3] | 78.3 [68.4, 87.3] | **27.9 [20.1, 36.5]** |

Continuous, same budget, mean DockQ:

| benchmark | selected | oracle | paired gap |
|---|---|---|---|
| SAbDab2 partition | 0.576 [0.479, 0.673] | 0.702 [0.619, 0.785] | **0.126 [0.091, 0.165]** |
| FoldBench | 0.524 [0.459, 0.590] | 0.700 [0.645, 0.754] | **0.175 [0.146, 0.207]** |
| PXM-2025 | 0.365 [0.285, 0.450] | 0.531 [0.457, 0.602] | **0.165 [0.134, 0.199]** |

Every gap interval excludes zero on both measures and on all three benchmarks.

**The deficit is not a threshold artefact.** If the gap were marginal crossings of 0.23, the
continuous means would converge while the acceptable fractions diverged. They do not: the mean
DockQ gap is 0.13–0.18 and grows with budget on all three. ipTM is leaving pose quality on the
table across the range, not only at the acceptable boundary.

**PXM-2025 has the largest gap**, 27.9% against the SAbDab2 partition's 13.7% — a difference of
14.2 points at **2.4 sigma** on the cluster bootstrap. Real, but an earlier version of this file
called it "far larger than either SEM" on the strength of binomial bands that were too narrow. It
is a difference worth acting on and not one to overstate.

### Why the curve ribbons are wide and the gap interval is not

Three multiplicative causes, all legitimate, none a bug:

1. **A 95% interval is ~1.96x a +/-1 SEM band.** Measured: x1.96, x2.04, x1.90 on the three
   benchmarks. Most of the widening against the earlier figures is this alone.
2. **Clustering costs effective sample size** -- x1.23, x1.14 and **x1.93**, the last because
   PXM's 292 interfaces come from 123 complexes.
3. **The quantity really is that variable.** The per-complex selected rate has an sd of 40-44
   points, because **69-89% of complexes sit at exactly 0% or 100%** -- ipTM either finds an
   acceptable pose for that complex or never does. With 53-123 complexes the mean cannot be
   pinned tighter.

But the two curves are measured on the SAME interfaces, so all of that variation cancels in their
difference:

| benchmark | selected CI half-width | paired gap | gap CI half-width |
|---|---|---|---|
| SAbDab2 partition | +/-11.3 | 13.7 | +/-8.3 |
| FoldBench | +/-8.3 | 17.4 | +/-6.2 |
| PXM-2025 | +/-10.7 | 27.9 | +/-8.2 |

**A reader who judges significance from the overlapping ribbons reaches the wrong conclusion.**
That is why the main figure drops them and annotates the paired gap instead.

### Error bars: cluster bootstrap, not binomial SEM

The bands resample **complexes** with replacement, because interfaces within a complex are not
independent draws — 39 of SAbDab2's 53 complexes, 52 of FoldBench's 104 and 73 of PXM's 123
contribute more than one interface, and PXM has complexes contributing ten. A per-interface
binomial SEM assumes that away and is optimistic, most severely where the clustering is worst:

| benchmark | binomial SEM | cluster SEM | ratio |
|---|---|---|---|
| SAbDab2 partition | 4.7 | 5.8 | 1.2x |
| FoldBench | 3.6 | 4.1 | 1.1x |
| PXM-2025 | 2.9 | 5.6 | **1.9x** |

PXM's effective sample size is nearer its 123 complexes than its 292 interfaces. `--binomial-band`
draws the naive interval alongside for comparison.

**No homology clusters are available** — none of the three registries carries a cluster column, so
the unit is the complex. That is a lower bound on the true correlation, since homologous complexes
under different PDB ids remain correlated, so these bands are conservative against a binomial SEM
and still optimistic against truth.

### Seed subsampling is reported separately, and is not an error bar

Two different questions, kept apart:

- **seed subsampling** — "if I had drawn a different N seeds for these same interfaces?" Measured
  within an interface across draws. It is large at small N (FoldBench 21.6 points at N=1) and
  collapses as N approaches the pool depth (2.5 at N=100, and exactly 0 on SAbDab2 where every
  interface has exactly 100 seeds, so the subset *is* the pool). It is a property of the budget.
- **test-set uncertainty** — "if I had a different sample of complexes?" This is the cluster
  bootstrap, and it is what the shaded bands show.

Only the second belongs on the plot as an interval around the mean.

### A claim withdrawn

An earlier version of this figure took the FIRST N seeds in sorted order, and on that basis I
reported the selected curve as non-monotonic — SAbDab2 partition 73.9 at N=25 falling to 71.7 at N=75 —
and called it the figure's central finding. **That was an artefact of the single deterministic
draw.** Under the bootstrap the largest decrease anywhere on any benchmark is 0.14 pp, which is
draw noise. The mechanism is real (a pose with higher ipTM and lower DockQ can displace the
previous pick) but it does not produce a net decrease in expectation, and the figure should not be
cited for it.

### Against the published Protenix curve

Checked against the Protenix-v2 FoldBench-AB panel, the same 104 complexes / 160 interfaces:

| N | ours (bootstrap) | ours (first-N) | published |
|---|---|---|---|
| 1 | 50.4% | 45.6% | ~56% |
| 5 | 62.3% | 60.6% | ~65% |
| 10 | 65.8% | 63.7% | ~67% |
| 20 | ~68% | 68.1% | ~68.5% |
| 50 | 71.5% | 74.4% | ~70.5% |
| 100 | 73.9% | 74.4% | ~71% |

From N=20 upward we agree with the published curve, slightly above it. A ~5 point gap remains at
N<=10. Two candidate explanations were tested and **rejected**: aggregating per complex instead of
per interface moves the number the wrong way (46.7% at N=1), and the first-N/bootstrap difference
is already accounted for above.

**The informative number is our ORACLE at N=1: 57.3%.** Their *selected* rate at N=1 is ~56%, so
on our poses no ranker whatsoever could reach their N=1 figure — a perfect oracle gets 57.3%. The
difference therefore cannot be in the ranking statistic; it has to be in the pool. Either their
poses are better at a single seed, or their N=1 pool is larger than ours (we have 5 diffusion
samples per seed, so our N=1 is 5 poses). That the gap closes by N=20 is consistent with both
curves saturating toward a similar oracle ceiling, where pool differences compress.

Until that is settled against their protocol, **the low-N end of our curve is not a reproduction
of theirs** and should not be presented as one.

## Method notes that matter

- **Selection is `max` by `(pair_iptm, -seed, -sample)`**, the same explicit tie-break
  `phase_k_eval._pick` uses, so a pose cannot be resolved differently here than in the evaluator.
- **Acceptable is DockQ >= 0.23.**
- **The pool at budget N is a RANDOM subset of N seeds**, bootstrapped over `--draws` draws
  (default 40), every sample of those seeds included. Not the first N by seed number: that is one
  deterministic draw and it disagrees with a bootstrap by ~5 points at N=1. The
  benchmarks use different seed ranges — SAbDab2 partition 400000+, FoldBench a mix of 101–200, 501–510
  and 600000+, PXM its own — so N is a position, not a seed value.
- **The MAIN figures carry no ribbons; the supplementary ones do.** The ribbons around the two
  curves are far wider than the interval on their paired difference, because complex difficulty
  moves oracle and selected together and cancels when they are differenced. Overlapping ribbons
  therefore read as "the gap is uncertain" when every paired gap interval excludes zero. The
  uncertainty is not hidden: the paired gap and its 95% interval are annotated on every panel, and
  `--bands` regenerates the ribboned version.
- **When drawn, the shaded band is a 95% cluster-bootstrap confidence interval**, taken as the 2.5th/97.5th
  percentiles of 2,000 replicates that resample COMPLEXES with replacement. Not a binomial SEM over
  interfaces (that is what this file used to say and the code used to do — see below), and not a
  +/-1 SEM band: percentiles do not assume symmetry, which matters where a proportion is bounded —
  FoldBench's oracle runs above 90% and a symmetric band there would extend past 100%.
- **The interface set is fixed across the axis.** Only interfaces with at least 100 seeds are
  counted, so the denominator does not change with N. Every interface of all three benchmarks
  clears 100, so nothing is dropped — but only after FoldBench's top-off was merged: 28 of its 160
  interfaces sat at 60–93 seeds until then, and without the merge they would have been silently
  excluded and the panel would read 132 interfaces instead of 160.
