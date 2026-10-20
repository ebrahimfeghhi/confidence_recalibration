# Figures

One script per figure, committed beside its output, so a figure in a draft can be traced to the
code and the tables that produced it. Each script reads the benchmark result tables directly and
writes its PNG atomically; none of them caches an intermediate, so re-running after a table changes
re-derives rather than re-uses.

Run from the repository root.

| figure | script | what it shows |
|---|---|---|
| `selection_vs_budget.png` | `selection_vs_budget.py` | For Validation, FoldBench and PXM-2025: the fraction of interfaces where ipTM selects an acceptable pose, where an acceptable pose exists but ipTM picks a decoy, and where the pool holds no acceptable pose — against candidate budget 1–100 seeds. |

## What `selection_vs_budget` is for

The three categories partition the interfaces, and the split between the last two is the point:

- **no acceptable pose in the pool** — a *generation* failure. More or better folding is the only
  fix; no confidence head can reach it.
- **acceptable pose exists, ipTM picks a decoy** — a *selection* failure. This band is the entire
  headroom available to this project, and its size at the free 50-seed budget is the ceiling on
  any reranking method.

Read at 50 seeds, the free budget, as % of interfaces +/- SEM:

| benchmark | interfaces | selected | decoy | none | oracle |
|---|---|---|---|---|---|
| Validation | 92 | 72.0 ± 4.7 | 13.7 ± 3.6 | 14.3 ± 3.6 | 85.7 |
| FoldBench | 160 | 71.5 ± 3.6 | 17.4 ± 3.0 | 11.1 ± 2.5 | 88.9 |
| PXM-2025 | 292 | 50.4 ± 2.9 | 27.9 ± 2.6 | 21.7 ± 2.4 | 78.3 |

**PXM-2025 is where the headroom is**: 27.9% of its interfaces have an acceptable pose that ipTM
fails to pick, against 13.7% on Validation — a gap far larger than either SEM. A selection method
evaluated only on the validation split is being measured where there is least to win.

**The decoy band widens with budget while the selected band flattens.** FoldBench runs 6.9% decoy
at N=1 to 17.5% at N=100 while selected goes 50.4 → 73.9; PXM runs 4.6% → 30.1% while selected
goes 38.4 → 51.5. So the oracle rises faster than ipTM can follow, and the difference is
accumulating selection failure — the quantity a better confidence head would convert.

### A claim withdrawn

An earlier version of this figure took the FIRST N seeds in sorted order, and on that basis I
reported the selected curve as non-monotonic — Validation 73.9 at N=25 falling to 71.7 at N=75 —
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
  benchmarks use different seed ranges — Validation 400000+, FoldBench a mix of 101–200, 501–510
  and 600000+, PXM its own — so N is a position, not a seed value.
- **The shaded band is the SEM of a proportion**, sqrt(p(1-p)/n) over the interface count — the
  uncertainty of a fraction of interfaces, not a spread over seeds. n is constant across the axis,
  so a band narrows only as p approaches 0 or 1.
- **The interface set is fixed across the axis.** Only interfaces with at least 100 seeds are
  counted, so the denominator does not change with N. Every interface of all three benchmarks
  clears 100, so nothing is dropped — but only after FoldBench's top-off was merged: 28 of its 160
  interfaces sat at 60–93 seeds until then, and without the merge they would have been silently
  excluded and the panel would read 132 interfaces instead of 160.
