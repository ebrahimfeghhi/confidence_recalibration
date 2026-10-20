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

Read at 50 seeds, the free budget:

| benchmark | interfaces | selected | decoy | none | oracle |
|---|---|---|---|---|---|
| Validation | 92 | 72.8% | 13.0% | 14.1% | 85.9% |
| FoldBench | 160 | 74.4% | 17.5% | 8.1% | 91.9% |
| PXM-2025 | 292 | 48.6% | 30.5% | 20.9% | 79.1% |

**PXM-2025 is where the headroom is**: 30.5% of its interfaces have an acceptable pose that ipTM
fails to pick, against 13.0% on Validation. A selection method evaluated only on the validation
split is being measured where there is least to win.

Both FoldBench and Validation are flat from 50 to 100 seeds in the selected band — the extra 50
seeds add almost nothing ipTM can use, which is the quantitative form of the free-budget argument.

## Method notes that matter

- **Selection is `max` by `(pair_iptm, -seed, -sample)`**, the same explicit tie-break
  `phase_k_eval._pick` uses, so a pose cannot be resolved differently here than in the evaluator.
- **Acceptable is DockQ >= 0.23.**
- **The pool at budget N is the first N seeds by sorted seed number**, every sample included. The
  benchmarks use different seed ranges — Validation 400000+, FoldBench a mix of 101–200, 501–510
  and 600000+, PXM its own — so N is a position, not a seed value.
- **The interface set is fixed across the axis.** Only interfaces with at least 100 seeds are
  counted, so the denominator does not change with N. Every interface of all three benchmarks
  clears 100, so nothing is dropped — but only after FoldBench's top-off was merged: 28 of its 160
  interfaces sat at 60–93 seeds until then, and without the merge they would have been silently
  excluded and the panel would read 132 interfaces instead of 160.
