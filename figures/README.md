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
| Validation | 92 | 72.8 ± 4.6 | 13.0 ± 3.5 | 14.1 ± 3.6 | 85.9 |
| FoldBench | 160 | 74.4 ± 3.5 | 17.5 ± 3.0 | 8.1 ± 2.2 | 91.9 |
| PXM-2025 | 292 | 48.6 ± 2.9 | 30.5 ± 2.7 | 20.9 ± 2.4 | 79.1 |

**PXM-2025 is where the headroom is**: 30.5% of its interfaces have an acceptable pose that ipTM
fails to pick, against 13.0% on Validation — and the gap is far larger than either SEM. A
selection method evaluated only on the validation split is being measured where there is least to
win.

**The `selected` curve is not monotonic in N, and the `oracle` is.** Validation runs 73.9 at N=25,
72.8 at 50, 71.7 at 75, 72.8 at 100; FoldBench 66.9 → 74.4 → 73.1 → 74.4. Adding seeds can
*lower* the fraction where ipTM selects an acceptable pose, because a new pose with higher ipTM and
lower DockQ displaces the previous pick. Meanwhile the oracle only rises, so the decoy band widens
with budget: 12.0 → 14.1 on Validation, 18.2 → 30.8 on PXM between N=5 and N=100.

That is the figure's central claim. **More candidates reliably improve what is achievable and not
what ipTM achieves**, and every bit of that growing gap is selection failure — exactly the quantity
a better confidence head would convert.

## Method notes that matter

- **Selection is `max` by `(pair_iptm, -seed, -sample)`**, the same explicit tie-break
  `phase_k_eval._pick` uses, so a pose cannot be resolved differently here than in the evaluator.
- **Acceptable is DockQ >= 0.23.**
- **The pool at budget N is the first N seeds by sorted seed number**, every sample included. The
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
