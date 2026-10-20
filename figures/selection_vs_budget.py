#!/usr/bin/env python
"""Where each interface stands as the candidate budget grows, on all three benchmarks.

THE QUESTION. Given N seeds of free folds for an interface, ipTM picks one pose. Three things can
happen, and they partition the interfaces:

  acceptable     ipTM selected a pose with DockQ >= 0.23
  decoy          an acceptable pose was in the pool and ipTM picked a worse one  <- SELECTION failure
  no acceptable  the pool contains no acceptable pose at all                     <- GENERATION failure

Separating the last two is the point. "No acceptable" can only be fixed by folding more or folding
better; "decoy" is the part a confidence head could recover, and it is the ceiling on everything
this project is doing. Plotting them against N shows how that ceiling moves with budget -- a method
that beats a 50-seed baseline by selecting better is worth something only if the decoy band is
large at 50 seeds.

SELECTION RULE, MATCHING THE EVALUATOR EXACTLY. max by (pair_iptm, -seed, -sample), the same
explicit tie-break phase_k_eval._pick uses, so two poses with equal ipTM cannot be resolved
differently here than there.

THE POOL IS THE FIRST N SEEDS BY SORTED SEED NUMBER, with every sample of those seeds. The three
benchmarks use different seed ranges -- validation 400000+, FoldBench a mix of 101-200, 501-510 and
600000+, PXM its own -- so "seed N" is a position, not a value.

A FIXED INTERFACE SET ACROSS THE WHOLE AXIS. Only interfaces with at least --max-budget seeds are
included, so the denominator does not change with N. Every interface of all three benchmarks
clears 100, so nothing is dropped at the default.

  figures/selection_vs_budget.py --out figures/selection_vs_budget.png
"""
import argparse
import collections
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACCEPTABLE = 0.23

# Each benchmark's table and the column names it happens to use. The schemas differ; the maths
# does not.
BENCH = [
    ("Validation", "extra_training/validation/fold/results.csv", "iface", "seed", "sample",
     "dockq", "iptm"),
    ("FoldBench", "experimental_data/protenix/foldbench/foldbench_baseline_results.csv", "iface",
     "seed", "sample", "dockq", "pair_iptm"),
    ("PXM-2025", "experimental_data/protenix/pxm2025/pxm2025_baseline_results.csv", "iface",
     "seed", "sample", "dockq", "pair_iptm"),
]
# acceptable / decoy / no-acceptable. Green for what works, amber for what a better head could
# recover, grey for what it could not.
COLORS = ["#2ca02c", "#e8a33d", "#9aa0a6"]
LABELS = ["ipTM selects acceptable", "acceptable exists, ipTM picks decoy", "no acceptable in pool"]
# DISTINGUISHABLE WHERE THEY OVERLAP, which they do: on Validation the decoy and no-acceptable
# curves converge around 13-14% and sit on top of each other, so colour alone loses one of them.
# Three defences -- a different dash pattern each, markers staggered so they do not stack at the
# same x, and an explicit draw order -- and any two coincident curves stay separable.
STYLES = ["-", "--", ":"]
MARKEVERY = [(0, 10), (3, 10), (6, 10)]
MARKERS = ["o", "s", "^"]


def sem(p, n):
    """Standard error of a proportion over n interfaces: sqrt(p(1-p)/n).

    The quantity on each curve is a fraction of interfaces, so its uncertainty is binomial in the
    interface count -- not a spread over seeds. n is fixed across the axis by construction (only
    interfaces reaching --max-budget are counted), so the band narrows only where p approaches 0
    or 1, which is the honest behaviour: a category at 5% is better determined than one at 50%.
    """
    return (p * (1.0 - p) / n) ** 0.5 if n else 0.0


def curves(path, ic, sc, smc, dc, qc, max_budget, draws, rng_seed=0):
    """Fractions in each category for N = 1..max_budget, BOOTSTRAPPED over random seed subsets.

    WHY NOT THE FIRST N SEEDS. The first version of this figure took seeds in sorted order, which
    is one deterministic draw rather than an expectation over which N seeds you happen to have.
    Checked against Protenix's published FoldBench-AB curve (104/160, the same set): first-N gave
    45.6% at N=1 against a bootstrap's 50.6%, and the two agree only once N is large enough that
    the particular subset stops mattering. The published curve carries a shaded band, which is a
    bootstrap by construction. At N=100 both methods give the same answer because the subset IS
    the pool.

    ONE PERMUTATION, SWEPT INCREMENTALLY. For each draw the interface's seeds are shuffled once and
    N walks 1..max_budget, carrying the running best DockQ and the running ipTM-selected pose.
    Re-pooling at every N would be max_budget times the work for the same answer.
    """
    by = collections.defaultdict(list)
    with open(path) as fh:
        for r in csv.DictReader(fh):
            try:
                sd, sm = int(r[sc]), int(r[smc])
                dq, ip = float(r[dc]), float(r[qc])
            except (ValueError, KeyError, TypeError):
                continue
            by[r[ic]].append((sd, sm, ip, dq))
    # EVERY INTERFACE MUST REACH max_budget, so the denominator is constant across the axis.
    kept = {}
    for ifc, rows in by.items():
        if len({t[0] for t in rows}) >= max_budget:
            kept[ifc] = rows
    acc = [0.0] * (max_budget + 1)
    dec = [0.0] * (max_budget + 1)
    non = [0.0] * (max_budget + 1)
    rng = random.Random(rng_seed)
    for ifc, rows in kept.items():
        byseed = collections.defaultdict(list)
        for t in rows:
            byseed[t[0]].append(t)
        seeds = sorted(byseed)
        for _ in range(draws):
            order = seeds[:]
            rng.shuffle(order)
            best_dq = -1.0
            sel = None
            for n in range(1, max_budget + 1):
                for sd, sm, ip, dq in byseed[order[n - 1]]:
                    if dq > best_dq:
                        best_dq = dq
                    cand = (ip, -sd, -sm, dq)
                    if sel is None or cand[:3] > sel[:3]:
                        sel = cand
                _tally(acc, dec, non, n, best_dq, sel)
    n_if = len(kept)
    denom = n_if * draws
    xs = list(range(1, max_budget + 1))
    return (n_if, xs,
            [acc[n] / denom for n in xs], [dec[n] / denom for n in xs],
            [non[n] / denom for n in xs])


def _tally(acc, dec, non, n, best_dq, sel):
    if best_dq < ACCEPTABLE:
        non[n] += 1
    elif sel is not None and sel[3] >= ACCEPTABLE:
        acc[n] += 1
    else:
        dec[n] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--max-budget", type=int, default=100)
    ap.add_argument("--draws", type=int, default=40,
                    help="bootstrap draws per interface. The curve is an expectation over which N "
                         "seeds you happen to have, not over the first N by seed number.")
    ap.add_argument("--out", default="figures/selection_vs_budget.png")
    a = ap.parse_args()

    series = []
    for name, rel, ic, sc, smc, dc, qc in BENCH:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"{name}: {path} absent, skipped")
            continue
        series.append((name,) + curves(path, ic, sc, smc, dc, qc, a.max_budget, a.draws))

    # The numbers, printed as well as drawn -- a figure is read off axes, a table is read exactly.
    for name, n_if, xs, acc, dec, non in series:
        print(f"\n{name}: {n_if} interfaces, {a.draws} bootstrap draws   (value % +/- SEM)")
        print(f"{'N':>4}  {'acceptable':>16} {'decoy':>16} {'none':>16}   {'oracle':>7}")
        for n in (1, 5, 10, 25, 50, 75, 100):
            if n > len(xs):
                continue
            i = n - 1
            print(f"{n:>4}  {acc[i]*100:>8.1f}+/-{sem(acc[i], n_if)*100:<5.1f} "
                  f"{dec[i]*100:>8.1f}+/-{sem(dec[i], n_if)*100:<5.1f} "
                  f"{non[i]*100:>8.1f}+/-{sem(non[i], n_if)*100:<5.1f}   "
                  f"{(acc[i]+dec[i])*100:>6.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    def band(ax, xs, ys, n_if, color):
        lo = [(y - sem(y, n_if)) * 100 for y in ys]
        hi = [(y + sem(y, n_if)) * 100 for y in ys]
        # 0.12 rather than 0.15: three bands that overlap at 0.15 each stack to a muddy block in
        # which none of the three is readable.
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0, zorder=1)

    for ax, (name, n_if, xs, acc, dec, non) in zip(axes, series):
        # BANDS FIRST, ALL OF THEM, then the lines on top: interleaving them lets a later band
        # wash out an earlier line where the curves cross.
        for ys, c in zip((acc, dec, non), COLORS):
            band(ax, xs, ys, n_if, c)
        for i, (ys, c, lbl) in enumerate(zip((acc, dec, non), COLORS, LABELS)):
            # MARKERS EVERY 10 SEEDS, not on all 100: a marker per point turns a 100-wide axis
            # into a solid band and hides the line it is meant to mark. The offsets stagger them
            # so three coincident curves do not pile markers on the same x.
            ax.plot(xs, [y * 100 for y in ys], color=c, label=lbl, lw=1.9,
                    ls=STYLES[i], marker=MARKERS[i], ms=4.0, markevery=MARKEVERY[i],
                    zorder=5 - i)
        ax.set_title(f"{name}  (n={n_if})", fontsize=10)
        ax.set_xlabel("seeds")
        ax.set_xlim(1, a.max_budget)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("% of interfaces")
    # ONE LEGEND, BELOW THE PANELS. Three legends inside three axes was what overlapped the data;
    # a figure-level legend under the row cannot collide with anything.
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3, fontsize=9,
               frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    tmp = a.out + ".tmp.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    os.replace(tmp, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
