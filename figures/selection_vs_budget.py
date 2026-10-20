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
    # NOT "Validation": that reads like a generic split, when it is specifically the held-out
    # partition of SAbDab2 this project trains against -- 53 structures, 92 interfaces -- and the
    # other two panels are named benchmarks.
    ("SAbDab2 partition", "extra_training/validation/fold/results.csv", "iface", "seed", "sample",
     "dockq", "iptm"),
    ("FoldBench", "experimental_data/protenix/foldbench/foldbench_baseline_results.csv", "iface",
     "seed", "sample", "dockq", "pair_iptm"),
    ("PXM-2025", "experimental_data/protenix/pxm2025/pxm2025_baseline_results.csv", "iface",
     "seed", "sample", "dockq", "pair_iptm"),
]
# acceptable / decoy / no-acceptable. Green for what works, amber for what a better head could
# recover, grey for what it could not.
# ORACLE AND SELECTED, with the gap between them as the point of the figure.
#
#   oracle   = an acceptable pose is somewhere in the pool of N seeds
#   selected = the pose ipTM actually picks is acceptable
#
# Their DIFFERENCE is the selection gap: interfaces the folding solved and the ranking then lost.
# That is the entire headroom available to a confidence head, and drawing both curves puts it on
# the page as a vertical distance instead of asking the reader to subtract. Everything below the
# oracle line is a generation failure no reranking can reach.
COLORS = ["#1f77b4", "#444444"]          # selected, oracle
LABELS = ["ipTM-selected", "oracle (best pose in pool)"]
MARKERS = ["o", "^"]
MARKEVERY = [(0, 10), (5, 10)]


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

    # EACH PANEL ITS OWN Y AXIS. Shared limits forced PXM's 50% band and FoldBench's 72% band onto
    # one 0-100 scale, which flattened both; per-panel ranges let each curve use its height.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    def band(ax, xs, ys, n_if, color):
        lo = [(y - sem(y, n_if)) * 100 for y in ys]
        hi = [(y + sem(y, n_if)) * 100 for y in ys]
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0, zorder=1)

    for ax, (name, n_if, xs, acc, dec, non) in zip(axes, series):
        # ORACLE = selected + decoy: an acceptable pose is in the pool, whoever picks it.
        orc = [a_ + d_ for a_, d_ in zip(acc, dec)]
        shown = (acc, orc)
        # THE GAP IS NOT SHADED. It was, and that was a mistake: this panel already uses shading
        # to mean UNCERTAINTY (the SEM bands), so filling a region that is not an uncertainty in
        # the same visual language invites reading it as a confidence interval. Two curves show
        # the distance between them without help.
        for ys, c in zip(shown, COLORS):
            band(ax, xs, ys, n_if, c)
        for i, (ys, c, lbl) in enumerate(zip(shown, COLORS, LABELS)):
            ax.plot(xs, [y * 100 for y in ys], color=c, label=lbl, lw=1.9, ls="-",
                    marker=MARKERS[i], ms=4.0, markevery=MARKEVERY[i], zorder=5 - i)
        # LIMITS FROM THE DATA PLUS THE BANDS, so a SEM band is never clipped at the frame.
        lo = min((y - sem(y, n_if)) * 100 for ys in shown for y in ys)
        hi = max((y + sem(y, n_if)) * 100 for ys in shown for y in ys)
        pad = max(1.0, (hi - lo) * 0.08)
        ax.set_ylim(max(0.0, lo - pad), min(100.0, hi + pad))
        ax.set_title(f"{name}  (n={n_if})", fontsize=10)
        ax.set_xlabel("# of seeds")
        ax.set_xlim(1, a.max_budget)
        ax.set_ylabel("DockQ \u2265 0.23 (%)")
        ax.grid(True, alpha=0.3)
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
