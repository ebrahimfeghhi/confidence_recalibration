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


def sem(p, n):
    """Standard error of a proportion over n interfaces: sqrt(p(1-p)/n).

    The quantity on each curve is a fraction of interfaces, so its uncertainty is binomial in the
    interface count -- not a spread over seeds. n is fixed across the axis by construction (only
    interfaces reaching --max-budget are counted), so the band narrows only where p approaches 0
    or 1, which is the honest behaviour: a category at 5% is better determined than one at 50%.
    """
    return (p * (1.0 - p) / n) ** 0.5 if n else 0.0


def curves(path, ic, sc, smc, dc, qc, max_budget):
    """Fractions in each category for N = 1..max_budget. One pass, then an incremental sweep."""
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
            kept[ifc] = sorted(rows)            # by seed, then sample
    acc = [0] * (max_budget + 1)
    dec = [0] * (max_budget + 1)
    non = [0] * (max_budget + 1)
    for ifc, rows in kept.items():
        seeds = sorted({t[0] for t in rows})[:max_budget]
        pos = {s: i + 1 for i, s in enumerate(seeds)}
        # INCREMENTAL: walk the poses in seed order, carrying the running best DockQ and the
        # running ipTM-selected pose. Recomputing the pool for each N would be 100x the work.
        best_dq = -1.0
        sel = None                               # (iptm, -seed, -sample, dockq)
        nxt = 1
        for sd, sm, ip, dq in rows:
            n = pos.get(sd)
            if n is None:
                continue
            while nxt < n:                       # close out budgets with no new pose
                _tally(acc, dec, non, nxt, best_dq, sel)
                nxt += 1
            if dq > best_dq:
                best_dq = dq
            cand = (ip, -sd, -sm, dq)
            if sel is None or cand[:3] > sel[:3]:
                sel = cand
        while nxt <= max_budget:
            _tally(acc, dec, non, nxt, best_dq, sel)
            nxt += 1
    n_if = len(kept)
    xs = list(range(1, max_budget + 1))
    return (n_if, xs,
            [acc[n] / n_if for n in xs], [dec[n] / n_if for n in xs], [non[n] / n_if for n in xs])


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
    ap.add_argument("--out", default="figures/selection_vs_budget.png")
    a = ap.parse_args()

    series = []
    for name, rel, ic, sc, smc, dc, qc in BENCH:
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            print(f"{name}: {path} absent, skipped")
            continue
        series.append((name,) + curves(path, ic, sc, smc, dc, qc, a.max_budget))

    # The numbers, printed as well as drawn -- a figure is read off axes, a table is read exactly.
    for name, n_if, xs, acc, dec, non in series:
        print(f"\n{name}: {n_if} interfaces   (value % +/- SEM)")
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
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)

    for ax, (name, n_if, xs, acc, dec, non) in zip(axes, series):
        for ys, c, lbl in zip((acc, dec, non), COLORS, LABELS):
            # MARKERS EVERY 10 SEEDS, not on all 100: a marker per point turns a 100-wide axis
            # into a solid band and hides the line it is meant to mark.
            ax.plot(xs, [y * 100 for y in ys], "-o", color=c, label=lbl, lw=1.8, ms=3.5,
                    markevery=10)
            band(ax, xs, ys, n_if, c)
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
