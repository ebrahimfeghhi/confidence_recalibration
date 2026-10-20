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
LABELS = ["ipTM selects an acceptable pose", "acceptable pose exists, ipTM picks a decoy",
          "no acceptable pose in the pool"]


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

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), sharey=True)
    for ax, (name, rel, ic, sc, smc, dc, qc) in zip(axes, BENCH):
        path = os.path.join(a.root, rel)
        if not os.path.exists(path):
            ax.set_title(f"{name}\n(table absent)")
            continue
        n_if, xs, acc, dec, non = curves(path, ic, sc, smc, dc, qc, a.max_budget)
        ax.stackplot(xs, acc, dec, non, colors=COLORS, labels=LABELS, edgecolor="none")
        ax.set_title(f"{name} — {n_if} interfaces")
        ax.set_xlabel("candidate budget (seeds)")
        ax.set_xlim(1, a.max_budget)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2, color="white", lw=0.6)
        # THE TWO NUMBERS WORTH READING OFF: the selection headroom at the free budget, and what
        # is unreachable by selection at all.
        i50 = min(len(xs), 50) - 1
        ax.annotate(f"at 50 seeds: {acc[i50]*100:.0f}% selected, {dec[i50]*100:.0f}% decoy, "
                    f"{non[i50]*100:.0f}% none",
                    xy=(0.5, -0.19), xycoords="axes fraction", ha="center", fontsize=8)
    axes[0].set_ylabel("fraction of interfaces")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Pose selection against candidate budget: the amber band is what a better "
                 "confidence head could recover; the grey band is what it could not",
                 fontsize=10, y=0.90)
    fig.tight_layout(rect=(0, 0.04, 1, 0.86))
    tmp = a.out + ".tmp.png"
    fig.savefig(tmp, dpi=130)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
