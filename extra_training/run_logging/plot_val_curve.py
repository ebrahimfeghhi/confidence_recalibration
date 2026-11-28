#!/usr/bin/env python
"""HELD-OUT SELECTION ACROSS EPOCHS: mean DockQ and frac acceptable, BOTH POOLINGS, two panels.

    plot_val_curve.py <out.png> <rid> [<rid2> ...] [--title T]

SIX PANELS: {mean DockQ, fraction acceptable, mean rank of the first acceptable pose} x
{max-pooled, mean-pooled ipTM}, one metric per panel and one pooling per ROW. The two poolings
were overlaid in one panel at first; separating them is what the owner asked for and it is the
better default -- max and mean disagree about which arms work (Arm S was positive on max at
epochs 4-5 while never beating shipped on mean), and a shared axis makes the smaller of the two
look like noise around the larger.

THE RANK PANEL IS LOWER-IS-BETTER and says so on its axis. The axis is NOT inverted: an inverted
axis silently reverses every comparison a reader makes in the first second.

rho IS THE PROJECT'S rho, NOT A NEW STATISTIC: the MEAN OVER INTERFACES of the within-interface
Spearman correlation between the pooled ipTM and DockQ across that interface's poses, exactly as
arm_table.py computes it (arm_table.py:98). A global Spearman over every pose at once would be a
different and easier quantity -- it would be carried by between-interface variation in difficulty
rather than by the within-interface ranking the selector actually performs.

ONE ARM PER CALL, its chain passed as several rids (fshpzx holds epochs 1-13 and 7mo52r 14-18,
so both are needed to draw one 18-epoch curve).

THE SET IS THE 139 LEAK-FREE INTERFACES, dropping 8z7j_AB and 8z7j_AC -- 8z7j is the one
validation structure sharing an antibody cluster with the training pool, so leaving it in would
let a trained-on cluster into a held-out claim.

SHIPPED IS RECOMPUTED FROM THE ARM'S OWN ROWS via stored_iptm, not read from a constant: it is
the shipped head's ipTM for exactly the poses this arm scored, so the two are paired. It is drawn
as a horizontal line per panel because it has no epoch.

BOTH POOLINGS, ALWAYS. max-pooled and mean-pooled ipTM disagree about which arms work -- Arm S's
max column was positive at epochs 4-5 while its mean column was never above shipped -- so one
pooling alone is a coin flip about which story gets told.
"""
import argparse, collections, csv, glob, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import plot_fullset as P                                           # noqa: E402
from val_two_tables import load, epochs_of, ACC                    # noqa: E402

RESTRICT = "training/valfull83/pool_noleak.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("rids", nargs="+")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}' for r in csv.DictReader(open(RESTRICT))}

    eps = {}
    for rid in a.rids:
        for e in epochs_of(rid):
            d = load(rid, e)
            if d:
                eps[e] = d                     # a later rid wins a shared epoch, by argument order
    if not eps:
        sys.exit("no val row csvs found for " + ", ".join(a.rids))

    keys = sorted(set.intersection(*[set(d) for d in eps.values()]) & keep)
    n = len(keys)
    if not n:
        sys.exit("no interfaces survive the restrict set")

    def top1(d, idx):
        return [max(d[k], key=lambda t: t[idx])[3] for k in keys]

    # WINNABLE INTERFACES ONLY FOR THE RANK PANEL. Rank is undefined where no candidate is
    # acceptable; censoring those at n+1 would make the mean a function of the candidate count.
    win = [k for k in keys if any(t[3] >= ACC for t in eps[min(eps)][k])]

    def rank(d, idx):
        out = []
        for k in win:
            order = sorted(d[k], key=lambda t: -t[idx])
            out.append(next(j for j, t in enumerate(order, 1) if t[3] >= ACC))
        return statistics.fmean(out)

    E = sorted(eps)
    series = {}
    for pname, idx in (("max", 1), ("mean", 2)):
        dq, fa, rk = [], [], []
        for e in E:
            v = top1(eps[e], idx)
            dq.append(statistics.fmean(v))
            fa.append(100.0 * sum(1 for x in v if x >= ACC) / n)
            rk.append(rank(eps[e], idx))
        series[pname] = (dq, fa, rk)
    sb = top1(eps[min(eps)], 0)
    ship = (statistics.fmean(sb), 100.0 * sum(1 for x in sb if x >= ACC) / n,
            rank(eps[min(eps)], 0))

    fig, axs = plt.subplots(2, 3, figsize=(18.0, 9.0))
    colour = {"max": "tab:blue", "mean": "tab:green"}
    fmt = (lambda v: f"shipped {v:.4f}", lambda v: f"shipped {v:.1f}%",
           lambda v: f"shipped {v:.2f}")
    titles = ("mean DockQ of the selected pose",
              "fraction acceptable (DockQ >= %.2f)" % ACC,
              "mean rank of 1st acceptable  (LOWER IS BETTER)")
    ylabs = ("mean DockQ (top-1)", "frac acceptable (top-1), %",
             "mean rank of 1st acceptable pose")
    for i, pname in enumerate(("max", "mean")):
        for j in range(3):
            ax = axs[i][j]
            ax.plot(E, series[pname][j], "-o", color=colour[pname], lw=1.9, ms=5,
                    label=f"{pname}-pooled ipTM")
            ax.axhline(ship[j], color="0.35", lw=1.2, ls=":", label=fmt[j](ship[j]))
            ax.set_xlabel("epoch")
            ax.set_ylabel(ylabs[j])
            ax.grid(alpha=.25)
            # ONE TICK PER EPOCH IS UNREADABLE PAST ~20 PASSES, so thin rather than rotate.
            ax.set_xticks(E if len(E) <= 20 else E[::max(1, len(E) // 14)])
            ax.legend(fontsize=9)
            ax.set_title(f"{pname}-pooled -- {titles[j]}", fontsize=10)
    fig.suptitle(a.title or " + ".join(a.rids) + f"  --  val83 leak-free, n={n}", y=.99)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}: epochs {E[0]}-{E[-1]} ({len(E)} pass(es)), n={n} "
          f"({len(win)} winnable), shipped {ship[0]:.4f} / {ship[1]:.1f}% / rank {ship[2]:.2f}")


if __name__ == "__main__":
    main()
