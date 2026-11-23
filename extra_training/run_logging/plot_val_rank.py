#!/usr/bin/env python
"""MEAN RANK OF THE FIRST ACCEPTABLE POSE, held out, across epochs, for several arms.

    plot_val_rank.py <out.png> "<label>=<rid>[+<rid2>...]" ... [--title T]

WHY THIS METRIC. top-1 DockQ and frac acceptable only move when the pick crosses a threshold, so
a reordering that lifts an acceptable pose from rank 40 to rank 5 is invisible in both. This is
the 1-based position of the first acceptable pose (DockQ >= 0.23) in the arm's own ranking, mean
over interfaces -- the quantity the selector controls, measured whether or not the improvement
reaches the top slot. On the in-sample pool it is the only statistic in this project that has
cleared significance (t = -3.68 at n=885) while every DockQ column stayed inside the noise.

LOWER IS BETTER AND THE AXIS IS LEFT ALONE. It was inverted at first so that "up = better"
matched the DockQ figures, and that was the wrong call: a reader looks at the numbers before the
caption, and an inverted axis silently reverses every comparison they make in the first second.
The axis now runs normally, so the best arm is the LOWEST band. --invert restores the old
behaviour for anyone who wants it.

ONE COMMON INTERFACE SET FOR EVERY ARM AND EVERY EPOCH, intersected across all of them and then
restricted to the 139 leak-free interfaces and to those with at least one acceptable pose among
their candidates. Rank is undefined without an acceptable pose, and a set that shifted between
arms would move the baseline with it.

SHIPPED COMES FROM THE ROWS via stored_iptm and is recomputed per arm; the figure prints the
spread rather than assuming they agree, because each arm's shards are its own.
"""
import argparse, csv, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
from val_two_tables import load, epochs_of, ACC                    # noqa: E402

RESTRICT = "training/valfull83/pool_noleak.csv"
COLOURS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("arms", nargs="+", help='"label=rid" or "label=rid1+rid2"')
    ap.add_argument("--title", default="")
    ap.add_argument("--invert", action="store_true",
                    help="flip the y axis so up = better; off by default")
    a = ap.parse_args()

    keep = {f'{r["pdb"]}_{r["chain1"]}{r["chain2"]}' for r in csv.DictReader(open(RESTRICT))}

    arms = []
    for spec in a.arms:
        label, rids = spec.split("=", 1)
        eps = {}
        for rid in rids.split("+"):
            for e in epochs_of(rid):
                d = load(rid, e)
                if d:
                    eps[e] = d
        if not eps:
            print(f"  {label}: no val rows; skipped", file=sys.stderr)
            continue
        arms.append((label, eps))
    if not arms:
        sys.exit("no arm had val rows")

    common = set.intersection(*[set(d) for _, eps in arms for d in eps.values()]) & keep
    probe = arms[0][1][min(arms[0][1])]
    keys = sorted(k for k in common if any(t[3] >= ACC for t in probe[k]))
    n = len(keys)
    if not n:
        sys.exit("no winnable interface survives the intersection")

    def mean_rank(d, idx):
        out = []
        for k in keys:
            order = sorted(d[k], key=lambda t: -t[idx])
            out.append(next(j for j, t in enumerate(order, 1) if t[3] >= ACC))
        return statistics.fmean(out)

    ships = [mean_rank(eps[min(eps)], 0) for _, eps in arms]
    fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.4))
    for j, (pname, idx) in enumerate((("max", 1), ("mean", 2))):
        for k, (label, eps) in enumerate(arms):
            E = sorted(eps)
            ax[j].plot(E, [mean_rank(eps[e], idx) for e in E], "-o",
                       color=COLOURS[k % len(COLOURS)], lw=1.9, ms=4, label=label)
        ax[j].axhline(statistics.fmean(ships), color="0.35", lw=1.2, ls=":",
                      label=f"shipped {statistics.fmean(ships):.2f}")
        if a.invert:
            ax[j].invert_yaxis()
        ax[j].set_xlabel("epoch")
        ax[j].set_ylabel("mean rank of 1st acceptable pose\n(LOWER IS BETTER)")
        ax[j].set_title(f"{pname}-pooled ipTM")
        ax[j].grid(alpha=.25)
        ax[j].legend(fontsize=9)
    fig.suptitle(a.title or f"val83 leak-free, {n} winnable interfaces, 150 candidates each",
                 y=.99)
    fig.tight_layout()
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}: n={n} winnable of {len(common)} common; "
          f"shipped per arm {['%.2f' % s for s in ships]}")
    for label, eps in arms:
        E = sorted(eps)
        print(f"  {label}: epochs {E[0]}-{E[-1]} ({len(E)} pass(es))")


if __name__ == "__main__":
    main()
