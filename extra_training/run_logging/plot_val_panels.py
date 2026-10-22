#!/usr/bin/env python
"""SIX HELD-OUT PANELS, PLUS TWO TRAIN-SIDE ONES KEPT SEPARATE AND LABELLED AS SUCH.

Ordered from the objective the optimiser sees to the number that would go in a paper. All val10,
budget 30, 18 interfaces.

  1. held-out PAE          the FT3 objective, on unseen structures. Lower is better.
  2. held-out RANK LOSS    the --rank-loss objective, on unseen structures. The trainer's own
                           per-epoch line is a TRAIN number; this is its held-out twin, so the
                           generalisation gap on the new objective is visible rather than assumed.
                           Lower is better; 0 is no alignment.
  3. frac acceptable       fraction of interfaces whose SELECTED pose clears DockQ 0.23, with the
                           standard error over the 18.
  4. mean DockQ            mean over interfaces of the selected pose's DockQ, with the standard
                           error over the 18.
  5. rho(ipTM, DockQ)      mean within-interface Spearman over all ~150 poses -- the ordering the
                           argmax consumes.
  6. within-seed TOP-1     for each seed group of 5, does the highest-ipTM pose also hold the
                           highest DockQ? Fraction of groups where it does, averaged over
                           interfaces. 0.20 is random choice among 5. Higher is better.

ALL FOUR INTERFACE-AVERAGED PANELS CARRY A BAND (3, 4, 5, 6). Banding two and not the other two
would imply the unbanded ones were exact; rho and top-1 are means over the same 18 interfaces and
carry the same kind of error.

THE BAND IS THE STANDARD ERROR, not the standard deviation. The sd over 18 interfaces is ~0.26 for
DockQ, which says only that interfaces differ in difficulty -- never in doubt, and it would swamp
the figure. sd/sqrt(18) answers the question actually being asked of panels 3 and 4: is an
epoch-to-epoch move larger than the noise in a mean over 18 interfaces?

EVERY REFERENCE LINE IS DERIVED, not passed, except the two oracle lines: the shipped head's rho,
rank loss and top-1 come from the `stored_iptm` column of each run's own shards, so the baseline
and the curve are the same measurement on the same interfaces.

ANNOTATIONS LIVE IN THE TITLES, not inside the axes. In-axes notes collided with the legends.

PANELS 7 AND 8 ARE THE ONLY TRAIN-SIDE ONES and are marked [TRAIN] so they cannot be mistaken for
held-out numbers. 7 overlays train and held-out PAE because train PAE alone says little -- what it
is for is the DISTANCE to held-out, which is panel 8. For scale: wpg274 reached a gap of 1.26 while
its held-out PAE went ABOVE the untrained head's.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_io as MIO
import rank_rho as RHO

COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
# rank_rho column indices, by name, so a reordering of COLS cannot silently mis-plot a panel.
IX = {n: i for i, n in enumerate(RHO.COLS)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run_id=label")
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--budget", type=int, default=30)
    ap.add_argument("--shipped-val-pae", type=float, default=2.9756)
    ap.add_argument("--oracle-dockq", type=float, default=0.5968)
    ap.add_argument("--oracle-frac", type=float, default=1.0)
    ap.add_argument("--title", default="HELD OUT -- val10: 10 structures, 18 interfaces, budget 30")
    ap.add_argument("--out", default="training/runs/val_panels.png")
    a = ap.parse_args()

    fig, ax = plt.subplots(2, 4, figsize=(26, 9.4))
    ax = ax.ravel()
    ship = {k: [] for k in ("rho", "rl", "top1", "dq", "fr")}
    for ci, spec in enumerate(a.runs):
        rid, _, label = spec.partition("=")
        label = label or rid
        c = COLORS[ci % len(COLORS)]
        M = MIO.load(a.root, rid, a.budget, 0)
        rr = RHO.load(a.root, rid, a.budget)

        def curve(k, xs, ys, base, err=None):
            """epoch 0 is the shipped head, joined to epoch 1 by the same line."""
            if base is not None:
                xs, ys = [0] + list(xs), [base] + list(ys)
                if err is not None:
                    err = [0.0] + list(err)
            if not xs:
                return
            ax[k].plot(xs, ys, marker="o", ms=4, lw=1.7, color=c, label=label)
            if err is not None:
                lo = [y - e for y, e in zip(ys, err)]
                hi = [y + e for y, e in zip(ys, err)]
                ax[k].fill_between(xs, lo, hi, color=c, alpha=0.18, lw=0)

        ve = sorted(M.val_pae)
        curve(0, ve, [M.val_pae[e][0] for e in ve], a.shipped_val_pae)
        # PANELS 7 AND 8: train side. Only the epochs holding BOTH metrics, so the gap is always
        # the same head's two numbers rather than two different epochs'.
        both = sorted(set(M.train_pae) & set(M.val_pae))
        if both:
            ax[6].plot(both, [M.train_pae[e] for e in both], marker="o", ms=4, lw=1.7,
                       color=c, label=f"{label} -- TRAIN")
            ax[6].plot(both, [M.val_pae[e][0] for e in both], marker="s", ms=3.5, lw=1.4,
                       ls="--", color=c, alpha=0.65, label=f"{label} -- held out")
            curve(7, both, [M.val_pae[e][0] - M.train_pae[e] for e in both], None)
        re = sorted(rr)
        if not re:
            continue
        f = rr[re[0]]
        ship["rl"].append(f[IX["rankloss_shipped"]])
        ship["rho"].append(f[IX["rho_shipped"]])
        ship["top1"].append(f[IX["top1_shipped"]])
        curve(1, re, [rr[e][IX["rankloss_trained"]] for e in re], f[IX["rankloss_shipped"]])
        curve(2, re, [rr[e][IX["frac_mean"]] for e in re], None,
              err=[rr[e][IX["frac_sem"]] for e in re])
        curve(3, re, [rr[e][IX["dockq_mean"]] for e in re], None,
              err=[rr[e][IX["dockq_sem"]] for e in re])
        curve(4, re, [rr[e][IX["rho_trained"]] for e in re], f[IX["rho_shipped"]],
              err=[rr[e][IX["rho_sem"]] for e in re])
        curve(5, re, [rr[e][IX["top1_trained"]] for e in re], f[IX["top1_shipped"]],
              err=[rr[e][IX["top1_sem"]] for e in re])

    def ref(k, v, lab, col, ls="--"):
        ax[k].axhline(v, ls=ls, lw=1.1, color=col)
        ax[k].annotate(f"{lab} {v:.4f}", xy=(0.995, v), xycoords=("axes fraction", "data"),
                       xytext=(0, 2), textcoords="offset points", ha="right", va="bottom",
                       fontsize=7.5, color=col)

    ref(0, a.shipped_val_pae, "shipped", "#888888")
    if ship["rl"]:
        ref(1, sum(ship["rl"]) / len(ship["rl"]), "shipped", "#888888")
    ax[1].axhline(0.0, ls=":", lw=1.0, color="#cccccc")
    ref(2, 0.5556, "shipped", "#888888"); ref(2, a.oracle_frac, "oracle", "#2ca02c")
    ref(3, 0.2909, "shipped", "#888888"); ref(3, a.oracle_dockq, "oracle", "#2ca02c")
    if ship["rho"]:
        ref(4, sum(ship["rho"]) / len(ship["rho"]), "shipped", "#888888")
    if ship["top1"]:
        ref(5, sum(ship["top1"]) / len(ship["top1"]), "shipped", "#888888")
    # THE RANDOM FLOOR on panel 6: one of five, chosen blind.
    ref(5, 0.2, "random 1 of 5", "#d62728", ls="-.")
    ax[7].axhline(0.0, ls=":", lw=1.0, color="#cccccc")

    titles = [
        ("1. held-out PAE  (lower better)", "PAE loss"),
        ("2. held-out RANK LOSS  (lower better, 0 = no alignment)", "weighted pairwise loss"),
        ("3. frac acceptable, DockQ >= 0.23  (band = s.e. over 18)", "frac acceptable"),
        ("4. mean DockQ of the SELECTED pose  (band = s.e. over 18)", "mean DockQ"),
        ("5. rho(ipTM, DockQ) within interface  (higher better, band = s.e.)",
         "mean Spearman"),
        ("6. within-seed TOP-1: best of 5 ranked first  (higher better, band = s.e.)",
         "fraction of seed groups"),
        ("7. [TRAIN] train PAE vs held-out PAE", "PAE loss"),
        ("8. [TRAIN] generalisation gap, held-out minus train", "PAE gap"),
    ]
    for k, (t, yl) in enumerate(titles):
        ax[k].set_title(t, fontsize=10.5)
        ax[k].set_ylabel(yl, fontsize=9)
        ax[k].set_xlabel("epoch   (0 = shipped head)", fontsize=9)
        ax[k].grid(alpha=0.25)
        ax[k].margins(x=0.02)
        if ax[k].get_legend_handles_labels()[0]:
            ax[k].legend(fontsize=8, loc="best", framealpha=0.9)
    fig.suptitle(a.title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
