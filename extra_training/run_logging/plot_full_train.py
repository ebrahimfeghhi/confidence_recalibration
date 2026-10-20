#!/usr/bin/env python
"""Four panels for the full-training-set run: train PAE, val PAE, val mean DockQ, val frac-acceptable.

WHY FOUR AND NOT THREE. plot_val10.py has the three the older runs could produce. The fourth --
held-out PAE -- is the one that mattered and did not exist: measured 2026-10-20 on the
30-structure anchor, training PAE fell 38% from epoch 1 to 142 while val10 PAE ROSE 51%
(2.8290 -> 4.2639, against 2.9756 for the shipped head), and the DockQ panels showed nothing
because ipTM barely orders poses within an interface -- Spearman 0.20 shipped, 0.10 trained. A plot
of panels 1, 3 and 4 alone was consistent with a run that was working.

PANEL 2 IS THE ONE TO READ. Panel 1 is what the optimiser minimises and is expected to fall on any
amount of data. Panels 3 and 4 are the goal but are noisy at 18 interfaces. Panel 2 is the same
quantity as panel 1 on structures the run never sees, so the gap between panels 1 and 2 IS the
overfit, visible directly.

The shipped-head line on panel 2 is drawn because "val PAE falling" is not the bar -- it has to
beat the untrained head, and on 30 structures it only did so for the first handful of epochs.

  run_logging/plot_full_train.py ydikys --out training/runs/full_train_plot.png
"""
import argparse
import collections
import csv
import os
import re

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_io as MIO

# The held-out DockQ eval reuses the train-eval code path, so it still prints "TRAIN:".
COLORS = ["#1f77b4", "#e67e22", "#16a085", "#c0392b", "#000000"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run id or run_id=label; a+b joins a lineage")
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--full-steps", type=int, default=250,
                    help="steps that make an epoch COMPLETE; incomplete epochs are dropped from "
                         "panel 1 because a partial mean is not comparable. 488 structures x 10 "
                         "seeds x 5 poses at 84/step is ~290, so 250 admits a short last shard "
                         "while still excluding a half-finished epoch.")
    # BUDGET 30, which is NOT the budget-100 pool. Derived from validation/fold/results.csv over
    # the same 18 interfaces: at 100 seeds the baseline is 0.2626 / 0.5000 and the oracle
    # 0.6662 / 1.0000. Quoting one against the other is a real error and these are the 30-seed set.
    ap.add_argument("--budget", type=int, default=30,
                    help="the seed budget the reference lines belong to; a metric json recorded at "
                         "a different budget is flagged in the subtitle rather than plotted silently")
    ap.add_argument("--shipped-dockq", type=float, default=0.2909)
    ap.add_argument("--shipped-frac", type=float, default=0.5556)
    ap.add_argument("--oracle-dockq", type=float, default=0.5968)
    ap.add_argument("--oracle-frac", type=float, default=1.0000)
    # The shipped head's held-out PAE, measured by analysis/oneoff/06_val10_loss_shipped.sh.
    ap.add_argument("--shipped-val-pae", type=float, default=2.9756)
    # EPOCH 0 IS THE SHIPPED HEAD, and plotting it as a POINT rather than only as a dashed line
    # makes each curve start where the untrained head sits -- so "is training helping at all" is
    # read off the first segment instead of inferred from a line's position.
    #
    # PANEL 1 HAS NO EPOCH-0 POINT and is not given a default. The training-set PAE of the
    # shipped head over these 488 structures has never been measured: it needs a --loss-only pass
    # over all 4880 units, ~24,400 poses, which is hours rather than the 3 minutes val10 costs.
    # A number borrowed from a different structure set would not be the same quantity, and
    # inventing one to make the panel symmetric is exactly the error the hardcoded 0.5775
    # baseline was. Pass --shipped-train-pae once it is measured.
    ap.add_argument("--shipped-train-pae", type=float, default=None,
                    help="the SHIPPED head's pose-weighted PAE over the TRAINING set. Unmeasured; "
                         "panel 1 shows no epoch-0 point until this is given.")
    ap.add_argument("--title", default="full training set (488 structures) -- held-out val10: "
                                      "10 validation structures, 18 interfaces, budget 30")
    ap.add_argument("--out", default="training/runs/full_train_plot.png")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 4, figsize=(22, 4.6))
    notes = []
    for ci, spec in enumerate(a.runs):
        rid, _, label = spec.partition("=")
        label = label or rid
        c = COLORS[ci % len(COLORS)]
        # EPOCH 0 IS THE FIRST POINT OF THE CURVE, not a separate symbol. The shipped head is
        # what every one of these metrics starts from, so joining it to epoch 1 with the same
        # line makes the first segment answer "did training help at all" directly -- an isolated
        # marker leaves the reader to judge a distance by eye.
        #
        # Panel 1 is not prepended: the shipped head's TRAIN PAE over these 488 structures is
        # unmeasured, and a fabricated start would misstate the one panel the optimiser drives.
        def _with_ep0(xs, ys, base):
            # WITH NO TRAINED EPOCHS YET, the epoch-0 point is still drawn on its own. The
            # baseline is a measurement, not a consequence of the run, so a panel waiting for its
            # first eval should show where the untrained head sits rather than render blank.
            if base is None:
                return xs, ys
            if not xs:
                return [0], [base]
            if xs[0] == 0:
                return xs, ys
            return [0] + list(xs), [base] + list(ys)

        M = MIO.load(a.root, rid, a.budget, a.full_steps)
        for _p, _why in M.rejected:
            notes.append(f"REJECTED {os.path.basename(_p)}: {_why}")
        _te = sorted(M.train_pae)
        tx, ty = _with_ep0(_te, [M.train_pae[e] for e in _te], a.shipped_train_pae)
        if tx:
            axes[0].plot(tx, ty, marker="o", ms=4.5, lw=1.8, color=c, label=label)
        _ve = sorted(M.val_pae)
        vx = _ve
        vy = [M.val_pae[e][0] for e in _ve]
        vn = [M.val_pae[e][1] for e in _ve]
        _x, _y = _with_ep0(vx, vy, a.shipped_val_pae)
        if _x:
            axes[1].plot(_x, _y, marker="o", ms=4.5, lw=1.8, color=c,
                         label=label if vx else f"{label} (epoch 0 only)")
        if vx and len(set(vn)) > 1:
            notes.append(f"{label}: held-out pose count varies {min(vn)}-{max(vn)}")
        # A JSON AT ANOTHER BUDGET NEVER REACHES HERE: metrics_io rejects it and the rejection is
        # already in `notes` above. The plot used to warn and plot it anyway.
        ex = sorted(M.dockq)
        dq = [M.dockq[e][0] for e in ex]
        fa = [M.dockq[e][1] for e in ex]
        for ax, ys, base in ((axes[2], dq, a.shipped_dockq), (axes[3], fa, a.shipped_frac)):
            _x, _y = _with_ep0(ex, ys, base)
            if _x:
                ax.plot(_x, _y, marker="o", ms=5, lw=1.8, color=c,
                        label=label if ex else f"{label} (epoch 0 only)")

    EP0 = "#7f7f7f"
    if a.shipped_train_pae is None:
        # SAID, NOT LEFT BLANK. An empty epoch-0 slot on panel 1 reads as "no baseline exists"
        # rather than "this one costs hours and has not been paid for".
        axes[0].annotate("no epoch-0 point: the shipped head's TRAIN PAE\nover these 488 "
                         "structures is unmeasured\n(needs a --loss-only pass over 4,880 units)",
                         xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7.5, color=EP0,
                         va="bottom", ha="left")

    axes[0].set_title("1. TRAIN PAE (pose-weighted)\nwhat the optimiser minimises")
    axes[0].set_ylabel("PAE loss")
    axes[1].set_title("2. HELD-OUT PAE\nthe same quantity on unseen structures")
    axes[1].set_ylabel("PAE loss")
    axes[2].set_title("3. held-out mean DockQ")
    axes[2].set_ylabel("mean DockQ")
    axes[3].set_title("4. held-out frac acceptable (DockQ >= 0.23)")
    axes[3].set_ylabel("frac acceptable")

    # The shipped head on the held-out PAE panel: falling is not the bar, beating UNTRAINED is.
    axes[1].axhline(a.shipped_val_pae, ls="--", lw=1.1, color="#888888")
    axes[1].annotate(f"shipped head {a.shipped_val_pae:.4f}", xy=(1, a.shipped_val_pae),
                     xycoords=("axes fraction", "data"), xytext=(-4, 3),
                     textcoords="offset points", ha="right", va="bottom",
                     fontsize=8, color="#888888")
    for ax, ship, orac in ((axes[2], a.shipped_dockq, a.oracle_dockq),
                           (axes[3], a.shipped_frac, a.oracle_frac)):
        for v, lab, col in ((ship, "shipped head", "#888888"), (orac, "oracle", "#2ca02c")):
            ax.axhline(v, ls="--", lw=1.1, color=col)
            ax.annotate(f"{lab} {v:.4f}", xy=(1, v), xycoords=("axes fraction", "data"),
                        xytext=(-4, 3), textcoords="offset points",
                        ha="right", va="bottom", fontsize=8, color=col)
    for ax in (axes[1], axes[2], axes[3]):
        ax.annotate("epoch 0 = shipped head", xy=(0.02, 0.97), xycoords="axes fraction",
                    fontsize=7.5, color=EP0, va="top", ha="left")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.25)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    sub = a.title + ("   [" + "; ".join(notes) + "]" if notes else "")
    fig.suptitle(sub, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(a.out, dpi=130)
    have = [n for n, ax in zip("1234", axes) if ax.get_legend_handles_labels()[0]]
    print(f"wrote {a.out}" + (f"  (panels with data: {','.join(have)})" if have
                              else "  (NO DATA YET -- no complete epoch, no held-out pass)"))


if __name__ == "__main__":
    main()
