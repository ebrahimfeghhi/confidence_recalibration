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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The held-out DockQ eval reuses the train-eval code path, so it still prints "TRAIN:".
DOCKQ = re.compile(r"epoch (\d+) TRAIN: mean_dockq ([0-9.]+)\s+frac_acc ([0-9.]+)")
COLORS = ["#1f77b4", "#e67e22", "#16a085", "#c0392b", "#000000"]


def val_pae(root, rid):
    """Held-out PAE per epoch, from val_loss.csv -- written by --val-loss-every.

    Prefers the csv over scraping the log: the csv is written by the hook itself and carries the
    pose count, so a pass that measured fewer poses than intended is visible rather than silently
    averaged in.
    """
    byep = {}
    for part in rid.split("+"):
        p = os.path.join(root, part, "val_loss.csv")
        if not os.path.exists(p):
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                try:
                    byep[int(r["epoch"])] = (float(r["val_pae"]), int(r["n_poses"]))
                except (ValueError, KeyError, TypeError):
                    continue
    if not byep:
        return [], [], []
    ks = sorted(byep)
    return ks, [byep[k][0] for k in ks], [byep[k][1] for k in ks]


def dockq(root, rid):
    byep = {}
    for part in rid.split("+"):
        log = os.path.join(root, part, "run.log")
        if not os.path.exists(log):
            continue
        for m in DOCKQ.finditer(open(log, errors="replace").read()):
            byep[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    ks = sorted(byep)
    return ks, [byep[k][0] for k in ks], [byep[k][1] for k in ks]


def train_pae(root, rid, full_steps):
    """Pose-weighted PAE per epoch. WEIGHTED BY batch_n, never a mean of per-step means.

    An epoch's last step is a remainder, so averaging step means over-weights it. That bias once
    made a stop rule and a curve disagree in sign.
    """
    d = collections.defaultdict(lambda: [0.0, 0, 0])
    for part in rid.split("+"):
        traj = os.path.join(root, part, "trajectory.csv")
        if not os.path.exists(traj):
            continue
        with open(traj) as fh:
            for r in csv.DictReader(fh):
                if r.get("step_taken") != "1":
                    continue
                try:
                    e, v, n = int(r["epoch"]), float(r["batch_pae_loss"]), int(r["batch_n"])
                except (ValueError, KeyError, TypeError):
                    continue
                d[e][0] += v * n; d[e][1] += n; d[e][2] += 1
    xs = sorted(e for e, v in d.items() if v[2] >= full_steps and v[1] > 0)
    return xs, [d[e][0] / d[e][1] for e in xs]


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
    ap.add_argument("--shipped-dockq", type=float, default=0.2909)
    ap.add_argument("--shipped-frac", type=float, default=0.5556)
    ap.add_argument("--oracle-dockq", type=float, default=0.5968)
    ap.add_argument("--oracle-frac", type=float, default=1.0000)
    # The shipped head's held-out PAE, measured by analysis/oneoff/06_val10_loss_shipped.sh.
    ap.add_argument("--shipped-val-pae", type=float, default=2.9756)
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
        tx, ty = train_pae(a.root, rid, a.full_steps)
        if tx:
            axes[0].plot(tx, ty, marker="o", ms=4.5, lw=1.8, color=c, label=label)
        vx, vy, vn = val_pae(a.root, rid)
        if vx:
            axes[1].plot(vx, vy, marker="o", ms=4.5, lw=1.8, color=c, label=label)
            if len(set(vn)) > 1:
                notes.append(f"{label}: held-out pose count varies {min(vn)}-{max(vn)}")
        ex, dq, fa = dockq(a.root, rid)
        for ax, ys in ((axes[2], dq), (axes[3], fa)):
            if ex:
                ax.plot(ex, ys, marker="o", ms=5, lw=1.8, color=c, label=label)

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
