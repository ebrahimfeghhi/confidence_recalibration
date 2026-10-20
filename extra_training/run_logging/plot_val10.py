#!/usr/bin/env python
"""Three panels for a run scored on held-out validation structures: train loss, then the two metrics.

SEPARATE FROM plot_train_eval.py ON PURPOSE. That plot overlays every arm of the 30-structure
probe, and those arms were scored on the pool they TRAIN on. These numbers come from
training/val10/pool.csv -- 10 validation structures the run never sees -- so putting them on the
same axes would invite exactly the comparison that is not valid.

Panel 1 is the training loss: pose-weighted PAE cross-entropy per epoch, which is what the run is
actually minimising. Panels 2 and 3 are the held-out metrics, mean DockQ and fraction acceptable,
each with the shipped head and the oracle drawn in so the distance left is visible.

  run_logging/plot_val10.py e4tvxe --out training/runs/val10_plot.png
"""
import argparse
import collections
import csv
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The eval reuses the train-eval code path, so it still prints "TRAIN:". The numbers are not.
LINE = re.compile(r"epoch (\d+) TRAIN: mean_dockq ([0-9.]+)\s+frac_acc ([0-9.]+)")
COLORS = ["#000000", "#e67e22", "#1f77b4", "#16a085", "#c0392b"]


def metrics(root, rid):
    """One curve across a JOINED lineage: "a+b" reads both logs into one series.

    A run relaunched to pick up a code change is still one experiment -- f8rqck resumed e4tvxe at
    epoch 6 with identical settings -- and plotting it as a second curve starting at epoch 7 would
    read as a new arm.
    """
    xs, dq, fa = [], [], []
    for part in rid.split("+"):
        log = os.path.join(root, part, "run.log")
        if not os.path.exists(log):
            continue
        for m in LINE.finditer(open(log, errors="replace").read()):
            xs.append(int(m.group(1))); dq.append(float(m.group(2))); fa.append(float(m.group(3)))
    if not xs:
        return [], [], []
    # one point per epoch, last write wins -- a resumed run can repeat an epoch line
    byep = {}
    for e, d, f in zip(xs, dq, fa):
        byep[e] = (d, f)
    ks = sorted(byep)
    return ks, [byep[k][0] for k in ks], [byep[k][1] for k in ks]


def pae(root, rid, full_steps):
    """Pose-weighted PAE per epoch. WEIGHTED BY batch_n, not a mean of the per-step means.

    The last step of an epoch is a remainder -- 72 poses against 84 -- so averaging step means
    gives it equal weight for a fraction of the data. That bias once read an epoch as a run low
    when the pose-weighted number was flat, which is how a stop rule and a curve ended up
    disagreeing in sign.
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
    ap.add_argument("runs", nargs="+", help="run id or run_id=label")
    ap.add_argument("--root", default="training/runs")
    ap.add_argument("--full-steps", type=int, default=18,
                    help="steps that make an epoch complete; incomplete epochs are dropped from "
                         "the loss panel because a partial mean is not comparable")
    # The val10 reference numbers, computed from validation/fold/results.csv over the 18
    # interfaces of the 10 highest-headroom structures. Defaults rather than magic numbers in the
    # code so a different pool can be plotted without editing this file.
    ap.add_argument("--shipped-dockq", type=float, default=0.2626)
    ap.add_argument("--shipped-frac", type=float, default=0.5000)
    ap.add_argument("--oracle-dockq", type=float, default=0.6662)
    ap.add_argument("--oracle-frac", type=float, default=1.0000)
    ap.add_argument("--title", default="held-out val10: 10 validation structures, 18 interfaces")
    ap.add_argument("--out", default="training/runs/val10_plot.png")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))
    any_pts = False
    for ci, spec in enumerate(a.runs):
        rid, _, label = spec.partition("=")
        label = label or rid
        c = COLORS[ci % len(COLORS)]
        px, py = pae(a.root, rid, a.full_steps)
        if px:
            axes[0].plot(px, py, marker="o", ms=4.5, lw=1.8, color=c, label=label)
            any_pts = True
        ex, dq, fa = metrics(a.root, rid)
        for ax, ys in ((axes[1], dq), (axes[2], fa)):
            if ex:
                ax.plot(ex, ys, marker="o", ms=5, lw=1.8, color=c, label=label)
                any_pts = True

    axes[0].set_title("training loss: PAE cross-entropy (pose-weighted)")
    axes[0].set_ylabel("PAE loss")
    axes[1].set_title("held-out mean DockQ")
    axes[1].set_ylabel("mean DockQ")
    axes[2].set_title("held-out fraction acceptable (DockQ >= 0.23)")
    axes[2].set_ylabel("frac acceptable")
    for ax, ship, orac in ((axes[1], a.shipped_dockq, a.oracle_dockq),
                           (axes[2], a.shipped_frac, a.oracle_frac)):
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
    fig.suptitle(a.title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(a.out, dpi=130)
    print(f"wrote {a.out}" + ("" if any_pts else "  (NO DATA YET -- no complete epoch or eval)"))


if __name__ == "__main__":
    main()
