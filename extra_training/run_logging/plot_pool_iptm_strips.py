#!/usr/bin/env python
"""All 150 poses' ipTM for one interface, shipped beside ours, with the best-DockQ pose marked.

Answers "did it raise the good pose or push the bad ones down" by showing the whole pool rather
than a summary: the bulk is the cloud, the good pose is the star, and the two heads sit on the
same axis because both were scored on the IDENTICAL ground-truth crop.

RAW ipTM IS COMPARABLE ACROSS HEADS ONLY BECAUSE OF THAT. Same crop, same d0, same poses -- the
only difference is the weights. It is NOT comparable across interfaces, so each panel is its own.
"""
import argparse, csv, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT = "7qu2_BC,7qu2_AC,9fvb_BD,9lme_AD,9cn2_AB,7tzh_AC"

ap = argparse.ArgumentParser()
ap.add_argument("--ifaces", default=DEFAULT)
ap.add_argument("--ours", default="training/runs/cropped_eval/valfull_ours_ep30")
ap.add_argument("--shipped", default="training/runs/cropped_eval/valfull_shipped")
ap.add_argument("--col", default="iptm", choices=("iptm", "iptm_mean"))
ap.add_argument("--out", default="figures/pool_iptm_strips.png")
a = ap.parse_args()

IFS = [s for s in a.ifaces.split(",") if s]


def load(d, col):
    by = {}
    fs = sorted(glob.glob(os.path.join(ROOT, d, "rows_b30_*.csv")))
    if not fs:
        raise SystemExit(f"STOP: no rows under {d}")
    for f in fs:
        for r in csv.DictReader(open(f)):
            by.setdefault(r["iface"], []).append((float(r[col]), float(r["dockq"])))
    return by


O, S = load(a.ours, a.col), load(a.shipped, a.col)
missing = [i for i in IFS if i not in O or i not in S]
if missing:
    raise SystemExit(f"STOP: not in both passes: {missing}")

ncol = 3
nrow = int(np.ceil(len(IFS) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.1 * nrow), squeeze=False)
rng = np.random.default_rng(0)

for k, iface in enumerate(IFS):
    ax = axes[k // ncol][k % ncol]
    o = np.array([p[0] for p in O[iface]]); s = np.array([p[0] for p in S[iface]])
    dq = np.array([p[1] for p in O[iface]])
    if not np.allclose(dq, [p[1] for p in S[iface]]):
        raise SystemExit(f"STOP: {iface} DockQ differs between passes -- the join is wrong")
    g = int(np.argmax(dq))

    for xpos, sc, nm in ((0, s, "shipped"), (1, o, "ours ep30")):
        jit = xpos + rng.uniform(-0.17, 0.17, len(sc))
        ax.scatter(jit, sc, s=13, c=dq, cmap="viridis", vmin=0, vmax=1,
                   alpha=0.72, lw=0, zorder=2)
        ax.scatter([xpos], [sc[g]], s=185, marker="*", c="#d62728",
                   edgecolor="white", lw=0.8, zorder=4)
        ax.scatter([xpos], [np.median(sc)], s=52, marker="_", c="black", lw=2.0, zorder=5)
        r = int((sc > sc[g]).sum()) + 1
        ax.annotate(f"rank {r}", (xpos, sc[g]), textcoords="offset points",
                    xytext=(11, -3), fontsize=8, color="#d62728", fontweight="bold")

    ax.plot([0, 1], [s[g], o[g]], color="#d62728", lw=1.0, ls="--", alpha=0.55, zorder=3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["shipped", "ours ep30"], fontsize=9)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylabel("ipTM (MAX-pooled)" if a.col == "iptm" else "ipTM (MEAN-pooled)", fontsize=8.5)
    nok = int((dq >= 0.23).sum())
    # two lines: at 3 columns a one-line title ran into its neighbour
    ax.set_title(f"{iface}\nbest DockQ {dq[g]:.3f}   {nok}/150 acceptable", fontsize=9.5)
    ax.grid(alpha=0.18, lw=0.6, axis="y")

for k in range(len(IFS), nrow * ncol):
    axes[k // ncol][k % ncol].axis("off")

sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 1))
cb = fig.colorbar(sm, ax=axes, fraction=0.019, pad=0.015)
cb.set_label("true DockQ of the pose", fontsize=9)

fig.suptitle("Every candidate's ipTM, identical ground-truth crop on both heads\n"
             "red star = the pool's best pose   |   black dash = pool median",
             fontsize=11.5)
op = os.path.join(ROOT, a.out)
os.makedirs(os.path.dirname(op), exist_ok=True)
fig.savefig(op, dpi=160, bbox_inches="tight")
print(f"  {len(IFS)} interface(s) -> {a.out}")
