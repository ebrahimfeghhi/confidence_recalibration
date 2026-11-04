#!/usr/bin/env python
"""train30's 77 interfaces vs the full 975: selected DockQ and fraction acceptable.

EVERYTHING IS SCORED BY ONE FUNCTION. jy6sah's rows and fawgtu's rows are both reduced with
phase_k_eval's canonical tie-break -- max by (score, -seed, -sample) -- and the canonical
threshold, dockq >= 0.23. plot_overfit30.py's own helper uses max(key=iptm) and a strict > 0.23,
which is the divergence that forced a round of numbers to be regenerated once already; mixing the
two conventions across panels would invent a difference between the runs that is not there.

BOTH SIDES ARE BUDGET 10, because the training fold holds only seeds 500000-500009. The two
populations are NOT two measurements of one thing: train30's 77 interfaces are a subset of the 975
chosen for the overfit experiment, and shipped scores 0.0420 on them against 0.4202 on the whole
pool -- they are the hard end. The headroom bar is drawn for exactly that reason.

jy6sah IS ABSENT FROM THE FULL-POOL PANEL, and that is a fact rather than an omission: it trained
on 30 structures and was never scored on the other 457, so there is no number to draw. The panel
says so rather than leaving a silent gap.

Usage:  python -I run_logging/plot_train30_vs_full.py [out.png]
"""
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 0.23


def canon(files, keep=None):
    """(n, model dockq, model frac, shipped dockq, shipped frac, oracle dockq, oracle frac)."""
    rows = {}
    for f in files:
        with open(f) as fh:
            for r in csv.DictReader(fh):
                if keep is not None and r["iface"] not in keep:
                    continue
                rows.setdefault(r["iface"], []).append(
                    (int(r["seed"]), int(r["sample"]), float(r["iptm"]),
                     float(r["dockq"]), float(r["stored_iptm"])))
    if not rows:
        sys.exit("  no rows matched")
    mdl = [max(v, key=lambda t: (t[2], -t[0], -t[1]))[3] for v in rows.values()]
    shp = [max(v, key=lambda t: (t[4], -t[0], -t[1]))[3] for v in rows.values()]
    orc = [max(t[3] for t in v) for v in rows.values()]
    fr = lambda a: sum(1 for x in a if x >= THRESH) / len(a)
    return (len(rows), float(np.mean(mdl)), fr(mdl), float(np.mean(shp)), fr(shp),
            float(np.mean(orc)), fr(orc))


keep30 = {f"{r['pdb']}_{r['chain1']}{r['chain2']}"
          for r in csv.DictReader(open(os.path.join(ROOT, "training/overfit30/pool.csv")))}
FAW = glob.glob(os.path.join(ROOT, "training/runs/fawgtu/traineval_ep35_b10/rows_b10_*.csv"))
JY = glob.glob(os.path.join(ROOT, "training/runs/jy6sah/train_eval_shards_ep136/rows_b*_*.csv"))

f30 = canon(FAW, keep30)
jy30 = canon(JY)
ffull = canon(FAW)
print(f"  train30: {f30[0]} interfaces (fawgtu), {jy30[0]} (jy6sah)   full: {ffull[0]}")
assert f30[0] == jy30[0] == 77, (f30[0], jy30[0])

# (label, dockq, frac) in draw order; None = not measured
GROUPS = [
    ("train30\n77 interfaces", [
        ("shipped", f30[3], f30[4], "#9e9e9e"),
        ("jy6sah\ntrained on these 30", jy30[1], jy30[2], "#1f7a1f"),
        ("fawgtu\ntrained on all 487", f30[1], f30[2], "#2f6fb0"),
        ("oracle", f30[5], f30[6], "#d9d9d9"),
    ]),
    ("full training set\n975 interfaces", [
        ("shipped", ffull[3], ffull[4], "#9e9e9e"),
        (None, None, None, None),                       # jy6sah: never scored here
        ("fawgtu\ntrained on all 487", ffull[1], ffull[2], "#2f6fb0"),
        ("oracle", ffull[5], ffull[6], "#d9d9d9"),
    ]),
]

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
W = 0.2
for col, (metric, ylab) in enumerate([(1, "mean selected DockQ"),
                                      (2, "fraction acceptable (DockQ >= 0.23)")]):
    A = ax[col]
    for gi, (gname, bars) in enumerate(GROUPS):
        for bi, (lab, dq, fa, color) in enumerate(bars):
            x = gi + (bi - 1.5) * W
            if lab is None:
                A.text(x, 0.012, "not\nscored", ha="center", va="bottom", fontsize=8,
                       color="#bbbbbb", style="italic")
                continue
            v = dq if metric == 1 else fa
            A.bar(x, v, W * 0.9, color=color,
                  edgecolor="#555555" if lab == "oracle" else "none",
                  linewidth=0.8, hatch="//" if lab == "oracle" else None,
                  label=lab.split("\n")[0] if gi == 0 else None)
            A.text(x, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5)
    A.set_xticks(range(len(GROUPS)))
    A.set_xticklabels([g[0] for g in GROUPS])
    A.set_ylabel(ylab)
    A.set_ylim(0, 1.0 if metric == 2 else 0.62)
    A.grid(axis="y", alpha=0.3)
    A.set_axisbelow(True)
    A.set_title(ylab)
ax[0].legend(loc="upper left", fontsize=9.5)
fig.suptitle("IN-SAMPLE selection, budget 10 both sides -- train30's 77 interfaces are the HARD "
             "end of the pool\nshipped scores 0.042 there against 0.420 over all 975, so the two "
             "panels' x-groups are not comparable to each other", fontsize=11, y=1.02)
fig.tight_layout()
out = sys.argv[1] if len(sys.argv) > 1 else "figures/train30_vs_full.png"
fig.savefig(os.path.join(ROOT, out), dpi=140, bbox_inches="tight")
print(f"  -> {out}")
for gname, bars in GROUPS:
    print(f"\n  {gname.replace(chr(10), ' ')}")
    for lab, dq, fa, _ in bars:
        if lab:
            print(f"    {lab.split(chr(10))[0]:>8}  dockq {dq:.4f}  frac {fa:.4f}")
