#!/usr/bin/env python
"""Held-out valfull selection: this arm against the shipped baseline, per eval epoch.

WHAT IS PLOTTED. Mean selected DockQ on the 92 valfull interfaces at budget 30, as a PAIRED
difference from the shipped head on the identical poses -- not two absolute curves, because the
two selectors see the same candidates and the paired difference has a much smaller error bar than
the difference of two means. The absolute values are annotated so the scale is still visible.

WHY THE SHIPPED LINE IS FLAT AT ZERO. It is the reference, not a measurement that moves: shipped
ipTM does not change between epochs. Its recorded absolute value is 0.5831 at b30 with oracle
0.6926 (paper_results.md section 4), and this script's own recomputation from the eval rows gives
0.5830 -- agreeing to 1e-4, which is the cross-check that the rows reproduce the record.

PRIOR ARMS ARE DRAWN AS BANDS, at their FINAL checkpoints, because this arm's epoch-5 number
sitting in their range is the honest comparison and two of them were positive:
  ip8ga4  +0.0073 +- 0.0040
  dntxmn  +0.0029 +- 0.0032
  tv6hue  -0.0084 +- 0.0054
Those are endpoints and this arm's are mid-run, so a point inside their band is not yet a match.

BOTH POOLINGS ARE SHOWN. The arm trains and selects by the MEAN, but the eval reports the MAX
from the same pass, and at epoch 5 the max was ahead (+0.0055 against +0.0027) -- the opposite of
what the design assumes. Hiding the max would hide that.

Usage:  python -I run_logging/plot_rel_heldout.py [run_id] [out.png]
"""
import csv
import glob
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RID = sys.argv[1] if len(sys.argv) > 1 else "o2qczq"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/rel_heldout_{RID}.png"
RUN = os.path.join(ROOT, "training", "runs", RID)
PRIOR = [("ip8ga4", 0.0073, 0.0040), ("dntxmn", 0.0029, 0.0032),
         ("tv6hue", -0.0084, 0.0054)]
SHIPPED_ABS = 0.5831          # paper_results.md section 4, valfull b30
ORACLE_ABS = 0.6926

rows = []
for d in sorted(glob.glob(os.path.join(RUN, "train_eval_shards_ep*"))):
    m = re.search(r"train_eval_shards_ep(\d+)$", d)
    if not m:
        continue
    ep = int(m.group(1))
    by = defaultdict(list)
    for f in glob.glob(os.path.join(d, "rows_b*_s*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                vm = r.get("iptm_mean")
                by[r["iface"]].append((float(r["iptm"]),
                                       float(vm) if vm not in (None, "") else None,
                                       float(r["stored_iptm"]) if r["stored_iptm"] != "" else None,
                                       float(r["dockq"])))
            except (ValueError, TypeError, KeyError):
                continue
    if not by:
        continue
    full = max({len(v) for v in by.values()}, default=0)
    keep = [i for i in sorted(by) if len(by[i]) == full
            and all(x[2] is not None for x in by[i])]
    if len(keep) < 10:
        continue
    have_mean = all(all(x[1] is not None for x in by[i]) for i in keep)
    pick = lambda j: np.array([max(by[i], key=lambda t: t[j])[3] for i in keep])  # noqa: E731
    base = pick(2)
    mx = pick(0)
    mn = pick(1) if have_mean else None
    orac = np.array([max(t[3] for t in by[i]) for i in keep])

    def pair(sel):
        if sel is None:
            return None, None
        d_ = sel - base
        return float(d_.mean()), float(d_.std(ddof=1) / math.sqrt(len(d_)))

    dmx, emx = pair(mx)
    dmn, emn = pair(mn)
    rows.append(dict(ep=ep, n=len(keep), cands=full, base=float(base.mean()),
                     orac=float(orac.mean()), dmx=dmx, emx=emx, dmn=dmn, emn=emn,
                     abs_mx=float(mx.mean()), abs_mn=None if mn is None else float(mn.mean())))

if not rows:
    sys.exit(f"  no train_eval_shards_ep*/rows_*.csv under {RID} yet")

# THE FORM FOLLOWS THE DATA. With one or two eval points a line chart is all furniture and no
# information -- the first version drew one errorbar point plus three overlapping grey bands with
# right-aligned labels colliding at the axis edge. Below three points this draws a plain
# horizontal comparison, which reads instantly at n=1 and is the right shape for "how does ours
# compare"; at three or more it becomes the trajectory it should be.
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white",
                     "xtick.labelsize": 9.5, "ytick.labelsize": 10})
last = rows[-1]

if len(rows) < 3:
    # ---- HORIZONTAL COMPARISON ----
    items = [(f"{nm}  (final)", v, e, "#aaaaaa") for nm, v, e in PRIOR]
    if last["dmx"] is not None:
        items.append((f"{RID} ep{last['ep']}  max pooling", last["dmx"], last["emx"], "#b03a3a"))
    if last["dmn"] is not None:
        items.append((f"{RID} ep{last['ep']}  MEAN pooling", last["dmn"], last["emn"], "#2f6fb0"))
    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    ys = range(len(items))
    for y, (nm, v, e, col) in zip(ys, items):
        ax.errorbar(v, y, xerr=e, color=col, marker="o", ms=7, capsize=4, lw=2)
        ax.annotate(f"{v:+.4f}", (v, y), color=col, fontsize=9.5, fontweight="bold",
                    xytext=(0, 9), textcoords="offset points", ha="center")
    ax.axvline(0.0, color="#333333", lw=1.3)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([nm for nm, _v, _e, _c in items])
    ax.set_ylim(-0.7, len(items) - 0.3)
    ax.set_xlabel("mean selected DockQ minus shipped ipTM   (paired, same poses)")
    ax.grid(axis="x", color="#eeeeee", lw=0.8)
    ax.set_title(f"held-out valfull, {last['n']} interfaces at budget 30\n"
                 f"shipped {last['base']:.4f}   oracle {last['orac']:.4f}   "
                 f"(oracle is +{last['orac'] - last['base']:.4f})")
else:
    # ---- TRAJECTORY ----
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    eps = [r["ep"] for r in rows]
    for nm, v, e in PRIOR:
        ax.axhline(v, color="#bbbbbb", lw=0.9, ls=":")
        ax.annotate(nm, (eps[-1], v), fontsize=8, color="#999999", va="center",
                    xytext=(8, 0), textcoords="offset points")
    ax.axhline(0.0, color="#333333", lw=1.3)
    ax.annotate("shipped", (0.01, 0.0), xycoords=("axes fraction", "data"), fontsize=9,
                color="#333333", va="bottom")
    ax.errorbar(eps, [r["dmn"] for r in rows], yerr=[r["emn"] for r in rows],
                color="#2f6fb0", lw=2, marker="o", ms=5, capsize=3, label="MEAN pooling")
    ax.errorbar(eps, [r["dmx"] for r in rows], yerr=[r["emx"] for r in rows],
                color="#b03a3a", lw=1.4, ls="--", marker="s", ms=4, capsize=3, alpha=0.8,
                label="max pooling")
    ax.set_xlabel("epoch")
    ax.set_ylabel("DockQ minus shipped  (paired)")
    ax.set_title(f"{RID}: held-out valfull, {last['n']} interfaces at budget 30")
    ax.grid(color="#eeeeee", lw=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.legend(loc="lower left", fontsize=9)

fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)), exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=150, bbox_inches="tight")
print(f"  {len(rows)} eval epoch(s) -> {OUT}")
print(f"  {'ep':>3} {'n':>4} {'cand':>5} {'shipped':>8} {'MEAN':>8} {'dMEAN':>9} "
      f"{'t':>6} {'MAX':>8} {'dMAX':>9} {'t':>6}")
for r in rows:
    tm = (r["dmn"] / r["emn"]) if r["dmn"] is not None and r["emn"] else float("nan")
    tx = (r["dmx"] / r["emx"]) if r["emx"] else float("nan")
    print(f"  {r['ep']:>3} {r['n']:>4} {r['cands']:>5} {r['base']:>8.4f} "
          f"{(r['abs_mn'] if r['abs_mn'] is not None else float('nan')):>8.4f} "
          f"{(r['dmn'] if r['dmn'] is not None else float('nan')):>+9.4f} {tm:>+6.2f} "
          f"{r['abs_mx']:>8.4f} {r['dmx']:>+9.4f} {tx:>+6.2f}")
