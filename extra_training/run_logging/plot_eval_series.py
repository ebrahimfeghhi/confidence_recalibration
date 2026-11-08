#!/usr/bin/env python
"""Held-out curves across epochs: 83 structures / 141 interfaces, budget 30.

All the data definitions live in run_logging/eval_curve.py; this file only draws.
Panels share one legend below the figure. The oracle is stated per panel rather than drawn,
because it sits ~0.15 above everything and including it collapses the differences being
measured -- thousandths -- into an unreadable sliver.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_curve as EC
import plot_three

RID = sys.argv[1] if len(sys.argv) > 1 else "suz564"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/eval_series_{RID}.png"
CHAIN = list(plot_three.run_chain(RID))
S, SER = EC.shipped(), EC.series(CHAIN)
RHO, RHO_SHIP = EC.rho_series(CHAIN)
if not SER:
    raise SystemExit(f"STOP: no 83-interface metric json for the chain {CHAIN}")

C_MEAN, C_MAX, C_SHIP = "tab:blue", "tab:orange", "#444"
plt.rcParams.update({"font.size": 11})


def span(v, pad=.25):
    lo, hi = min(v), max(v)
    if hi - lo < 1e-9:
        lo, hi = lo - 5e-4, hi + 5e-4
    return lo - (hi - lo) * pad, hi + (hi - lo) * pad


def panel(a, xs, mean_y, max_y, ship_mean, ship_max, ylab, title):
    vals = []
    for ys, col, ms in ((mean_y, C_MEAN, 11), (max_y, C_MAX, 7)):
        pts = [(x, y) for x, y in zip(xs, ys) if y == y]
        if pts:
            a.plot(*zip(*pts), "-s", lw=1.8, ms=ms, color=col, mec=col,
                   mfc="none" if ms > 9 else col, mew=2)
            vals += [y for _x, y in pts]
    for v, ls in ((ship_mean, "-"), (ship_max, "--")):
        if v is not None and v == v:
            a.axhline(v, color=C_SHIP, ls=ls, lw=1.6)
            vals.append(v)
    if vals:
        a.set_ylim(*span(vals))
    a.set_xlabel("epoch"); a.set_ylabel(ylab); a.set_title(title, fontsize=11.5)
    a.grid(alpha=.3); a.set_xticks(xs); a.set_xlim(min(xs) - .6, max(max(xs), min(xs) + 1) + .6)


eps = sorted(SER)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
panel(ax[0], eps, [SER[e].get("mean_dockq") for e in eps],
      [SER[e].get("mean_dockq_select_max") for e in eps],
      S and S["mean_dockq"], S and S["mean_dockq_select_max"],
      "mean DockQ", "mean DockQ of the selected pose"
      + (f"\noracle {S['oracle_mean_dockq']:.4f}" if S else ""))
panel(ax[1], eps, [SER[e].get("frac_acc") for e in eps],
      [SER[e].get("frac_acc_select_max") for e in eps],
      S and S["frac_acc"], S and S["frac_acc_select_max"],
      "frac acceptable", r"fraction acceptable (DockQ $\geq$ 0.23)"
      + (f"\noracle {S['oracle_frac_acc']:.4f}" if S else ""))
re_ = sorted(RHO)
panel(ax[2], re_ or eps, [RHO[e]["mean"] for e in re_] or [float("nan")] * len(eps),
      [RHO[e]["max"] for e in re_] or [float("nan")] * len(eps),
      None, RHO_SHIP, "median within-interface Spearman",
      "ranking quality\nshipped reference is max-pooled only")

fig.legend(handles=[
    Line2D([], [], ls="none", marker="s", ms=11, mec=C_MEAN, mfc="none", mew=2,
           label="arm, mean-pooled ipTM  (what it selects on)"),
    Line2D([], [], ls="none", marker="s", ms=7, color=C_MAX, label="arm, max-pooled ipTM"),
    Line2D([], [], ls="-", lw=1.6, color=C_SHIP, label="shipped, mean-pooled"),
    Line2D([], [], ls="--", lw=1.6, color=C_SHIP, label="shipped, max-pooled")],
    loc="lower center", ncol=4, frameon=False, fontsize=10.5, bbox_to_anchor=(.5, -.01))
fig.suptitle(f"Held out: 83 structures / 141 interfaces, budget 30 seeds  --  {RID}", fontsize=13)
fig.tight_layout(rect=[0, .07, 1, .92])
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"-> {OUT}")
for e in eps:
    d = SER[e]
    print(f"  ep{e}: mean {d['mean_dockq']:.4f}  max {d['mean_dockq_select_max']:.4f}  "
          f"frac {d['frac_acc']:.4f}" + ("  (combined 92+49)" if d.get("_combined") else ""))
if S:
    print(f"  shipped: mean {S['mean_dockq']:.4f}  max {S['mean_dockq_select_max']:.4f}  "
          f"frac {S['frac_acc']:.4f}")
for e in re_:
    print(f"  rho ep{e}: mean-pooled {RHO[e]['mean']:+.4f}  max-pooled {RHO[e]['max']:+.4f}  "
          f"(shipped {RHO_SHIP:+.4f})")
