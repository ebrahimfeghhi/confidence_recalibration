#!/usr/bin/env python
"""Held-out valfull: shipped vs this arm's two poolings. Bars with per-interface dots.

THREE SELECTORS ON THE SAME POSES: the shipped head's ipTM, and this arm's score pooled by MEAN
(what it trains) and by MAX (the shipped readout). All three choose from the identical candidate
set, so the only difference is the scoring function.

LEFT: mean selected DockQ. The bar is the mean, the dots are the 92 individual interfaces. The
dots are the point -- a 0.003 difference in a mean over 92 interfaces whose own values span 0 to 1
is invisible in the bar alone, and the dots show that the distribution is what it is regardless of
which bar is slightly taller.

RIGHT: fraction of interfaces whose selected pose clears DockQ 0.23. Bars only -- per interface
that quantity is 0 or 1, so dots would be two stripes and say nothing.

Usage:  python -I run_logging/plot_rel_bars.py [run_id] [epoch] [out.png]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RID = sys.argv[1] if len(sys.argv) > 1 else "o2qczq"
EP = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else None
# --best PICKS THE EPOCH WITH THE HIGHEST HELD-OUT MEAN DockQ under MEAN pooling, which is the
# quantity the arm trains. IT IS A BIASED NUMBER and the script says so on the figure: choosing
# the best of N epochs on the same metric you then report inflates it, by roughly the per-epoch
# noise times a factor that grows with N. The per-epoch table printed below always lists EVERY
# eval epoch, so the selection is visible rather than implied.
BEST = "--best" in sys.argv
OUT = (sys.argv[3] if len(sys.argv) > 3
       else f"figures/rel_bars_{RID}.png")
RUN = os.path.join(ROOT, "training", "runs", RID)

dirs = sorted(glob.glob(os.path.join(RUN, "train_eval_shards_ep*")),
              key=lambda d: int(re.search(r"ep(\d+)$", d).group(1)))
if EP is not None:
    dirs = [d for d in dirs if int(re.search(r"ep(\d+)$", d).group(1)) == EP]
if not dirs:
    sys.exit(f"  no train_eval_shards_ep* for {RID}" + (f" epoch {EP}" if EP else ""))


def load(dd):
    g = defaultdict(list)
    for f in glob.glob(os.path.join(dd, "rows_b*_s*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                vm = r.get("iptm_mean")
                g[r["iface"]].append((float(r["iptm"]),
                                      float(vm) if vm not in (None, "") else None,
                                      float(r["stored_iptm"]) if r["stored_iptm"] != "" else None,
                                      float(r["dockq"])))
            except (ValueError, TypeError, KeyError):
                continue
    fl = max({len(v) for v in g.values()}, default=0)
    kp = [i for i in sorted(g) if len(g[i]) == fl
          and all(x[2] is not None for x in g[i]) and all(x[1] is not None for x in g[i])]
    return g, fl, kp


# every eval epoch, so the table can show them all and --best can choose honestly
ALL = []
for dd in dirs:
    e_ = int(re.search(r"ep(\d+)$", dd).group(1))
    g_, fl_, kp_ = load(dd)
    if not kp_:
        continue
    pk = lambda j: np.array([max(g_[i], key=lambda t: t[j])[3] for i in kp_])  # noqa: E731
    ALL.append((e_, dd, float(pk(1).mean()), float(pk(0).mean()), float(pk(2).mean()), len(kp_)))
if not ALL:
    sys.exit("  no eval epoch has a complete candidate set with both columns")
d = max(ALL, key=lambda r: r[2])[1] if BEST else ALL[-1][1]
ep = int(re.search(r"ep(\d+)$", d).group(1))

by, full, keep = load(d)
n = len(keep)
pick = lambda j: np.array([max(by[i], key=lambda t: t[j])[3] for i in keep])  # noqa: E731
S = [("shipped ipTM", pick(2), "#8a8a8a"),
     ("ours, MEAN pooling", pick(1), "#2f6fb0"),
     ("ours, MAX pooling", pick(0), "#b03a3a")]
base = S[0][1]

# PER-INTERFACE SPEARMAN for the third panel. Computed here rather than only printed, because it
# has more statistical power than the selection delta -- it reads all 150 candidates per interface
# while selection reads only the argmax -- so it belongs on the figure, not in a log line.
RHO = {}
try:
    from scipy import stats as _stt
    for _nm, _j in (("shipped ipTM", 2), ("ours, MEAN pooling", 1), ("ours, MAX pooling", 0)):
        _v = []
        for _ifc in keep:
            _a = np.array([[x[_j], x[3]] for x in by[_ifc]], dtype=float)
            if _a[:, 0].std() > 1e-9 and _a[:, 1].std() > 1e-9:
                _v.append(_stt.spearmanr(_a[:, 0], _a[:, 1]).statistic)
        RHO[_nm] = np.array(_v)
except ImportError:
    RHO = {}

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.labelsize": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "xtick.labelsize": 9.5})
_ncol = 3 if RHO else 2
fig, _axs = plt.subplots(1, _ncol, figsize=(5.4 * _ncol, 5.2),
                         gridspec_kw={"width_ratios": ([1.3, 0.85, 1.3] if RHO
                                                       else [1.45, 1])})
ax, bx = _axs[0], _axs[1]
cx = _axs[2] if RHO else None
rng = np.random.default_rng(0)          # jitter only; fixed so the figure is reproducible
xs = np.arange(len(S))

for i, (nm, v, col) in enumerate(S):
    ax.bar(i, v.mean(), width=0.62, color=col, alpha=0.30, edgecolor=col, linewidth=1.4,
           zorder=1)
    ax.scatter(i + rng.uniform(-0.19, 0.19, n), v, s=11, color=col, alpha=0.45,
               edgecolor="none", zorder=2)
    ax.plot([i - 0.33, i + 0.33], [v.mean()] * 2, color=col, lw=2.4, zorder=3)
    # THE VALUE SITS JUST ABOVE ITS OWN BAR, one line, inside the axes. The first version
    # stacked a two-line label per bar above the axes at y=1.015, so three of them collided with
    # each other and with the title. The delta moves into the tick label, where it has its own
    # column of space.
    ax.annotate(f"{v.mean():.4f}", (i, v.mean()), ha="center", va="bottom", fontsize=10,
                color=col, fontweight="bold", xytext=(0, 5), textcoords="offset points",
                zorder=4)
ax.set_xticks(xs)
_tl = []
for i, (nm, v, _c) in enumerate(S):
    if i == 0:
        _tl.append(f"{nm}\n(reference)")
    else:
        dd = v - base
        se = dd.std(ddof=1) / math.sqrt(n)
        _tl.append(f"{nm}\n{dd.mean():+.4f}  t={dd.mean()/se:+.2f}")
ax.set_xticklabels(_tl, fontsize=9.5)
ax.set_ylabel("selected DockQ")
ax.set_ylim(0, 1.04)
ax.set_title(f"mean selected DockQ   ({n} interfaces, one dot each)", pad=10)
ax.grid(axis="y", color="#eeeeee", lw=0.8)
ax.set_axisbelow(True)

for i, (nm, v, col) in enumerate(S):
    fa = float((v >= 0.23).mean())
    bx.bar(i, fa, width=0.62, color=col, alpha=0.30, edgecolor=col, linewidth=1.4)
    bx.annotate(f"{fa:.4f}", (i, fa), ha="center", va="bottom", fontsize=10, color=col,
                fontweight="bold", xytext=(0, 5), textcoords="offset points")
bx.set_xticks(xs)
_bl = []
_fb = float((base >= 0.23).mean())
for i, (nm, v, _c) in enumerate(S):
    short = nm.replace("ours, ", "").replace(" pooling", "").replace("shipped ipTM", "shipped")
    _bl.append(short if i == 0
               else f"{short}\n{float((v >= 0.23).mean()) - _fb:+.4f}")
bx.set_xticklabels(_bl, fontsize=9.5)
bx.set_ylabel("fraction with selected DockQ >= 0.23")
bx.set_ylim(0, 0.92)
bx.set_title("fraction acceptable  (DockQ >= 0.23)", pad=10)
bx.grid(axis="y", color="#eeeeee", lw=0.8)
bx.set_axisbelow(True)

# THIRD PANEL: within-interface Spearman, bar at the MEDIAN with one dot per interface. The
# median rather than the mean because that is the statistic reported everywhere else for this
# quantity, and a per-interface rho distribution is skewed and has negative values.
if cx is not None:
    _rb = np.median(RHO["shipped ipTM"])
    for i, (nm, _v, col) in enumerate(S):
        _r = RHO.get(nm)
        if _r is None or not len(_r):
            continue
        _m = float(np.median(_r))
        cx.bar(i, _m, width=0.62, color=col, alpha=0.30, edgecolor=col, linewidth=1.4,
               zorder=1)
        cx.scatter(i + rng.uniform(-0.19, 0.19, len(_r)), _r, s=11, color=col, alpha=0.45,
                   edgecolor="none", zorder=2)
        cx.plot([i - 0.33, i + 0.33], [_m] * 2, color=col, lw=2.4, zorder=3)
        cx.annotate(f"{_m:+.4f}", (i, _m), ha="center", va="bottom", fontsize=10, color=col,
                    fontweight="bold", xytext=(0, 6), textcoords="offset points", zorder=4)
    cx.axhline(0.0, color="#999999", lw=0.9)
    cx.set_xticks(xs)
    cx.set_xticklabels(
        [(nm.replace("ours, ", "").replace(" pooling", "").replace("shipped ipTM", "shipped")
          + ("" if i == 0 else
             f"\n{float(np.median(RHO[nm])) - _rb:+.4f}"))
         for i, (nm, _v, _c) in enumerate(S)], fontsize=9.5)
    cx.set_ylabel("within-interface Spearman vs DockQ")
    cx.set_ylim(-1.05, 1.05)
    cx.set_title("ranking quality  (median, one dot per interface)", pad=10)
    cx.grid(axis="y", color="#eeeeee", lw=0.8)
    cx.set_axisbelow(True)

# ONE compact header instead of a suptitle plus two padded panel titles competing for the
# same strip of space.
fig.text(0.012, 0.985, f"{RID} epoch {ep}  -  held-out valfull, budget 30, "
         f"paired on identical poses", fontsize=11, fontweight="bold", va="top")
if BEST and len(ALL) > 1:
    fig.text(0.012, 0.945, f"BEST of {len(ALL)} eval epochs by mean-pooled DockQ -- a "
             f"best-of-N number is biased upward; see the per-epoch table",
             fontsize=8.5, color="#b03a3a", va="top")
fig.tight_layout(rect=(0, 0, 1, 0.955))
os.makedirs(os.path.join(ROOT, os.path.dirname(OUT)), exist_ok=True)
fig.savefig(os.path.join(ROOT, OUT), dpi=150, bbox_inches="tight")
# HELD-OUT WITHIN-INTERFACE SPEARMAN, the diagnostic the trainer logs for TRAIN only. The
# original request was for train AND validation and only train was implemented; it needs no GPU,
# because the eval rows already carry iptm_mean and dockq per pose.
#
# IT HAS MORE POWER THAN THE SELECTION DELTA and should be read first: it uses all 150 candidates
# per interface while selection reads only the argmax. At epoch 5 the selection delta was
# +0.0027 (t=0.48) while the Spearman gain over shipped was +0.0920 in the median.
try:
    from scipy import stats as _st
    print("\n  held-out WITHIN-INTERFACE SPEARMAN (median over interfaces):")
    _rhos = {}
    for _nm, _j in (("shipped ipTM", 2), ("ours, MEAN", 1), ("ours, MAX", 0)):
        _v = []
        for _ifc in keep:
            _a = np.array([[x[_j], x[3]] for x in by[_ifc]], dtype=float)
            if _a[:, 0].std() > 1e-9 and _a[:, 1].std() > 1e-9:
                _v.append(_st.spearmanr(_a[:, 0], _a[:, 1]).statistic)
        _v = np.array(_v)
        _rhos[_nm] = _v
        _d = ("" if _j == 2 else
              f"   {np.median(_v) - np.median(_rhos['shipped ipTM']):+.4f} vs shipped")
        print(f"    {_nm:>14}  median {np.median(_v):+.4f}  mean {_v.mean():+.4f}  "
              f"negative {100*(_v < 0).mean():.0f}%{_d}")
except ImportError:
    print("  (scipy unavailable; skipping the held-out Spearman)")

print(f"  epoch {ep}{' (BEST of %d)' % len(ALL) if BEST and len(ALL) > 1 else ''}, "
      f"{n} interfaces at {full} candidates -> {OUT}")
if len(ALL) > 1:
    print(f"\n  EVERY eval epoch (so the --best selection is visible):")
    print(f"    {'ep':>4} {'shipped':>8} {'MEAN':>8} {'dMEAN':>9} {'MAX':>8} {'dMAX':>9}")
    for e_, _d, m_, x_, b_, _k in ALL:
        print(f"    {e_:>4} {b_:>8.4f} {m_:>8.4f} {m_-b_:>+9.4f} {x_:>8.4f} {x_-b_:>+9.4f}"
              + ("   <- shown" if e_ == ep else ""))
    print()
for nm, v, _c in S:
    dd = v - base
    se = dd.std(ddof=1) / math.sqrt(n) if nm != S[0][0] else 0.0
    print(f"    {nm:>20}  mean {v.mean():.4f}  frac {(v>=0.23).mean():.4f}"
          + ("" if nm == S[0][0] else
             f"  delta {dd.mean():+.4f} +- {se:.4f} (t={dd.mean()/se:+.2f})"))
