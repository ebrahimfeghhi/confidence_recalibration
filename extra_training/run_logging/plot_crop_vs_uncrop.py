#!/usr/bin/env python
"""CROPPED vs UNCROPPED arm, overlaid, with the one comparison that is legitimate separated
from the three that are not.

THE TRAP THIS FIGURE IS BUILT AROUND. The two arms differ in ONE setting, --crop-margin, but that
setting changes what the TRAINING metrics are computed over. The cropped arm's PAE, rel term,
within-interface Spearman and spread ratio are all measured on CROPPED inputs (median 333 of 617
tokens); the uncropped arm's are measured on full structures. Overlaying them as if they were one
quantity would compare a number to a different number -- the exact error section 8c was written
about. The training panels therefore carry that warning in the axis label, and the figure's
conclusion is drawn only from the held-out panels.

THE HELD-OUT PANELS ARE COMPARABLE, and that is why this figure exists. BOTH arms' DockQ eval runs
UNCROPPED -- phase_k_eval's default is --crop-margin 0 and the cropped arm predates the crop path
entirely -- so the held-out series measure the same thing on the same 92 interfaces and the same
13,800 poses, against the same shipped baseline. They differ only in how the head was trained.

CADENCE DIFFERS AND IS SHOWN AS IT IS. The cropped arm evaluated every 5 epochs (5, 10, 15, 20,
25, 30); the uncropped arm every epoch. Points are plotted where they exist and NOT interpolated,
because a line through one cropped-arm point per five epochs would imply a trajectory nobody
measured.
"""
import glob
import json
import os
import re
import sys
import collections
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_tied  # noqa: E402

ROOT = "/home/ebrahim/confidence_recalibration/extra_training"
UNCROP = ["erb7nd"]
CROP = ["o2qczq", "t4jtjr", "s6szr8", "thvp73"]
MAXEP = int(sys.argv[1]) if len(sys.argv) > 1 else 10
OUT = sys.argv[2] if len(sys.argv) > 2 else "figures/crop_vs_uncrop.png"


def train_table(rids):
    """epoch -> (pae, rel, rho, sd) from the run logs, completed epochs only."""
    out = {}
    for rid in rids:
        lg = os.path.join(ROOT, "training/runs", rid, "run.log")
        if not os.path.exists(lg):
            continue
        txt = open(lg, errors="replace").read()
        for m in re.finditer(r"\[rank 0\] epoch (\d+) rank-loss \(rel[^)]*\): ([+-]?[\d.]+)", txt):
            out.setdefault(int(m.group(1)), {})["rel"] = float(m.group(2))
        for m in re.finditer(r"\[rank \d\] epoch (\d+) WITHIN-INTERFACE SPEARMAN[^:]*: median "
                             r"([+-]?[\d.]+)", txt):
            out.setdefault(int(m.group(1)), {})["rho"] = float(m.group(2))
        for m in re.finditer(r"\[rank 0\] epoch (\d+) SCORE SPREAD:.*?median ratio ([\d.]+)", txt):
            out.setdefault(int(m.group(1)), {})["sd"] = float(m.group(2))
        tj = os.path.join(ROOT, "training/runs", rid, "trajectory.csv")
        if os.path.exists(tj):
            acc = collections.defaultdict(list)
            for r in csv.DictReader(open(tj)):
                e = r.get("epoch")
                if e in (None, ""):
                    continue
                try:
                    acc[int(float(e))].append(float(r["pae_loss"]))
                except (TypeError, ValueError, KeyError):
                    pass
            for e, v in acc.items():
                if len(v) >= 6000:          # completed epochs only
                    out.setdefault(e, {})["pae"] = float(np.mean(v))
    return out


def heldout(rids):
    """epoch -> (dMEAN, rho_ours, rho_shipped), recomputed from the shard rows."""
    out = {}
    for rid in rids:
        for d in glob.glob(os.path.join(ROOT, "training/runs", rid, "valfull_eval_shards_ep*")):
            ep = int(d.split("_ep")[-1])
            by = collections.defaultdict(list)
            for f in glob.glob(os.path.join(d, "rows_*.csv")):
                for r in csv.DictReader(open(f)):
                    if r["dockq"] == "" or r["stored_iptm"] == "":
                        continue
                    by[r["iface"]].append((float(r["iptm_mean"]), float(r["iptm"]),
                                           float(r["stored_iptm"]), float(r["dockq"])))
            if not by:
                continue
            ks = sorted(by)
            ours = np.array([select_tied.expected_pick(by[k], 0, 3)[0] for k in ks])
            shp = np.array([select_tied.expected_pick(by[k], 2, 3)[0] for k in ks])
            om = np.array([spearmanr([p[0] for p in by[k]], [p[3] for p in by[k]]).correlation
                           for k in ks])
            sh = np.array([spearmanr([p[2] for p in by[k]], [p[3] for p in by[k]]).correlation
                           for k in ks])
            m = np.isfinite(om) & np.isfinite(sh)
            # BOTH POOLINGS AND FRAC ACCEPTABLE. The arm trains the MEAN, so MEAN is primary,
            # but a readout nothing optimises crossing the one that is optimised is evidence the
            # optimised one is noise -- it happened at epoch 6 -- so MAX is carried too.
            mx = np.array([select_tied.expected_pick(by[k], 1, 3)[0] for k in ks])
            fa_o = np.mean([select_tied.expected_pick(by[k], 0, 3, 0.23)[1] for k in ks])
            fa_s = np.mean([select_tied.expected_pick(by[k], 2, 3, 0.23)[1] for k in ks])
            out[ep] = (float(ours.mean() - shp.mean()), float(np.median(om[m])),
                       float(np.median(sh[m])), len(ks),
                       float(mx.mean() - shp.mean()), float(fa_o), float(fa_s))
    return out


tc, tu = train_table(CROP), train_table(UNCROP)
hc, hu = heldout(CROP), heldout(UNCROP)

CC, CU = "#b2182b", "#2166ac"
fig, ax = plt.subplots(4, 2, figsize=(13.4, 14.6))
fig.suptitle("CROPPED vs UNCROPPED arm, same loss (beta 75, mean-pooled), "
             f"epochs 1-{MAXEP}", fontsize=14, fontweight="bold", y=0.985, va="top")
fig.text(0.5, 0.945,
         "ROWS 1-2: training metrics -- NOT COMPARABLE between arms, the cropped arm's are "
         "computed over 333 of 617 tokens.\nROWS 3-4: held out, 92 interfaces, BOTH arms "
         "evaluated UNCROPPED -- this is the comparison that counts.   "
         "Selection is MEAN-pooled ipTM unless a panel says MAX.",
         ha="center", va="top", fontsize=9.5, color="#333333")


def line(axis, tbl, key, c, lab, maxep=MAXEP):
    e = sorted(k for k in tbl if k <= maxep and key in tbl[k])
    if not e:
        return
    axis.plot(e, [tbl[k][key] for k in e], "o-", color=c, lw=1.9, ms=4.6, label=lab)


for axis, key, title in ((ax[0][0], "pae", "PAE loss (train)"),
                         (ax[0][1], "rel", "relative-iTM term (train)"),
                         (ax[1][0], "rho", "within-interface Spearman (train)"),
                         (ax[1][1], "sd", "std(predicted)/std(true iTM) (train)")):
    line(axis, tc, key, CC, "cropped (o2qczq)")
    line(axis, tu, key, CU, "uncropped (erb7nd)")
    axis.set_title(title + "   [different inputs -- levels not comparable]", fontsize=10)
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
    axis.margins(y=0.22)
    axis.legend(fontsize=8.5, loc="best")

# --- held out, the comparable row -------------------------------------------------------------
for axis, idx, title, zline in ((ax[2][0], 0, "HELD OUT: selected DockQ - shipped  (MEAN)", True),
                                (ax[2][1], 1, "HELD OUT: within-interface Spearman  (MEAN)",
                                 False)):
    # THE CROPPED ARM IS DASHED BECAUSE IT WAS MEASURED EVERY FIVE EPOCHS. A solid line between
    # epoch 5 and epoch 10 would draw a trajectory through four epochs nobody evaluated, which is
    # the same overclaim as interpolating. Dashed + markers says "two measurements" honestly.
    for tbl, c, lab, st in ((hc, CC, "cropped (every 5 ep, eval uncropped)", "o--"),
                            (hu, CU, "uncropped (every epoch)", "o-")):
        e = sorted(k for k in tbl if k <= MAXEP)
        if e:
            axis.plot(e, [tbl[k][idx] for k in e], st, color=c, lw=2.1, ms=6.4, label=lab,
                      alpha=0.95)
    if zline:
        axis.axhline(0, color="#555555", ls="--", lw=1.2)
        axis.text(0.015, 0.03, "0 = shipped head", transform=axis.transAxes, fontsize=8.5,
                  color="#555555")
    else:
        sh = (list(hu.values()) + list(hc.values()))[0][2]
        axis.axhline(sh, color="#555555", ls="--", lw=1.2)
        axis.text(0.015, 0.03, f"shipped {sh:+.4f}", transform=axis.transAxes, fontsize=8.5,
                  color="#555555")
    axis.set_title(title, fontsize=10.5, fontweight="bold")
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
    axis.margins(y=0.3)
    axis.legend(fontsize=8.5, loc="best")

_step = 1 if MAXEP <= 12 else 2
_ticks = list(range(1, MAXEP + 1, _step))
for row in ax:
    for axis in row:
        axis.set_xticks(_ticks)
        axis.set_xlim(0.6, MAXEP + 0.4)

# --- row 4: frac acceptable, and the MAX-pooled readout the loss does not optimise -----------
for axis, idx, title, ref in ((ax[3][0], 5, "HELD OUT: frac acceptable  (MEAN)", 6),
                              (ax[3][1], 4, "HELD OUT: selected DockQ - shipped  (MAX)", None)):
    for tbl, c, lab, st in ((hc, CC, "cropped (every 5 ep)", "o--"),
                            (hu, CU, "uncropped (every epoch)", "o-")):
        e = sorted(k for k in tbl if k <= MAXEP)
        if e:
            axis.plot(e, [tbl[k][idx] for k in e], st, color=c, lw=2.1, ms=6.4, label=lab,
                      alpha=0.95)
    if ref is not None:
        sv = (list(hu.values()) + list(hc.values()))[0][ref]
        axis.axhline(sv, color="#555555", ls="--", lw=1.2)
        axis.text(0.015, 0.04, f"shipped {sv:.4f}   (a step is 1/92 = 0.0109)",
                  transform=axis.transAxes, fontsize=8.5, color="#555555")
    else:
        axis.axhline(0, color="#555555", ls="--", lw=1.2)
        axis.text(0.015, 0.04, "0 = shipped head   (the loss trains the MEAN, not this)",
                  transform=axis.transAxes, fontsize=8.5, color="#555555")
    axis.set_title(title, fontsize=10.5, fontweight="bold")
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.25)
    axis.margins(y=0.32)
    axis.legend(fontsize=8.5, loc="best")

fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(ROOT, OUT), dpi=135)
print(f"-> {OUT}")
print(f"\nHELD OUT, both arms evaluated UNCROPPED, 92 interfaces:")
print("        cropped arm                     uncropped arm")
print("   ep   dMEAN    dMAX    rho     facc   dMEAN    dMAX    rho     facc")
for ep in sorted(set(hc) | set(hu)):
    if ep > MAXEP:
        continue
    def f(t, i, w=7, p=4):
        return ("%+.*f" % (p, t[i])) if t else "     --"
    def g(t, i):
        return ("%.4f" % t[i]) if t else "    --"
    c, u = hc.get(ep), hu.get(ep)
    print(f"   {ep:2d}  {f(c,0)} {f(c,4)} {f(c,1)}  {g(c,5)}  "
          f"{f(u,0)} {f(u,4)} {f(u,1)}  {g(u,5)}")
_sh = (list(hu.values()) + list(hc.values()))[0]
print(f"\n   shipped: facc {_sh[6]:.4f}, rho {_sh[2]:+.4f}; one interface of 92 = 0.0109 of facc")
