#!/usr/bin/env python
"""Two figures: training over the cropped arm's full 30 epochs, and held out split by pooling.

WHAT IS COMPARABLE AND WHAT IS NOT, which is the whole reason this is two figures and not one:

  TRAINING (figure 1) is NOT comparable between arms. The cropped arm's PAE, rel term, Spearman
  and spread ratio are computed over 333 of 617 tokens; the uncropped arm's over all of them.
  Plotted together only so each arm's SHAPE can be read, never their levels.

  HELD OUT, UNCROPPED EVAL (figure 2, circles) IS comparable. Both arms' DockQ eval defaults to
  --crop-margin 0, so the same 92 interfaces, the same 13,800 poses, the same shipped baseline.

  HELD OUT, CROPPED EVAL (figure 2, the single starred point) IS NOT A DEPLOYMENT NUMBER and is
  drawn apart from the line for that reason. It is section 9b: the epoch-30 cropped head scored on
  ground-truth-derived interface crops against a shipped head scored on the SAME crops. The crop
  comes from `label_coordinate`, so it cannot be built at inference. It is shown because it is the
  arm's strongest result and the thing the uncropped arm was launched to test, not because it can
  be compared to the circles.
"""
import collections
import csv
import glob
import os
import re
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import select_tied  # noqa: E402

ROOT = "/home/ebrahim/confidence_recalibration/extra_training"
CROP = ["o2qczq", "t4jtjr", "s6szr8", "thvp73"]
UNCROP = ["erb7nd"]
CC, CU, CX = "#b2182b", "#2166ac", "#1a9850"
ACC = 0.23


def train_table(rids):
    out = {}
    for rid in rids:
        lg = os.path.join(ROOT, "training/runs", rid, "run.log")
        if os.path.exists(lg):
            txt = open(lg, errors="replace").read()
            for m in re.finditer(r"\[rank 0\] epoch (\d+) rank-loss \(rel[^)]*\): "
                                 r"([+-]?[\d.]+)", txt):
                out.setdefault(int(m.group(1)), {})["rel"] = float(m.group(2))
            for m in re.finditer(r"\[rank \d\] epoch (\d+) WITHIN-INTERFACE SPEARMAN[^:]*: median "
                                 r"([+-]?[\d.]+)", txt):
                out.setdefault(int(m.group(1)), {})["rho"] = float(m.group(2))
            for m in re.finditer(r"\[rank 0\] epoch (\d+) SCORE SPREAD:.*?median ratio ([\d.]+)",
                                 txt):
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
                if len(v) >= 6000:
                    out.setdefault(e, {})["pae"] = float(np.mean(v))
    return out


def rows_of(d):
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(d, "rows_*.csv")):
        for r in csv.DictReader(open(f)):
            if r["dockq"] == "" or r["stored_iptm"] == "":
                continue
            by[r["iface"]].append((float(r["iptm_mean"]), float(r["iptm"]),
                                   float(r["stored_iptm"]), float(r["dockq"])))
    return by


def metrics(by, base_by=None):
    """(dMEAN, dMAX, rhoMEAN, rhoMAX, faccMEAN, faccMAX, shipped_facc, shipped_rho)."""
    ks = sorted(by)
    out = {}
    # THE BASELINE MUST COME FROM THE SAME PASS. For an uncropped eval the shipped column is
    # stored_iptm on the same rows. For the CROPPED eval, stored_iptm is the fold's UNCROPPED
    # value -- comparing a cropped head to an uncropped baseline would be the section 8c error
    # again -- so base_by carries a shipped pass scored on the SAME crops.
    if base_by is None:
        shp = np.array([select_tied.expected_pick(by[k], 2, 3)[0] for k in ks])
        shf = np.mean([select_tied.expected_pick(by[k], 2, 3, ACC)[1] for k in ks])
        shr = np.median([r for r in (spearmanr([p[2] for p in by[k]],
                                               [p[3] for p in by[k]]).correlation for k in ks)
                         if r == r])
    else:
        bk = [k for k in ks if k in base_by]
        ks = bk
        shp = np.array([select_tied.expected_pick(base_by[k], 1, 3)[0] for k in ks])
        shf = np.mean([select_tied.expected_pick(base_by[k], 1, 3, ACC)[1] for k in ks])
        shr = np.median([r for r in (spearmanr([p[1] for p in base_by[k]],
                                               [p[3] for p in base_by[k]]).correlation
                                     for k in ks) if r == r])
    for j, tag in ((0, "MEAN"), (1, "MAX")):
        sel = [select_tied.expected_pick(by[k], j, 3, ACC) for k in ks]
        out["d" + tag] = float(np.mean([s[0] for s in sel]) - shp.mean())
        out["facc" + tag] = float(np.mean([s[1] for s in sel]))
        rr = [r for r in (spearmanr([p[j] for p in by[k]], [p[3] for p in by[k]]).correlation
                          for k in ks) if r == r]
        out["rho" + tag] = float(np.median(rr))
    out["shipped_facc"], out["shipped_rho"], out["n"] = float(shf), float(shr), len(ks)
    return out


def heldout(rids):
    out = {}
    for rid in rids:
        for d in glob.glob(os.path.join(ROOT, "training/runs", rid, "valfull_eval_shards_ep*")):
            by = rows_of(d)
            if by:
                out[int(d.split("_ep")[-1])] = metrics(by)
    return out


hc, hu = heldout(CROP), heldout(UNCROP)
tc, tu = train_table(CROP), train_table(UNCROP)

CE = os.path.join(ROOT, "training/runs/cropped_eval")
cropped_eval = None
if os.path.isdir(os.path.join(CE, "valfull_ours_ep30")):
    cropped_eval = metrics(rows_of(os.path.join(CE, "valfull_ours_ep30")),
                           base_by=rows_of(os.path.join(CE, "valfull_shipped")))

MAXEP = max(list(tc) + list(tu) + [30])


def ticks(axis, hi):
    st = 1 if hi <= 12 else (2 if hi <= 24 else 5)
    axis.set_xticks(list(range(0, hi + 1, st)) or [1])
    axis.set_xlim(0.4, hi + 0.6)
    axis.set_xlabel("epoch")
    axis.grid(alpha=0.22)
    axis.margins(y=0.26)


# ============================== FIGURE 1: training ============================================
f1, a1 = plt.subplots(2, 2, figsize=(13.2, 8.4))
f1.suptitle("TRAINING metrics  (in-sample, the 368 structures each arm fits)",
            fontsize=14, fontweight="bold", y=0.985, va="top")
f1.text(0.5, 0.932, "LEVELS ARE NOT COMPARABLE BETWEEN ARMS: the cropped arm's are computed over "
                    "333 of 617 tokens. Read each curve's shape, not the gap.",
        ha="center", va="top", fontsize=9.5, color="#444444")
for axis, key, ttl in ((a1[0][0], "pae", "PAE loss"), (a1[0][1], "rel", "relative-iTM term"),
                       (a1[1][0], "rho", "within-interface Spearman"),
                       (a1[1][1], "sd", "std(predicted) / std(true iTM)")):
    for tbl, c, lab in ((tc, CC, "cropped (o2qczq chain)"), (tu, CU, "uncropped (erb7nd)")):
        e = sorted(k for k in tbl if key in tbl[k])
        if e:
            axis.plot(e, [tbl[k][key] for k in e], "o-", color=c, lw=1.9, ms=4.2, label=lab)
    axis.set_title(ttl, fontsize=11, fontweight="bold")
    ticks(axis, MAXEP)
    axis.legend(fontsize=8.5, loc="best")
f1.tight_layout(rect=[0, 0, 1, 0.915])
f1.savefig(os.path.join(ROOT, "figures/arms_train.png"), dpi=135)

# ============================== FIGURE 2: held out, by pooling =================================
f2, a2 = plt.subplots(2, 3, figsize=(16.2, 9.0))
f2.suptitle("HELD OUT  (valfull, 92 interfaces, never trained on)",
            fontsize=14, fontweight="bold", y=0.986, va="top")
f2.text(0.5, 0.936,
        "Circles: BOTH arms evaluated UNCROPPED -- comparable to each other and to shipped.    "
        "Star: the cropped arm on GROUND-TRUTH interface crops (section 9b),\nagainst a shipped "
        "head on the SAME crops -- NOT a deployment number, the crop needs labels. "
        "Top row MEAN-pooled ipTM, bottom row MAX-pooled.",
        ha="center", va="top", fontsize=9.5, color="#444444")

shipped_rho = (list(hu.values()) + list(hc.values()))[0]["shipped_rho"]
shipped_facc = (list(hu.values()) + list(hc.values()))[0]["shipped_facc"]

for row, tag in ((0, "MEAN"), (1, "MAX")):
    for col, (key, ttl, ref) in enumerate((
            ("d" + tag, f"selected DockQ - shipped   ({tag})", 0.0),
            ("rho" + tag, f"within-interface Spearman   ({tag})", shipped_rho),
            ("facc" + tag, f"frac acceptable   ({tag})", shipped_facc))):
        axis = a2[row][col]
        for tbl, c, lab, st in ((hc, CC, "cropped arm", "o--"), (hu, CU, "uncropped arm", "o-")):
            e = sorted(tbl)
            if e:
                axis.plot(e, [tbl[k][key] for k in e], st, color=c, lw=2.0, ms=5.8, label=lab,
                          alpha=0.95)
        if cropped_eval is not None:
            axis.plot([30], [cropped_eval[key]], "*", color=CX, ms=19, zorder=6, mec="#0b3d20",
                      mew=0.8, label="cropped arm, CROPPED eval")
        axis.axhline(ref, color="#555555", ls=":", lw=1.3)
        axis.set_title(ttl, fontsize=11, fontweight="bold")
        ticks(axis, MAXEP)
        if col == 2:
            axis.text(0.02, 0.04, "one interface = 1/92 = 0.0109", transform=axis.transAxes,
                      fontsize=8, color="#666666")
        if row == 0 and col == 0:
            axis.legend(fontsize=8.5, loc="upper left")
f2.tight_layout(rect=[0, 0, 1, 0.918])
f2.savefig(os.path.join(ROOT, "figures/arms_heldout.png"), dpi=135)

print("-> figures/arms_train.png")
print("-> figures/arms_heldout.png\n")
hdr = f"{'ep':>4} {'arm':<10} {'eval':<9} {'dMEAN':>8} {'dMAX':>8} {'rhoMEAN':>8} " \
      f"{'rhoMAX':>8} {'faccMEAN':>9} {'faccMAX':>8}"
print(hdr); print("-" * len(hdr))
for ep in sorted(set(hc) | set(hu)):
    for tbl, nm in ((hc, "cropped"), (hu, "uncropped")):
        if ep in tbl:
            m = tbl[ep]
            print(f"{ep:>4} {nm:<10} {'uncropped':<9} {m['dMEAN']:>+8.4f} {m['dMAX']:>+8.4f} "
                  f"{m['rhoMEAN']:>+8.4f} {m['rhoMAX']:>+8.4f} {m['faccMEAN']:>9.4f} "
                  f"{m['faccMAX']:>8.4f}")
if cropped_eval is not None:
    m = cropped_eval
    print(f"{30:>4} {'cropped':<10} {'CROPPED':<9} {m['dMEAN']:>+8.4f} {m['dMAX']:>+8.4f} "
          f"{m['rhoMEAN']:>+8.4f} {m['rhoMAX']:>+8.4f} {m['faccMEAN']:>9.4f} {m['faccMAX']:>8.4f}")
    print(f"     (its own shipped baseline, same crops: facc {m['shipped_facc']:.4f}, "
          f"rho {m['shipped_rho']:+.4f}, n={m['n']})")
print(f"\nshipped, uncropped: facc {shipped_facc:.4f}, rho {shipped_rho:+.4f}")
