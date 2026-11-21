#!/usr/bin/env python
"""Learning-rate sweep at epoch 5: the three held-out metrics, one bar per arm.

    plot_lr_sweep.py [OUT.png]

NOT the default figure. The per-epoch plots are curves because an arm is judged on its trajectory;
this one is bars because every arm is frozen at the SAME epoch and only the learning rate differs,
so the comparison is categorical. It exists for this sweep and should not outlive it.

Arms are discovered from their epoch-5 metric json, so the figure fills in as the sweep runs rather
than needing an edit per arm. An arm with no epoch-5 eval yet is simply absent.

The shipped head is a horizontal reference, not a bar: it is the thing being beaten, not a fourth
arm. Bars start at the axis minimum rather than zero and the axis is annotated as such -- a DockQ
bar from 0 would make a 0.002 difference invisible, and these differences are thousandths.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_curve as EC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else "figures/lr_sweep_ep5.png"

#: (label, run id). The 0.01 arm's epoch 5 predates the 83-structure pool and was scored as two
#: passes, which eval_curve.series() combines; the rest write metric_valfull83_ep5.json directly.
ARMS = [("0.01", "5dgrsu"), ("3e-3", "qip9ta"), ("1e-3", None), ("8e-4", None)]


def rid_for(label):
    f = os.path.join(ROOT, "..", "scratchpad")
    for tag in ("lr1e3", "lr8e4"):
        p = f"/tmp/claude-1004/-home-ebrahim-confidence-recalibration/d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/{tag}_rid"
        if os.path.exists(p) and label.replace("-", "") in tag.replace("lr", "").replace("e", "e-"):
            return open(p).read().strip()
    return None


def load(label, rid):
    if rid is None:
        rid = {"1e-3": "lr1e3", "8e-4": "lr8e4"}.get(label)
        p = (f"/tmp/claude-1004/-home-ebrahim-confidence-recalibration/"
             f"d68d923a-c929-4ce4-bdef-d5300b0cc024/scratchpad/{rid}_rid")
        rid = open(p).read().strip() if os.path.exists(p) else None
    if not rid:
        return None
    ser = EC.series([rid])
    if 5 not in ser:
        return None
    d = dict(ser[5])
    rho, _shp = EC.rho_series([rid])
    d["rho"] = rho.get(5, {}).get("mean", float("nan"))
    d["_rid"] = rid
    return d


data = [(lbl, load(lbl, rid)) for lbl, rid in ARMS]
have = [(l, d) for l, d in data if d]
if not have:
    raise SystemExit("STOP: no arm has an epoch-5 eval yet")
S = EC.shipped()
_, shp_rho = EC.rho_series([have[0][1]["_rid"]])

PANELS = [("mean_dockq", "mean DockQ of the selected pose", S and S["mean_dockq"]),
          ("frac_acc", r"fraction acceptable (DockQ $\geq$ 0.23)", S and S["frac_acc"]),
          ("rho", "median within-interface Spearman", shp_rho)]

plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(14, 4.8))
for a, (key, title, ref) in zip(ax, PANELS):
    labels = [l for l, _d in have]
    vals = [d.get(key, float("nan")) for _l, d in have]
    xs = np.arange(len(labels))
    a.bar(xs, vals, 0.55, color="tab:blue")
    if ref == ref:
        a.axhline(ref, color="#444", lw=1.6)
        a.text(len(labels) - .5, ref, " shipped", color="#333", fontsize=9, va="bottom", ha="right")
    lo = min(list(vals) + ([ref] if ref == ref else []))
    hi = max(list(vals) + ([ref] if ref == ref else []))
    # A FLOOR ON THE WINDOW. frac_acc is identical across arms to four decimals, and zooming to
    # the true spread put matplotlib into 1e-6 offset notation -- bars that differ by a millionth
    # drawn as if they differed hugely, with the real value hidden in a corner label. When the
    # spread is negligible the axis shows a readable window around the value instead, so
    # "these are the same" reads as the same.
    span = hi - lo
    floor = max(abs(hi), 1e-3) * 0.01
    if span < floor:
        mid = (hi + lo) / 2
        a.set_ylim(mid - floor, mid + floor)
    else:
        a.set_ylim(lo - span * .35, hi + span * .35)
    a.ticklabel_format(axis="y", useOffset=False, style="plain")
    for x, v in zip(xs, vals):
        a.text(x, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    a.set_xticks(xs); a.set_xticklabels(labels)
    a.set_xlabel("learning rate"); a.set_title(title, fontsize=11)
    a.grid(axis="y", alpha=.3)
ax[0].set_ylabel("value  (axis does NOT start at zero)")
fig.suptitle("LR sweep at epoch 5 -- held out, 83 structures / 141 interfaces, budget 30\n"
             "identical in every other setting: seed 0, accum 30, world 3, same frozen tensors",
             fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, .88])
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"-> {OUT}")
print(f"  {'lr':>6} {'mean_dockq':>11} {'frac_acc':>9} {'rho':>8}   run")
for l, d in have:
    print(f"  {l:>6} {d['mean_dockq']:>11.4f} {d['frac_acc']:>9.4f} {d['rho']:>8.4f}   {d['_rid']}")
print(f"  {'shipped':>6} {S['mean_dockq']:>11.4f} {S['frac_acc']:>9.4f} {shp_rho:>8.4f}")
missing = [l for l, d in data if not d]
if missing:
    print(f"  (still running: {', '.join(missing)})")
