#!/usr/bin/env python
"""Three panels for a PAE-ONLY arm: training PAE loss, held-out mean DockQ, held-out frac-acc.

WHY A SEPARATE PLOTTER. plot_rel_arm.py draws four panels, three of which -- the relative-iTM
term, within-interface Spearman and score spread -- come from the RANKING objective. An arm with
--pae-only and no --rank-loss has none of those, so three of its four panels stay empty for the
whole run. That is not a degraded view of the arm, it is the wrong instrument.

WHAT IS ACTUALLY WORTH LOOKING AT HERE. The PAE loss is the only thing being optimised, and the
held-out selection metrics are the only things that decide whether the arm worked. Training loss
going down is what any working optimiser does; this project has recorded cropped training
improving a training metric WITHOUT improving held-out selection four times, so the two are drawn
side by side and the shipped baseline is drawn as a line rather than quoted in a caption.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_curve as EC                                            # noqa: E402
import plot_three                                                  # noqa: E402

RID = (sys.argv[1] if len(sys.argv) > 1 else "privsp").split(",")[0]
OUT = sys.argv[2] if len(sys.argv) > 2 else f"figures/{RID}_pae.png"
CHAIN = list(plot_three.run_chain(RID))

# training PAE per epoch, from the same helper plot_rel_arm uses
EP = plot_three.epoch_table(CHAIN) if hasattr(plot_three, "epoch_table") else None
if EP is None:
    import csv
    import collections
    acc = collections.defaultdict(list)
    for rid in CHAIN:
        p = f"training/runs/{rid}/trajectory.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            try:
                e = int(float(r["epoch"]))
                v = float(r["pae_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if v == v:
                acc[e].append(v)
    # A PARTIAL EPOCH IS DRAWN HOLLOW, not hidden and not drawn solid: the last epoch of a live
    # run is always partial and its mean moves as it fills.
    # PARTIAL EPOCHS ARE DROPPED, NOT DRAWN HOLLOW. At the user's direction. The last epoch of a
    # live run is always incomplete and its mean moves as it fills, so plotting it puts a number
    # on the page that will be different in ten minutes -- a hollow marker says so but does not
    # stop anyone reading it. A COMPLETED EPOCH IS ONE THAT WROTE ITS CHECKPOINT.
    import glob as _glob
    import re as _re
    done = set()
    for rid in CHAIN:
        for f in _glob.glob(f"training/runs/{rid}/head_ep*.pt"):
            m = _re.search(r"head_ep(\d+)_", f)
            if m:
                done.add(int(m.group(1)))
    EP = sorted((e, sum(v) / len(v), True) for e, v in acc.items() if e in done)

S, SER = EC.shipped(), EC.series(CHAIN)
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
fig.suptitle(f"PAE-only arm -- {RID}    (held-out: 83 structures / 141 interfaces, budget 30)")

# 1. the only thing being optimised
xs = [e for e, _v, _d in EP]
ys = [v for _e, v, _d in EP]
if xs:
    ax[0].plot(xs, ys, "-o", color="tab:blue", ms=5)
else:
    ax[0].text(.5, .5, "no completed epoch yet", ha="center", va="center",
               transform=ax[0].transAxes, color="#666")
ax[0].set_title("training PAE loss")
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("PAE loss (all ranks)")
ax[0].grid(alpha=.3)

# 2 and 3. the only things that decide whether it worked
eps = sorted(SER)
for k, (col, lab, ship, orc) in enumerate((
        ("mean_dockq", "mean DockQ of the selected pose", S.get("mean_dockq"), S.get("oracle_mean_dockq")),
        ("frac_acc", "fraction acceptable (DockQ >= 0.23)", S.get("frac_acc"), S.get("oracle_frac_acc")))):
    # THE SHIPPED LINE MUST MATCH THE POOLING THE ARM SELECTS BY, or the plot invites the same
    # apples-to-oranges reading. eval_curve.shipped() returns the MEAN-pooled baseline; the metric
    # json carries the MAX-pooled one as baseline_*, written by the same pass. Reading this arm's
    # max-pooled 0.5850 against the mean-pooled 0.5735 is the mistake this prevents.
    _sb = next((SER[e].get("select_by") for e in eps if SER[e].get("select_by")), "max")
    if _sb == "max":
        _bv = next((SER[e].get("baseline_" + col) for e in eps
                    if SER[e].get("baseline_" + col) is not None), None)
        if _bv is not None:
            ship = _bv
    a = ax[k + 1]
    if eps:
        a.plot(eps, [SER[e].get(col) for e in eps], "-s", color="tab:blue", ms=6,
               label=f"arm, {_sb}-pooled ipTM (what it selects on)")
        # THE OTHER POOLING, WHICHEVER IT IS. A run that SELECTS by the mean writes its max
        # column as `_select_max`, and one that selects by the max writes its mean as
        # `_select_mean`; the json's own `select_by` says which it is. Drawing whichever is
        # present, labelled by what it actually is, stops the two being confused -- which is
        # exactly the mistake that made me read this arm's max-pooled 0.5850 against the shipped
        # MEAN baseline of 0.5735 instead of the max baseline of 0.5813.
        _selby = next((SER[e].get("select_by") for e in eps if SER[e].get("select_by")), "max")
        _other = "mean" if _selby == "max" else "max"
        mx = f"{col}_select_{_other}"
        if any(SER[e].get(mx) is not None for e in eps):
            a.plot(eps, [SER[e].get(mx) for e in eps], "-s", color="tab:orange", ms=6,
                   label=f"arm, {_other}-pooled ipTM")
    else:
        a.text(.5, .5, "no held-out eval yet\n(first at epoch 5)", ha="center", va="center",
               transform=a.transAxes, color="#666")
    if ship is not None:
        a.axhline(ship, color="#444", lw=1.4, label=f"shipped ({ship:.4f})")
    a.set_title(lab + (f"\noracle {orc:.4f}" if orc else ""))
    a.set_xlabel("epoch"); a.grid(alpha=.3)
ax[1].set_ylabel("mean DockQ"); ax[2].set_ylabel("frac acceptable")
h, l = ax[1].get_legend_handles_labels()
if h:
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.04))
fig.tight_layout(rect=(0, .04, 1, .95))
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
fig.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"-> {OUT}")
print(f"\n{RID}  (chain {' -> '.join(CHAIN)})")
print(f"{'ep':>5} {'train PAE':>11}")
for e, v, d in EP:
    print(f"{e:5d} {v:11.4f}")
if eps:
    print(f"\n{'ep':>5} {'mean':>9} {'max':>9} {'frac':>9}   shipped "
          f"{S.get('mean_dockq'):.4f} / {S.get('frac_acc'):.4f}")
    for e in eps:
        print(f"{e:5d} {SER[e].get('mean_dockq'):9.4f} "
              f"{SER[e].get('mean_dockq_select_max') or float('nan'):9.4f} "
              f"{SER[e].get('frac_acc'):9.4f}")
else:
    print("\nno held-out eval yet -- first at epoch 5")
