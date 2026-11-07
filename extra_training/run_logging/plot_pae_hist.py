#!/usr/bin/env python
"""Per-STRUCTURE training PAE at one epoch, as a histogram.

PER STRUCTURE, NOT PER INTERFACE. trajectory.csv carries one pae_loss per pose, and that scalar is
already pooled over every scored chain pair of that structure -- 382 of 487 structures hold more
than one interface, so a multi-interface structure's bar blends an easy interface with a hard one.
There is no per-interface PAE on disk and this does not invent one.

THE MEAN IS BELOW THE MEDIAN and the plot says so, because the aggregate number everyone quotes is
pulled down by a tail of structures the head has essentially solved; reading 1.78 as "the typical
structure" would be wrong by 0.12.

Usage:  python -I run_logging/plot_pae_hist.py <run_id> [out.png] [--epoch N] [--chain]
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pos = [x for x in sys.argv[1:] if not x.startswith("--")]
rid = pos[0] if pos else sys.exit("need a run id")
out = pos[1] if len(pos) > 1 else f"figures/pae_hist_{rid}.png"
want_ep = None
for a in sys.argv[1:]:
    if a.startswith("--epoch"):
        want_ep = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])

rids = [rid]
if "--chain" in sys.argv:
    sys.path.insert(0, os.path.join(ROOT, "run_logging"))
    import plot_three as P
    rids = list(P.run_chain(rid))

n_by_ep = defaultdict(int)
recs = []
for r_ in rids:
    p = os.path.join(ROOT, "training", "runs", r_, "trajectory.csv")
    if not os.path.exists(p):
        continue
    for r in csv.DictReader(open(p)):
        if r.get("epoch") and r.get("pae_loss"):
            e = int(float(r["epoch"]))
            n_by_ep[e] += 1
            recs.append((e, r["pdb"], float(r["pae_loss"])))
if not recs:
    sys.exit(f"  no pae_loss rows for {rid}")
full = max(n_by_ep.values())
ep = want_ep if want_ep is not None else max(e for e, n in n_by_ep.items() if n >= 0.9 * full)
per = defaultdict(list)
for e, pdb, v in recs:
    if e == ep:
        per[pdb].append(v)
vals = np.array([np.mean(v) for v in per.values()])
print(f"  epoch {ep}: {len(vals)} structures, {sum(len(v) for v in per.values())} poses")

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})
fig, ax = plt.subplots(figsize=(9.5, 5.4))
ax.hist(vals, bins=40, color="#b03a3a", edgecolor="white", linewidth=0.5)
mean, med = vals.mean(), float(np.median(vals))
ax.axvline(mean, color="#1f3b73", lw=2, label=f"mean {mean:.3f}")
ax.axvline(med, color="#1f7a1f", lw=2, ls="--", label=f"median {med:.3f}")
ax.axvline(1.0, color="#777777", lw=1.4, ls=":", label="1.0")
ax.set_xlabel("training PAE per structure")
ax.set_ylabel("structures")
ax.set_title(f"{rid} \u2014 epoch {ep}")
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)
fig.tight_layout()
os.makedirs(os.path.join(ROOT, os.path.dirname(out)) or ".", exist_ok=True)
fig.savefig(os.path.join(ROOT, out), dpi=140, bbox_inches="tight")
print(f"  -> {out}")
for q in (0, 10, 25, 50, 75, 90, 100):
    print(f"    p{q:<3} {np.percentile(vals, q):.4f}")
