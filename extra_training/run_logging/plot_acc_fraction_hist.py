#!/usr/bin/env python
"""Per-interface fraction of ACCEPTABLE candidates: train against val, as histograms.

ONE NUMBER PER INTERFACE, not per pose: of the candidates that interface has, what share clear
DockQ 0.23. A pool where most interfaces sit near 0 is one where selection rarely has anything to
find; a pool where most sit near 1 is one where selection can hardly go wrong. The shape between
those ends is what decides how much a selector can earn.

TRAIN IS RESTRICTED TO 15-SEED-COMPLETE INTERFACES and val carries 30 seeds, so the two have
different candidate counts (75 against 150). The fraction is the quantity that survives that --
a count would not.
"""
import collections, csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ACC, ROOT = 0.23, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

def fractions(path, depth, require_trunk):
    by = collections.defaultdict(list)
    seeds = collections.defaultdict(set)
    for r in csv.DictReader(open(os.path.join(ROOT, path))):
        if r.get("arm") not in (None, "", "free"):
            continue
        if require_trunk and r.get("has_trunk") != "1":
            continue
        try:
            by[r["iface"]].append(float(r["dockq"])); seeds[r["iface"]].add(int(r["seed"]))
        except (ValueError, KeyError):
            continue
    return {k: sum(1 for q in v if q >= ACC) / len(v)
            for k, v in by.items() if len(seeds[k]) >= depth}

tr = fractions("training/fold/results_with_trunks.csv", 15, True)
va = fractions("validation/fold/results.csv", 30, False)
bins = np.linspace(0, 1, 21)
fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
for a_, (d, name, col) in zip(ax, [(tr, f"TRAIN, 15-seed-complete ({len(tr)} interfaces)", "0.45"),
                                   (va, f"VAL83, 30 seeds ({len(va)} interfaces)", "tab:blue")]):
    v = np.array(list(d.values()))
    a_.hist(v, bins=bins, color=col, edgecolor="white", weights=np.full(len(v), 100.0 / len(v)))
    a_.set_title(f"{name}\nmedian {np.median(v):.2f}   mean {v.mean():.2f}", fontsize=11)
    a_.set_xlabel("fraction of this interface's candidates that are acceptable (DockQ >= 0.23)")
    a_.grid(alpha=.3)
    a_.axvline(np.median(v), color="black", lw=1.2, ls="--")
ax[0].set_ylabel("% of interfaces")
fig.suptitle("How many of an interface's candidates are acceptable", fontsize=13)
fig.subplots_adjust(top=.84, bottom=.14, left=.07, right=.97, wspace=.08)
out = os.path.join(ROOT, "figures/acc_fraction_hist.png")
fig.savefig(out, dpi=130)
for d, n in ((tr, "TRAIN"), (va, "VAL83")):
    v = np.array(list(d.values()))
    print(f"  {n}: {len(v)} interfaces, median {np.median(v):.3f}, mean {v.mean():.3f}, "
          f"zero {100*np.mean(v == 0):.1f}%, all {100*np.mean(v == 1):.1f}%")
print(f"  wrote {out}")
