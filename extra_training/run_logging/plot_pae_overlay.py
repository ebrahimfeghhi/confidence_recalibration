#!/usr/bin/env python
"""Overlay two PAE-only arms on the SAME held-out pool, each pooling against its OWN baseline.

    python -I run_logging/plot_pae_overlay.py \
        "ea9vy2,7roo28,q6mgyc=dropout 0.25 + L2-SP 1e-3" \
        "yw8rtt=dropout 0.25 only" --out figures/overlay.png

WHY A SEPARATE PLOTTER AND NOT plot_arms.py. plot_arms.py recomputes everything from each run's
train_eval shards and rank_rho, which a PAE-only arm never writes -- it has no ranking objective.
These arms are compared through metric_valfull83_ep<N>.json, the same files plot_pae_arm.py reads.

THE ENCODING. COLOUR IS THE POOLING, as everywhere else in this project: blue is max-pooled,
orange is mean-pooled, and the two shipped baselines are drawn in those same colours. LINE STYLE
is the ARM: solid for the first, dashed for the second. Overloading colour with both would make
the one comparison that matters -- an arm against ITS OWN pooling's baseline -- unreadable, and
reading a mean-pooled point against the max-pooled line is the specific error this project has
made before.

THE NEWEST RUN IN A CHAIN OWNS AN EPOCH. A resume at a boundary leaves the abandoned run holding
a partial next epoch, so concatenating a chain averages a partial attempt together with the
complete redo of the same epoch number. Measured on ea9vy2 -> 7roo28: 3,938 partial rows against
11,541 complete ones for epoch 3.
"""
import argparse
import collections
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_curve as EC                                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("arms", nargs="+", help='"rid[,rid...]=label"')
ap.add_argument("--out", default="figures/pae_overlay.png")
a = ap.parse_args()

ARMS = []
for spec in a.arms:
    chain, _, label = spec.partition("=")
    ARMS.append(([r for r in chain.split(",") if r], label or chain))


def train_series(chain):
    """epoch -> mean training PAE, newest run in the chain winning an epoch outright."""
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for rid in chain:
        p = f"training/runs/{rid}/trajectory.csv"
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            try:
                e, v = int(float(r["epoch"])), float(r["pae_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            if v == v:
                per[rid][e].append(v)
    acc = {}
    for rid in chain:
        for e, vs in per.get(rid, {}).items():
            acc[e] = sum(vs) / len(vs)
    done = {int(f.split("head_ep")[1].split("_")[0])
            for rid in chain for f in glob.glob(f"training/runs/{rid}/head_ep*.pt")}
    return {e: v for e, v in acc.items() if e in done}


S = EC.shipped()
C_MAX, C_MEAN = "tab:blue", "tab:orange"
STYLES = ["-", "--", ":"]
plt.rcParams.update({"font.size": 11})
fig, ax = plt.subplots(1, 4, figsize=(19, 4.6))
fig.suptitle("PAE-only arms overlaid    (held-out: 83 structures / 141 interfaces, budget 30)")

for i, (chain, label) in enumerate(ARMS):
    ls = STYLES[i % len(STYLES)]
    T = train_series(chain)
    if T:
        xs = sorted(T)
        ax[0].plot(xs, [T[e] for e in xs], ls, color="tab:blue", lw=1.6, label=label)

    SER = EC.series(chain)
    eps = sorted(SER)
    RHO, RHO_SHIP = EC.rho_series(chain)
    for k, col in enumerate(("mean_dockq", "frac_acc")):
        if not eps:
            continue
        sb = next((SER[e].get("select_by") for e in eps if SER[e].get("select_by")), "max")
        other = "mean" if sb == "max" else "max"
        ax[k + 1].plot(eps, [SER[e].get(col) for e in eps], ls + "s",
                       color=(C_MAX if sb == "max" else C_MEAN), ms=5,
                       label=f"{label} -- {sb}-pooled")
        alt = f"{col}_select_{other}"
        if any(SER[e].get(alt) is not None for e in eps):
            ax[k + 1].plot(eps, [SER[e].get(alt) for e in eps], ls + "s",
                           color=(C_MAX if other == "max" else C_MEAN), ms=5,
                           label=f"{label} -- {other}-pooled")
    _re = sorted(RHO)
    if _re:
        ax[3].plot(_re, [RHO[e]["max"] for e in _re], ls + "s", color=C_MAX, ms=5,
                   label=f"{label} -- max-pooled")
        ax[3].plot(_re, [RHO[e]["mean"] for e in _re], ls + "s", color=C_MEAN, ms=5,
                   label=f"{label} -- mean-pooled")
    if i == 0 and RHO_SHIP == RHO_SHIP:
        ax[3].axhline(RHO_SHIP, color=C_MAX, lw=1.4,
                      label=f"shipped, max-pooled ({RHO_SHIP:+.4f})")
        ax[3].plot([], [], " ", label="shipped mean-pooled rho: not computed")

# THE TWO SHIPPED BASELINES, from eval_curve.shipped() -- which carries BOTH poolings, so neither
# is hardcoded and neither waits on an arm reaching its first eval.
for k, col in enumerate(("mean_dockq", "frac_acc")):
    bmax = S.get(col + "_select_max") if S else None
    bmean = S.get(col) if S else None
    if bmax is not None:
        ax[k + 1].axhline(bmax, color=C_MAX, lw=1.4, label=f"shipped, max-pooled ({bmax:.4f})")
    if bmean is not None:
        ax[k + 1].axhline(bmean, color=C_MEAN, lw=1.4, label=f"shipped, mean-pooled ({bmean:.4f})")

ax[0].set_title("training PAE loss"); ax[0].set_ylabel("PAE loss (all ranks)")
ax[1].set_title("mean DockQ of the selected pose"); ax[1].set_ylabel("mean DockQ")
ax[2].set_title("fraction acceptable (DockQ >= 0.23)"); ax[2].set_ylabel("frac acceptable")
ax[3].set_title("within-interface Spearman, ipTM vs DockQ"); ax[3].set_ylabel("median rho")
for x in ax:
    x.set_xlabel("epoch"); x.grid(alpha=.3)
ax[0].legend(fontsize=8, frameon=False)
h, l = ax[1].get_legend_handles_labels()
for _h, _l in zip(*ax[3].get_legend_handles_labels()):
    if _l not in l:
        h.append(_h); l.append(_l)
fig.tight_layout(rect=(0, .17, 1, .95))
fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, .005), fontsize=8)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
fig.savefig(a.out, dpi=110, bbox_inches="tight")
print(f"-> {a.out}")

for chain, label in ARMS:
    SER, (RHO, _) = EC.series(chain), EC.rho_series(chain)
    print(f"\n{label}   ({' -> '.join(chain)})")
    print(f"  {'ep':>3} {'max-pool':>9} {'mean-pool':>10} {'frac':>8} {'frac(mn)':>9} "
          f"{'rho max':>9} {'rho mean':>9}")
    for e in sorted(SER):
        d = SER[e]
        sb = d.get("select_by", "max")
        mx = d.get("mean_dockq") if sb == "max" else d.get("mean_dockq_select_max")
        mn = d.get("mean_dockq_select_mean") if sb == "max" else d.get("mean_dockq")
        fx = d.get("frac_acc") if sb == "max" else d.get("frac_acc_select_max")
        fn = d.get("frac_acc_select_mean") if sb == "max" else d.get("frac_acc")
        r = RHO.get(e, {})
        print(f"  {e:3d} {mx:9.4f} {mn:10.4f} {fx:8.4f} {fn:9.4f} "
              f"{r.get('max', float('nan')):+9.4f} {r.get('mean', float('nan')):+9.4f}")
