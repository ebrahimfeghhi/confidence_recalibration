#!/usr/bin/env python
"""How the ipTM score distribution moves across epochs, on held-out data.

    python -I run_logging/spread_plot.py <rid> [<rid> ...] [--out path] [--budget 30]

WHY SPREAD IS WORTH WATCHING. rank_loss.py's docstring records the failure the softplus was
adopted to prevent: the LINEAR ranking loss is L = sum_i c_i s_i, so scaling every score by lambda
scales the loss by lambda and WIDENING THE SPREAD PAYS WITHOUT LIMIT -- better than reordering, by
about 71x at the real within-seed gap scale. Run ntwjfp is what that looks like: spread 0.0114 ->
0.0371 over 8 epochs while the held-out ordering got WORSE. So an arm whose spread inflates is
suspect no matter what its loss curve does.

The softplus bounds that, because sigmoid(-m/tau) decays once a pair is comfortably ordered. This
plot is the check that it actually did.

FOUR MEASURES, all from the run's own eval shards, all CPU:

  within-seed std     mean over (interface, seed) of the std of that seed's pose scores. The same
                      quantity rank_rho stores as `spread`, recomputed here so the plot does not
                      depend on which cache version a run has.
  within-iface std    std over ALL poses of an interface, so it includes between-seed variation.
                      ipTM is 82.3% between-seed, so this is mostly NOT the within-seed number.
  top1-top2 gap       median margin that actually decides the argmax. argmax_margin.py measured
                      0.00087 on 8fkrsu against a tau of 0.1429, which is how a loss can be
                      indifferent to exactly the comparison that picks the pose.
  full range          median (max - min) within an interface, the envelope the scores occupy.

The shipped head's value is drawn as a dashed line from the same shards' stored_iptm, so every
curve has its own baseline on the same poses.
"""
import argparse, csv, glob, os, statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("rids", nargs="+")
ap.add_argument("--out", default="training/runs/spread.png")
ap.add_argument("--budget", type=int, default=30)
a = ap.parse_args()


def epochs_of(rid):
    out = {}
    for d in glob.glob(f"training/runs/{rid}/train_eval_shards_ep*"):
        ep = int(d.rsplit("ep", 1)[1])
        rows = []
        for f in glob.glob(f"{d}/rows_b{a.budget}_*.csv"):
            with open(f) as fh:
                rows += list(csv.DictReader(fh))
        if rows:
            out[ep] = rows
    return out


def measures(rows, col):
    byif = defaultdict(lambda: defaultdict(list))
    for r in rows:
        byif[r["iface"]][r["seed"]].append(float(r[col]))
    ws, wi, g12, rng = [], [], [], []
    for ifc, seeds in byif.items():
        allv = [v for vs in seeds.values() for v in vs]
        if len(allv) < 2:
            continue
        for vs in seeds.values():
            if len(vs) > 1:
                ws.append(st.pstdev(vs))
        wi.append(st.pstdev(allv))
        s = sorted(allv, reverse=True)
        g12.append(s[0] - s[1])
        rng.append(s[0] - s[-1])
    m = lambda x: st.median(x) if x else float("nan")
    return (sum(ws) / len(ws) if ws else float("nan"),
            sum(wi) / len(wi) if wi else float("nan"), m(g12), m(rng))


TITLES = ["within-seed std (mean)", "within-interface std (mean)",
          "median top1-top2 gap", "median full range"]
fig, axes = plt.subplots(1, 4, figsize=(5.0 * 4, 4.4), layout="constrained")
COLORS = ["#1f77b4", "#e67e22", "#16a085", "#8e44ad"]

for k, rid in enumerate(a.rids):
    eps = epochs_of(rid)
    if not eps:
        print(f"  {rid}: no shards")
        continue
    xs = sorted(eps)
    tr = [measures(eps[e], "iptm") for e in xs]
    sh = measures(eps[xs[-1]], "stored_iptm")
    for i, ax in enumerate(axes):
        ax.plot(xs, [t[i] for t in tr], color=COLORS[k % len(COLORS)], lw=2.0,
                marker="o", ms=3.5, label=rid)
        if k == 0:
            ax.axhline(sh[i], color="#c0392b", ls="--", lw=1.1)
            ax.text(0.98, sh[i], f"shipped {sh[i]:.5f}", transform=ax.get_yaxis_transform(),
                    color="#c0392b", fontsize=8, va="bottom", ha="right")
        ax.set_title(TITLES[i], fontsize=11.5)
        ax.set_xlabel("epoch", fontsize=10.5)
        ax.grid(alpha=0.15)
    print(f"  {rid}: ep{xs[0]}..{xs[-1]}  within-seed {tr[0][0]:.5f} -> {tr[-1][0]:.5f}"
          f"   top1-top2 {tr[0][2]:.5f} -> {tr[-1][2]:.5f}   (shipped {sh[0]:.5f} / {sh[2]:.5f})")

axes[0].set_ylabel("ipTM", fontsize=10.5)
for ax in axes:
    ax.legend(fontsize=8.5, loc="best", framealpha=0.9)
fig.suptitle("ipTM spread on held-out poses, per epoch "
             "(dashed = shipped head on the same poses)", fontsize=11)
plt.savefig(a.out, dpi=140)
print(f"wrote {a.out}")
