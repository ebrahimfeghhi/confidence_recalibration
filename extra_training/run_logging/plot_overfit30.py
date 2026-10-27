#!/usr/bin/env python
"""The overfit30 chain: training PAE against IN-SAMPLE DockQ/frac, over 138 epochs.

WHY THIS MATTERS. Every arm on the full training set plateaus near train PAE 2.61 and captures
about 2% of valfull's headroom. This chain pushed train PAE to 1.34 on 30 structures and captured
most of overfit30's own headroom -- shipped picks 0 of 77 interfaces above DockQ 0.23, the trained
head picks 39%. So a PAE objective CAN convert into pose selection; the question is whether it
survives new complexes.

IN-SAMPLE MEANS IN-SAMPLE. The DockQ here is scored on the same 30 structures the chain trained
on, so it is a fit measurement, not a result. It is drawn against the shipped baseline computed
from the SAME rows (stored_iptm), never against a number from another pool.
"""
import collections
import csv
import glob
import os
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_three as P  # noqa: E402

THRESH = 0.23


def pae_by_epoch(rid):
    out = collections.defaultdict(list)
    try:
        with open(os.path.join(ROOT, "training", "runs", rid, "trajectory.csv")) as fh:
            for r in csv.DictReader(fh):
                v = r.get("batch_pae_loss") or r.get("pae_loss")
                if v and r.get("epoch"):
                    try:
                        out[int(float(r["epoch"]))].append(float(v))
                    except ValueError:
                        pass
    except OSError:
        pass
    return {k: statistics.mean(v) for k, v in out.items()}


def evals(rid):
    """epoch -> (trained dockq, shipped dockq, trained frac, shipped frac, oracle)."""
    out = {}
    for d in glob.glob(os.path.join(ROOT, "training", "runs", rid, "train_eval_shards_ep*")):
        try:
            ep = int(d.split("_ep")[-1])
        except ValueError:
            continue
        by = collections.defaultdict(list)
        for f in glob.glob(os.path.join(d, "rows_b*_*.csv")):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                           float(r["dockq"])))
        if not by:
            continue
        tr = [max(v, key=lambda x: x[0])[2] for v in by.values()]
        sh = [max(v, key=lambda x: x[1])[2] for v in by.values()]
        oc = [max(x[2] for x in v) for v in by.values()]
        out[ep] = (statistics.mean(tr), statistics.mean(sh),
                   sum(1 for x in tr if x > THRESH) / len(tr),
                   sum(1 for x in sh if x > THRESH) / len(sh), statistics.mean(oc))
    return out


def main():
    chain = P.run_chain(sys.argv[1] if len(sys.argv) > 1 else "jy6sah")
    pae, ev = {}, {}
    for r in chain:
        pae.update(pae_by_epoch(r))
        ev.update(evals(r))
    print(f"  chain {chain}: {len(pae)} epoch(s) with train PAE, {len(ev)} with an in-sample eval")
    ep_p = sorted(pae)
    ep_e = sorted(ev)

    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": "#dddddd", "legend.frameon": False,
                         "figure.facecolor": "white"})
    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ax[0].plot(ep_p, [pae[e] for e in ep_p], color="#b03a3a", lw=1.8)
    ax[0].set_title("training PAE loss\nthe chain drives it from 2.9 to 1.34")
    ax[0].set_ylabel("PAE loss")

    oc = statistics.mean(v[4] for v in ev.values()) if ev else 0
    for i, (idx, lab, shv) in enumerate(((0, "mean DockQ", 1), (2, "fraction DockQ > 0.23", 3)), start=1):
        ax[i].plot(ep_e, [ev[e][idx] for e in ep_e], color="#2f6fb0", lw=2, marker="o", ms=3,
                   label="trained head")
        ax[i].axhline(statistics.mean(ev[e][shv] for e in ep_e), color="#9a9a9a", ls="--", lw=1.3,
                      label="shipped (same rows)")
        if idx == 0:
            ax[i].axhline(oc, color="#2e9e83", ls=":", lw=1.3, label="oracle")
        ax[i].set_title(f"IN-SAMPLE {lab}\nscored on the 30 structures it trained on")
        ax[i].set_ylabel(lab)
        ax[i].legend(fontsize=9, loc="best")
    for a in ax:
        a.set_xlabel("epoch")
        a.xaxis.set_major_locator(MaxNLocator(integer=True, nbins="auto"))
        a.grid(alpha=0.6, lw=0.7)
    fig.suptitle("overfit30 chain — IN-SAMPLE fit, 77 interfaces, budget 10 "
                 "· not a generalisation result", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = sys.argv[2] if len(sys.argv) > 2 else "figures/overfit30_chain.png"
    fig.savefig(out, dpi=150)
    print("wrote", out)
    if ep_e:
        f, l = ep_e[0], ep_e[-1]
        print(f"  first eval ep{f}: dockq {ev[f][0]:.4f} frac {ev[f][2]:.4f}  |  "
              f"last ep{l}: dockq {ev[l][0]:.4f} frac {ev[l][2]:.4f}  |  "
              f"shipped {ev[l][1]:.4f}/{ev[l][3]:.4f}  oracle {ev[l][4]:.4f}")


if __name__ == "__main__":
    sys.exit(main())
