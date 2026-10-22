#!/usr/bin/env python
"""Three held-out panels: mean DockQ, frac acceptable, and Spearman(ipTM, DockQ) -- each with s.e.

Usage: python -I run_logging/plot_three.py <rid> [--out path]

ALL THREE ARE HELD OUT, on val10's 18 interfaces at budget 30, and all three carry the standard
error ACROSS INTERFACES rather than across poses. That matters: the paired s.e. on mean DockQ over
these 18 interfaces is 0.026, so the set resolves ~0.05 DockQ and nothing smaller. A band that
overlaps the dashed line is not a result.

The dashed line is the SHIPPED head, derived from this run's own stored_iptm column -- a baseline,
not another arm.
"""
import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rank_rho as RHO

SHIPPED = {"dockq": 0.2909, "frac": 0.5556, "rho": 0.1958}
ORACLE = {"dockq": 0.5968, "frac": 1.0000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rid")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=30)
    a = ap.parse_args()

    base = os.path.join("training", "runs", a.rid)
    cache = os.path.join(base, f"rank_rho_b{a.budget}.csv")
    if not os.path.exists(cache):
        raise SystemExit(f"no {cache} yet")
    rows = list(csv.DictReader(open(cache)))
    if not rows:
        raise SystemExit(f"{cache} is empty")
    ep = [int(r["epoch"]) for r in rows]
    g = lambda k: [float(r[k]) for r in rows]

    panels = [
        ("mean DockQ of the selected pose", g("dockq_mean"), g("dockq_sem"),
         SHIPPED["dockq"], ORACLE["dockq"]),
        ("fraction acceptable (DockQ > 0.23)", g("frac_mean"), g("frac_sem"),
         SHIPPED["frac"], ORACLE["frac"]),
        ("Spearman(ipTM, DockQ) within interface", g("rho_trained"), g("rho_sem"),
         SHIPPED["rho"], None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, (title, y, se, ship, orac) in zip(axes, panels):
        lo = [v - s for v, s in zip(y, se)]
        hi = [v + s for v, s in zip(y, se)]
        ax.fill_between(ep, lo, hi, color="#1f77b4", alpha=0.18, lw=0)
        ax.plot(ep, y, color="#1f77b4", lw=2.0, marker="o", ms=3.5)
        ax.axhline(ship, color="#d62728", ls="--", lw=1.6)
        ax.text(ax.get_xlim()[1], ship, f" shipped {ship:.4f}", color="#d62728",
                fontsize=9, va="bottom", ha="right")
        if orac is not None:
            ax.axhline(orac, color="#2ca02c", ls=":", lw=1.4)
            ax.text(ax.get_xlim()[1], orac, f" oracle {orac:.4f}", color="#2ca02c",
                    fontsize=9, va="bottom", ha="right")
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("epoch", fontsize=10.5)
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("held out, 18 interfaces, +- s.e. across interfaces", fontsize=10)
    fig.suptitle(f"{a.rid} -- ranking-only, cross-seed softplus  (bands are s.e. over the 18 "
                 f"interfaces; the set resolves ~0.05 DockQ)", fontsize=11.5, y=1.02)
    out = a.out or os.path.join("training", "runs", "three_panels.png")
    plt.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({len(ep)} epoch(s): {ep[0]}..{ep[-1]})")


if __name__ == "__main__":
    main()
