#!/usr/bin/env python
"""ipTM against DockQ for every pose of one complex: self-templating vs free + compute-matched.

SUMMARY
    Selection picks the pose with the highest ipTM, so whether an arm succeeds on an interface depends
    on where its confident poses sit. This plots every pose of each interface of one complex, one row
    per interface: left, the self-templated folds (the experiment's results.csv); right, free +
    compute-matched (the first 50 free seeds plus the complex's matched extra free seeds, chosen by
    compare_arms.matched_seeds, exactly as the reporters count them). The pose each arm selects is
    ringed; the dashed line is the acceptable threshold (DockQ 0.23).

INPUTS
    --dataset:  pxm2025 or foldbench. Default pxm2025.
    --pdb:      the complex. Required.
    --exp:      the templated experiment. Default the dataset's self_template_fixed_50_seed.
    --out:      the figure. Default analysis/scatter_<dataset>_<pdb>.png (a .pdf beside it too).

OUTPUTS
    <out> and its .pdf; stdout: per interface and arm, poses, acceptable poses, the selected pose

EXAMPLE
    analysis/scatter_iptm_dockq.py --dataset foldbench --pdb 7yxu

FUNCTIONS
    arms(): one complex -> {interface: {arm: [(ipTM, DockQ)]}}
    main(): gather, report, draw
"""
import argparse
import collections
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

import compare_arms as C  # noqa: E402
import paths  # noqa: E402
from score_experiment import ACCEPTABLE, arm_defaults, classes_at, pool_by_seed  # noqa: E402

BLUE, GRAY, INK, MUTED, GRID = "#2a78d6", "#83827c", "#0b0b0b", "#52514e", "#e6e5e1"
TEMPLATED, MATCHED = "self-template", "free + compute-matched"


def arms(pdb, exp, extra_csv, pool_csv):
    """in: a complex, the experiment, free results csvs, the pool csv.
    out: ({interface: {arm: [(ipTM, DockQ)]}}, {interface: class}, matched seeds, donor count)"""
    pool, first = pool_by_seed(50, pool_csv)
    cls = classes_at(pool, first)
    ifaces = sorted(i for i in pool if i.split("_")[0] == pdb)
    if not ifaces:
        raise SystemExit(f"{pdb}: not in the pool")
    per, _ = C.donors_used(exp, first)
    n = len(per.get(pdb, ()))
    extra = C.extra_free(extra_csv, first)
    seeds = C.matched_seeds(extra, ifaces, n)
    out = {}
    for i in ifaces:
        free = [(ip, dq) for s, _, dq, ip in pool[i] if s in first[pdb]]
        out[i] = {MATCHED: free + [v for s in seeds for v in extra[i][s]], TEMPLATED: []}
    for r in csv.DictReader(open(f"{exp}/results.csv")):
        if r["iface"] in out:
            out[r["iface"]][TEMPLATED].append((float(r["iptm"]), float(r["dockq"])))
    return out, cls, seeds, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=paths.DATASET_NAMES, default="pxm2025")
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--exp", default=None)
    ap.add_argument("--out", default=None)
    A = ap.parse_args()
    ds, exp, extra_csv = arm_defaults(A.dataset)
    A.exp = A.exp or exp
    A.out = A.out or f"{HERE}/scatter_{A.dataset}_{A.pdb}.png"

    data, cls, seeds, n = arms(A.pdb, A.exp, extra_csv, str(ds.pool_csv))
    if len(seeds) < n:
        print(f"  NOTE {A.pdb}: {len(seeds)} of {n} matched free seeds scored so far; the free + "
              f"matched panel shows what exists")
    # say what the free panel holds: until the top-up is folded it is the free 50 plus what exists
    free_name = MATCHED if len(seeds) >= n else f"free 50 + {len(seeds)}/{n} matched seeds"
    fig, axes = plt.subplots(len(data), 2, figsize=(9, 3.6 * len(data)), squeeze=False,
                             sharex=True, sharey=True)
    for row, (iface, arm) in enumerate(data.items()):
        for col, (name, colour) in enumerate(((TEMPLATED, BLUE), (MATCHED, GRAY))):
            ax, v = axes[row][col], arm[name]
            name = name if name == TEMPLATED else free_name
            if not v:
                ax.text(0.5, 0.5, "no poses", ha="center", va="center", color=MUTED,
                        transform=ax.transAxes)
                continue
            pick = max(v)
            ax.scatter([p[0] for p in v], [p[1] for p in v], s=10, color=colour, alpha=0.55,
                       edgecolors="none", zorder=2)
            ax.scatter([pick[0]], [pick[1]], s=90, facecolors="none", edgecolors=INK, linewidths=1.4,
                       zorder=3)
            ax.axhline(ACCEPTABLE, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
            ax.grid(color=GRID, linewidth=0.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            acc = sum(q >= ACCEPTABLE for _, q in v)
            ax.set_title(f"{iface} ({cls.get(iface, '?')}) — {name}\n{len(v)} poses, {acc} acceptable; "
                         f"selected: ipTM {pick[0]:.2f}, DockQ {pick[1]:.2f}", fontsize=8.5,
                         color=INK, loc="left")
            print(f"  {iface:9s} {name:24s} poses {len(v):5d}  acceptable {acc:4d}  "
                  f"selected ipTM {pick[0]:.3f} DockQ {pick[1]:.3f}")
            if row == len(data) - 1:
                ax.set_xlabel("ipTM (model confidence at this interface)", color=INK)
            if col == 0:
                ax.set_ylabel("DockQ against the crystal", color=INK)
    fig.suptitle(f"{A.dataset} {A.pdb}: every pose, ipTM vs DockQ (ringed: the pose selection picks; "
                 f"dashed: acceptable, DockQ {ACCEPTABLE})", fontsize=9, color=MUTED, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(A.out, dpi=200)
    fig.savefig(os.path.splitext(A.out)[0] + ".pdf")
    print(f"  {n} donors; {len(seeds)} matched free seeds  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
