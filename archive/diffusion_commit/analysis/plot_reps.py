#!/usr/bin/env python
"""Run-to-run variation of an identical fold, per complex, against the original pool value.

WHY. Every single-run comparison in this project assumes a difference between two runs was caused by
the thing we changed. That assumption broke twice in one day -- the trunk-resume looked
non-equivalent, and refolds looked like they lost good poses -- and neither could be judged because
the noise floor had never been measured. This plots it: one seed per complex, folded five times with
identical settings.

Two panels, because they answer different questions:
  MEAN DockQ over the 5 diffusion samples -- how the whole fold behaves
  BEST DockQ of the 5 samples             -- what selection actually gets to pick from

bar  = mean across the repeat runs
dots = the individual runs (one per repeat)
star = the value from the ORIGINAL pool run of that same seed (September), which is the number every
       earlier table in this project was built on

Usage: plot_reps.py --reps <reps.csv> --out <png>
"""
import argparse
import collections
import csv
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

POOL = str(paths.PXM2025_POOL_CSV)
# seed per complex: the donor seed these repeats re-fold
SEEDS = {"9x1h_AF": "300014", "9g9o_AC": "300009", "9r1g_AC": "300024", "8w5m_BC": "300001",
         "8w5i_AE": "300017", "8vqf_AC": "300048", "9dbo_AC": "300040", "8w5f_BD": "300047",
         "8yw9_BC": "300003", "9s8w_AC": "300027"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    runs = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(A.reps)):
        runs[r["iface"]][r["arm"]].append(float(r["dockq"]))

    pool = collections.defaultdict(list)
    for r in csv.DictReader(open(POOL)):
        iface = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
        if iface in SEEDS and r["seed"] == SEEDS[iface]:
            pool[iface].append(float(r["dockq"]))

    ifaces = sorted(runs, key=lambda i: -st.mean([max(v) for v in runs[i].values()]))
    fig, axes = plt.subplots(2, 1, figsize=(max(7, 1.15 * len(ifaces)), 8.2), sharex=True)
    fig.subplots_adjust(hspace=0.16)

    for ax, mode, title in ((axes[0], "mean", "MEAN DockQ over the 5 diffusion samples"),
                            (axes[1], "best", "BEST DockQ of the 5 diffusion samples")):
        xs, bars, dotsets, stars = [], [], [], []
        for i, iface in enumerate(ifaces):
            per_run = [st.mean(v) if mode == "mean" else max(v) for v in
                       (runs[iface][a] for a in sorted(runs[iface]))]
            xs.append(i)
            bars.append(st.mean(per_run))
            dotsets.append(per_run)
            p = pool.get(iface)
            stars.append((st.mean(p) if mode == "mean" else max(p)) if p else None)

        ax.bar(xs, bars, width=0.62, color="#9fb8c8", edgecolor="#3c5a6b", linewidth=0.9,
               zorder=1, label="mean of repeats")
        for x, dots in zip(xs, dotsets):
            ax.scatter([x] * len(dots), dots, s=26, color="#16323f", zorder=3,
                       label="individual run" if x == 0 else None)
        sx = [x for x, s in zip(xs, stars) if s is not None]
        sy = [s for s in stars if s is not None]
        ax.scatter(sx, sy, marker="*", s=215, color="#d1495b", edgecolor="#6b1f2b",
                   linewidth=0.6, zorder=4, label="original pool run")
        ax.axhline(0.23, color="#888", lw=0.9, ls="--", zorder=0)
        ax.text(len(ifaces) - 0.45, 0.245, "DockQ 0.23 (acceptable)", fontsize=7.5,
                color="#666", ha="right")
        ax.set_ylabel("DockQ")
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_ylim(0, max(0.75, max([b for b in bars] + [s for s in sy]) * 1.18))
        ax.grid(axis="y", lw=0.4, color="#ddd", zorder=0)
        ax.set_axisbelow(True)
        for n, (x, dots) in enumerate(zip(xs, dotsets)):
            ax.text(x, 0.012, f"n={len(dots)}", ha="center", fontsize=7, color="#39505c")

    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[1].set_xticks(range(len(ifaces)))
    axes[1].set_xticklabels(ifaces, rotation=32, ha="right", fontsize=8.5)
    axes[0].set_xlim(-0.6, len(ifaces) - 0.4)
    fig.suptitle("Identical fold repeated 5x per complex, versus the original pool run",
                 fontsize=11.5, y=0.965)
    fig.savefig(A.out, dpi=170, bbox_inches="tight")
    print(f"  wrote {A.out}  ({len(ifaces)} complexes)")


if __name__ == "__main__":
    main()
sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402  (needs the path insert above)
