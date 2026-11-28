#!/usr/bin/env python
"""FIGURE 1. What five diffusion samples per seed buy, and what we actually collect.

THE COMPARISON. Every interface that carries all 15 seeds x 5 samples is scored twice:

  all 5     the full 75-pose pool -- the protocol as run
  1 of 5    one sample drawn at random from each seed, a 15-pose pool, redrawn --reps times

Both pools are scored the same two ways, and the difference between those two ways is the point:

  ipTM      the pose the shipped confidence head selects (highest ipTM; ties to the lowest seed
            then the lowest sample, phase_k_eval.py:795) -- what the pipeline delivers
  oracle    the best pose in the pool by ground-truth DockQ -- what the pool contains

THREE METRICS over the interfaces: mean DockQ of the selected pose, and the fraction of
interfaces whose selected pose clears the two CAPRI thresholds, 0.23 acceptable and 0.49 medium.

The all-5 condition is DETERMINISTIC -- there is nothing to resample -- so it is drawn as a
marker, not a box. Only the 1-of-5 condition is bootstrapped, and its box shows the spread over
draws, i.e. how much the answer depends on WHICH sample you happened to keep. That spread is not
a confidence interval on the all-5 number and the two should not be read as overlapping errors.
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACCEPTABLE, MEDIUM = 0.23, 0.49


def argmax_like_eval(cand):
    """cand = [(dockq, iptm, seed, sample)] -> the index the eval would select."""
    return sorted(range(len(cand)),
                  key=lambda i: (-cand[i][1], cand[i][2], cand[i][3]))[0]


def metrics(pools):
    """(mean DockQ, frac >= 0.23, frac >= 0.49) for ipTM selection and for the oracle."""
    import numpy as np
    sel, orc = [], []
    for cand in pools:
        sel.append(cand[argmax_like_eval(cand)][0])
        orc.append(max(c[0] for c in cand))
    s, o = np.asarray(sel), np.asarray(orc)
    return ((float(s.mean()), float((s >= ACCEPTABLE).mean()), float((s >= MEDIUM).mean())),
            (float(o.mean()), float((o >= ACCEPTABLE).mean()), float((o >= MEDIUM).mean())))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--seeds", type=int, default=15,
                    help="seeds per interface to use. An interface with MORE is truncated to its "
                         "first N by seed id, so every interface contributes the same depth and "
                         "the pool size is not a hidden covariate; one with fewer is dropped.")
    ap.add_argument("--label", default="", help="dataset name for the figure title")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--out", default="figures/fig1_depth_vs_selection.png")
    ap.add_argument("--csv-out", default="")
    a = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pose = collections.defaultdict(dict)
    with open(a.results) as fh:
        rdr = csv.DictReader(fh)
        cols = rdr.fieldnames or []
        # THE BENCHMARK TABLES NAME IT pair_iptm AND CARRY NO has_trunk/arm COLUMNS. Detected
        # rather than branched on a dataset name, so a new table with either spelling just works.
        ipcol = "iptm" if "iptm" in cols else "pair_iptm"
        gate_trunk, gate_arm = "has_trunk" in cols, "arm" in cols
        print(f"{os.path.relpath(a.results, ROOT)}: ipTM column '{ipcol}'"
              f"{', gated on has_trunk' if gate_trunk else ''}"
              f"{', gated on arm=free' if gate_arm else ''}")
        for r in rdr:
            if gate_trunk and r.get("has_trunk") != "1":
                continue
            if gate_arm and r.get("arm") != "free":
                continue
            try:
                pose[r["iface"]][(int(r["seed"]), int(r["sample"]))] = (
                    float(r["dockq"]), float(r[ipcol]))
            except (ValueError, KeyError, TypeError):
                continue

    full, drop_seeds, drop_samples = {}, 0, 0
    for iface, d in pose.items():
        # A SEED IS KEPT ONLY IF IT IS COMPLETE. FoldBench has seeds with 4 samples; a 4-sample
        # seed in a 5-sample comparison would quietly lower that interface's oracle.
        ok = [sd for sd in sorted({s for s, _ in d})
              if sum(1 for (s, _) in d if s == sd) == int(a.samples)]
        if len(ok) < int(a.seeds):
            if len({s for s, _ in d}) < int(a.seeds):
                drop_seeds += 1
            else:
                drop_samples += 1
            continue
        keep = ok[:int(a.seeds)]
        full[iface] = (keep, {k: v for k, v in d.items() if k[0] in set(keep)})
    print(f"interfaces kept: {len(full)} of {len(pose)}  "
          f"(dropped {drop_seeds} with < {a.seeds} seeds, "
          f"{drop_samples} left short once incomplete seeds were removed)")
    if not full:
        raise SystemExit(f"STOP: no interface has {a.seeds} seeds x {a.samples} complete samples")

    all5 = [[(d[(sd, i)][0], d[(sd, i)][1], sd, i) for sd in seeds for i in range(a.samples)]
            for seeds, d in full.values()]
    m5_sel, m5_orc = metrics(all5)

    rng = np.random.default_rng(int(a.rng))
    draws_sel, draws_orc = [], []
    for _ in range(int(a.reps)):
        pools = []
        for seeds, d in full.values():
            pools.append([(lambda k: (d[k][0], d[k][1], k[0], k[1]))(
                (sd, int(rng.integers(a.samples)))) for sd in seeds])
        s, o = metrics(pools)
        draws_sel.append(s)
        draws_orc.append(o)
    draws_sel = np.asarray(draws_sel)
    draws_orc = np.asarray(draws_orc)

    names = ["mean DockQ", f"frac >= {ACCEPTABLE} (acceptable)", f"frac >= {MEDIUM} (medium)"]
    print(f"\n{'metric':30s} {'1 of 5, ipTM':>14s} {'all 5, ipTM':>12s} "
          f"{'1 of 5, oracle':>15s} {'all 5, oracle':>14s}")
    for j, nm in enumerate(names):
        print(f"{nm:30s} {draws_sel[:, j].mean():14.4f} {m5_sel[j]:12.4f} "
              f"{draws_orc[:, j].mean():15.4f} {m5_orc[j]:14.4f}")
    print(f"\n{'metric':30s} {'ipTM delta':>12s} {'oracle delta':>14s} {'converted':>11s}")
    for j, nm in enumerate(names):
        ds = m5_sel[j] - draws_sel[:, j].mean()
        do = m5_orc[j] - draws_orc[:, j].mean()
        print(f"{nm:30s} {ds:+12.4f} {do:+14.4f} {ds / do if do else float('nan'):11.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.8))
    GREY, BLUE = "0.45", "#2f6fd0"
    for j, (ax, nm) in enumerate(zip(axes, names)):
        bp = ax.boxplot([draws_sel[:, j], draws_orc[:, j]], positions=[0, 1], widths=0.42,
                        patch_artist=True, medianprops=dict(color="white", lw=1.4),
                        flierprops=dict(marker="o", ms=2.5, mfc="0.6", mec="none"))
        for patch, c in zip(bp["boxes"], (GREY, BLUE)):
            patch.set_facecolor(c)
            patch.set_edgecolor(c)
            patch.set_alpha(0.85)
        for e in ("whiskers", "caps"):
            for art in bp[e]:
                art.set_color("0.55")
        ax.scatter([0], [m5_sel[j]], marker="D", s=74, color=GREY,
                   edgecolor="black", lw=0.9, zorder=5)
        ax.scatter([1], [m5_orc[j]], marker="D", s=74, color=BLUE,
                   edgecolor="black", lw=0.9, zorder=5)
        for pos, box_v, five_v in ((0, draws_sel[:, j].mean(), m5_sel[j]),
                                   (1, draws_orc[:, j].mean(), m5_orc[j])):
            ax.annotate("", xy=(pos + 0.26, five_v), xytext=(pos + 0.26, box_v),
                        arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.1,
                                        shrinkA=0, shrinkB=0))
            ax.text(pos + 0.31, (box_v + five_v) / 2, f"{five_v - box_v:+.3f}",
                    fontsize=8.5, va="center", color="0.25")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["ipTM-selected", "oracle"], fontsize=9.5)
        ax.set_title(nm, fontsize=11)
        ax.grid(axis="y", alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(-0.5, 1.95)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    head = (f"{a.label}: " if a.label else "") + "one diffusion sample per seed vs all five"
    nstruct = len({k.split("_")[0] for k in full})
    fig.suptitle(f"{head}\n"
                 f"n = {len(full)} interfaces over {nstruct} structures, "
                 f"{a.seeds} seeds x {a.samples} samples = {len(full) * a.seeds * a.samples:,} poses",
                 fontsize=12.5, y=0.985)
    fig.text(0.5, 0.875, f"box = {a.reps} random 1-of-5 draws     "
                         f"diamond = all {a.samples} samples (deterministic)",
             ha="center", fontsize=9.5, color="0.35")
    fig.tight_layout(rect=(0, 0, 1, 0.855))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.savefig(a.out, dpi=170)
    print(f"\nwrote {a.out}")

    if a.csv_out:
        with open(a.csv_out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "condition", "selector", "value"])
            for j, nm in enumerate(names):
                w.writerow([nm, "all5", "iptm", m5_sel[j]])
                w.writerow([nm, "all5", "oracle", m5_orc[j]])
                for v in draws_sel[:, j]:
                    w.writerow([nm, "one_of_five", "iptm", v])
                for v in draws_orc[:, j]:
                    w.writerow([nm, "one_of_five", "oracle", v])
        print(f"wrote {a.csv_out}")


if __name__ == "__main__":
    main()
