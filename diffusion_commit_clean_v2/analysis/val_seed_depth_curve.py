#!/usr/bin/env python
"""Fraction of validation interfaces that are acceptable / decoy / no-acceptable, versus seed depth.

    ./val_seed_depth_curve.py --out ../../extra_training/docs/figures/val_seed_depth
    ./val_seed_depth_curve.py --arm "all new" --out .../val_seed_depth_allnew

Writes <out>.png, <out>.pdf and <out>.csv.

CLASSES, per interface, for a given draw of k seeds:
  no acceptable  no pose in the draw reaches DockQ --thresh        a sampling failure
  decoy          an acceptable pose is in the draw but the top pair-ipTM pose is below it
                                                                   a selection failure
  acceptable     the top pair-ipTM pose is above it

WEIGHTING is the frozen convention: fractions are computed within each cluster, then averaged over
clusters with equal weight, so a cluster contributing two representatives counts once.

DRAWS. k=1 enumerates all --universe single-seed draws exactly; there are only 100 of them, so
sampling would be strictly worse than counting. k=--universe has exactly one possible draw and
therefore zero spread by construction. In between, --boots draws without replacement under a fixed
seed. The band is +-1 sd ACROSS DRAWS -- the spread of the estimate at that depth, not a confidence
interval on a population.

The x axis is log-scaled: the depths are 1, 5, 10 then multiples of 10, and on a linear axis the
region where the curves actually move is compressed into the first centimetre.
"""
import argparse
import collections
import csv
import math
import os
import random
import statistics as st
import sys

sys.path[:0] = ["/home/ebrahim/tts_abag"]
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
from matplotlib.ticker import FixedLocator, FixedFormatter  # noqa: E402
import paths                             # noqa: E402

CLASSES = ("acceptable", "decoy", "no acceptable")
COLOURS = {"acceptable": "#2d7d46", "decoy": "#c2691f", "no acceptable": "#8c2d3f"}


def load(dataset, jobs_manifest, split_csv, universe):
    """out: ({cluster: [ {seed: [(dockq, iptm), ...]} per interface ]}, {cluster: kind}, dropped)"""
    d = paths.DATASETS[dataset]
    pdb2cl = {m["pdb"]: m["cluster"] for m in csv.DictReader(open(jobs_manifest))}
    kind = {r["cluster"]: r["kind"] for r in csv.DictReader(open(split_csv))}
    rows = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(os.path.join(str(d.free), "results.csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                dq, ip = float(r["dockq"]), float(r["iptm"])
            except Exception:
                continue
            if math.isnan(dq) or math.isnan(ip):
                continue
            rows[r["iface"]][int(r["seed"])].append((dq, ip))
    byclust, dropped = collections.defaultdict(list), 0
    for k, sd in rows.items():
        cl = pdb2cl.get(k.rsplit("_", 1)[0])
        if cl is None or len(sd) < universe:
            dropped += 1
            continue
        byclust[cl].append({s: sd[s] for s in sorted(sd)[:universe]})
    return byclust, kind, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="sabdab_val2")
    ap.add_argument("--jobs-manifest",
                    default="/home/ebrahim/tts_abag/extra_training/val2_jobs/manifest.csv")
    ap.add_argument("--split-csv",
                    default="/home/ebrahim/tts_abag/extra_training/results/val_split_proposal.csv")
    ap.add_argument("--universe", type=int, default=100)
    ap.add_argument("--depths", default="1,5,10,20,30,40,50,60,70,80,90,100")
    ap.add_argument("--boots", type=int, default=50)
    ap.add_argument("--thresh", type=float, default=0.23)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="all", choices=["all", "all new", "mixed"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    byclust, kind, dropped = load(a.dataset, a.jobs_manifest, a.split_csv, a.universe)
    clusters = sorted(c for c in byclust if a.arm == "all" or kind.get(c) == a.arm)
    if not clusters:
        raise SystemExit(f"STOP: no clusters match arm {a.arm!r}")
    nif = sum(len(byclust[c]) for c in clusters)
    rng = random.Random(a.seed)

    def classify(sd, seeds):
        poses = [p for s in seeds for p in sd[s]]
        oracle = max(p[0] for p in poses)
        chosen = max(poses, key=lambda p: p[1])[0]
        if oracle < a.thresh:
            return "no acceptable"
        return "acceptable" if chosen >= a.thresh else "decoy"

    def fracs(pick):
        out = {}
        for cls in CLASSES:
            out[cls] = st.mean([
                st.mean([1.0 if classify(sd, pick(sd)) == cls else 0.0 for sd in byclust[c]])
                for c in clusters])
        return out

    depths = [int(x) for x in a.depths.split(",") if 0 < int(x) <= a.universe]
    print(f"{a.dataset} arm={a.arm}: {len(clusters)} clusters, {nif} interfaces"
          + (f" ({dropped} interfaces dropped for < {a.universe} seeds)" if dropped else ""))
    table = []
    for k in depths:
        if k == 1:
            allseeds = sorted(byclust[clusters[0]][0].keys())
            draws = [fracs(lambda sd, s=s: [s]) for s in allseeds]
        elif k >= a.universe:
            draws = [fracs(lambda sd: list(sd))]
        else:
            draws = [fracs(lambda sd: rng.sample(list(sd), k)) for _ in range(a.boots)]
        row = {"seeds": k, "draws": len(draws)}
        for cls in CLASSES:
            v = [d[cls] for d in draws]
            row[cls] = st.mean(v)
            row[cls + " sd"] = st.stdev(v) if len(v) > 1 else 0.0
        table.append(row)
        print(f"  {k:3d} seeds ({len(draws):3d} draws): " + "  ".join(
            f"{cls} {100*row[cls]:.1f}+-{100*row[cls+' sd']:.1f}%" for cls in CLASSES))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]), lineterminator="\n")
        w.writeheader()
        for r in table:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=200)
    xs = [r["seeds"] for r in table]
    for cls in CLASSES:
        ys = [100 * r[cls] for r in table]
        es = [100 * r[cls + " sd"] for r in table]
        ax.plot(xs, ys, "-o", color=COLOURS[cls], label=cls, ms=4, lw=1.8, zorder=3)
        ax.fill_between(xs, [y - e for y, e in zip(ys, es)], [y + e for y, e in zip(ys, es)],
                        color=COLOURS[cls], alpha=0.18, lw=0, zorder=2)
    ax.set_xscale("log")
    # EVERY depth gets a tick, but only a legible subset gets a label: on a log axis 50 through
    # 100 sit within a few millimetres of one another and the labels overprint.
    label_at = {1, 5, 10, 20, 30, 50, 100}
    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.xaxis.set_major_formatter(FixedFormatter([str(x) if x in label_at else "" for x in xs]))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    ax.set_xlabel("seeds per interface (log scale)")
    ax.set_ylabel("% of validation interfaces")
    arm = "all clusters" if a.arm == "all" else f"{a.arm} clusters"
    ax.set_title(f"Validation set, {arm}: {len(clusters)} clusters, {nif} interfaces", fontsize=11)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    fig.text(0.005, 0.015,
             f"x: seeds drawn per interface.  y: % of interfaces in each class, averaged within "
             f"cluster then over clusters equally.\n"
             f"band = ±1 sd over draws.  k=1 enumerates all {a.universe} single-seed draws; "
             f"k={a.universe} has one possible draw; otherwise {a.boots} draws without replacement.",
             fontsize=6.5, color="#555", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 1))
    fig.savefig(a.out + ".png")
    fig.savefig(a.out + ".pdf")
    print(f"\nwrote {a.out}.png / .pdf / .csv")


if __name__ == "__main__":
    main()
