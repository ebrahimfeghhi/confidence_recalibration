#!/usr/bin/env python
"""THE SATURATION CURVE. Oracle and selected DockQ against the number of diffusion samples per
seed, 1 to 10, on units that now carry all ten.

Each (interface, seed) unit has the fold's original 5 poses plus 5 more from the SAME cached
trunk, so k samples are drawn from a pool of 10 that differ only in diffusion noise. For each k
the draw is repeated --reps times and the oracle (best DockQ in the draw) and the selected DockQ
(the pose with the highest ipTM, ties to the lowest sample id) are averaged.

TWO CURVES, because the two halves of the pool are not quite equivalent. "original" uses only the
fold's own 5, which are the matched-protocol poses and so carry no frame-mismatch bias, and runs
to k=5. "all" uses the pool of 10 and runs to k=10. Where they overlap they should agree; the gap
between them IS the bias, shown rather than argued.
"""
import argparse
import collections
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=f"{ROOT}/training/fold/results_with_trunks.csv")
    ap.add_argument("--new-dockq", default=f"{ROOT}/training/fold/poses_extra/dockq_extra.csv")
    ap.add_argument("--new-iptm", default=f"{ROOT}/training/fold/poses_extra/results_extra.csv")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--rng", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    import numpy as np

    old = collections.defaultdict(dict)
    with open(a.old) as fh:
        for r in csv.DictReader(fh):
            if r.get("arm") != "free" or r.get("has_trunk") != "1":
                continue
            try:
                old[(r["iface"], int(r["seed"]))][int(r["sample"])] = (
                    float(r["dockq"]), float(r["iptm"]))
            except (ValueError, KeyError):
                continue
    ndq, nip = collections.defaultdict(dict), collections.defaultdict(dict)
    with open(a.new_dockq) as fh:
        for r in csv.DictReader(fh):
            try:
                ndq[(r["iface"], int(r["seed"]))][int(r["sample"])] = float(r["dockq"])
            except (ValueError, KeyError):
                continue
    with open(a.new_iptm) as fh:
        for r in csv.DictReader(fh):
            try:
                nip[(r["iface"], int(r["seed"]))][int(r["sample"])] = float(r["iptm"])
            except (ValueError, KeyError):
                continue

    units = []
    for k, d in ndq.items():
        if k not in old or len(old[k]) != a.nsample or len(d) != a.nsample:
            continue
        ip = nip.get(k)
        if not ip or len(ip) != a.nsample:
            continue
        o = [(s, old[k][s][0], old[k][s][1]) for s in sorted(old[k])]
        n = [(s, d[s], ip[s]) for s in sorted(d)]
        units.append((o, o + n))
    print(f"{len(units)} (interface, seed) unit(s) with all 10 poses\n")

    rng = np.random.default_rng(int(a.rng))

    def curve(which, kmax):
        out = {}
        for k in range(1, kmax + 1):
            orc, sel = [], []
            for _ in range(int(a.reps)):
                o_, s_ = [], []
                for pools in units:
                    cand = pools[0] if which == "original" else pools[1]
                    idx = rng.choice(len(cand), size=k, replace=False)
                    sub = [cand[i] for i in idx]
                    o_.append(max(c[1] for c in sub))
                    s_.append(sorted(sub, key=lambda t: (-t[2], t[0]))[0][1])
                orc.append(np.mean(o_))
                sel.append(np.mean(s_))
            out[k] = (float(np.mean(orc)), float(np.mean(sel)))
        return out

    co = curve("original", 5)
    ca = curve("all", 10)
    print(f"{'k':>3s} {'oracle (orig 5)':>16s} {'sel (orig 5)':>13s} "
          f"{'oracle (all 10)':>16s} {'sel (all 10)':>13s}")
    for k in range(1, 11):
        a1 = f"{co[k][0]:16.4f}" if k in co else " " * 16
        b1 = f"{co[k][1]:13.4f}" if k in co else " " * 13
        print(f"{k:3d} {a1} {b1} {ca[k][0]:16.4f} {ca[k][1]:13.4f}")
    print(f"\nORACLE   1->5  {ca[5][0] - ca[1][0]:+.4f}    5->10 {ca[10][0] - ca[5][0]:+.4f}"
          f"    1->10 {ca[10][0] - ca[1][0]:+.4f}")
    print(f"SELECTED 1->5  {ca[5][1] - ca[1][1]:+.4f}    5->10 {ca[10][1] - ca[5][1]:+.4f}"
          f"    1->10 {ca[10][1] - ca[1][1]:+.4f}")

    if a.out:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ks = list(range(1, 11))
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        ax.plot(ks, [ca[k][0] for k in ks], "o-", color="#2f6fd0", lw=2, ms=6,
                label="oracle (best pose in the pool)")
        ax.plot(ks, [ca[k][1] for k in ks], "s-", color="0.45", lw=2, ms=6,
                label="selected by ipTM")
        ax.axvline(5, color="0.8", lw=1, ls="--")
        ax.text(5.08, ax.get_ylim()[0], " the fold's own depth", fontsize=8, color="0.5",
                va="bottom")
        ax.set_xlabel("diffusion samples per seed")
        ax.set_ylabel("mean DockQ")
        ax.set_xticks(ks)
        ax.set_title(f"Depth generates, selection does not deliver\n"
                     f"{len(units)} (interface, seed) units, {a.reps} draws per k", fontsize=11)
        ax.grid(alpha=0.22, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, loc="center right")
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        fig.savefig(a.out, dpi=170)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
