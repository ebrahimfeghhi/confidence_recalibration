#!/usr/bin/env python
"""Margin over failures only, margin in units of score spread, and a between-run variance split.

THE MARGIN AS PLOTTED IS OVER ALL WINNABLE ITEMS, zeros included -- an item the selector gets
right contributes exactly 0. That makes the plotted number a product of two things: how often it is
right, and how far off it is when it is wrong. Averaged over failures only, it answers the second
on its own, which is the question "is it close" actually asks.

SCALE. A margin is a difference of scores, so it is only interpretable against how far apart the
scores in an item are at all. Divided by the item's own score spread it says how many spreads the
selector would have to move, which is comparable across epochs even as the scores compress.

THE VARIANCE SPLIT asks where a selector's score variation within an item lives: between the
generator runs, or between the samples inside one run. A selector that only tells runs apart cannot
pick the one good sample out of five siblings.
"""
import argparse
import csv
import glob
import os
import re
import statistics

ACC = 0.23


def load(ep, rids):
    for rid in rids:
        for nm in ("hardall15", "hard15"):
            ds = glob.glob(f"training/runs/{rid}/{nm}_eval_shards_ep{ep}")
            for d in ds:
                by = {}
                for f in glob.glob(os.path.join(d, "rows_*.csv")):
                    for r in csv.DictReader(open(f)):
                        try:
                            by.setdefault(r["iface"], []).append(
                                (float(r["iptm"]), float(r["stored_iptm"]),
                                 float(r["dockq"]), r["seed"], int(r["sample"])))
                        except (TypeError, ValueError, KeyError):
                            pass
                if len(by) >= 100:
                    return by, rid, nm
    return None, None, None


def margins(by, col):
    i = 0 if col == "ours" else 1
    allw, fails, ratios = [], [], []
    for cand in by.values():
        acc = [c for c in cand if c[2] >= ACC]
        if not acc:
            continue
        g = max(c[i] for c in cand) - max(c[i] for c in acc)
        allw.append(g)
        sd = statistics.pstdev([c[i] for c in cand]) or float("nan")
        if g > 1e-12:
            fails.append(g)
            ratios.append(g / sd)
    return allw, fails, ratios


def between_run(by, col):
    i = 0 if col == "ours" else 1
    fracs = []
    for cand in by.values():
        runs = {}
        for c in cand:
            runs.setdefault(c[3], []).append(c[i])
        if len(runs) < 2:
            continue
        allv = [v for vs in runs.values() for v in vs]
        tot = statistics.pvariance(allv)
        if tot <= 0:
            continue
        # within = mean of per-run variances weighted by run size; between = total - within
        within = sum(statistics.pvariance(vs) * len(vs) for vs in runs.values() if len(vs) > 1)
        within /= len(allv)
        fracs.append(max(0.0, (tot - within)) / tot)
    return fracs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="5,25,50,100,150")
    ap.add_argument("--rids", default="ambgom,55reqf,ctgfwj,3mwrwq,xi5gbj,5rru5z,7uhgmw,upgkqf")
    a = ap.parse_args()
    rids = a.rids.split(",")

    print("MARGIN -- as plotted (all winnable, zeros included) vs failures only")
    print(f"{'ep':>5} {'n_win':>6} {'as plotted':>11} {'n_fail':>7} {'failures only':>14} "
          f"{'score sd':>9} {'margin/sd':>10}")
    last = None
    for ep in [int(x) for x in a.epochs.split(",")]:
        by, rid, nm = load(ep, rids)
        if by is None:
            print(f"{ep:>5}   no shards")
            continue
        allw, fails, ratios = margins(by, "ours")
        sds = [statistics.pstdev([c[0] for c in cand]) for cand in by.values()]
        print(f"{ep:>5} {len(allw):>6} {statistics.mean(allw):>+11.4f} {len(fails):>7} "
              f"{statistics.mean(fails):>+14.4f} {statistics.mean(sds):>9.4f} "
              f"{statistics.mean(ratios):>10.2f}")
        last = (ep, by, rid, nm)

    if last:
        ep, by, rid, nm = last
        print(f"\nVARIANCE SPLIT at epoch {ep} ({nm} under {rid}, {len(by)} items)")
        for col, lab in (("ours", "trained selector"), ("shipped", "shipped selector")):
            fr = between_run(by, col)
            print(f"  {lab:>17}: between-run share of within-item score variance "
                  f"{statistics.mean(fr):.1%} (median {statistics.median(fr):.1%}, n={len(fr)})")


if __name__ == "__main__":
    main()
