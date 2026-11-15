#!/usr/bin/env python
"""One table per scoring point: the fixed training slice and val side by side, shipped as reference.

FLIPS ARE PER ITEM, against shipped on the SAME candidates: an item shipped gets right and this
head gets wrong is a LOSS, the reverse is a GAIN. The aggregate rate hides them -- two heads can
sit at the same top-1 while disagreeing on a quarter of the items -- and the direction of the
disagreement is what says whether a change is worth keeping.

THE BINS COME FROM THE SLICE FILE, not recomputed here: an item's bin is a property of the
generator's output, and recomputing it per scoring run would let it move with the thing being
measured.
"""
import argparse
import collections
import csv
import glob
import os
import statistics

ACC = 0.23


def read_rows(work):
    by = collections.defaultdict(list)
    for f in glob.glob(os.path.join(work, "rows_*.csv")):
        for r in csv.DictReader(open(f)):
            try:
                by[r["iface"]].append((float(r["iptm"]), float(r["stored_iptm"]),
                                       float(r["dockq"]), r["seed"]))
            except (TypeError, ValueError, KeyError):
                pass
    return by


def bin_of(cand):
    by_seed = collections.defaultdict(list)
    for sc, ss, d, seed in cand:
        by_seed[seed].append(d)
    k = max(sum(1 for d in v if d >= ACC) for v in by_seed.values())
    return f"{k}/5"


def summarise(by, bins, col):
    i = 0 if col == "ours" else 1
    per = {}
    for iface, cand in by.items():
        top = max(cand, key=lambda t: t[i])
        per[iface] = (top[2] >= ACC, top[2])
    grp = collections.defaultdict(list)
    for iface, v in per.items():
        grp[bins.get(iface, bin_of(by[iface]))].append(v)
    return per, grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-work", help="shard dir for the 280-item training slice")
    ap.add_argument("--val-work", help="shard dir for val")
    ap.add_argument("--slice-items", default="training/slice280/items.csv")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    bins = {r["iface"]: r["bin"] for r in csv.DictReader(open(a.slice_items))} \
        if os.path.exists(a.slice_items) else {}

    sides = []
    for work, name in ((a.slice_work, "TRAIN SLICE (280, in-sample)"), (a.val_work, "VAL (141, held out)")):
        if not work or not glob.glob(os.path.join(work, "rows_*.csv")):
            continue
        by = read_rows(work)
        ours, g_ours = summarise(by, bins, "ours")
        ship, g_ship = summarise(by, bins, "shipped")
        gained = [i for i in ours if ours[i][0] and not ship[i][0]]
        lost = [i for i in ours if ship[i][0] and not ours[i][0]]
        sides.append((name, by, ours, ship, g_ours, g_ship, gained, lost))

    if not sides:
        raise SystemExit("STOP: no rows under either --slice-work or --val-work")

    print(f"=== {a.label} ===" if a.label else "")
    print(f"{'':<28} {'this head':>22}   {'shipped':>22}")
    print(f"{'':<28} {'top-1':>10} {'mean q':>11}   {'top-1':>10} {'mean q':>11}   "
          f"{'gained':>7} {'lost':>5}")
    for name, by, ours, ship, g_o, g_s, gained, lost in sides:
        so = sum(1 for v in ours.values() if v[0]) / len(ours)
        ss = sum(1 for v in ship.values() if v[0]) / len(ship)
        qo = statistics.mean(v[1] for v in ours.values())
        qs = statistics.mean(v[1] for v in ship.values())
        print(f"{name:<28} {so:>10.1%} {qo:>11.4f}   {ss:>10.1%} {qs:>11.4f}   "
              f"{len(gained):>7} {len(lost):>5}")

    for name, by, ours, ship, g_o, g_s, gained, lost in sides:
        print(f"\n  {name} by bin")
        print(f"  {'bin':<26} {'n':>4} {'this head':>20}   {'shipped':>20}")
        for b in sorted(g_o):
            o, s = g_o[b], g_s[b]
            print(f"  {b:<26} {len(o):>4} "
                  f"{sum(1 for x in o if x[0]) / len(o):>9.0%} {statistics.mean(x[1] for x in o):>10.3f}   "
                  f"{sum(1 for x in s if x[0]) / len(s):>9.0%} {statistics.mean(x[1] for x in s):>10.3f}")
        if lost:
            print(f"  lost vs shipped ({len(lost)}): " + ", ".join(sorted(lost)[:8])
                  + (" ..." if len(lost) > 8 else ""))


if __name__ == "__main__":
    main()
