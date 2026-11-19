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
import json
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
        b = bins.get(iface, bin_of(by[iface]))
        grp[b].append(v)
        # THE SLICE SPLITS 5/5 THREE WAYS AND VAL DOES NOT, so without this the easy bin -- the one
        # val can actually resolve -- has no row in common between the two tables.
        if b.startswith("5/5 ("):
            grp["5/5"].append(v)
    return per, grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice-work", help="shard dir for the 280-item training slice")
    ap.add_argument("--val-work", help="shard dir for val")
    ap.add_argument("--slice-items", default="training/slice280/items.csv")
    ap.add_argument("--label", default="")
    ap.add_argument("--expect-slice", type=int, default=280)
    ap.add_argument("--expect-val", type=int, default=141)
    ap.add_argument("--expect-cands", default="slice=75,val=150",
                    help="candidates per item each side must have, or the table is over a subset")
    # ---- MIXED-DEPTH IN-SAMPLE PASSES. The whole training pool cannot be scored at one depth:
    # 947 of its 1,688 interfaces have 15 folded seeds (75 candidates) and 741 have one (5). Given
    # the manifest that phase_k_eval --mixed-budget was run with, the in-sample side is split into
    # ONE TABLE PER DEPTH instead of being pooled, because "frac acceptable at top 1" over a
    # 5-candidate item and a 75-candidate item are not the same quantity and their pooled rate
    # moves with the depth mix alone. The per-item candidate assertion is kept -- it just becomes
    # per structure instead of one global number.
    ap.add_argument("--slice-depths", default="",
                    help="seed manifest (mixed:true) giving each structure's depth. Splits the "
                         "in-sample table by depth and asserts each item's candidate count "
                         "against ITS structure's depth x nsample.")
    ap.add_argument("--slice-pool", default="",
                    help="the pool actually scored. Used to assert the per-depth item counts and "
                         "to NAME the table after the pool, so a 1,688-interface in-sample number "
                         "cannot be read as the 280-item slice's.")
    a = ap.parse_args()
    want = dict(kv.split("=") for kv in a.expect_cands.split(","))

    bins = {r["iface"]: r["bin"] for r in csv.DictReader(open(a.slice_items))} \
        if os.path.exists(a.slice_items) else {}

    # THE IN-SAMPLE TABLE IS NAMED AFTER ITS POOL, and only the real 280-item slice may be called
    # "TRAIN SLICE": plot_fullset.report_points greps ^TRAIN SLICE\s*\([^)]*\) out of these
    # reports to draw an arm's slice panel, so a bigger or mixed pool wearing that label would be
    # drawn into the same curve as the 280-item one and the two pools would be averaged together.
    # AND AN OMITTED --slice-pool CANNOT MEAN "this is slice280". That inference is what made the
    # guard keyable to the argument instead of the data: review reproduced a 280-interface
    # MIXED-depth dir, run with --slice-work alone, printing "TRAIN SLICE (280, in-sample)" --
    # straight into plot_fullset's slice panel. The default is only allowed to stand for the pass
    # it was written for: 280 items, no depth split. Anything else must say which pool it is.
    if a.slice_work and not a.slice_pool and (a.expect_slice != 280 or a.slice_depths):
        raise SystemExit(
            f"STOP: --slice-pool is required here (--expect-slice {a.expect_slice}"
            + (", --slice-depths given" if a.slice_depths else "")
            + "). Without it this table would be labelled 'TRAIN SLICE (280, in-sample)', and "
              "plot_fullset.report_points greps ^TRAIN SLICE out of these reports -- a different "
              "pool would be drawn into the 280-item slice's curve and the two averaged.")
    _spool = os.path.basename(os.path.dirname(os.path.abspath(a.slice_pool))) if a.slice_pool else ""
    slice_name = ("TRAIN SLICE (280, in-sample)"
                  if _spool in ("", "slice280") and not a.slice_depths
                  else f"TRAIN POOL {_spool}")

    depth_of, nsample = {}, 5
    if a.slice_depths:
        man = json.load(open(a.slice_depths))
        if not man.get("mixed"):
            raise SystemExit(f"STOP: --slice-depths {a.slice_depths} does not declare mixed:true, "
                             f"so splitting by depth would report one table per depth for a pass "
                             f"that has only one. Drop the flag.")
        nsample = int(man.get("nsample") or 5)
        if not a.slice_pool:
            raise SystemExit("STOP: --slice-depths needs --slice-pool, or the per-depth item "
                             "counts cannot be asserted and each table could be over a subset.")
        _seeds = {p: v["seeds"] for p, v in man["structures"].items()}
        for r in csv.DictReader(open(a.slice_pool)):
            _i = f"{r['pdb'].strip()}_{r['chain1'].strip()}{r['chain2'].strip()}"
            if r["pdb"].strip() not in _seeds:
                raise SystemExit(f"STOP: {a.slice_pool} holds {_i} but {a.slice_depths} does not "
                                 f"name structure {r['pdb'].strip()}; its depth is undefined.")
            depth_of[_i] = len(_seeds[r["pdb"].strip()])

    jobs = []
    for work, name in ((a.slice_work, slice_name), (a.val_work, "VAL (141, held out)")):
        if not work or not glob.glob(os.path.join(work, "rows_*.csv")):
            continue
        by = read_rows(work)
        side = "val" if name.startswith("VAL") else "slice"
        if side == "slice" and depth_of:
            # EVERY INTERFACE OF THE POOL, BEFORE ANY SPLIT. Checked once over the union, so a
            # lost shard cannot hide inside one depth's table.
            # SET EQUALITY, NOT JUST THE COUNT: one pool interface absent while another not in
            # the pool is present leaves the counts equal, and the per-depth subset would then
            # raise KeyError instead of saying what is wrong.
            if set(by) != set(depth_of):
                _miss, _extra = sorted(set(depth_of) - set(by)), sorted(set(by) - set(depth_of))
                raise SystemExit(f"STOP: {name} holds {len(by)} item(s) against "
                                 f"{len(depth_of)} in {a.slice_pool}: {len(_miss)} pool "
                                 f"interface(s) missing (e.g. {_miss[:4]}), {len(_extra)} "
                                 f"unexpected (e.g. {_extra[:4]}). The tables would be over a "
                                 f"subset of the pool.")
            for d in sorted(set(depth_of.values()), reverse=True):
                _ifs = [i for i, dd in depth_of.items() if dd == d]
                jobs.append((f"{name} d{d} ({len(_ifs)}, in-sample)",
                             {i: by[i] for i in _ifs}, len(_ifs), d * nsample))
        else:
            _named = side == "slice" and not name.endswith("in-sample)")
            jobs.append((f"{name} ({len(by)}, in-sample)" if _named else name, by,
                         a.expect_slice if side == "slice" else a.expect_val,
                         int(want.get(side, 0))))

    sides = []
    for name, by, exp_n, exp_c in jobs:
        # A TABLE OVER A SUBSET IS WORSE THAN NO TABLE. A pass that lost a shard still leaves
        # rows_*.csv behind, and every rate below would read as if the whole set had been scored.
        if exp_n and len(by) != exp_n:
            raise SystemExit(f"STOP: {name} has {len(by)} item(s), expected {exp_n}. "
                             f"The pass is incomplete; the table would be over a subset.")
        if exp_c:
            bad = {i: len(v) for i, v in by.items() if len(v) != exp_c}
            if bad:
                ex = list(bad.items())[:4]
                raise SystemExit(f"STOP: {len(bad)} {name} item(s) do not have {exp_c} "
                                 f"candidate(s), e.g. {ex}")
        ours, g_ours = summarise(by, bins, "ours")
        ship, g_ship = summarise(by, bins, "shipped")
        gained = [i for i in ours if ours[i][0] and not ship[i][0]]
        lost = [i for i in ours if ship[i][0] and not ours[i][0]]
        sides.append((name, by, ours, ship, g_ours, g_ship, gained, lost))

    if not sides:
        raise SystemExit("STOP: no rows under either --slice-work or --val-work")

    print(f"=== {a.label} ===" if a.label else "")
    # THE NAME COLUMN WIDENS TO FIT, because a per-depth in-sample row carries its pool and its
    # depth ("TRAIN POOL fulltrain_mixed d15 (947, in-sample)") and a 28-wide field pushed the
    # numbers out of their columns -- and plot_fullset.report_points reads these rows back with a
    # regex anchored on the name.
    nw = max(28, max(len(n) for n, *_ in sides))
    print(f"{'':<{nw}} {'this head':>22}   {'shipped':>22}")
    print(f"{'':<{nw}} {'top-1':>10} {'mean q':>11}   {'top-1':>10} {'mean q':>11}   "
          f"{'gained':>7} {'lost':>5}")
    for name, by, ours, ship, g_o, g_s, gained, lost in sides:
        so = sum(1 for v in ours.values() if v[0]) / len(ours)
        ss = sum(1 for v in ship.values() if v[0]) / len(ship)
        qo = statistics.mean(v[1] for v in ours.values())
        qs = statistics.mean(v[1] for v in ship.values())
        print(f"{name:<{nw}} {so:>10.1%} {qo:>11.4f}   {ss:>10.1%} {qs:>11.4f}   "
              f"{len(gained):>7} {len(lost):>5}")
    if depth_of and any(" d" in n and n.endswith("in-sample)") for n, *_ in sides):
        # KEYED OFF THE ROWS THAT WERE ACTUALLY PRINTED, not off the flag: with SIDES=val there is
        # no in-sample side at all, and the footnote appeared under a val-only table explaining a
        # split that was not there.
        # NO POOLED IN-SAMPLE ROW IS PRINTED AT ALL for a mixed pass, and this says why rather
        # than leaving a reader to wonder where it went: one row per depth is the whole point.
        print(f"  the in-sample side is split by folded depth ({len(set(depth_of.values()))} "
              f"depths, {nsample} sample(s) per seed): a top-1 over 5 candidates and a top-1 over "
              f"75 are different problems, so they are NOT pooled into one rate here. Each row's "
              f"shipped column is the reference over THAT row's items and candidates.")

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
