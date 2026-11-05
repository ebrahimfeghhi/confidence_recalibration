#!/usr/bin/env python
"""Every pair pool.csv names, measured against the reference: does it touch, and was it scored?

THE QUESTION THIS SETTLES. pool.csv pairs every antibody chain with every antigen chain and never
checks whether they are in contact, so it names interfaces that are two chains sitting apart in the
structure. That alone only inflates the denominator. The thing that would be serious is the other
direction: a DockQ row written for a pair that does NOT touch in the ground truth, because such a
row is a number with no interface behind it and it would be averaged in with real ones.

So this crosses the two: measured contact against rows actually present. Four cells, and only one
of them is a problem.

  touching  +  scored      expected
  touching  +  no score    a gap -- the 900 already reported
  no contact + no score    correct: nothing to score
  no contact + SCORED      BAD DATA: a score for an interface that does not exist

Usage:
  python -I run_logging/pool_contact_audit.py --e training/fold --refs training/refs
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/confidence_recalibration/run_model")
import contacts                                                            # noqa: E402

_argv, sys.argv = sys.argv, ["folder.py"]
import folder                                                              # noqa: E402
sys.argv = _argv

ap = argparse.ArgumentParser()
ap.add_argument("--e", default="training/fold")
ap.add_argument("--refs", default="training/refs")
ap.add_argument("--cutoff", type=float, default=contacts.CUTOFF)
ap.add_argument("--out", default="")
a = ap.parse_args()

rows, _ = folder.open_csv(os.path.join(a.e, "results.csv"))
scored = {}
for (ifc, _seed, _arm), ks in rows.items():
    scored[ifc] = scored.get(ifc, 0) + len(ks)

pool = []
for r in csv.DictReader(open(os.path.join(a.e, "pool.csv"))):
    pool.append((r["pdb"], r["chain1"], r["chain2"]))

cells = {"touch_scored": [], "touch_unscored": [], "apart_unscored": [], "apart_scored": [],
         "uncheckable": []}
for pdb, c1, c2 in pool:
    ifc = f"{pdb}_{c1}{c2}"
    n_rows = scored.get(ifc, 0)
    ref = os.path.join(a.refs, f"{pdb}.cif")
    n, mn = (None, None)
    if os.path.exists(ref):
        n, mn = contacts.contacts(ref, c1, c2, a.cutoff)
    if n is None:
        cells["uncheckable"].append((ifc, n_rows, None))
        continue
    key = ("touch" if n else "apart") + ("_scored" if n_rows else "_unscored")
    cells[key].append((ifc, n_rows, mn))

print(f"pool interfaces: {len(pool)}   cutoff {a.cutoff} A   references {a.refs}")
print(f"  touching, scored            : {len(cells['touch_scored'])}")
print(f"  touching, NO score (a gap)  : {len(cells['touch_unscored'])}")
print(f"  no contact, no score (right): {len(cells['apart_unscored'])}")
print(f"  no contact, SCORED (BAD)    : {len(cells['apart_scored'])}")
print(f"  contact uncheckable         : {len(cells['uncheckable'])}")
for lbl in ("touch_unscored", "apart_scored", "uncheckable"):
    for ifc, n_rows, mn in sorted(cells[lbl])[:15]:
        d = "n/a" if mn is None else f"{mn:.2f} A"
        print(f"     {lbl}: {ifc} rows={n_rows} nearest={d}")
if a.out:
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["iface", "cell", "rows", "nearest_atom_A"])
        for lbl, items in cells.items():
            for ifc, n_rows, mn in items:
                w.writerow([ifc, lbl, n_rows, "" if mn is None else f"{mn:.3f}"])
    print(f"  wrote {a.out}")
sys.exit(1 if cells["apart_scored"] else 0)
