#!/usr/bin/env python
"""Drop the scored interfaces that have no DockQ label, so a ranking arm can be launched.

    python -I run_logging/build_labelled_pool.py --pool training/fullset/pool.csv \
        --manifest training/fullset/structures.csv --out training/fullset/pool_labelled.csv

WHY THIS EXISTS. --rank-loss refuses to start when a (pdb, seed, interface) it will train has no
DockQ: those poses pay a forward and contribute no ranking gradient, which makes the arm a partial
experiment reported as a whole one. On training/fullset, 17 of 1,679 interfaces are short -- 13
missing every seed and 4 missing one -- so the refusal is correct and the pool is what has to
change.

ONE POOL FOR BOTH TERMS, which is the choice this script encodes. The scored-pairs file sets the
PAE mask as well as the ranking groups, so dropping an interface removes it from BOTH. The
alternative -- tolerate unlabelled groups in the ranking term while PAE keeps them -- needs the
trainer to carry a new tolerance, and a tolerance for "some groups train on one objective and some
on two" is exactly the silent partial experiment the gate was written to stop. Removing 1.0% of
the interfaces from both is the smaller and more legible deviation, and it is REPORTED: an arm
built on this pool is not the same arm as one built on pool.csv, and the printed counts say so.

AN INTERFACE SHORT OF EVEN ONE SEED IS DROPPED WHOLE. Keeping it for its labelled seeds would
make the candidate list depend on which seeds happened to be scored, and under a softmax over the
item's candidates that changes the objective per item rather than merely shrinking it.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phases"))
import rank_loss as RKL                                                 # noqa: E402
import phase_k_eval as PKE                                              # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--labels", default="training/fold/results.csv")
ap.add_argument("--roles", default="training/fold/chain_roles_all.csv")
ap.add_argument("--out", required=True)
ap.add_argument("--skip", default="")
a = ap.parse_args()

lab, _bad = RKL.load_labels(a.labels, a.roles)
have = set(lab)
ifp, _ = PKE.interface_chain_pairs(a.pool, a.roles)
skip = {x.strip().lower() for x in a.skip.split(",") if x.strip()}

decl = {}
for r in csv.DictReader(open(a.manifest)):
    p = (r.get("pdb") or "").strip().lower()
    sd = (r.get("seeds") or "").strip()
    if not p or p in skip or not sd:
        continue
    out = []
    for tok in sd.split(","):
        if "-" in tok:
            lo, hi = tok.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(tok))
    decl[p] = out

bad = set()
for p, seeds in decl.items():
    for ifc, x, y in ifp.get(p, ()):
        k = (min(x, y), max(x, y))
        if any((p, s, k) not in have for s in seeds):
            bad.add((p, k))

rows = list(csv.DictReader(open(a.pool)))
kept, dropped = [], []
for r in rows:
    p = (r.get("pdb") or "").strip().lower()
    # THE ROW'S OWN CHAINS, RESOLVED THROUGH THE SAME ROLES FILE the label lookup used, so the
    # asym pair this compares is the asym pair that was found missing. Matching on the chain
    # letters instead would silently keep a row whose roles join to a dropped pair.
    hit = None
    for ifc, x, y in ifp.get(p, ()):
        if {(r.get("chain1") or "").strip(), (r.get("chain2") or "").strip()} == set(ifc.split("_")[-1]):
            hit = (min(x, y), max(x, y))
            break
    (dropped if (p, hit) in bad else kept).append(r)

if not kept:
    sys.exit("STOP: every row was dropped -- the roles join is wrong, not the labels.")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(kept)
print(f"-> {a.out}")
print(f"  {len(rows)} pooled row(s) in, {len(kept)} kept, {len(dropped)} dropped")
print(f"  {len(bad)} interface(s) short of a DockQ label over the {len(decl)} declared structure(s)")
for p, k in sorted(bad):
    print(f"    dropped {p} asym {k}")
