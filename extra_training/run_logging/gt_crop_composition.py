#!/usr/bin/env python
"""How many ANTIBODY and ANTIGEN tokens does the ground-truth crop actually keep?

WHY. A fixed-size per-pose crop needs a budget, and 256 would be a number picked from Protenix's
training config rather than from this data. This reports what the crop that DID help (section 9b:
contact 5 A, margin 10 A, on label coordinates, one mask shared by all poses) kept on each side.

THE BUDGET IS PER SCORED CHAIN, NOT PER SIDE. pool.csv lists H-ag and L-ag as SEPARATE rows, so
ipTM is computed per chain pair and its d0 = 1.24*(n-15)^(1/3) - 1.8 uses that pair's own n. A
budget of "K antibody tokens" split across H and L would leave each pair's n free to move; the
budget has to fix each scored chain's kept count.

d0 ONLY HAS TO BE CONSTANT WITHIN A STRUCTURE. Poses are never ranked across structures, so a
chain smaller than the budget can simply be kept whole -- capping at min(K, chain size) still
gives every pose in that structure the same n. That is why the antigen-size distribution below
matters for coverage, not for comparability.

Roles come from chain_roles.csv; model chains are lettered A, B, C... in asym_id order, so a model
chain letter's alphabet position IS its asym index (phase_k_eval.py:55).

    python -I run_logging/gt_crop_composition.py --pool training/valfull/pool.csv \
        --trunks validation/fold/trunks --labels validation/fold/labels \
        --roles validation/fold/chain_roles.csv --seed-base 400000
"""
import argparse
import csv
import json
import os
import statistics as st
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phases"))
import phase_j_build_cache as BC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--trunks", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--roles", required=True)
ap.add_argument("--contact", type=float, default=5.0)
ap.add_argument("--margin", type=float, default=10.0)
ap.add_argument("--seed-base", type=int, default=400000)
ap.add_argument("--out", default="")
a = ap.parse_args()

roles = {}
for r in csv.DictReader(open(a.roles)):
    roles.setdefault(r["pdb"], []).append((r["model_chain"], r["role"]))

pdbs = []
for r in csv.DictReader(open(a.pool)):
    if r["pdb"] not in pdbs:
        pdbs.append(r["pdb"])


def side(role):
    return "ab" if role in ("H", "L") else "ag"


rows = []
for pdb in pdbs:
    tp = os.path.join(a.trunks, pdb, f"free_{a.seed_base}.npz")
    lp = os.path.join(a.labels, pdb, f"free_{a.seed_base}.npz")
    if pdb not in roles or not (os.path.exists(tp) and os.path.exists(lp)):
        continue
    z = np.load(tp, allow_pickle=True)
    lab = np.load(lp, allow_pickle=True)
    keep, _ = BC.interface_crop_tokens(z, lab, a.contact, a.margin, all_poses=True)
    if keep is None:
        continue
    asym = np.asarray(z["ci_feat_asym_id"]).reshape(-1)
    per = {}
    for mc, role in roles[pdb]:
        ai = ord(mc) - ord("A")           # model chains are lettered in asym order
        m = asym == ai
        if not m.any():
            continue
        per[role] = dict(chain=mc, role=role, side=side(role),
                         full=int(m.sum()), kept=int((m & keep).sum()))
    if not per:
        continue
    rows.append(dict(pdb=pdb, n_tok=int(asym.shape[0]), kept=int(keep.sum()), chains=per))

if not rows:
    sys.exit("  nothing measured")


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(p * (len(v) - 1))))]


def table(title, items, n=None):
    if not items:
        print(f"  {title}: none")
        return
    # n is the number of CHAINS/STRUCTURES behind the rows, not the number of rows -- printing
    # len(items) said "n=3" for a 53-structure table because items is the column list.
    n = len(items[0][1]) if n is None else n
    print(f"  {title:<28}n={n:<5}"
          f"{'min':>8}{'p10':>8}{'median':>9}{'p90':>8}{'max':>8}")
    for nm, v in items:
        print(f"    {nm:<28}      {min(v):>8.0f}{q(v,.1):>8.0f}{st.median(v):>9.0f}"
              f"{q(v,.9):>8.0f}{max(v):>8.0f}")


print(f"  GROUND-TRUTH crop, contact {a.contact} A, margin {a.margin} A -- "
      f"{len(rows)} structure(s)")
print(f"  {sum(1 for r in rows if 'L' in r['chains'])} have a light chain, "
      f"{sum(1 for r in rows if 'L' not in r['chains'])} are heavy-only")
print()

for role in ("H", "L"):
    v = [c[role] for r in rows for c in [r["chains"]] if role in c]
    if v:
        table(f"chain {role}", [("full chain, tokens", [x["full"] for x in v]),
                                ("KEPT by the crop", [x["kept"] for x in v]),
                                ("kept / full", [100.0 * x["kept"] / x["full"] for x in v])])
        print()

agv = [c for r in rows for k, c in r["chains"].items() if c["side"] == "ag"]
table("antigen chains (all)", [("full chain, tokens", [x["full"] for x in agv]),
                               ("KEPT by the crop", [x["kept"] for x in agv]),
                               ("kept / full", [100.0 * x["kept"] / x["full"] for x in agv])])
print()

ab_tot = [sum(c["kept"] for c in r["chains"].values() if c["side"] == "ab") for r in rows]
ag_tot = [sum(c["kept"] for c in r["chains"].values() if c["side"] == "ag") for r in rows]
table("per structure, kept totals", [("antibody side", ab_tot), ("antigen side", ag_tot),
                                     ("whole crop", [r["kept"] for r in rows]),
                                     ("whole structure", [r["n_tok"] for r in rows])])
print()

small = [x for x in agv if x["full"] < 50]
print(f"  antigen chains smaller than 50 tokens: {len(small)} of {len(agv)}"
      f"  -> {sorted(x['full'] for x in small)[:12]}")
print("  a chain smaller than the budget is kept whole; n stays constant WITHIN that structure,")
print("  which is all d0 comparability needs.")

if a.out:
    json.dump(rows, open(a.out, "w"), indent=1)
    print(f"\n  per-structure rows -> {a.out}")
