#!/usr/bin/env python
"""Add the newly folded interfaces to the hard TRAINING pool, by the pool's own criteria.

THE CRITERIA ARE THE ORIGINAL ONES, GROUND TRUTH ONLY: at least one acceptable candidate
(DockQ >= 0.23) and at most 10% acceptable. No head's score enters, or the overfit test would be
measuring the thing it is supposed to test.

HARDNESS IS COMPUTED OVER TRUNK-AVAILABLE POSES, not over every row in results.csv. That
distinction is what made 19 of the 21 budget-1 lists unwinnable by construction the first time: the
rows described 50 poses while the trainer could load 5. A pose counts here only if its
trunks/<pdb>/free_<seed>.npz exists.

WHAT IS WRITTEN: pool.csv and structures.csv grow; pool_b15.csv and pool_b1.csv -- the EVAL pools --
are left exactly as they are, so the epoch-by-epoch curve stays one series. The new interfaces get
their own pool_new15.csv.
"""
import argparse
import collections
import csv
import glob
import os
import shutil

ACC = 0.23
MAX_FRAC = 0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="the structures that were re-folded")
    ap.add_argument("--out", required=True, help="training/overfit_hard")
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--pool", default="training/fold/pool.csv")
    ap.add_argument("--trunks", default="training/fold/trunks")
    ap.add_argument("--roles-all", default="training/fold/chain_roles_all.csv")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    want = [l.strip() for l in open(a.list) if l.strip() and l[0] != "#"]
    have = {p: {os.path.basename(g).split("free_")[1][:-4]
                for g in glob.glob(os.path.join(a.trunks, p, "free_*.npz"))} for p in want}

    dq = collections.defaultdict(list)
    for r in csv.DictReader(open(a.results)):
        p = r["iface"].split("_")[0]
        if p not in have or r.get("arm") not in (None, "free"):
            continue
        if r["seed"] not in have[p]:          # no trunk, no pose the trainer can load
            continue
        try:
            dq[r["iface"]].append(float(r["dockq"]))
        except (TypeError, ValueError):
            continue

    keep, why = [], []
    for iface, ds in sorted(dq.items()):
        n_acc = sum(1 for d in ds if d >= ACC)
        frac = n_acc / len(ds)
        ok = n_acc >= 1 and frac <= MAX_FRAC
        why.append((iface, len(ds), n_acc, frac, ok))
        if ok:
            keep.append(iface)

    cur = list(csv.DictReader(open(os.path.join(a.out, "pool.csv"))))
    cur_if = {f"{r['pdb']}_{r['chain1']}{r['chain2']}" for r in cur}
    new = [i for i in keep if i not in cur_if]

    print(f"candidates with a loadable pose: {len(dq)} interface(s) over {len(have)} structure(s)")
    print(f"  qualify (>=1 acceptable, <={MAX_FRAC:.0%} acceptable): {len(keep)}")
    print(f"  already in the training pool: {len(keep) - len(new)}")
    print(f"  NEW: {len(new)}")
    for iface, n, k, f, ok in why:
        if ok and iface in new:
            print(f"    + {iface}: {k} of {n} acceptable ({f:.1%})")
    if not new:
        print("nothing to add")
        return
    if not a.apply:
        print("dry run; pass --apply")
        return

    # pool.csv and structures.csv grow; the eval pools are untouched.
    for name in ("pool.csv", "structures.csv"):
        shutil.copy2(os.path.join(a.out, name), os.path.join(a.out, name + ".before_extend"))
    with open(os.path.join(a.out, "pool.csv"), "a", newline="") as fh:
        w = csv.writer(fh)
        for iface in new:
            pdb, pair = iface.split("_")
            w.writerow([pdb, pair[0], pair[1]])
    sfile = os.path.join(a.out, "structures.csv")
    rows = list(csv.reader(open(sfile)))
    head, body = rows[0], rows[1:]
    known = {r[0] for r in body}
    add = sorted({i.split("_")[0] for i in new} - known)
    with open(sfile, "a", newline="") as fh:
        w = csv.writer(fh)
        for p in add:
            w.writerow([p] + [""] * (len(head) - 1))
    # the new interfaces get their own eval pool, at the same budget as hard15
    with open(os.path.join(a.out, "pool_new15.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain1", "chain2"])
        for iface in new:
            pdb, pair = iface.split("_")
            w.writerow([pdb, pair[0], pair[1]])
    # chain roles for the structures that are new to this pool
    roles = {r["pdb"]: r for r in csv.DictReader(open(a.roles_all))} if os.path.exists(a.roles_all) else {}
    rfile = os.path.join(a.out, "chain_roles.csv")
    if os.path.exists(rfile) and roles:
        cur_roles = list(csv.DictReader(open(rfile)))
        fields = cur_roles[0].keys() if cur_roles else None
        missing = [p for p in add if p in roles]
        if missing and fields:
            with open(rfile, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(fields))
                for p in missing:
                    w.writerow({k: roles[p].get(k, "") for k in fields})
            print(f"  chain_roles.csv += {len(missing)} structure(s)")
    print(f"added {len(new)} interface(s) over {len(add)} new structure(s); "
          f"pool_new15.csv written for the separate eval")


if __name__ == "__main__":
    main()
