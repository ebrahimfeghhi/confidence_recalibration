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
SEEDS = ()      # set from --seeds in main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="the structures that were re-folded")
    ap.add_argument("--out", required=True, help="training/overfit_hard")
    ap.add_argument("--results", default="training/fold/results.csv")
    ap.add_argument("--pool", default="training/fold/pool.csv")
    ap.add_argument("--trunks", default="training/fold/trunks")
    ap.add_argument("--roles-all", default="training/fold/chain_roles_all.csv")
    ap.add_argument("--structures-src", default="train_val_splits/cache_train_structures.csv",
                    help="where a manifest row for a structure new to this pool is copied from")
    ap.add_argument("--seeds", default="500000-500014",
                    help="the seed span the re-folded structures now carry")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    global SEEDS
    _lo, _hi = (int(x) for x in a.seeds.split("-"))
    SEEDS = tuple(range(_lo, _hi + 1))

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
        print("nothing NEW qualifies; the seed span still has to be written")
    if not a.apply:
        print("dry run; pass --apply")
        return

    # pool.csv and structures.csv grow; the eval pools are untouched.
    for name in ("pool.csv", "structures.csv", "chain_roles.csv"):
        src = os.path.join(a.out, name)
        if os.path.exists(src):
            shutil.copy2(src, src + ".before_extend")
    with open(os.path.join(a.out, "pool.csv"), "a", newline="") as fh:
        w = csv.writer(fh)
        for iface in new:
            pdb, pair = iface.split("_")
            w.writerow([pdb, pair[0], pair[1]])

    # ---- structures.csv. THE MANIFEST IS DECLARED, NOT DISCOVERED (phase_k_train.py:3769): the
    # `seeds` column is what the trainer expands into units, so a re-folded structure whose row
    # still says 500000 contributes exactly one seed however many trunks exist on disk. Updating
    # that column IS the step that makes the re-fold reach training; without it the whole fold is
    # a no-op. And `pdb` is column 1, not column 0 -- column 0 is `cluster`.
    sfile = os.path.join(a.out, "structures.csv")
    with open(sfile) as fh:
        rdr = csv.DictReader(fh)
        head = rdr.fieldnames
        body = list(rdr)
    assert head[1] == "pdb" and "seeds" in head, f"unexpected manifest schema: {head}"
    seed_lo, seed_hi = min(SEEDS), max(SEEDS)
    span = f"{seed_lo}-{seed_hi}"
    touched = 0
    known = {r["pdb"] for r in body}
    for r in body:
        if r["pdb"] in have:
            # EVERY re-folded structure, not only the ones that gained an interface: the 18 that
            # were already in the pool are precisely the ones whose hard poses were unreachable.
            if r["seeds"] != span:
                r["seeds"] = span
                touched += 1
    add = sorted({i.split("_")[0] for i in new} - known)
    src = {r["pdb"]: r for r in csv.DictReader(open(a.structures_src))} if add else {}
    for p in add:
        if p not in src:
            raise SystemExit(f"STOP: {p} is not in {a.structures_src}, so its manifest row cannot "
                             f"be built. Refusing to write a row with empty columns.")
        row = {k: src[p].get(k, "") for k in head}
        row["pdb"] = p
        row["seeds"] = span
        body.append(row)
    with open(sfile, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=head)
        w.writeheader()
        w.writerows(body)
    print(f"  structures.csv: {touched} row(s) moved to seeds {span}, {len(add)} row(s) added")

    # the new interfaces get their own eval pool, at the same budget as hard15
    with open(os.path.join(a.out, "pool_new15.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain1", "chain2"])
        for iface in new:
            pdb, pair = iface.split("_")
            w.writerow([pdb, pair[0], pair[1]])

    # ---- chain roles. ONE ROW PER CHAIN, not per structure: keying a dict on pdb keeps only the
    # last chain and the pool-local file silently loses the rest.
    rfile = os.path.join(a.out, "chain_roles.csv")
    if os.path.exists(rfile) and os.path.exists(a.roles_all) and add:
        with open(rfile) as fh:
            rdr = csv.DictReader(fh)
            rfields = rdr.fieldnames
            have_rows = {(r["pdb"], r.get("chain", "")) for r in rdr}
        extra = [r for r in csv.DictReader(open(a.roles_all))
                 if r["pdb"] in add and (r["pdb"], r.get("chain", "")) not in have_rows]
        if extra:
            with open(rfile, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=rfields)
                for r in extra:
                    w.writerow({k: r.get(k, "") for k in rfields})
            print(f"  chain_roles.csv += {len(extra)} chain row(s) over {len(add)} structure(s)")
    print(f"added {len(new)} interface(s) over {len(add)} new structure(s); "
          f"pool_new15.csv written for the separate eval")


if __name__ == "__main__":
    main()
