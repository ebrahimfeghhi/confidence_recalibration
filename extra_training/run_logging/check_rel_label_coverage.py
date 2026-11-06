#!/usr/bin/env python
"""Does every (pdb, seed, sample, iface) the run will train have a true-iTM label?

WHY A PRE-LAUNCH GATE AND NOT THE TRAINER'S OWN CHECK. phase_k_train resolves a rel label per pose
as it reaches that pose, so a gap is found when the epoch gets there -- which on this population is
up to 3.6 hours in. Raised in review: the launcher's existing gate is directory-level, checking that
each trainable pdb has a <pdb>.npz and that the label build reported zero unmapped rows, and that
cannot see a structure whose file exists but is short of a seed, a sample or an interface. This
walks the exact key set instead, for about a second per hundred structures.

WHAT THE KEY IS. rank_loss keys the true-iTM labels by (seed, sample_index, iface NAME) -- the
interface name and not an asym pair, because the asym pair is resolved per pose by OST and is not
stable across poses, while DockQ is a property of the interface. So the expected set is every
declared (pdb, seed) times every interface the pool names for that pdb times samples 0..nsample-1.

IT IS READ-ONLY and names what is missing rather than summarising a count, because the useful
output is which structure to rebuild.
"""
import argparse, collections, csv, os, sys

import numpy as np


def parse_seed_field(txt):
    out = set()
    for part in str(txt).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.update(range(int(lo), int(hi) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="training/fold/itm_labels_extended")
    ap.add_argument("--manifest", default="train_val_splits/cache_train_structures_extended.csv")
    ap.add_argument("--pool", default="training/combined/pool.csv")
    ap.add_argument("--skip", default="9mx1,4zto,9q5m")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--exclude-ifaces", default="docs/unrecoverable_interfaces.csv",
                    help="interfaces no route can score; excused from the expected set by name")
    ap.add_argument("--limit", type=int, default=10, help="how many gaps to print")
    a = ap.parse_args()

    skip = {x.strip().lower() for x in a.skip.split(",") if x.strip()}
    declared = {}
    for r in csv.DictReader(open(a.manifest)):
        p = r["pdb"].strip().lower()
        if p in skip:
            continue
        sd = parse_seed_field(r.get("seeds", "") or "")
        if sd:
            declared[p] = sorted(sd)

    ifs = collections.defaultdict(set)
    for r in csv.DictReader(open(a.pool)):
        p = r["pdb"].strip().lower()
        if p in declared:
            ifs[p].add(f"{p}_{''.join(sorted((r['chain1'], r['chain2'])))}")

    excused = {}
    if a.exclude_ifaces and os.path.exists(a.exclude_ifaces):
        for r in csv.DictReader(open(a.exclude_ifaces)):
            excused[r["iface"].strip()] = parse_seed_field(str(r.get("seeds", "") or "").strip())

    missing, no_file, checked, excused_hit = [], [], 0, 0
    for p in sorted(declared):
        fp = os.path.join(a.labels, f"{p}.npz")
        if not os.path.exists(fp):
            no_file.append(p)
            continue
        try:
            z = np.load(fp, allow_pickle=True)
            keys = set()
            # STORED AS PARALLEL ARRAYS OR AS A MAPPING, depending on the builder's version; both
            # are read rather than assuming one, because guessing wrong would report everything
            # missing and look like a data disaster instead of a reader bug.
            if {"seed", "sample", "iface"} <= set(z.files):
                for sd, sm, ifc in zip(z["seed"], z["sample"], z["iface"]):
                    keys.add((int(sd), int(sm), str(ifc)))
            else:
                for k in z.files:
                    parts = str(k).split("|")
                    if len(parts) == 3 and parts[0].isdigit() and parts[1].isdigit():
                        keys.add((int(parts[0]), int(parts[1]), parts[2]))
            if not keys:
                no_file.append(f"{p} (unreadable key set: files={list(z.files)[:4]})")
                continue
        except Exception as e:
            no_file.append(f"{p} ({type(e).__name__})")
            continue
        for sd in declared[p]:
            for ifc in sorted(ifs[p]):
                ex = excused.get(ifc)
                if ex is not None and sd in ex:
                    # COUNTED IN KEYS, NOT IN GROUPS. Raised in review: this incremented once per
                    # (interface, seed) while printing "key(s) excused", understating by nsample.
                    excused_hit += a.nsample
                    continue
                for sm in range(a.nsample):
                    checked += 1
                    if (sd, sm, ifc) not in keys:
                        missing.append((p, sd, sm, ifc))

    print(f"rel labels {a.labels}")
    print(f"  {len(declared)} trained structure(s), {checked} expected (seed, sample, iface) key(s)")
    if excused_hit:
        print(f"  {excused_hit} key(s) excused by name via {os.path.basename(a.exclude_ifaces)}")
    if no_file:
        print(f"  {len(no_file)} structure(s) with no readable label: {no_file[:a.limit]}")
    if missing:
        byp = collections.Counter(m[0] for m in missing)
        print(f"  {len(missing)} MISSING key(s) over {len(byp)} structure(s); "
              f"worst: {byp.most_common(5)}")
        for m in missing[:a.limit]:
            print(f"      {m[0]} seed {m[1]} sample {m[2]} {m[3]}")
    if not missing and not no_file:
        print("  COVERAGE COMPLETE: every expected key is present")
    return 1 if (missing or no_file) else 0


if __name__ == "__main__":
    sys.exit(main())
