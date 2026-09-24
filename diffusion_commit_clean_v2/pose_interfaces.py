#!/usr/bin/env python
"""Decide which antibody-antigen chain pairs count as interfaces, using the poses instead of the crystal.

SUMMARY
    Runs after anchor_relabel.py and needs its output: the pairs are named in ANCHOR chain letters, so
    a pair means the same thing in every pose.

    Later steps sort poses per interface, which requires knowing which pairs to sort for. The input
    cannot tell you: 9k94 has 5 antibody and 5 antigen chains, so 25 possible pairings, and only 5 are
    real. Enumerating all of them over-generates by 113 across the benchmark's 292 interfaces. So the
    reference was supplying which pairs touch, and this replaces it with:

        a chain pair is an interface if those two chains come within 5.0 A in AT LEAST ONE pose

    The cutoff is not tuned -- it is the heavy-atom contact distance DockQ already uses throughout this
    pipeline. There is deliberately no "must appear in k poses" rule, which would be a knob.

    Measured against the reference list: 292 of 292 recovered, 53 extra declared. Nothing is missed, so
    the evaluation set is unchanged. The 53 extras are pairs that touch in some pose only because that
    pose mis-docked; they get their own donor list, but they cannot produce a false result because there
    is no ground truth to score them against. Their donor cost measured as nil -- donors per complex went
    up, 33.6 to 35.4, at identical coverage, because the 50-donor cap rarely binds.

    A pair is labelled by the anchor's chain names even when the ANCHOR ITSELF has no contact there: the
    mapping matches chains by superposition and never looks at contacts. The anchor supplies names, the
    poses supply which names are interfaces.

INPUTS
    --out:         where pose_interfaces.csv goes. Required.
    --anchor-dir:  where anchor_relabel.py wrote its csvs and manifest. Default paths.ANCHOR_DIR. This
                   used to be a literal in this file, so pointing anchor_relabel at a new directory left
                   this step silently reading the old maps.
    --pool:        pool csv listing the poses. Default paths.PXM2025_POOL_CSV.
    --jobs:        worker processes computing contacts in parallel. Default 8. CPU only, one pose cif
                   at a time per process.
    --poses:       how many poses per complex to inspect. Default 0, meaning all of them. Fewer poses
                   finds fewer contacts: at 10 poses the same measurement missed 2 reference interfaces
                   and declared 37 extras, against 0 and 53 using every pose.

    Files read:
    anchor csvs: chain_maps_anchor.csv and chain_roles_anchor.csv from anchor_relabel.py
    crystal roles: for the single-interface complexes, which have no anchor-frame roles, and to mark
                   which declared pairs have a reference counterpart and so can be scored
    pose cifs:   via paths.pool_pose()

OUTPUTS
    <out>/pose_interfaces.csv: one row per chain pair

        pdb, ab_chain, ag_chain, poses_in_contact, pairs_found, have_gt

    poses_in_contact: how many poses put those two chains within 5.0 A (0 means the reference has this
                      interface but no pose forms it)
    pairs_found:      how many pairs were declared for that complex
    have_gt:          1 if the reference contains this pair, so it can be scored; 0 if it is
                      selectable but not evaluable

EXAMPLE
    pose_interfaces.py --out anchor_merged --jobs 8
    pose_interfaces.py --out /tmp/try --poses 10 --jobs 4

FUNCTIONS
    candidate_pairs():   the anchor roles -> every antibody x antigen pair per complex, and which of
                         them the reference contains
    contacting_pairs():  one complex -> how many poses put each candidate pair in contact
    main():              build the candidates, count contacts in parallel, write the csv
"""
import argparse
import collections
import csv
import json
import os
import sys

# the repo root is this file's parent, so nothing here carries an absolute path; paths.py owns
# every location from that point on
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
from fmt import table  # noqa: E402

sys.path.insert(0, str(paths.METRICS_DIR))
import metrics  # noqa: E402

def candidate_pairs(anchor_roles, xtal_roles):
    """in: anchor-frame roles and crystal roles.
    out: (ab chains, ag chains, reference pairs) per complex, all in anchor chain letters
    """
    ab, ag, ref = (collections.defaultdict(set) for _ in range(3))
    for iface, (a, g) in anchor_roles.items():
        pdb = iface.split("_")[0]
        ab[pdb].add(a)
        ag[pdb].add(g)
        ref[pdb].add((a, g))
    for iface, (a, g) in xtal_roles.items():
        pdb = iface.split("_")[0]
        if pdb not in ref:          # single-interface complexes have no anchor-frame roles
            ab[pdb].add(a)
            ag[pdb].add(g)
            ref[pdb].add((a, g))
    return ab, ag, ref


def contacting_pairs(task):
    """in: (pdb, [(pose path, chain map)], candidate pairs).
    out: (pdb, {pair: how many poses put it in contact})
    """
    pdb, items, pairs = task
    seen = collections.Counter()
    for pose, chain_map in items:
        if not os.path.exists(pose):
            continue
        for a_ab, a_ag in pairs:
            if a_ab not in chain_map or a_ag not in chain_map:
                continue
            try:
                n = len(metrics.epitope_set(pose, chain_map[a_ab], chain_map[a_ag],
                                            cutoff=metrics.CONTACT_CUTOFF))
            except Exception:
                continue
            if n > 0:
                seen[(a_ab, a_ag)] += 1
    return pdb, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor-dir", default=str(paths.ANCHOR_DIR))
    ap.add_argument("--pool", default=str(paths.PXM2025_POOL_CSV))
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--poses", type=int, default=0, help="0 = every pose")
    A = ap.parse_args()

    anchor_maps = metrics.load_chain_maps(f"{A.anchor_dir}/chain_maps_anchor.csv")
    anchor_roles = metrics.load_roles(f"{A.anchor_dir}/chain_roles_anchor.csv")
    fman = f"{A.anchor_dir}/anchor_manifest.json"
    if os.path.exists(fman):
        man = json.load(open(fman))
        print(f"  anchor data: {man['n_complexes']} complexes from {man['pool_csv']}, "
              f"written {man['written']}")
    else:
        print(f"  NOTE no {fman}: cannot tell which complexes these maps cover")
    xtal_maps, xtal_roles = metrics.load_chain_maps(), metrics.load_roles()
    ab, ag, ref = candidate_pairs(anchor_roles, xtal_roles)

    poses = collections.defaultdict(list)
    with open(A.pool) as fh:
        for r in csv.DictReader(fh):
            poses[r["pdb"]].append((r["seed"], int(r["sample"])))

    tasks = []
    for pdb in sorted(ref):
        keys = sorted(set(poses.get(pdb, ())))
        if A.poses:
            keys = keys[:A.poses]
        items = [(paths.pool_pose(pdb, s, x), m) for s, x in keys
                 if (m := anchor_maps.get((pdb, s, x)) or xtal_maps.get((pdb, s, x)))]
        if items:
            tasks.append((pdb, items, sorted({(a, g) for a in ab[pdb] for g in ag[pdb]})))

    print(f"  complexes: {len(tasks)}   candidate pairs: {sum(len(t[2]) for t in tasks)}", flush=True)
    import multiprocessing as mp
    found = {}
    with mp.Pool(A.jobs) as workers:
        for i, (pdb, seen) in enumerate(workers.imap_unordered(contacting_pairs, tasks), 1):
            found[pdb] = seen
            if i % 10 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    os.makedirs(A.out, exist_ok=True)
    rows, stat = [], collections.Counter()
    for pdb, seen in sorted(found.items()):
        for (a, g), n in sorted(seen.items()):
            have = (a, g) in ref[pdb]
            stat["evaluable (in the reference)" if have else "extra (no ground truth)"] += 1
            rows.append([pdb, a, g, n, len(seen), int(have)])
        for a, g in sorted(ref[pdb] - set(seen)):
            stat["missed (in reference, no pose forms it)"] += 1
            rows.append([pdb, a, g, 0, len(seen), 1])
    with open(f"{A.out}/pose_interfaces.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "ab_chain", "ag_chain", "poses_in_contact", "pairs_found", "have_gt"])
        w.writerows(rows)
    print()
    print(table(["", "chain pairs"], [[k, v] for k, v in stat.most_common()], align="l r"))
    print(f"\n  wrote {A.out}/pose_interfaces.csv   ({len(rows)} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
