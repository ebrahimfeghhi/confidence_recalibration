#!/usr/bin/env python
"""Name every pose's chains after one chosen pose, so "interface A-B" means the same thing in all of them.

SUMMARY
    Protenix names duplicate chains arbitrarily: the antigen copy it calls B in one pose can be C in
    the next. Later steps sort poses per interface, so they need one consistent naming across the pool.
    This picks one pose per complex as the anchor and expresses every other pose in its terms.

    WHICH CHAINS ARE ANTIBODY AND ANTIGEN is taken as known (see pose_roles). Nothing else here reads
    the crystal: which pairs touch comes from the pose's own coordinates, the confidence from the
    pose's own summary json. A chain the benchmark leaves unannotated takes part in no pair.

    WHEN IT DOES ANYTHING -- only when a complex has more than one copy of an entity. With one antibody
    and one antigen there is a single possible assignment, so every mapping is the identity.

    WHY THE LABEL MATTERS when the copies are identical -- they share a sequence but not a position, and
    in a given pose the antibody touches one and not the other:

        pose 1: antibody A binds the copy labelled B  ->  (A,B) is a real interface, (A,C) is nothing
        pose 2: antibody A binds the copy labelled C  ->  (A,C) is a real interface, (A,B) is nothing

    ONE ANCHOR PER COMPLEX, not per interface. The mapping pairs each anchor chain with exactly one pose
    chain and no two share a partner, so the anchor's (A,B) and (A,C) necessarily land on different chain
    pairs of the pose.

    THE ANCHOR is the pose whose antibody-antigen pairs in contact have the highest mean
    chain_pair_iptm. Averaged over the pairs the pose actually forms, so a pose is not penalised for
    copies it leaves untouched.

INPUTS
    --poses:   csv of pdb, seed, sample -- the pool. Required.
    --out-dir: where the csvs and the manifest go. Required.
    --jobs:    worker processes. Default 8. CPU only.

OUTPUTS
    <out-dir>/pose_contacts.csv:     per pose, every antibody-antigen chain pair in contact, in the
                                     POSE's letters, with its chain_pair_iptm -- read by
                                     pose_interfaces.py, so no pose is parsed twice
    <out-dir>/chain_maps_anchor.csv: per pose, {anchor_chain: pose_chain}
    <out-dir>/anchor_chains.csv:     per complex, each anchor chain's role
    <out-dir>/anchor_manifest.json:  the anchor chosen per complex, and what was mapped or failed

EXAMPLE
    anchor_relabel.py --poses exp/poses.csv --out-dir exp/chain_frame

FUNCTIONS
    pose_roles():     every pose -> {pose chain: antibody | antigen}, every copy of a molecule alike
    pose_residues():  one pose -> each chain's residue names
    pose_pairs():     one pose -> its antibody-antigen pairs in contact, with their ipTM
    map_to_anchor():  one pose -> its chain mapping onto the anchor, as a csv row
    main():           contacts for every pose, pick the anchors, map every pose
"""
import argparse
import collections
import csv
import datetime
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402

logging.disable(logging.WARNING)   # pxmeter logs a warning per unmapped ligand, thousands of poses

MAP_FIELDS = ["pdb", "seed", "sample", "chain_mapping", "anchor_seed", "anchor_sample", "seconds"]
ANTIBODY, ANTIGEN = "antibody", "antigen"


def pose_roles(poses):
    """in: [(pdb, seed, sample)].  out: {(pdb, seed, sample): {pose chain: "antibody" | "antigen"}}

    Which chains are antibody and antigen is taken as KNOWN, and read the way the benchmark records it:
    its role file names the crystal's antibody and antigen chains, and each pose's crystal chain map
    says which pose chain each of those is. Only the role is used -- any copy of an antibody is an
    antibody -- never which chains bind. A chain the benchmark does not annotate gets no role. Raises
    if a pose has no map covering its role chains: a role must never be guessed.
    """
    import pool_index
    xmaps = pool_index.load_chain_maps(f"{paths.PXM2025_DIR}/chain_maps.csv")
    ab, ag = collections.defaultdict(set), collections.defaultdict(set)
    for iface, (a, g) in pool_index.load_roles(str(paths.PXM2025_CHAIN_ROLES)).items():
        ab[iface.split("_")[0]].add(a)
        ag[iface.split("_")[0]].add(g)
    out, bad = {}, []
    for p, s, x in poses:
        m = xmaps.get((p, s, x))
        if not m or not (ab[p] | ag[p]) <= set(m):
            bad.append((p, s, x))
            continue
        role = {**{m[c]: ANTIBODY for c in ab[p]}, **{m[c]: ANTIGEN for c in ag[p]}}
        # A ROLE BELONGS TO THE MOLECULE, NOT THE COPY. The benchmark often annotates one copy, and
        # the crystal map lands it on whichever copy of a pose superposes best -- so labelling only
        # that chain would smuggle in which copy binds. Every chain carrying identical residues (a copy
        # of the same input entity) takes the role.
        res = pose_residues(paths.pool_pose(p, s, x))
        for c, r in list(role.items()):
            for d, seq in res.items():
                if seq == res[c]:
                    if role.get(d, r) != r:
                        raise SystemExit(f"{p} {s} {x}: copies {c} and {d} annotated as both roles")
                    role[d] = r
        out[(p, s, x)] = role
    if bad:
        raise SystemExit(f"{len(bad)} poses have no crystal chain map covering their role chains, "
                         f"e.g. {bad[:3]}")
    return out


def pose_residues(path):
    """in: a pose cif.  out: {chain: tuple of residue names} -- identical for copies of one entity"""
    import gemmi
    return {ch.name: tuple(r.name for r in ch) for ch in gemmi.read_structure(path)[0]}


def pose_pairs(task):
    """in: (pdb, seed, sample, {pose chain: role}).
    out: (pdb, seed, sample, [(ab chain, ag chain, chain_pair_iptm)] in contact, error or "")
    """
    import contacts
    import pool_index
    pdb, seed, sample, role = task
    path = paths.pool_pose(pdb, seed, sample)
    try:
        ab = sorted(c for c, r in role.items() if r == ANTIBODY)
        ag = sorted(c for c, r in role.items() if r == ANTIGEN)
        touching = contacts.contacting(path, [(a, g) for a in ab for g in ag])
        return (pdb, seed, sample,
                [(a, g, pool_index.pair_iptm_at(path, a, g)) for a, g in touching], "")
    except Exception as e:
        return pdb, seed, sample, [], f"{type(e).__name__}: {e}"


def map_to_anchor(task):
    """in: (pdb, seed, sample, anchor seed, anchor sample)
    out: a csv row; chain_mapping is "" and anchor_sample carries the reason when mapping fails
    """
    pdb, seed, sample, aseed, asamp = task
    from pxmeter.mapping import MappingResult
    t = time.time()
    base = {"pdb": pdb, "seed": seed, "sample": sample, "anchor_seed": aseed, "anchor_sample": asamp}
    apath = paths.pool_pose(pdb, aseed, asamp)
    if (seed, sample) == (aseed, asamp):
        import gemmi                              # the anchor against itself is the identity
        cm = {c.name: c.name for c in gemmi.read_structure(apath)[0]}
    else:
        try:
            m = MappingResult.from_cifs(apath, paths.pool_pose(pdb, seed, sample))
        except Exception as e:
            return {**base, "chain_mapping": "", "seconds": round(time.time() - t, 2),
                    "anchor_sample": f"FAILED: {type(e).__name__}"}
        cm = {str(k): str(v) for k, v in m.chain_mapping.items()}
    return {**base, "chain_mapping": json.dumps(cm, sort_keys=True),
            "seconds": round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=8)
    A = ap.parse_args()
    os.makedirs(A.out_dir, exist_ok=True)

    poses = sorted({(r["pdb"], r["seed"], int(r["sample"])) for r in csv.DictReader(open(A.poses))})
    roles = pose_roles(poses)
    import multiprocessing as mp

    # 1. every pose's antibody-antigen pairs in contact, with their confidence
    t0 = time.time()
    pairs, failed = {}, []
    fcon = f"{A.out_dir}/pose_contacts.csv"
    with open(fcon, "w", newline="") as fh, mp.Pool(A.jobs) as pool:
        w = csv.writer(fh)
        w.writerow(["pdb", "seed", "sample", "ab_chain", "ag_chain", "pair_iptm"])
        tasks = [(p, s, x, roles[(p, s, x)]) for p, s, x in poses]
        for i, (p, s, x, got, err) in enumerate(pool.imap_unordered(pose_pairs, tasks, 8), 1):
            if err:
                failed.append(f"{p} {s} {x}: {err}")
                continue
            pairs[(p, s, x)] = got
            w.writerows([p, s, x, a, g, f"{v:.6f}"] for a, g, v in got)
            if i % 2000 == 0 or i == len(tasks):
                print(f"  contacts {i}/{len(tasks)}  {time.time() - t0:.0f}s  failed {len(failed)}",
                      flush=True)
    if failed:
        raise SystemExit(f"{len(failed)} poses unreadable, e.g. {failed[:3]}")

    # 2. the anchor: highest mean ipTM over the antibody-antigen pairs the pose forms
    anchors, no_contact = {}, collections.Counter()
    for (p, s, x), got in sorted(pairs.items()):
        if not got:
            no_contact[p] += 1
            continue
        v = sum(i for _, _, i in got) / len(got)
        if p not in anchors or v > anchors[p][2]:
            anchors[p] = (s, x, v)
    missing = sorted({p for p, _, _ in poses} - set(anchors))
    if missing:
        raise SystemExit(f"no pose of {missing} forms any antibody-antigen contact")

    # 3. every pose mapped onto its complex's anchor
    fmap = f"{A.out_dir}/chain_maps_anchor.csv"
    t0, nfail = time.time(), 0
    tasks = [(p, s, x, anchors[p][0], anchors[p][1]) for p, s, x in poses]
    with open(fmap, "w", newline="") as fh, mp.Pool(A.jobs) as pool:
        w = csv.DictWriter(fh, fieldnames=MAP_FIELDS)
        w.writeheader()
        for i, row in enumerate(pool.imap_unordered(map_to_anchor, tasks, 8), 1):
            w.writerow(row)
            fh.flush()          # a kill loses the pose in flight, not the run
            nfail += not row["chain_mapping"]
            if i % 2000 == 0 or i == len(tasks):
                print(f"  mapped {i}/{len(tasks)}  {time.time() - t0:.0f}s  failed {nfail}", flush=True)

    fch = f"{A.out_dir}/anchor_chains.csv"
    with open(fch, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain", "role"])
        for p, (s, x, _) in sorted(anchors.items()):
            w.writerows([p, c, r] for c, r in sorted(roles[(p, s, x)].items()))

    json.dump({"poses": A.poses, "n_complexes": len(anchors),
               "anchors": {p: {"seed": s, "sample": x, "mean_pair_iptm": round(v, 6)}
                           for p, (s, x, v) in sorted(anchors.items())},
               "poses_without_contact": dict(no_contact), "poses_mapped": len(tasks),
               "poses_failed": nfail,
               "written": datetime.datetime.now().isoformat(timespec="seconds")},
              open(f"{A.out_dir}/anchor_manifest.json", "w"), indent=1)
    print(f"  {len(anchors)} anchors; {len(tasks)} poses mapped, {nfail} failed -> {A.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
