#!/usr/bin/env python
"""Choose which poses to self-template on: most confident first, keeping only what is structurally new.

SUMMARY
    The algorithm, per complex:

      1. for each interface, order its poses by pair_iptm, highest first
      2. walk down that list and keep a pose only if its pose-to-pose DockQ to everything already kept
         is below 0.23, stopping at the cap
      3. round-robin the per-interface lists into one set for the complex -- each interface's 1st pick,
         then each 2nd, and so on -- until the cap is reached

    Step 2 is the point: templating on 50 versions of the same wrong answer buys nothing, so the budget
    goes on 50 different guesses. 0.23 is the CAPRI acceptable threshold, the same number used to decide
    whether a pose is correct.

    Step 3 is per COMPLEX because a template is a whole complex and a fold scores every interface of it,
    so "donors for interface A-B" is not something a fold can deliver separately. Round-robin rather than
    a majority vote: capping by "poses the most interfaces asked for" starves the interface with unusual
    poses, which on one complex retained 14% of its own picks against 36% under round-robin.

    NOTHING HERE READS GROUND TRUTH. The order comes from pair_iptm, the filter from pose-to-pose DockQ,
    and the chain pinning from the anchor frame.

    REUSE is keyed on (donor pose, seed, pinned chains). A fold's output is fixed by those three, so a
    fold already done elsewhere can be carried over instead of repeated. The tag is only a name, and
    naming was load-bearing once by accident: two interfaces of one complex sharing tag K00 silently
    swapped donors. Carried rows are re-emitted with the pose and chains filled in, and a claim directory
    is created so a worker treats them as done.

INPUTS
    --out:       experiment directory to write into. Required.
    --cap:       donors per complex. Default 50.
    --seed:      the seed every templated fold uses. Default 430000.
    --old-exp:   an earlier experiment whose folds can be reused. Default the crystal-frame sweep.
    --no-reuse:  refold everything, ignoring --old-exp.
    --pose-interfaces: also select on the interfaces the predictions declare by contact, which have no
                 reference counterpart. Default off.

    Files read:
    matrices:    the .npz files from pose_to_pose_dockq.py -- anchor frame where it exists, crystal frame
                 for the single-interface complexes where the two are identical by construction
    anchor csvs: chain maps and roles, to pin each donor's chains
    old experiment: its queue, results csv and template namespace, to decide what can be reused

OUTPUTS
    <out>/queue.tsv:     one line per fold still to run -- kind, pdb, interface, tag, seed, donor pose
    <out>/results.csv:   rows carried over from the reused folds, with provenance filled in
    <out>/claims/:       a claim per reused donor, so a worker skips it
    <out>/frame_maps.csv, <out>/frame_roles.csv: the exact frame this queue was selected in, so the
                         template builder pins the same chains the selection ranked

EXAMPLE
    structural_diversity_filter.py --out experimental_data/reffree_50 --pose-interfaces \
                                   --old-exp experimental_data/anchor_50

FUNCTIONS
    leaders():      one interface -> its ipTM-ordered, diversity-filtered pose list
    round_robin():  per-interface lists -> one donor set per complex
    old_folds():    an earlier experiment -> what it folded, by (pdb, pose, seed)
    main():         select per complex, carry over what is already folded, write the queue
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

# the repo root is this file's parent, so nothing here carries an absolute path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402
import pool_index  # noqa: E402

DOCKQ_THRESHOLD = 0.23      # CAPRI "acceptable": two poses closer than this are the same answer


def round_robin(per, cap):
    """in: {interface: ordered pose list}, the cap.
    out: [(pose, the interface that picked it)] -- each interface's 1st pick, then each 2nd, ...

    The interface is carried because the template builder needs it to pin the donor's chains. It does
    not restrict scoring: every fold is scored against every interface of its complex.
    """
    out, seen, r = [], set(), 0
    while len(out) < cap and any(r < len(v) for v in per.values()):
        for i in sorted(per):
            if r < len(per[i]):
                p = per[i][r]
                if p not in seen:
                    seen.add(p)
                    out.append((p, i))
                if len(out) >= cap:
                    break
        r += 1
    return out

ANCHOR_MATRICES = str(paths.ANCHOR_MATRICES)
# predicted-only interfaces live in their own directory: a filename cannot distinguish them, since the
# anchor-frame pair 8w5i_AD and the real crystal-frame interface 8w5i_AD produce the same string
SPUR_MATRICES = str(paths.SPUR_MATRICES)
XTAL_MATRICES = str(paths.XTAL_MATRICES)
ANCHOR_DIR = str(paths.ANCHOR_DIR)
NEW_HEADER = "iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm\n"


def leaders(mdir, iface, cap):
    """in: a matrices directory, an interface, the cap.
    out: (seed, sample) of the poses kept -- ipTM order, each new one unlike everything before it
    """
    z = np.load(f"{mdir}/{iface}.npz")
    M, ip = z["dockq"].astype(float), z["pair_iptm"].astype(float)
    keys = [(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])]
    L = []
    for p in np.argsort(-ip):
        p = int(p)
        if all(M[p, j] < DOCKQ_THRESHOLD for j in L):
            L.append(p)
            if len(L) >= cap:
                break
    return [keys[j] for j in L]


def old_folds(exp):
    """What the crystal sweep produced, indexed by (pdb, pose, seed) -> (old_tag, pinned chain sets).

    The old csv cannot answer this alone: its donor_pose column is "-1". The pose comes from the old
    queue via the tag, and the chains from the template namespace that fold actually consumed.
    """
    pose_of, iface_of = {}, {}
    q = f"{exp}/queue.tsv"
    if os.path.exists(q):
        for line in open(q):
            f = line.rstrip("\n").split("\t")
            if len(f) >= 6:
                pose_of[(f[1], f[3])] = f[5]
                iface_of[(f[1], f[3])] = f[2]
    pinned = collections.defaultdict(set)
    jd = next((f"{exp}/{d}/jobs" for d in ("templates", "ns") if os.path.isdir(f"{exp}/{d}/jobs")),
              f"{exp}/templates/jobs")      # older experiments named the namespace ns/
    if os.path.isdir(jd):
        for d in os.listdir(jd):
            p = f"{jd}/{d}/chosen.json"
            if os.path.exists(p):
                pdb = d.split("_")[0]
                for c in json.load(open(p)):
                    pinned[(pdb, c["cif"])].add(tuple(sorted(c.get("chains", []))))
    rows = collections.defaultdict(list)
    idx = {}
    if os.path.exists(f"{exp}/results.csv"):
        for line in open(f"{exp}/results.csv"):
            f = line.rstrip("\n").split(",")
            # PARSE BY WIDTH, NOT BY HEADER. The old file is a mix: rows written before row.py gained
            # a tmpl_chains column have 7 fields, rows written after have 8, and the header says 7.
            # Feeding the whole file to DictReader silently shifts every 8-field row -- sample lands in
            # dockq, so 180 rows arrived carrying DockQ values of 0,1,2,3,4 and a class mean above 1.0.
            # Both widths end in (sample, dockq, iptm), so reading from the RIGHT is unambiguous.
            if len(f) < 7 or f[0] == "iface":
                continue
            iface, seed, arm = f[0], f[1], f[2]
            sample, dockq, iptm = f[-3], f[-2], f[-1]
            pdb = iface.split("_")[0]
            # PREFER THE ROW'S OWN PROVENANCE. Resolving the donor through the tag fails for rows
            # CARRIED from an earlier sweep: a carried donor's tag is deliberately absent from this
            # queue (that is what "reused" means), so pose_of misses and the row is dropped -- 18,730
            # rows silently unusable, reuse collapsing from ~2,100 folds to 476. donor_pose/tmpl_chains
            # exist precisely so a result identifies its own template without consulting a queue file.
            chains_here = set()
            pose = None
            if len(f) >= 8 and f[3] not in ("", "-1"):
                pose = f[3].split()[0]
                for grp in f[4].split():
                    chains_here |= {tuple(sorted(grp.split("+")))}
            if pose is None:
                pose = pose_of.get((pdb, arm))
            if not pose:
                continue
            r = {"iface": iface, "seed": seed, "arm": arm,
                 "sample": sample, "dockq": dockq, "iptm": iptm}
            rows[(pdb, pose, seed)].append(r)
            have = pinned.get((pdb, pose), set()) | chains_here
            idx[(pdb, pose, seed)] = (arm, have)
    return rows, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=430000)
    ap.add_argument("--old-exp", default=f"{paths.DATA_DIR}/top_50_complex")
    ap.add_argument("--no-reuse", action="store_true")
    ap.add_argument("--pose-interfaces", action="store_true",
                    help="also select on interfaces the predictions declare by contact (no reference)")
    ap.add_argument("--anchor-dir", default=None,
                    help="chain maps and roles to select in. Default paths.ANCHOR_DIR")
    ap.add_argument("--xtal-matrices", default=None)
    ap.add_argument("--anchor-matrices", default=None,
                    help="matrix directories. The npz files carry the pose list, so restricting which "
                         "seeds may be selected means pointing these at matrices built from a "
                         "restricted pool -- there is no pool csv on this code path")
    ap.add_argument("--spur-matrices", default=None)
    A = ap.parse_args()

    global ANCHOR_DIR, XTAL_MATRICES, ANCHOR_MATRICES, SPUR_MATRICES
    ANCHOR_DIR = A.anchor_dir or ANCHOR_DIR
    XTAL_MATRICES = A.xtal_matrices or XTAL_MATRICES
    ANCHOR_MATRICES = A.anchor_matrices or ANCHOR_MATRICES
    SPUR_MATRICES = A.spur_matrices or SPUR_MATRICES

    amaps = pool_index.load_chain_maps(f"{ANCHOR_DIR}/chain_maps_anchor.csv")
    aroles = pool_index.load_roles(f"{ANCHOR_DIR}/chain_roles_anchor.csv")
    # chain_roles_all adds the predicted-only pairs; both frames are keyed the same way
    ALL_ROLES = dict(aroles)
    if os.path.exists(f"{ANCHOR_DIR}/chain_roles_all.csv"):
        ALL_ROLES.update(pool_index.load_roles(f"{ANCHOR_DIR}/chain_roles_all.csv"))

    # every interface we can select on: anchor frame where it exists, crystal frame for the
    # single-interface complexes, where the two are identical by construction (one chain pair)
    bypdb = collections.defaultdict(dict)
    for mdir in (XTAL_MATRICES, ANCHOR_MATRICES):
        for f in sorted(os.listdir(mdir)):
            if f.endswith(".npz"):
                i = f[:-4]
                bypdb[i.split("_")[0]][i] = mdir       # anchor dir wins, it is iterated second
    if A.pose_interfaces and os.path.isdir(SPUR_MATRICES):
        for f in sorted(os.listdir(SPUR_MATRICES)):
            if f.endswith(".npz"):
                i = f[:-4]
                # keyed "SPUR:<iface>" so it cannot collide with a real interface of the same name
                bypdb[i.split("_")[0]]["SPUR:" + i] = SPUR_MATRICES
    single = {p for p, v in bypdb.items() if len(v) == 1}

    rows_old, idx_old = ({}, {}) if A.no_reuse else old_folds(A.old_exp)
    os.makedirs(A.out, exist_ok=True)
    os.makedirs(f"{A.out}/claims", exist_ok=True)

    lines, carried, stat = [], [], collections.Counter()
    for pdb, ifs in sorted(bypdb.items()):
        per = {i: leaders(d, i.split("SPUR:")[-1], A.cap) for i, d in sorted(ifs.items())}
        sel = round_robin(per, A.cap)
        for n, (key, iface) in enumerate(sel):
            pose = paths.pool_pose(pdb, key[0], key[1])
            if not os.path.exists(pose):
                stat["donor pose missing"] += 1
                continue
            tag = f"A{n:02d}"
            # the chains this frame requires for THIS donor; crystal frame for single-interface
            # complexes, whose roles file is the only one that names them
            # a predicted-only interface is named in anchor chains already; strip the key prefix
            spur = iface.startswith("SPUR:")
            iface = iface.split("SPUR:")[-1]
            use_a = spur or ifs.get(iface, ifs.get("SPUR:" + iface)) == ANCHOR_MATRICES
            m = (amaps if use_a else pool_index.load_chain_maps()).get((pdb, key[0], key[1]))
            rl = (ALL_ROLES if use_a else pool_index.load_roles()).get(iface)
            need = None
            if m and rl and rl[0] in m and rl[1] in m:
                need = {m[rl[0]], m[rl[1]]}
            k = (pdb, pose, str(A.seed))
            old = idx_old.get(k)
            # RECORD THE FULL CHAIN SET, not the pair. A template IS the full set of chains written
            # into it; the pair is only a constraint on which must be present. Writing the pair here
            # while folder.py writes the full set made tmpl_chains mean two different things, so the
            # same fold could not be recognised as the same fold. Verified once on 1625 carried
            # donors: given the same required pair, the contact heuristic picks the same extras, so
            # the matched template equals what this frame would have built.
            match = next((t for t in old[1] if need and need <= set(t)), None) if old else None
            if match is not None:
                stat["reused (rows carried over)"] += 1
                full = "+".join(sorted(match))
                for r in rows_old.get(k, []):
                    carried.append(f"{r['iface']},{r['seed']},{tag},{pose},"
                                   f"{full},{r['sample']},{r['dockq']},{r['iptm']}")
                # the claim name fold.py computes is "<pdb>_<target>_<tag>_<seed>" with target = the
                # queue's interface column, verified against the existing claims dir
                os.makedirs(f"{A.out}/claims/{pdb}_{iface}_{tag}_{A.seed}", exist_ok=True)
                continue
            stat["refold: pinned chains differ" if old else "refold: not folded before"] += 1
            lines.append("\t".join(["tmpl", pdb, iface, tag, str(A.seed), pose]))

    # EMIT THE EXACT FRAME THIS QUEUE WAS SELECTED IN, so build_templates pins the same chains the
    # selection ranked. A complex is entirely in one frame -- anchor if it has several interfaces,
    # crystal if it has one (where the two are identical, there being a single chain pair) -- so mixing
    # them per complex is consistent, while passing only the anchor files would leave every
    # single-interface complex with no roles entry and silently unpinned.
    multi_pdbs = {p for p in bypdb if p not in single}
    xm_rows = [r for r in csv.DictReader(open(f"{paths.PXM2025_DIR}/chain_maps.csv"))
               if r["pdb"] in single]
    am_rows = [r for r in csv.DictReader(open(f"{ANCHOR_DIR}/chain_maps_anchor.csv"))
               if r["pdb"] in multi_pdbs]
    with open(f"{A.out}/frame_maps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "seed", "sample", "chain_mapping"])
        for r in xm_rows + am_rows:
            w.writerow([r["pdb"], r["seed"], r["sample"], r["chain_mapping"]])
    xr = [r for r in csv.DictReader(open(str(paths.PXM2025_CHAIN_ROLES))) if r["pdb"] in single]
    ar = [r for r in csv.DictReader(open(f"{ANCHOR_DIR}/chain_roles_anchor.csv"))
          if r["pdb"] in multi_pdbs]
    with open(f"{A.out}/frame_roles.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain"])
        for r in xr + ar:
            w.writerow([r["pdb"], r["chain_1"], r["chain_2"], r["ab_chain"], r["ag_chain"]])
    print(f"  frame files: {len(xm_rows)} crystal + {len(am_rows)} anchor pose mappings, "
          f"{len(xr)} + {len(ar)} interface roles")

    open(f"{A.out}/queue.tsv", "w").write("\n".join(lines) + "\n")
    with open(f"{A.out}/results.csv", "w") as fh:
        fh.write(NEW_HEADER)
        fh.write("\n".join(carried) + ("\n" if carried else ""))
    print(f"  complexes: {len(bypdb)}  (single-interface, crystal frame reused: {len(single)})")
    for k, v in stat.most_common():
        print(f"    {k}: {v}")
    print(f"  folds to run: {len(lines)}   rows carried over: {len(carried)}   -> {A.out}/queue.tsv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
