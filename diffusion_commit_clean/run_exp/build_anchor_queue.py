#!/usr/bin/env python
"""Anchor-frame donor queue, reusing every fold the crystal-frame sweep already produced.

WHAT CHANGES FROM build_complex_queue.py. Only the frame. Selection is the same algorithm -- per
interface, order by pair_iptm, keep a pose whose pose-pose DockQ to everything kept is below 0.23,
round-robin the per-interface lists into one per-complex set, cap 50. What differs is that the pose-pose
DockQ and the chain pinning come from matrices built against the complex's highest-mean-ipTM POSE
instead of against the crystal, so nothing in the selection path consults the native structure.

REUSE IS KEYED ON (donor pose, seed, pinned chains), NOT ON THE TAG. A fold's output is determined by
the pose it templated, the seed, and which chains of that pose were pinned. The tag is just a name, and
naming was already load-bearing once by accident: sibling interfaces sharing tag K00 silently swapped
donors. Measured here: 728 anchor donors are poses the crystal frame ALSO chose but pinned to a
different antigen copy. Those are different templates and must be refolded; reusing them on pose+seed
alone would substitute one fold for another that never ran.

CARRIED ROWS ARE REWRITTEN, NOT COPIED. The old csv recorded donor_pose="-1", so a carried row would
arrive with no provenance and the reuse rule could not be applied to it a second time. Each carried row
is re-emitted under its NEW tag with the pose and chains filled in, and a claim directory is created so
a worker treats it as already done instead of refolding it.

    build_anchor_queue.py --out DIR [--cap 50] [--seed 430000] [--old-exp .../top_50_complex]
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402
from build_complex_queue import T, round_robin  # noqa: E402

ANCHOR_MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices_anchor_all"
# Interfaces the PREDICTIONS declare by contact but the reference does not contain. They live in their
# own directory because a matrix filename cannot distinguish them: the anchor-frame pair 8w5i_AD and the
# real crystal-frame interface 8w5i_AD produce the same string. Measured: adding them does NOT dilute --
# the 50-donor cap rarely binds, so extra lists contribute extra poses (33.6 -> 35.4 per multi-interface
# complex) at identical coverage and MORE good donors per interface.
SPUR_MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices_spur"
XTAL_MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
ANCHOR_DIR = str(paths.ANCHOR_DIR)
NEW_HEADER = "iface,seed,arm,donor_pose,tmpl_chains,sample,dockq,iptm\n"


def leaders(mdir, iface, cap):
    z = np.load(f"{mdir}/{iface}.npz")
    M, ip = z["dockq"].astype(float), z["pair_iptm"].astype(float)
    keys = [(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])]
    L = []
    for p in np.argsort(-ip):
        p = int(p)
        if all(M[p, j] < T for j in L):
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
    jd = f"{exp}/ns/jobs"
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
    ap.add_argument("--old-exp", default="/home/ebrahim/tts_abag/experimental_data/top_50_complex")
    ap.add_argument("--no-reuse", action="store_true")
    ap.add_argument("--pose-interfaces", action="store_true",
                    help="also select on interfaces the predictions declare by contact (no reference)")
    A = ap.parse_args()

    amaps = metrics.load_chain_maps(f"{ANCHOR_DIR}/chain_maps_anchor.csv")
    aroles = metrics.load_roles(f"{ANCHOR_DIR}/chain_roles_anchor.csv")
    # chain_roles_all adds the predicted-only pairs; both frames are keyed the same way
    ALL_ROLES = dict(aroles)
    if os.path.exists(f"{ANCHOR_DIR}/chain_roles_all.csv"):
        ALL_ROLES.update(metrics.load_roles(f"{ANCHOR_DIR}/chain_roles_all.csv"))

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
            m = (amaps if use_a else metrics.load_chain_maps()).get((pdb, key[0], key[1]))
            rl = (ALL_ROLES if use_a else metrics.load_roles()).get(iface)
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
