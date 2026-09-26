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
    poses, which on one complex retained 14% of its own picks against 36% under round-robin. Interfaces
    take their turn in name order.

    NOTHING HERE READS THE CRYSTAL. The order comes from pair_iptm and the filter from pose-to-pose
    DockQ, both in the anchor frame, both from pose_to_pose_dockq.py's matrices.

INPUTS
    --matrices: pose_to_pose_dockq.py's directory, one npz per interface. Required.
    --out:      the queue to write. Required.
    --cap:      donors per complex. Default 50.
    --seed:     the seed every templated fold uses. Default 430000.

OUTPUTS
    <out>: one line per fold -- kind, pdb, interface, tag, seed, donor pose. The interface is the one
           whose list picked the donor; the template builder pins its chains. Tags are A00, A01, ... in
           pick order within a complex.

EXAMPLE
    structural_diversity_filter.py --matrices exp/pose_dockq_matrices --out exp/queue.tsv

FUNCTIONS
    leaders():      one interface -> its ipTM-ordered, diversity-filtered pose list
    round_robin():  per-interface lists -> one donor set per complex
    main():         select per complex, write the queue
"""
import argparse
import collections
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402

DOCKQ_THRESHOLD = 0.23      # CAPRI "acceptable": two poses closer than this are the same answer


def leaders(npz, cap):
    """in: one interface's matrix file, the cap.
    out: (seed, sample) of the poses kept -- ipTM order, each new one unlike everything before it
    """
    z = np.load(npz)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=430000)
    A = ap.parse_args()

    bypdb = collections.defaultdict(dict)
    for f in sorted(os.listdir(A.matrices)):
        if f.endswith(".npz"):
            bypdb[f.split("_")[0]][f[:-4]] = f"{A.matrices}/{f}"

    lines, missing = [], 0
    for pdb, ifs in sorted(bypdb.items()):
        per = {i: leaders(p, A.cap) for i, p in ifs.items()}
        for n, (key, iface) in enumerate(round_robin(per, A.cap)):
            pose = paths.pool_pose(pdb, key[0], key[1])
            if not os.path.exists(pose):
                missing += 1
                continue
            lines.append("\t".join(["tmpl", pdb, iface, f"A{n:02d}", str(A.seed), pose]))
    if missing:
        raise SystemExit(f"{missing} selected donor poses are not on disk")
    open(A.out, "w").write("\n".join(lines) + "\n")
    print(f"  {len(bypdb)} complexes, {sum(len(v) for v in bypdb.values())} interfaces, "
          f"{len(lines)} folds -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
