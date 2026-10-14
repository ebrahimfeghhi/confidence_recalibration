#!/usr/bin/env python
"""Per-COMPLEX donor selection: the design that matches what a fold actually produces.

WHY THIS REPLACES THE PER-INTERFACE QUEUE. A template is a whole complex and a fold scores every
interface of that complex, so "leaders for interface A-B" is not a thing the fold can deliver
separately. The previous queue selected per interface and tagged them K00..K49 PER INTERFACE, so tags
collided across siblings: once A-B's K00 fold had written rows for every interface under tag K00,
A-C's K00 looked already-scored and was skipped. Net effect, whichever interface was claimed first
supplied the donors for the whole complex and the others inherited them -- on 8rwb_AC, 8w5t_BC,
8yw9_AD and others, ZERO of their own leaders were ever folded.

THE PROCEDURE
  1. per interface: order poses by ipTM, keep one only if its pose-pose DockQ to everything kept is
     below 0.23, stop at 50  -> that interface's ranked leader list
  2. per complex: round-robin those lists -- each interface's 1st pick, then each 2nd, ... skipping
     poses already taken -- until 50 distinct poses
  3. one fold per pose, one template, target=* so every interface is scored from it

ROUND-ROBIN, NOT MAJORITY VOTE. Capping by "poses the most interfaces asked for" keeps shared
structure but starves the idiosyncratic interface: on 9uza one interface retained 14% of its own
picks. Round-robin gives every interface an equal share and raises that floor to 36%.

REUSE. A pose already folded at this seed by the k50 run produces the same output, so those folds are
skipped and their existing rows are pooled at evaluation time. 1223 of 3313 come for free.

    build_complex_queue.py --out run_exp/queue_complex.tsv [--cap 50] [--seed 430000]
"""
import argparse
import collections
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/analysis")

import paths  # noqa: E402

MATRICES = "/home/ebrahim/tts_abag/diffusion_commit_clean/matrices"
T = 0.23


def leaders(iface, cap):
    """Ranked leader list for one interface: ipTM order, keep only if distinct from all kept."""
    z = np.load(f"{MATRICES}/{iface}.npz")
    M = z["dockq"].astype(float)
    ip = z["pair_iptm"].astype(float)
    pdb = iface.split("_")[0]
    pose = [paths.pool_pose(pdb, s, int(x)) for s, x in zip(z["seed"], z["sample"])]
    L = []
    for p in np.argsort(-ip):
        p = int(p)
        if all(M[p, j] < T for j in L):
            L.append(p)
            if len(L) >= cap:
                break
    return [pose[j] for j in L]


def round_robin(per, cap):
    """Each interface's 1st pick, then each 2nd, ... until `cap` distinct poses.

    Returns (pose, selecting_interface). The interface is kept because the template builder needs one
    to pin the donor's chains -- a multi-copy assembly has several chains per entity and the pose must
    be templated on the pair it was RANKED on. It does NOT restrict scoring: fold.py scores every
    interface of the complex from every fold regardless of target.
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


def already_folded(seed, single_only):
    """Poses the k50 run already folded at this seed, restricted to SINGLE-interface complexes.

    Those are safe to reuse: one interface means one tag per fold, so a stored row maps to exactly one
    donor. On MULTI-interface complexes 114 pdb+tag combinations were claimed by two interfaces, so a
    row cannot be traced back to its template -- and the new design pools by complex, so mixing in
    rows from donors that did not survive the round-robin would silently enlarge the candidate set.
    Those are refolded instead.
    """
    qline = {}
    q = f"{paths.DATA_DIR}/k50/queue.tsv"
    if not os.path.exists(q):
        return set()
    for line in open(q):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 6 and f[4] == str(seed):
            qline[f"{f[1]}_{f[2]}_{f[3]}_{f[4]}"] = f[5]
    cd = f"{paths.DATA_DIR}/k50/claims"
    if not os.path.isdir(cd):
        return set()
    return {qline[c] for c in os.listdir(cd)
            if c in qline and c.split("_")[0] in single_only}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--seed", type=int, default=430000)
    ap.add_argument("--no-reuse", action="store_true",
                    help="refold everything, including single-interface complexes")
    A = ap.parse_args()

    bypdb = collections.defaultdict(list)
    for f in sorted(glob.glob(f"{MATRICES}/*.npz")):
        i = os.path.basename(f)[:-4]
        bypdb[i.split("_")[0]].append(i)

    single_only = {p for p, v in bypdb.items() if len(v) == 1}
    done = set() if A.no_reuse else already_folded(A.seed, single_only)
    lines, reused, n_per = [], 0, {}
    for pdb, ifs in sorted(bypdb.items()):
        sel = round_robin({i: leaders(i, A.cap) for i in ifs}, A.cap)
        n_per[pdb] = len(sel)
        for n, (pose, iface) in enumerate(sel):
            if pose in done:
                reused += 1
                continue
            if not os.path.exists(pose):
                continue
            # target names the SELECTING interface so the builder can pin chains; the C## tag is
            # unique per COMPLEX so sibling interfaces can no longer collide on (seed, tag)
            lines.append("\t".join(["tmpl", pdb, iface, f"C{n:02d}", str(A.seed), pose]))

    os.makedirs(os.path.dirname(A.out) or ".", exist_ok=True)
    open(A.out, "w").write("\n".join(lines) + "\n")
    tot = sum(n_per.values())
    print(f"  complexes: {len(n_per)}   donors selected: {tot}   median per complex: "
          f"{int(np.median(list(n_per.values())))}")
    print(f"  reused from the k50 run: {reused}")
    print(f"  folds to run: {len(lines)}  ->  {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
