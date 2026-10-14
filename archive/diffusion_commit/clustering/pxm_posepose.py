#!/usr/bin/env python
"""Pose-level pose-pose DockQ clustering on the PXM-2025 pool, to build a DIVERSITY shortlist.

THE QUESTION. On 91 confident decoys that do contain an acceptable pose, ranking seeds by ipTM puts a
good seed in the top 10 only 44% of the time -- the most confident seeds are the same dominant wrong
basin repeated. Shortlisting for COVERAGE instead does much better; epitope-Jaccard leader clustering
reaches 77% at 10. This asks whether clustering on exact pose-pose DockQ does better still, since
DockQ also sees register and orientation, which an epitope set does not.

WHY POSE LEVEL, NOT SEED LEVEL. The epitope version took each seed's best-ipTM pose as its
representative. That is the operation we have repeatedly shown is untrustworthy, and on interfaces with
1-4 good poses out of 250 it can discard the answer before clustering starts. Here every pose is
clustered and the shortlist is one SEED per cluster, so no ipTM filter is applied anywhere.

CHAIN HANDLING. dockq_torch.load_poses takes one chain pair PER POSE because OST remaps native onto
whichever model copy fits, and that mapping can change pose to pose. That hazard does not exist in this
pool: all 91 interfaces have entirely distinct protein sequences, so there is exactly one copy of each
molecule and the model chain letters are fixed across poses. Verified, not assumed -- the script
re-checks and refuses to run on any interface with a duplicated sequence.

The model chain letters are taken by matching sequences in the job json against the antibody/antigen
assignment in chain_roles.csv, rather than assuming the native letters carry over.

Usage:
    pxm_posepose.py --iface 8vqf_AC [--thresh 0.23] [--k 3 5 10 20] [--device cuda]
    pxm_posepose.py --all --n 20
"""
import argparse
import collections
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")
sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402  (needs the path insert above)
from fmt import table

REPO = str(paths.DATA_DIR)
POOL = f"{REPO}/pxm2025/dockq_pxm2025.csv"
JOBS = f"{REPO}/pxm2025/jsons"
ROLES = f"{REPO}/pxm2025/chain_roles.csv"
# the PXM pool's own reference structures -- the same files row.py scores against. NOT the
# FoldBench ground_truths dir, which covers a different 117-target set and lacks these pdbs.
GT = f"{REPO}/pxm2025/cif"
MATDIR = f"{REPO}/posepose_pxm"


def roles_map():
    """interface -> (antibody native chain, antigen native chain), from the verified assignment.

    chain_roles.csv keys interfaces as "9jo7_A_B"; the pool csv uses "9jo7_AB".
    """
    out = {}
    for r in csv.DictReader(open(ROLES)):
        if r["interface"] == "interface":
            continue
        out[f'{r["pdb"]}_{r["chain_1"]}{r["chain_2"]}'] = (r["ab_chain"], r["ag_chain"])
    return out


def model_chains(pdb, ab_native, ag_native):
    """Map NATIVE chain letters to MODEL chain letters by sequence.

    The job json carries no chain ids -- Protenix names model chains A, B, C... in the order the
    sequences appear (verified on 8vqf: job sequences of 212/235/156 residues appear as model chains
    A/B/C of exactly those lengths). The native letters are NOT guaranteed to agree, so mapping by
    sequence is the only safe route; assuming the letters carry over is the mistake that once had 45%
    of interfaces measured on the wrong chains.

    REFUSES on a duplicated sequence: that is the one case where a single fixed chain pair per
    interface would be ambiguous.
    """
    import gemmi

    j = json.load(open(f"{JOBS}/{pdb}-final-updated.json"))
    e = j[0] if isinstance(j, list) else j
    job_seqs = [v["sequence"] for s_ in e["sequences"] for k, v in s_.items()
                if k == "proteinChain" and v.get("sequence")]
    if len(set(job_seqs)) != len(job_seqs):
        raise ValueError(f"{pdb}: duplicated sequence -- a fixed chain pair is unsafe here")
    # model chains are assigned in job order
    letters = [chr(ord("A") + i) for i in range(len(job_seqs))]

    gt = f"{GT}/{pdb}.cif"
    if not os.path.exists(gt):
        raise FileNotFoundError(f"no ground truth for {pdb}: {gt}")
    # READ label_asym_id, NOT gemmi's chain names. gemmi.read_structure names chains by auth_asym_id,
    # while chain_roles.csv (and OST, and the pool csv) speak label_asym_id -- they differ on many
    # complexes, so using ch.name looks up the WRONG native chain. Reading the raw loop is what
    # scripts/decoy_set.py:native_chains does, for exactly this reason: doing it the convenient way
    # silently compared the wrong molecules on 70 of 91 interfaces here.
    block = gemmi.cif.read(gt).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "label_comp_id", "group_PDB")]
    seen = collections.defaultdict(dict)
    for a, sq, comp, g in zip(*cols):
        if g != "ATOM":
            continue
        seen[a][sq] = comp
    nat = {}
    for a, res in seen.items():
        order = sorted(res, key=lambda x: int(x) if x.isdigit() else 0)
        seq = gemmi.one_letter_code([res[k] for k in order])
        if len(seq) >= 20:
            nat[a] = seq.upper()

    def best(nseq):
        """Match a native chain sequence to a job sequence.

        Position-wise comparison does NOT work: the ground truth is offset and truncated relative to
        the job sequence (8vqf native C starts "ALGKD..." where the job has "QDTPALGKD...", and native
        A is 218 residues against the job's 212). SequenceMatcher handles offset and indels, which a
        zip() comparison cannot -- that version scored a true match at 0.04.
        """
        import difflib
        scored = sorted(((difflib.SequenceMatcher(None, nseq, js).ratio(), i)
                         for i, js in enumerate(job_seqs)), reverse=True)
        if not scored or scored[0][0] < 0.7:
            raise ValueError(f"{pdb}: no job sequence matches native chain "
                             f"(best ratio {scored[0][0]:.2f} if any)")
        if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.05:
            raise ValueError(f"{pdb}: ambiguous chain match "
                             f"({scored[0][0]:.2f} vs {scored[1][0]:.2f})")
        return letters[scored[0][1]]

    for c in (ab_native, ag_native):
        if c not in nat:
            raise ValueError(f"{pdb}: native chain {c} not in {sorted(nat)}")
    return best(nat[ab_native]), best(nat[ag_native])


def pose_list(iface, ab, ag):
    """Every pose of every seed for this interface: (path, model_c1, model_c2, iptm, dockq_gt, seed, sample).

    No best-ipTM-per-seed filter -- see the module docstring.
    """
    pdb = iface.split("_")[0]
    c1, c2 = iface.split("_")[1][0], iface.split("_")[1][1]
    rows = []
    for r in csv.DictReader(open(POOL)):
        if r["pdb"] != pdb or r["chain1"] != c1 or r["chain2"] != c2:
            continue
        p = paths.pool_pose(pdb, r["seed"], r["sample"])
        if os.path.exists(p):
            rows.append((p, ab, ag, float(r["pair_iptm"]), float(r["dockq"]),
                         r["seed"], int(r["sample"])))
    return rows


def leaders(order, sim, thresh):
    """Leader clustering over pose indices. sim(i, j) is DockQ similarity in [0, 1]."""
    out = []
    for i in order:
        for L in out:
            if sim(i, L["leader"]) >= thresh:
                L["members"].append(i)
                break
        else:
            out.append({"leader": i, "members": [i]})
    return out


def run_iface(iface, roles, A):
    from dockq_torch import dockq_matrix

    pdb = iface.split("_")[0]
    ab_n, ag_n = roles[iface]
    ab, ag = model_chains(pdb, ab_n, ag_n)
    rows = pose_list(iface, ab, ag)
    if len(rows) < 200:
        return None, f"only {len(rows)} poses"

    os.makedirs(MATDIR, exist_ok=True)
    npy = f"{MATDIR}/dockq_{iface}.npy"
    idx = f"{MATDIR}/dockq_{iface}_index.csv"
    if os.path.exists(npy):
        M = np.load(npy)
        if M.shape[0] != len(rows):
            return None, f"cached matrix is {M.shape[0]} poses, pool has {len(rows)}"
    else:
        M = dockq_matrix([(r[0], r[1], r[2]) for r in rows], device=A.device, verbose=False)
        M = M.astype(np.float32)
        np.save(npy, M)
        with open(idx, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["row", "seed", "sample", "iptm", "dockq_gt"])
            for i, r in enumerate(rows):
                w.writerow([i, r[5], r[6], f"{r[3]:.4f}", f"{r[4]:.4f}"])

    # DockQ is not symmetric (reference vs model), so use the mean of the two directions
    S = (M + M.T) / 2.0
    n = len(rows)
    # cluster in POOL ORDER (seed, sample) so the shortlist cannot depend on ipTM in any way
    order = sorted(range(n), key=lambda i: (rows[i][5], rows[i][6]))
    cl = leaders(order, lambda i, j: float(S[i, j]), A.thresh)

    good_seeds = {r[5] for r in rows if r[4] >= 0.23}
    by_size = sorted(cl, key=lambda L: -len(L["members"]))
    div_seeds, seen = [], set()
    for L in by_size:                                   # one SEED per cluster, biggest clusters first
        s = rows[L["leader"]][5]
        if s not in seen:
            seen.add(s)
            div_seeds.append(s)
    seed_iptm = collections.defaultdict(float)
    for r in rows:
        seed_iptm[r[5]] = max(seed_iptm[r[5]], r[3])
    by_iptm = sorted(seed_iptm, key=lambda s: -seed_iptm[s])

    res = [iface, n, len(cl), len(good_seeds)]
    for k in A.k:
        res.append("yes" if any(s in good_seeds for s in by_iptm[:k]) else "no")
        res.append("yes" if any(s in good_seeds for s in div_seeds[:k]) else "no")
    return res, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", nargs="+")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--thresh", type=float, default=0.23,
                    help="pose-pose DockQ above which two poses are the same basin")
    ap.add_argument("--k", type=int, nargs="+", default=[3, 5, 10, 20])
    ap.add_argument("--device", default="cuda")
    A = ap.parse_args()

    roles = roles_map()
    P = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(open(POOL)):
        P[f"{r['pdb']}_{r['chain1']}{r['chain2']}"][r["seed"]].append(
            (float(r["pair_iptm"]), float(r["dockq"])))
    cands = []
    for iface, seeds in P.items():
        if len(seeds) < 50 or iface not in roles:
            continue
        allp = [p for v in seeds.values() for p in v]
        if max(allp)[1] >= 0.23 or not any(d >= 0.23 for _, d in allp):
            continue
        cands.append(iface)
    cands.sort()

    todo = A.iface if A.iface else cands[:A.n]
    print(f"  confident decoys: {len(cands)}; running {len(todo)} | "
          f"pose-pose DockQ threshold {A.thresh}", flush=True)
    rows = []
    for iface in todo:
        try:
            res, err = run_iface(iface, roles, A)
        except Exception as e:
            print(f"    {iface}: {type(e).__name__}: {str(e)[:110]}", flush=True)
            continue
        if err:
            print(f"    {iface}: {err}", flush=True)
            continue
        rows.append(res)
        print(f"    {iface}: {res[1]} poses, {res[2]} clusters, {res[3]} good seeds", flush=True)

    if not rows:
        print("  nothing measured")
        return 1
    hdr = ["interface", "poses", "clusters", "good seeds"]
    for k in A.k:
        hdr += [f"ipTM@{k}", f"div@{k}"]
    print()
    print(table(hdr, rows, align="l r r r" + " l l" * len(A.k)))
    n = len(rows)
    print(f"\n  n = {n} interfaces")
    for i, k in enumerate(A.k):
        ip = sum(1 for r in rows if r[4 + 2 * i] == "yes")
        dv = sum(1 for r in rows if r[5 + 2 * i] == "yes")
        print(f"    shortlist {k:2d}:  ipTM {ip:3d}/{n} ({100*ip/n:3.0f}%)   "
              f"pose-pose DockQ diversity {dv:3d}/{n} ({100*dv/n:3.0f}%)")
    cc = [r[2] for r in rows]
    print(f"    clusters per interface: median {sorted(cc)[len(cc)//2]}, range {min(cc)}-{max(cc)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
