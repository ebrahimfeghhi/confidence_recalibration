#!/usr/bin/env python
"""POSE-TO-POSE DockQ: are two poses of the same seed structurally more similar than two poses
from different seeds, at the SAME true-DockQ gap?

WHY THIS AND NOT THE SPREAD FIGURES. figures/poses_vs_seeds_matched.png shows both selectors
converting within-seed oracle gain at ~0%, and figures/within_seed_spread_panels.png rules out
"the score cannot see the pose" as the reason (DeepRank-Ab's within-seed variance share is 0.65,
higher than DockQ's own). The remaining reading is that the comparison is simply harder: two poses
from one seed may differ in quality while barely differing in geometry, and no selector can grade a
difference that is not there to read. That is a statement about the POSES, not about any score, so
it is measured with no score involved.

DockQ-TO-NATIVE IS NOT A SIMILARITY. Two poses can share a DockQ and sit 90 A apart (9fve/400073),
so pose similarity has to be measured by scoring one pose AGAINST THE OTHER. That makes the
reference a pose cif rather than training/refs, and DockQ is asymmetric in its two arguments -- it
normalises fnat and the RMSDs on the reference's contacts -- so every pair is scored BOTH WAYS and
the two are averaged. The two directions are kept in the csv: a large gap between them is itself a
sign the pair is badly mapped, not merely dissimilar.

MATCHED ON THE TRUE-DockQ GAP, INSIDE ONE CHAIN PAIR. Same-seed poses are closer in quality than
different-seed poses, so an unmatched comparison would answer a question nobody asked. Each
sampled within-seed pair is matched to an across-seed pair of the SAME chain pair whose
|dDockQ| agrees within --tol; the printout reports both mean gaps, and they must agree.

by_label=False ON BOTH SIDES, AND THE LETTERS HAVE TO BE TRANSLATED FIRST.
phase_j_fill_scores.write_subset selects reference residues by label_asym_id and model residues by
chain name, which is right when the reference is a deposited structure. Here the reference IS a
pose cif and carries MODEL chain naming, so both sides select by chain name -- and the interface
letters in the results CSVs are REF chains, which are permuted against the model names (10fj model
A = ref B). Passing the ref letters straight through left one of the two chains missing on 32% of
the probed pairs, and write_subset returns the one chain it found rather than failing, so the pair
dropped out silently. Ref -> model comes from chain_roles for sabdab and from chain_maps.csv, which
is per (pdb, seed, sample), for foldbench.
"""
import argparse
import collections
import csv
import glob
import os
import random
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, os.path.join(REPO, "eval"))

# dataset -> the fold roots whose poses/<pdb>/<seed>-*/predictions hold the sample cifs
FOLDS = {
    "sabdab": [f"{ROOT}/training/fold", f"{ROOT}/validation/fold"],
    "foldbench": [f"{REPO}/experimental_data/protenix/foldbench/foldbench_baseline_remaining_folds"],
}


def pair_dockq(job):
    """Symmetric pose-to-pose DockQ. Returns the two directions, or None if either failed.

    Each side carries its OWN chain letters because the ref->model permutation is per pose file.
    """
    import ost_dockq
    key, (a1, a2), (b1, b2), cif_a, cif_b, scratch = job
    import phase_j_fill_scores as FS
    vals = []
    for i, (r_cif, m_cif, (ca, cb), (ma, mb)) in enumerate(
            ((cif_a, cif_b, (a1, a2), (b1, b2)),
             (cif_b, cif_a, (b1, b2), (a1, a2)))):
        tag = f"{key}_{i}"
        rs, ms = f"{scratch}/{tag}_r.cif", f"{scratch}/{tag}_m.cif"
        out = f"{scratch}/{tag}.json"
        try:
            FS.write_subset(r_cif, {ca, cb}, rs, False)
            FS.write_subset(m_cif, {ma, mb}, ms, False)
            if not ost_dockq.invoke(ms, rs, out):
                return None
            v = ost_dockq.interface(ost_dockq.parse(out), ca, cb)
            if not v or v.get("dockq") is None:
                return None
            vals.append(float(v["dockq"]))
        except Exception:                                      # noqa: BLE001
            return None
        finally:
            for f in (rs, ms, out):
                try:
                    os.remove(f)
                except OSError:
                    pass
    return vals


def ref_to_model(ds):
    """(pdb, seed, sample, ref chain) -> model chain, as a lookup function with an identity fallback.

    sabdab: chain_roles carries ref_chain and model_chain per pdb, constant over seeds.
    foldbench: chain_maps.csv stores one mapping per (pdb, seed, sample).
    """
    per_pdb, per_unit = {}, {}
    if ds == "sabdab":
        for pth in (f"{ROOT}/training/fold/chain_roles_all.csv",
                    f"{ROOT}/validation/fold/chain_roles_all83.csv"):
            if not os.path.exists(pth):
                continue
            with open(pth) as fh:
                for r in csv.DictReader(fh):
                    per_pdb.setdefault(r["pdb"], {})[r["ref_chain"]] = r["model_chain"]
    elif ds == "foldbench":
        import json
        pth = f"{REPO}/experimental_data/protenix/foldbench/chain_maps.csv"
        if os.path.exists(pth):
            with open(pth) as fh:
                for r in csv.DictReader(fh):
                    try:
                        per_unit[(r["pdb"], int(r["seed"]), int(r["sample"]))] = \
                            json.loads(r["chain_mapping"])
                    except (ValueError, TypeError):
                        continue

    def look(pdb, seed, sample, c):
        m = per_unit.get((pdb, seed, sample)) or per_pdb.get(pdb) or {}
        return m.get(c, c)
    return look


def cif_index(folds, pdb, seed):
    """sample id -> cif path for one (pdb, seed), across whichever fold root holds it."""
    import phase_j_build_cache as BC
    for root in folds:
        dirs = [d for d in glob.glob(f"{root}/poses/{pdb}/{seed}-*")
                if os.path.exists(os.path.join(d, "run.json"))]
        if len(dirs) != 1:
            continue
        cifs = BC.sample_cifs(dirs[0], pdb)
        if not cifs:
            continue
        out = {}
        for c in cifs:
            stem = os.path.basename(c)[:-4]
            tail = stem.rsplit("_sample_", 1)
            if len(tail) == 2 and tail[1].isdigit():
                out[int(tail[1])] = c
        if out:
            return out
    return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master", default=f"{ROOT}/training/fold/abag_all_results.csv")
    ap.add_argument("--datasets", default="sabdab,foldbench")
    ap.add_argument("--poses", type=int, default=5)
    ap.add_argument("--min-seeds", type=int, default=10)
    ap.add_argument("--chain-pairs", type=int, default=60,
                    help="chain pairs sampled per dataset")
    ap.add_argument("--pairs-per", type=int, default=12,
                    help="matched within/across pose pairs per chain pair")
    ap.add_argument("--min-gap", type=float, default=0.05,
                    help="only pairs whose true DockQ differs by at least this; a pair of equally "
                         "good poses has nothing for a selector to get right")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--scratch", default="/tmp/pose_pair_dockq")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{ROOT}/docs/pose_pair_dockq.csv")
    a = ap.parse_args()

    # 48 CORES IS THE CEILING ON THIS BOX, AND A WORKER IS NOT A CORE. 48 workers measured at
    # 59 cores because each OST process spawns BLAS/OpenMP threads, so the thread envs are pinned
    # to 1 here -- before the pool forks -- rather than trusting the worker count to bound usage.
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    MAX_WORKERS = 40
    if a.workers > MAX_WORKERS:
        print(f"  --workers {a.workers} clamped to {MAX_WORKERS}")
        a.workers = MAX_WORKERS

    import numpy as np
    rng = random.Random(a.seed)
    os.makedirs(a.scratch, exist_ok=True)

    want = [d.strip() for d in a.datasets.split(",") if d.strip()]
    cell = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.master)):
        if r["pose_set"] != "orig" or r["dataset"] not in want:
            continue
        try:
            cell[(r["dataset"], r["structure"], r["interface"])].setdefault(
                int(r["seed"]), {})[int(r["pose"])] = float(r["dockq"])
        except (ValueError, TypeError):
            continue

    jobs, meta = [], []
    maps = {}
    picked = collections.Counter()
    for (ds, pdb, iface), seeds in sorted(cell.items()):
        if picked[ds] >= a.chain_pairs:
            continue
        full = {s: v for s, v in seeds.items() if len(v) == int(a.poses)}
        if len(full) < int(a.min_seeds):
            continue
        letters = iface.split("_")[-1]
        if len(letters) != 2:
            continue
        ca, cb = letters[0], letters[1]
        look = maps.setdefault(ds, ref_to_model(ds))

        order = sorted(full)
        cifs = {}
        for s in order:
            ix = cif_index(FOLDS.get(ds, []), pdb, s)
            if len(ix) >= a.poses:
                cifs[s] = ix
        order = [s for s in order if s in cifs]
        if len(order) < int(a.min_seeds):
            continue

        wit, acr = [], []
        for i, s1 in enumerate(order):
            for p1 in range(a.poses):
                for p2 in range(p1 + 1, a.poses):
                    g = abs(full[s1][p1] - full[s1][p2])
                    if g >= a.min_gap:
                        wit.append((g, s1, p1, s1, p2))
            for s2 in order[i + 1:]:
                for p1 in range(a.poses):
                    for p2 in range(a.poses):
                        g = abs(full[s1][p1] - full[s2][p2])
                        if g >= a.min_gap:
                            acr.append((g, s1, p1, s2, p2))
        if not wit or not acr:
            continue

        rng.shuffle(wit)
        rng.shuffle(acr)
        used = set()
        n = 0
        for g, s1, p1, s2, p2 in wit:
            if n >= a.pairs_per:
                break
            cand = [k for k, x in enumerate(acr)
                    if k not in used and abs(x[0] - g) <= a.tol]
            if not cand:
                continue
            k = cand[0]
            used.add(k)
            n += 1
            for kind, (gg, t1, q1, t2, q2) in (("within", (g, s1, p1, s2, p2)),
                                               ("across", acr[k])):
                key = f"{pdb}_{iface}_{kind}_{t1}_{q1}_{t2}_{q2}"
                la = (look(pdb, t1, q1, ca), look(pdb, t1, q1, cb))
                lb = (look(pdb, t2, q2, ca), look(pdb, t2, q2, cb))
                if la[0] == la[1] or lb[0] == lb[1]:
                    continue
                jobs.append((key, la, lb, cifs[t1][q1], cifs[t2][q2], a.scratch))
                meta.append({"dataset": ds, "structure": pdb, "interface": iface,
                             "kind": kind, "gap": gg,
                             "seed_a": t1, "pose_a": q1, "seed_b": t2, "pose_b": q2,
                             "dockq_a": full[t1][q1], "dockq_b": full[t2][q2]})
        if n:
            picked[ds] += 1

    print(f"chain pairs sampled: {dict(picked)}")
    print(f"pose pairs: {len(jobs)}  ->  {2 * len(jobs)} OST calls", flush=True)
    if not jobs:
        print("nothing to score")
        return

    import time
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, v in enumerate(ex.map(pair_dockq, jobs, chunksize=1), 1):
            if v:
                m = dict(meta[i - 1])
                m["dockq_ab"], m["dockq_ba"] = v[0], v[1]
                m["dockq_sym"] = 0.5 * (v[0] + v[1])
                rows.append(m)
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{len(jobs)} pairs, {len(rows)} scored, {i / el:.1f} pair/s, "
                      f"~{(len(jobs) - i) / (i / el) / 60:.0f} min left", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cols = ["dataset", "structure", "interface", "kind", "gap", "seed_a", "pose_a",
            "seed_b", "pose_b", "dockq_a", "dockq_b", "dockq_ab", "dockq_ba", "dockq_sym"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nrows -> {a.out}  ({len(rows)} of {len(jobs)} scored)")

    print()
    print(f"{'dataset':>12s} {'kind':>8s} {'n':>6s} {'mean gap':>9s} "
          f"{'mean pose-pose DockQ':>21s} {'median':>8s} {'|ab-ba|':>8s}")
    for ds in want:
        for kind in ("within", "across"):
            v = [r for r in rows if r["dataset"] == ds and r["kind"] == kind]
            if not v:
                continue
            s = np.array([r["dockq_sym"] for r in v])
            g = np.array([r["gap"] for r in v])
            d = np.array([abs(r["dockq_ab"] - r["dockq_ba"]) for r in v])
            print(f"{ds:>12s} {kind:>8s} {len(v):6d} {g.mean():9.4f} "
                  f"{s.mean():21.4f} {np.median(s):8.4f} {d.mean():8.4f}")


if __name__ == "__main__":
    main()
