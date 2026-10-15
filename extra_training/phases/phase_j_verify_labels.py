"""Verify every cached label ON DISK against its job cif, and name the structures needing a rebuild.

WHY ON DISK AND NOT FROM THE BUILDER'S REPORT. phase_j_build_cache asserts its own work as it writes,
but a label can be wrong for reasons the writing run never saw: it was written by an EARLIER run
whose code had a bug (review round 7's masking branch queried sorted coordinates and applied the
result to the original atom order, retaining invalid atoms and masking valid ones), or its inputs
changed afterwards. What training reads is the file, so the file is what is checked.

WHAT IS CHECKED, per pose of per unit:
  1. every atom still marked RESOLVED lies within --tol of some atom of that structure's job cif
  2. every pose's resolved count agrees, since each pose's label is a permutation of the same set
  3. every unit of the manifest has a label at all

The output is a REPAIR LIST -- the pdb ids to pass to `phase_j_build_cache.py --only` -- so the
corrective pass follows from measurement instead of from a list someone typed. Exit is non-zero when
anything needs repair, so a caller can gate on it.
"""
import argparse
import csv
import glob
import os
import sys

import numpy as np


def main():
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--labels", default="")
    ap.add_argument("--mmcif-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--selection", default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--seeds", default="500000-500009")
    ap.add_argument("--skip", default="9mx1",
                    help="comma-separated pdb ids the manifest must not expect, each a documented "
                         "exclusion. 9mx1 is 2,941 residues against Protenix's 2,560-token limit.")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--out", default="", help="write the repair list here, one pdb per line")
    a = ap.parse_args()
    a.labels = a.labels or os.path.join(a.fold, "labels")

    import gemmi
    from scipy.spatial import cKDTree

    lo, hi = (int(x) for x in a.seeds.split("-"))
    skip = {x.strip().lower() for x in a.skip.split(",") if x.strip()}
    with open(a.selection) as fh:
        pdbs = [r["pdb"].strip().lower() for r in csv.DictReader(fh)]
    pdbs = [p for p in pdbs if p not in skip]
    print(f"manifest: {len(pdbs)} structure(s) x seeds {lo}-{hi} = {len(pdbs)*(hi-lo+1)} unit(s); "
          f"excluded by name: {sorted(skip) or 'none'}")

    repair, missing_units, checked, bad_poses = set(), 0, 0, 0
    for n, pdb in enumerate(pdbs, 1):
        cif = os.path.join(a.mmcif_dir, f"{pdb}.cif")
        if not os.path.exists(cif):
            print(f"  {pdb}: NO JOB CIF -- cannot verify, queued for repair")
            repair.add(pdb); continue
        # ONE TREE PER STRUCTURE, reused across its units; a tree per unit made this unusably slow
        st = gemmi.read_structure(cif); st.setup_entities(); st.remove_hydrogens()
        pts = np.array([[at.pos.x, at.pos.y, at.pos.z]
                        for ch in st[0] for r in ch for at in r], dtype=np.float64)
        if not len(pts):
            print(f"  {pdb}: job cif has no atoms"); repair.add(pdb); continue
        tree = cKDTree(pts)
        for sd in range(lo, hi + 1):
            p = os.path.join(a.labels, pdb, f"free_{sd}.npz")
            if not os.path.exists(p):
                missing_units += 1; repair.add(pdb); continue
            try:
                z = np.load(p, allow_pickle=True)
                lc, lm = z["label_coordinate"], z["label_coordinate_mask"]
            except Exception as e:
                print(f"  {pdb}/free_{sd}: unreadable ({type(e).__name__})"); repair.add(pdb); continue
            counts = []
            for i in range(lc.shape[0]):
                m = lm[i].astype(bool)
                counts.append(int(m.sum()))
                d, _j = tree.query(np.ascontiguousarray(lc[i][m], dtype=np.float64), k=1)
                far = int((d > a.tol).sum())
                checked += 1
                if far:
                    bad_poses += 1; repair.add(pdb)
                    if bad_poses <= 12:
                        print(f"  BAD {pdb}/free_{sd} pose {i}: {far} resolved atom(s) more than "
                              f"{a.tol} A from the job cif (worst {d.max():.3f} A)")
            if len(set(counts)) > 1:
                print(f"  BAD {pdb}/free_{sd}: resolved counts differ across poses {counts}, but "
                      f"every pose is a permutation of the same set")
                repair.add(pdb)
        if n % 100 == 0:
            print(f"  ...{n}/{len(pdbs)} structures, {len(repair)} needing repair", flush=True)

    print()
    print(f"poses verified      : {checked}")
    print(f"poses with a bad atom: {bad_poses}")
    print(f"units with no label : {missing_units}")
    print(f"STRUCTURES TO REPAIR: {len(repair)}" + (f" -> {sorted(repair)}" if repair else ""))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write("\n".join(sorted(repair)) + ("\n" if repair else ""))
        print(f"repair list -> {a.out}")
    return 1 if repair else 0


if __name__ == "__main__":
    sys.exit(main())
