#!/usr/bin/env python
"""Fold the FoldBench split arms back into one directory each, and archive the off-block folds.

    ./consolidate_arms.py            # dry run
    ./consolidate_arms.py --apply

Each logical arm had drifted across two directories, and an analysis that read only one of them
silently reported a column as empty -- the decoy row sat at 0 for hours because `free_vs_matched.py`
read `selftmpl_mask_on_nc10` and not its `_b151` continuation. One directory per arm removes that
whole class of bug.

  free_matched_topup/      ->  free_compute_matched/        extra free seeds, one home
  selftmpl_mask_on_nc10_b151/ -> selftmpl_mask_on_nc10/     the method, one home

Then every method fold whose seed is OUTSIDE 151-200 is moved to the archive. Those are the folds
queued through the old `matched_seeds` draw, which took the lowest-numbered seed available and handed
8aok seeds 1 and 7; the `_b151` run refolded those 29 complexes onto the clean block, so the originals
are superseded, not lost.

LAYOUTS DIFFER and the merge normalises them: `free_compute_matched/` stores poses at
`<pdb>/<seed>-<run_id>/` while `free_matched_topup/` uses `poses/<pdb>/<seed>-<run_id>/`. Each
destination keeps its own existing convention.

SAFETY. Verified before anything moves, and asserted again here: no pose directory, no (complex, seed)
and no results row-key collides between the pair being merged. Rows are appended, never rewritten, and
poses are moved within one filesystem. Nothing is deleted -- folds do not reproduce.

A worker must not be running against either side. Stop them first; claims and results.csv make the
restart idempotent.
"""
import argparse
import csv
import glob
import os
import shutil

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
ARCHIVE = "/home/ebrahim/tts_abag/experimental_data/archive"
BLOCK = (151, 200)

# (source, destination, source pose root, destination pose root)
MERGES = [
    (f"{FB}/free_matched_topup", f"{FB}/free_compute_matched", "poses", ""),
    (f"{FB}/selftmpl_mask_on_nc10_b151", f"{FB}/selftmpl_mask_on_nc10", "poses", "poses"),
]


def pose_dirs(root, sub):
    base = f"{root}/{sub}" if sub else root
    return [d for d in glob.glob(f"{base}/*/*") if os.path.isdir(d)]


def row_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r["iface"], r["seed"], r["arm"], r["sample"]) for r in csv.DictReader(open(path))}


def merge(src, dst, ssub, dsub, apply):
    sd = pose_dirs(src, ssub)
    dd = pose_dirs(dst, dsub)
    skeys = {(d.split("/")[-2], d.split("/")[-1]) for d in sd}
    dkeys = {(d.split("/")[-2], d.split("/")[-1]) for d in dd}
    sseed = {(p, n.split("-")[0]) for p, n in skeys}
    dseed = {(p, n.split("-")[0]) for p, n in dkeys}
    assert not skeys & dkeys, f"pose dir collision: {sorted(skeys & dkeys)[:3]}"
    assert not sseed & dseed, f"(complex, seed) collision: {sorted(sseed & dseed)[:3]}"
    srows = list(csv.DictReader(open(f"{src}/results.csv")))
    assert not (row_keys(f"{src}/results.csv") & row_keys(f"{dst}/results.csv")), "row-key collision"
    print(f"{os.path.basename(src)} -> {os.path.basename(dst)}: "
          f"{len(sd)} pose dirs, {len(srows)} rows")
    if not apply:
        return
    dest_base = f"{dst}/{dsub}" if dsub else dst
    for d in sd:
        pdb, name = d.split("/")[-2], d.split("/")[-1]
        os.makedirs(f"{dest_base}/{pdb}", exist_ok=True)
        tgt = f"{dest_base}/{pdb}/{name}"
        if not os.path.exists(tgt):
            shutil.move(d, tgt)
    with open(f"{dst}/results.csv") as fh:
        cols = csv.DictReader(fh).fieldnames
    with open(f"{dst}/results.csv", "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=cols).writerows(srows)
    for c in glob.glob(f"{src}/claims/*"):
        os.makedirs(f"{dst}/claims", exist_ok=True)
        tgt = f"{dst}/claims/{os.path.basename(c)}"
        if not os.path.exists(tgt):
            shutil.move(c, tgt)
    # the source queue is kept beside the destination: it records which folds this batch covered
    if os.path.exists(f"{src}/queue.tsv"):
        shutil.copy2(f"{src}/queue.tsv", f"{dst}/queue_{os.path.basename(src)}.tsv")
    print(f"  merged into {dst}")


def archive_offblock(arm, name, apply):
    """Move every fold of `arm` whose seed is outside BLOCK into the archive, poses and rows."""
    dest = f"{ARCHIVE}/{name}"
    lo, hi = BLOCK
    off = [d for d in pose_dirs(arm, "poses")
           if not (d.split("/")[-1].split("-")[0].isdigit()
                   and lo <= int(d.split("/")[-1].split("-")[0]) <= hi)]
    keep, move = [], []
    with open(f"{arm}/results.csv") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        for r in rd:
            (keep if (r["seed"].isdigit() and lo <= int(r["seed"]) <= hi) else move).append(r)
    print(f"archive off-block from {os.path.basename(arm)}: {len(off)} pose dirs, {len(move)} rows "
          f"({len(keep)} rows stay)")
    if not apply:
        return
    for d in off:
        pdb, n = d.split("/")[-2], d.split("/")[-1]
        os.makedirs(f"{dest}/poses/{pdb}", exist_ok=True)
        tgt = f"{dest}/poses/{pdb}/{n}"
        if not os.path.exists(tgt):
            shutil.move(d, tgt)
    os.makedirs(dest, exist_ok=True)
    with open(f"{dest}/results_archived.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(move)
    # poses are safely moved before the arm's csv is rewritten, so an interruption leaves rows
    # pointing at archived poses rather than poses with no rows
    if not os.path.exists(f"{arm}/results.csv.bak"):
        shutil.copy2(f"{arm}/results.csv", f"{arm}/results.csv.bak")
    with open(f"{arm}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(keep)
    print(f"  archived to {dest}; {os.path.basename(arm)}/results.csv now {len(keep)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    for src, dst, ss, ds in MERGES:
        if os.path.isdir(src):
            merge(src, dst, ss, ds, A.apply)
    archive_offblock(f"{FB}/selftmpl_mask_on_nc10", "foldbench_selftmpl_nc10_offblock_seeds", A.apply)
    if not A.apply:
        print("\ndry run -- pass --apply")


if __name__ == "__main__":
    main()
