#!/usr/bin/env python
"""Move score rows whose TRUNK NO LONGER EXISTS out of results.csv into an archive beside it.

WHY THEY HAVE TO GO. The DockQ-SD<0.05 prune deleted the trunk tensors for nine of ten seeds while
keeping all fifty DockQ labels. A re-fold does NOT reproduce a seed's poses (phase_i's own header
says so), so those labels describe poses nothing can ever load, and the new fold will write rows at
the SAME (iface, seed, sample) keys. Left in place they would read as duplicates of the new draws.

WHAT IS NOT TOUCHED. Rows at a seed whose trunk is still present -- for these structures seed
500000 -- because label and features still match there. Nothing is deleted: every moved row is
written to the archive first and the row count is asserted on both sides before the swap.
"""
import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="training/fold/results.csv")
    ap.add_argument("--structures", default="docs/degenerate_regen_structures.txt")
    ap.add_argument("--trunks", default="training/fold/trunks")
    ap.add_argument("--archive", required=True, help="csv the orphaned rows are moved into")
    ap.add_argument("--apply", action="store_true", help="without this, only reports")
    a = ap.parse_args()

    want = [l.strip() for l in open(a.structures) if l.strip() and l[0] != "#"]
    # THE KEPT SEEDS ARE READ OFF DISK, not assumed: the trunk file is the only authority on
    # which (structure, seed) a pose can still be loaded for.
    keep = {}
    for p in want:
        keep[p] = {os.path.basename(g).split("free_")[1][:-4]
                   for g in glob.glob(os.path.join(a.trunks, p, "free_*.npz"))}
    lines = open(a.csv).read().splitlines(True)
    head, body = lines[0], lines[1:]
    cols = next(csv.reader([head]))
    i_if, i_sd, i_arm = cols.index("iface"), cols.index("seed"), cols.index("arm")

    keepl, movel = [], []
    for ln in body:
        f = next(csv.reader([ln]))
        pdb = f[i_if].split("_")[0]
        (movel if pdb in keep and f[i_sd] not in keep[pdb] else keepl).append(ln)

    print(f"{a.csv}: {len(body)} row(s)")
    for p in want:
        print(f"  {p}: trunk seed(s) {sorted(keep[p]) or 'NONE'}")
    # THE TRUNK NAME CARRIES THE TAG -- <tag>_<seed>.npz -- and the kept-seed set above was built
    # from free_*.npz only, so a templated row's trunk would not have been looked for. Refuse rather
    # than archive a row this script never checked.
    bad = [ln for ln in movel if next(csv.reader([ln]))[i_arm] != "free"]
    assert not bad, f"{len(bad)} non-free row(s) selected; the kept-seed glob only covers free_*"
    print(f"  orphaned -> archive: {len(movel)}")
    print(f"  staying:             {len(keepl)}")
    assert len(movel) + len(keepl) == len(body), "row accounting"
    if not movel:
        print("nothing to move"); return
    if not a.apply:
        print("dry run; pass --apply"); return

    os.makedirs(os.path.dirname(os.path.abspath(a.archive)), exist_ok=True)
    new = not os.path.exists(a.archive)
    with open(a.archive, "a") as fh:
        if new:
            fh.write(head)
        fh.writelines(movel)
    n_arch = sum(1 for _ in open(a.archive)) - 1
    assert n_arch >= len(movel), f"archive holds {n_arch}, moved {len(movel)}"
    # ATOMIC, AND ON THE SAME FILESYSTEM -- os.replace raises EXDEV across devices.
    tmp = a.csv + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(head); fh.writelines(keepl)
    os.replace(tmp, a.csv)
    print(f"archived {len(movel)} row(s) -> {a.archive} ({n_arch} total there)")
    print(f"rewrote {a.csv} with {len(keepl)} row(s)")


if __name__ == "__main__":
    main()
