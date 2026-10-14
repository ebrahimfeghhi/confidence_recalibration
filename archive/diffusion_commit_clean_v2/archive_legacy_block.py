#!/usr/bin/env python
"""Move the FoldBench free pool's undocumented seed 151-200 folds out to the archive.

    ./archive_legacy_block.py            # dry run
    ./archive_legacy_block.py --apply

WHY. Eight complexes carry a full 151-200 block inside the free pool, in directories named
`<seed>-legacy-no-provenance`: no run.json, no recorded ncycle, precision or template setting. They
are not the same folds as the clean block the matched baseline is built from. On 7yxu seed 151 the
legacy copy scores DockQ 0.010 and the provenance-recorded refold scores 0.673 -- same complex, same
seed, so the settings cannot have matched. Leaving them in the pool means a (complex, seed) resolves
to two different answers and `extra_free()` can pick up either, which is the contamination that made
the old compute-matched baseline look like adding free seeds HURT.

WHAT MOVES. Only seeds 151-200 of those eight complexes, poses and their results.csv rows together.
Nothing is deleted -- folds do not reproduce, so the archive keeps them readable -- and no other seed
of any complex is touched. The pool's own 101-150 (the "free 50" everything is measured against) is
untouched by construction: this script never looks outside 151-200.

AFTER THIS, the rule holds: a (complex, seed) exists in exactly one place, and the copy with
provenance is the data.
"""
import argparse
import csv
import os
import shutil
import sys

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
DEST = "/home/ebrahim/tts_abag/experimental_data/archive/foldbench_free_legacy_151_200"
LO, HI = 151, 200


def legacy_complexes():
    """out: {complex: [pose dir names]} for every free-pool complex holding folds in [LO, HI]"""
    out = {}
    for p in sorted(os.listdir(f"{FB}/free")):
        d = f"{FB}/free/{p}"
        if not os.path.isdir(d):
            continue
        hit = [x for x in sorted(os.listdir(d))
               if x.split("-")[0].isdigit() and LO <= int(x.split("-")[0]) <= HI]
        if hit:
            out[p] = hit
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    found = legacy_complexes()
    n_dirs = sum(len(v) for v in found.values())
    print(f"{len(found)} complexes, {n_dirs} pose directories in seeds {LO}-{HI}")
    for p, v in found.items():
        print(f"  {p}: {len(v)}")

    src_csv = f"{FB}/free/results.csv"
    keep, move = [], []
    with open(src_csv) as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        for r in rd:
            s = r["seed"]
            (move if (r["iface"].split("_")[0] in found and s.isdigit()
                      and LO <= int(s) <= HI) else keep).append(r)
    print(f"results.csv: {len(move)} rows move, {len(keep)} stay")
    if not A.apply:
        print("\ndry run -- pass --apply")
        return

    os.makedirs(f"{DEST}/poses", exist_ok=True)
    moved = 0
    for p, v in found.items():
        os.makedirs(f"{DEST}/poses/{p}", exist_ok=True)
        for d in v:
            dst = f"{DEST}/poses/{p}/{d}"
            if os.path.exists(dst):          # a rerun of this script: already archived, skip
                continue
            shutil.move(f"{FB}/free/{p}/{d}", dst)
            moved += 1
    with open(f"{DEST}/results_archived.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(move)
    # the pool csv is rewritten only after the poses are safely in the archive, so an interruption
    # leaves rows pointing at moved poses rather than poses with no rows
    if not os.path.exists(src_csv + ".bak"):
        shutil.copy2(src_csv, src_csv + ".bak")
    with open(src_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(keep)
    print(f"moved {moved} pose dirs and {len(move)} rows to {DEST}")
    print(f"free/results.csv rewritten ({len(keep)} rows); backup at results.csv.bak")


if __name__ == "__main__":
    main()
