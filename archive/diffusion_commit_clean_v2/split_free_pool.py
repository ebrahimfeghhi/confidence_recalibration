#!/usr/bin/env python
"""Reduce the FoldBench free pool to the two blocks the paper uses; park the rest beside it.

    ./split_free_pool.py            # dry run
    ./split_free_pool.py --apply

AFTER THIS there are exactly two result directories for the experiment:

    free/                   the pool (seeds 101-150) AND the compute-matched extras (151-200)
    selftmpl_mask_on_nc10/  the method

    free_other_seeds/       everything else that used to sit in free/, kept and readable

WHY SPLIT RATHER THAN ARCHIVE. `free/` holds 10,217 pose directories across eight seed blocks and
only 5,550 of them are the pool. The leftovers are not junk to throw away -- the 1327-1453 and
1915-2036 blocks are the depth-robustness runs behind "selection is depth-robust at 1024 seeds" -- but
they must not sit in the same directory as the experiment's own seeds, because `extra_free()` reads
"every seed outside the pool's first 50" and `matched_seeds()` then takes the LOWEST available. That
pair drew seeds 0-100 for the handful of complexes that had been experimented on before, and those
extras measured 0.060 worse than the pool on DockQ -- which is why adding free seeds appeared to make
the baseline WORSE. The blocks are concentrated, not spread: 75 of 111 complexes hold only pool seeds,
while 8pn0 alone carries 449 extras.

WHY MERGING THE MATCHED EXTRAS IN IS SAFE NOW. Nothing in `free/` will remain between 151 and 200
once this runs, so the clean block cannot collide with a legacy twin. Those eight twins were archived
earlier the same day (`archive/foldbench_free_legacy_151_200/`).

DONORS ARE UNAFFECTED, verified before writing: every donor path in every FoldBench queue and in the
method's `donor_pose` column resolves to a pool seed 101-150. Moving the other blocks cannot orphan a
template. The script re-asserts this and refuses to move anything if it ever stops being true.
"""
import argparse
import collections
import csv
import glob
import os
import re
import shutil

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
FREE = f"{FB}/free"
OTHER = f"{FB}/free_other_seeds"
MATCHED = f"{FB}/free_compute_matched"
POOL = (101, 150)
EXTRA = (151, 200)
DONOR = re.compile(r"/free/([0-9a-z]{4})/(\d+)-")


def keeps(seed):
    """out: whether this seed belongs to the experiment's own two blocks"""
    return POOL[0] <= seed <= POOL[1] or EXTRA[0] <= seed <= EXTRA[1]


def assert_donors_in_pool():
    """Refuse to run if any donor path points outside the pool block -- moving would orphan it."""
    bad = set()
    for q in glob.glob(f"{FB}/*/queue*.tsv") + glob.glob(f"{FB}/*/*/queue*.tsv"):
        for line in open(q):
            for m in DONOR.finditer(line):
                if not POOL[0] <= int(m.group(2)) <= POOL[1]:
                    bad.add((os.path.relpath(q, FB), m.group(1), m.group(2)))
    for r in csv.DictReader(open(f"{FB}/selftmpl_mask_on_nc10/results.csv")):
        m = DONOR.search(r.get("donor_pose", "") or "")
        if m and not POOL[0] <= int(m.group(2)) <= POOL[1]:
            bad.add(("method results.csv", m.group(1), m.group(2)))
    assert not bad, f"donors outside the pool block -- refusing to move: {sorted(bad)[:5]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    assert_donors_in_pool()

    move, stay = [], []
    for d in glob.glob(f"{FREE}/*/*"):
        if not os.path.isdir(d):
            continue
        s = d.split("/")[-1].split("-")[0]
        (stay if (s.isdigit() and keeps(int(s))) else move).append(d)
    blocks = collections.Counter()
    for d in move:
        s = d.split("/")[-1].split("-")[0]
        blocks[int(s) // 50 * 50 if s.isdigit() else -1] += 1
    print(f"free/: {len(stay)} pose dirs stay, {len(move)} move to free_other_seeds/")
    for k in sorted(blocks):
        print(f"   seeds {k}-{k + 49}: {blocks[k]}")

    keep_rows, move_rows = [], []
    with open(f"{FREE}/results.csv") as fh:
        rd = csv.DictReader(fh)
        cols = rd.fieldnames
        for r in rd:
            s = r["seed"]
            (keep_rows if (s.isdigit() and keeps(int(s))) else move_rows).append(r)
    print(f"results.csv: {len(move_rows)} rows move, {len(keep_rows)} stay")

    mdirs = [d for d in glob.glob(f"{MATCHED}/*/*") if os.path.isdir(d)]
    mrows = list(csv.DictReader(open(f"{MATCHED}/results.csv"))) if os.path.exists(
        f"{MATCHED}/results.csv") else []
    print(f"free_compute_matched -> free/: {len(mdirs)} pose dirs, {len(mrows)} rows")
    if not A.apply:
        print("\ndry run -- pass --apply")
        return

    # 1. move the other blocks OUT before the matched extras come IN, so free/ is never holding both
    #    a legacy and a clean copy of one seed
    for d in move:
        pdb, name = d.split("/")[-2], d.split("/")[-1]
        os.makedirs(f"{OTHER}/{pdb}", exist_ok=True)
        tgt = f"{OTHER}/{pdb}/{name}"
        if not os.path.exists(tgt):
            shutil.move(d, tgt)
    os.makedirs(OTHER, exist_ok=True)
    with open(f"{OTHER}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(move_rows)

    # 2. now fold the matched extras in
    left = {(d.split("/")[-2], d.split("/")[-1].split("-")[0]) for d in stay}
    for d in mdirs:
        pdb, name = d.split("/")[-2], d.split("/")[-1]
        assert (pdb, name.split("-")[0]) not in left, f"collision on {pdb} seed {name}"
        os.makedirs(f"{FREE}/{pdb}", exist_ok=True)
        tgt = f"{FREE}/{pdb}/{name}"
        if not os.path.exists(tgt):
            shutil.move(d, tgt)
    for c in glob.glob(f"{MATCHED}/claims/*"):
        os.makedirs(f"{FREE}/claims", exist_ok=True)
        tgt = f"{FREE}/claims/{os.path.basename(c)}"
        if not os.path.exists(tgt):
            shutil.move(c, tgt)
    for f in glob.glob(f"{MATCHED}/queue*.tsv") + glob.glob(f"{MATCHED}/*.sh"):
        shutil.copy2(f, f"{FREE}/{os.path.basename(f)}")

    if not os.path.exists(f"{FREE}/results.csv.split.bak"):
        shutil.copy2(f"{FREE}/results.csv", f"{FREE}/results.csv.split.bak")
    with open(f"{FREE}/results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(keep_rows)
        w.writerows([{k: r.get(k, "") for k in cols} for r in mrows])
    print(f"free/results.csv now {len(keep_rows) + len(mrows)} rows "
          f"({len(keep_rows)} pool/extra + {len(mrows)} matched)")
    print(f"free_other_seeds/: {len(move)} pose dirs, {len(move_rows)} rows")


if __name__ == "__main__":
    main()
