#!/usr/bin/env python
"""Mark which rows of training/fold/results.csv can actually be SCORED, and emit the scoreable
subset beside it.

WHY THIS EXISTS. results.csv keeps a DockQ row for every pose that was ever folded. Structures
trimmed to one seed still carry their old 10 seeds of DockQ, and the evaluator recomputes ipTM
from the TRUNK -- so a seed whose trunk was reclaimed has a label and no way to produce a score.
Reading results.csv as "what we have" is therefore wrong, and it misled three separate analyses
in one morning: it produced a claim that 21% of training poses were unused, then a claim that the
shallow structures carry 10 usable seeds, then a 947-interface pool built to dodge a depth
mismatch that was really 15-vs-1 rather than 15-vs-10.

TWO OUTPUTS, ONE SOURCE.
  * results.csv gains a `has_trunk` column (1/0). The truth travels WITH the row, so a reader who
    reaches for the obvious filename still sees it. Every reader in this repo uses csv.DictReader,
    so an appended column changes nothing for them.
  * results_with_trunks.csv holds only has_trunk=1 rows, for readers that want the scoreable set
    without filtering.

DERIVED, NEVER HAND-MAINTAINED, AND SELF-INVALIDATING. Both outputs carry the trunk-set digest
they were built from in a sidecar, and --check recomputes it and fails if the trunks have moved
since. A second hand-kept file is the same failure one layer along: it goes stale quietly, which
is exactly what results.csv did.

NOTHING IS DELETED. results.csv keeps every row it ever had -- it remains the record of what was
folded, which is a different and still useful question from what can be scored.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "training/fold/results.csv")
TRUNKS = os.path.join(ROOT, "training/fold/trunks")
OUT = os.path.join(ROOT, "training/fold/results_with_trunks.csv")
SIDE = os.path.join(ROOT, "training/fold/results_with_trunks.meta.json")
SEED_RE = re.compile(r"free_(\d+)\.npz$")


def trunk_index():
    """pdb -> set of seeds with a trunk on disk, plus a digest of the whole set."""
    have, n = {}, 0
    for pdb in sorted(os.listdir(TRUNKS)):
        d = os.path.join(TRUNKS, pdb)
        if not os.path.isdir(d):
            continue
        seeds = set()
        for f in os.listdir(d):
            m = SEED_RE.search(f)
            if m:
                seeds.add(int(m.group(1)))
        if seeds:
            have[pdb] = seeds
            n += len(seeds)
    h = hashlib.sha256()
    for pdb in sorted(have):
        h.update(f"{pdb}:{','.join(str(s) for s in sorted(have[pdb]))};".encode())
    return have, n, h.hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="verify the derived file matches the trunks on disk; write nothing")
    a = ap.parse_args()

    have, n_trunks, digest = trunk_index()
    rows = list(csv.DictReader(open(RESULTS)))
    if not rows:
        raise SystemExit(f"STOP: {RESULTS} has no rows")
    # THE KEY IS (pdb, seed). `iface` is pdb+chains concatenated -- '10fj_AB', not '10fj_A_B' --
    # and the pdb is what owns a trunk, so the id is taken before the first underscore.
    for r in rows:
        pdb = r["iface"].split("_")[0]
        r["has_trunk"] = "1" if int(r["seed"]) in have.get(pdb, ()) else "0"
    keep = [r for r in rows if r["has_trunk"] == "1"]
    drop = len(rows) - len(keep)

    if a.check:
        if not os.path.exists(SIDE):
            raise SystemExit(f"STOP: {os.path.basename(SIDE)} is absent -- the derived file was "
                             f"never built, or was built by an older version. Re-run without "
                             f"--check.")
        side = json.load(open(SIDE))
        if side.get("trunk_digest") != digest:
            raise SystemExit(
                f"STOP: the trunks have changed since results_with_trunks.csv was built "
                f"(digest {side.get('trunk_digest')} -> {digest}).\n"
                f"  It is STALE. Re-run this script; do not read it as it stands.")
        print(f"  results_with_trunks.csv is current ({side['kept']} row(s), "
              f"trunk digest {digest})")
        return 0

    cols = list(rows[0].keys())
    for p, data in ((RESULTS, rows), (OUT, keep)):
        tmp = p + ".tmp"
        with open(tmp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(data)
        if os.path.exists(p):
            os.chmod(tmp, os.stat(p).st_mode & 0o7777)
        os.replace(tmp, p)

    json.dump({"trunk_digest": digest, "n_trunks": n_trunks,
               "rows_total": len(rows), "kept": len(keep), "dropped_no_trunk": drop,
               "source": "training/fold/results.csv",
               "note": "kept = rows whose (pdb, seed) has a trunk on disk and can be SCORED. "
                       "results.csv keeps every row and now carries has_trunk."},
              open(SIDE, "w"), indent=1)

    print(f"  trunks on disk      : {n_trunks} over {len(have)} structure(s), digest {digest}")
    print(f"  results.csv rows    : {len(rows)}  (+ has_trunk column)")
    print(f"  scoreable (kept)    : {len(keep)}")
    print(f"  no trunk (dropped)  : {drop}  -> these have a DockQ label and cannot be scored")
    print(f"  wrote {os.path.relpath(OUT, ROOT)} and its .meta.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
