#!/usr/bin/env python
"""Write validation/fold/queue.tsv -- the file folder.py actually consumes -- from the selection.

WHY THIS EXISTS. val30_cpu_chain.sh built val30_queue.json for the pool builder but never built
queue.tsv, and its final check only verified that whatever queue.tsv already held had pool rows and
references. An old, shorter queue therefore passed: the 13-structure / 1,300-unit queue did pass,
and the full queue was restored by hand, which is not reproducible. Raised in review.

THE UNIT SET IS STATED, NOT INHERITED: every selected structure times every seed in the range, one
row each, and the result is checked against that product before anything is written -- exact row
count, exact structure set, exact seed set per structure, and no duplicate (pdb, seed).

Columns are folder.py's: kind, pdb, target, tag, seed. target "*" means "score every interface of
this complex from the pool", which is what a free fold does.
"""
import argparse, csv, collections, os, shutil, sys

ap = argparse.ArgumentParser()
ap.add_argument("--selection", default="validation/full_val/val30_structures.csv")
ap.add_argument("--out", default="validation/fold/queue.tsv")
ap.add_argument("--seed0", type=int, default=400000)
ap.add_argument("--nseed", type=int, default=100)
ap.add_argument("--kind", default="free")
ap.add_argument("--target", default="*")
ap.add_argument("--tag", default="free")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

pdbs = [r["pdb"] for r in csv.DictReader(open(a.selection))]
dup = [p for p, n in collections.Counter(pdbs).items() if n > 1]
if dup:
    raise SystemExit(f"STOP: the selection lists {dup} more than once; nothing written")

seeds = list(range(a.seed0, a.seed0 + a.nseed))
rows = [(a.kind, p, a.target, a.tag, str(s)) for p in pdbs for s in seeds]

want_n = len(pdbs) * a.nseed
print(f"selection {len(pdbs)} structure(s) x {a.nseed} seed(s) "
      f"({seeds[0]}-{seeds[-1]}) = {want_n} unit(s)")

# verify the product before writing, so a malformed queue is never the thing on disk
seen = collections.Counter((r[1], r[4]) for r in rows)
bad = [k for k, n in seen.items() if n > 1]
by = collections.defaultdict(set)
for r in rows:
    by[r[1]].add(int(r[4]))
problems = []
if len(rows) != want_n:
    problems.append(f"built {len(rows)} rows, expected {want_n}")
if bad:
    problems.append(f"{len(bad)} duplicate (pdb, seed) unit(s), e.g. {bad[:3]}")
if set(by) != set(pdbs):
    problems.append(f"structure set differs from the selection")
off = {p: sorted(set(seeds) ^ by[p])[:4] for p in by if by[p] != set(seeds)}
if off:
    problems.append(f"seed set differs for {len(off)} structure(s), e.g. {list(off.items())[:2]}")
for w in problems:
    print(f"  PROBLEM: {w}")
if problems:
    raise SystemExit("STOP: nothing written")
print(f"  verified: {want_n} unique unit(s), {len(by)} structure(s), "
      f"seeds {seeds[0]}-{seeds[-1]} for every one")

if not a.apply:
    if os.path.exists(a.out):
        cur = [ln for ln in open(a.out) if ln.strip()]
        curset = {tuple(ln.rstrip("\n").split("\t")) for ln in cur}
        print(f"  on disk now: {len(cur)} row(s); "
              f"{'IDENTICAL' if curset == set(rows) else 'DIFFERENT -- would be replaced'}")
    print("report only; pass --apply to write")
    sys.exit(0)

if os.path.exists(a.out):
    bak = a.out + ".bak"
    shutil.copy2(a.out, bak)
    print(f"  previous queue backed up to {bak}")
with open(a.out, "w") as fh:
    for r in rows:
        fh.write("\t".join(r) + "\n")
n = sum(1 for ln in open(a.out) if ln.strip())
print(f"wrote {a.out}: {n} unit(s)")
if n != want_n:
    raise SystemExit(f"STOP: wrote {n} rows, expected {want_n}")
