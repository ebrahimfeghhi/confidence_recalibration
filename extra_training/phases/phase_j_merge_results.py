"""Merge a refold root's results.csv into the destination root's, replacing rows by key.

WHY THIS EXISTS. STAGE 0 of queue_ft3_pilot.sh merged a refold into validation/fold by moving
trunks/ and poses/ and NOTHING ELSE, so validation/fold ended up holding the refold's trunks and
poses beside the ORIGINAL fold's score rows for 7tzh, 8v8f and 9cn2 -- 2,147 of 3,000 rows differed.
That is the "a new trunk must not be paired with an old pose" error in score-row form, and it is
worse than a missing row: the evaluator selects a pose using the new trunk's confidence and then
reads the OLD fold's DockQ for it, scoring the wrong thing with no symptom. The deferred merge did
this correctly (13,000 of 13,000 rows identical), which is why only one stage was affected.

Repeating a seed does not reproduce its poses -- the same seed refolded has gone from DockQ 0.022 to
0.644 -- so a score row belongs to exactly one fold of exactly one seed, and merging means REPLACING
the destination's row, never keeping both or preferring either by file order.

WHAT IT REFUSES. The key is (iface, seed, sample). A src key absent from dst is a row for a unit dst
never had, which means the roots are not what the caller thinks they are; that is refused rather
than appended, because the repair case is always a replacement. Duplicate keys within either file
are refused for the same reason. Nothing is written without --apply, and the destination is copied
aside first.
"""
import argparse, collections, csv, os, shutil, sys

KEY = ("iface", "seed", "sample")


def load(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"{path} has no rows")
    counts = collections.Counter(tuple(r[k] for k in KEY) for r in rows)
    dup = [k for k, n in counts.items() if n > 1]
    if dup:
        sys.exit(f"{path} has {len(dup)} duplicate {KEY} key(s), e.g. {dup[:3]}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the refold root's results.csv")
    ap.add_argument("--dst", required=True, help="the destination root's results.csv")
    ap.add_argument("--apply", action="store_true", help="without this, nothing is written")
    a = ap.parse_args()

    src, dst = load(a.src), load(a.dst)
    if [f for f in src[0]] != [f for f in dst[0]]:
        sys.exit(f"column mismatch:\n  src {list(src[0])}\n  dst {list(dst[0])}")
    cols = list(dst[0])
    val = [c for c in cols if c not in KEY]

    by_key = {tuple(r[k] for k in KEY): r for r in src}
    absent = [k for k in by_key if k not in {tuple(r[k] for k in KEY) for r in dst}]
    if absent:
        sys.exit(f"STOP: {len(absent)} src key(s) are absent from dst, e.g. {absent[:3]}. A merge "
                 f"here is a REPLACEMENT; appending would change the unit count.")

    same = diff = 0
    changed_ifaces = collections.Counter()
    for r in dst:
        k = tuple(r[k_] for k_ in KEY)
        s = by_key.get(k)
        if s is None:
            continue
        if all(r[c] == s[c] for c in val):
            same += 1
        else:
            diff += 1
            changed_ifaces[r["iface"]] += 1
            for c in val:
                r[c] = s[c]

    print(f"src rows {len(src)}  dst rows {len(dst)}  matched {same + diff}  "
          f"already identical {same}  REPLACED {diff}")
    for i, n in sorted(changed_ifaces.items()):
        print(f"    {i}: {n} row(s)")
    if not a.apply:
        print("\ndry run, nothing written. Re-run with --apply.")
        return

    bak = a.dst + ".pre_merge"
    if not os.path.exists(bak):
        shutil.copy2(a.dst, bak)
        print(f"copied the destination aside: {bak}")
    tmp = a.dst + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(dst)
    os.replace(tmp, a.dst)
    print(f"wrote {a.dst}: {len(dst)} row(s), {diff} replaced")


if __name__ == "__main__":
    main()
