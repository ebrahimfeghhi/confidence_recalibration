#!/usr/bin/env python
"""Promote recovered DockQ rows into a results.csv as ONE checked transaction.

WHY A SEPARATE TOOL. phase_j produces rows; this decides whether they may become part of the
canonical record. Those are different questions and want different gates. phase_j's --skip-existing
is right for exploratory output -- it stops the same sweep re-emitting what it already emitted --
but canonical ingestion needs a contract, because once a row is in results.csv every average over
its seed includes it and nothing downstream will ever question where it came from.

WHAT IT REFUSES, and why each clause is here rather than left to inspection:

  AN UNEXPECTED ROW OR INTERFACE COUNT. The recovery is a known, closed set: 17 interfaces x 10
  seeds x 5 samples = 850. A sweep that returns 849 or 851 has either missed a pose or scored one
  twice, and both are invisible in a file this size. The count is passed in rather than inferred so
  that a wrong sweep cannot define its own success criterion.

  A KEY ALREADY IN THE CANONICAL FILE. A duplicate (iface, seed, arm, sample) double-weights that
  pose in every average over its seed -- the exact defect dedupe_results.py exists to resolve, and
  there is no reason to create more of them.

  THE SAME KEY FROM TWO SOURCES. 4zto_EF can be produced both by pair isolation and by the
  full-complex route, and the two agree to four decimals, which makes a double-append easy to miss.
  Each key must come from exactly one input file, and the file it came from is recorded.

  A ROW THE PIPELINE WOULD NOT READ AS A SCORE. run_model/folder.py's open_csv does
  `float(r["dockq"]), float(r["iptm"])` and skips the row when either does not parse or is not
  finite. A blank iptm would therefore sit in the csv looking present while the pipeline counted
  the sample as still to do -- worse than an honest gap. Provenance columns are checked for the
  same reason: arm, donor_pose and tmpl_chains are what distinguish a free fold from a templated
  one, and phase_j hardcodes them.

  AN INTERFACE THAT IS NOT IN THE POOL. The pool defines which interfaces exist. A row naming one
  it does not know is either a typo or a mis-attribution, and section 2 of SCORING_RECOVERY.md is
  about how expensive the latter is to detect later.

  A CANONICAL FILE THAT IS NOT ALREADY CLEAN. If it holds duplicate keys of its own, they are
  carried into the combined file and the post-merge uniqueness check becomes meaningless. Run
  dedupe_results.py first; this will not paper over them.

HOW IT WRITES. It builds the combined file under a temporary name, re-reads the canonical file to
confirm nothing appended while it worked, takes the same `<csv>.rescore.lock` the rescorer and
dedupe_results.py use, copies the canonical file aside, and os.replaces -- atomic within one
filesystem, so no reader ever sees a partial csv. Without --replace it writes the snapshot and the
audit and touches nothing.
"""
import argparse, collections, csv, fcntl, math, os, shutil, subprocess, sys

CANON = ["iface", "seed", "arm", "donor_pose", "tmpl_chains", "sample", "dockq", "iptm"]
KEY = ("iface", "seed", "arm", "sample")
ARM, DONOR, TMPL = "free", "-1", ""


def key_of(r):
    return tuple(r[c] for c in KEY)


def finite(x):
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def read_rows(path):
    with open(path) as fh:
        rd = csv.DictReader(fh)
        if rd.fieldnames != CANON:
            raise SystemExit(f"STOP: {path} header is {rd.fieldnames}, expected {CANON}. "
                             f"A column order that differs from scorer.py's is not appendable.")
        return list(rd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="the canonical results.csv")
    ap.add_argument("--rows", nargs="+", required=True, help="recovered rows, canonical columns")
    ap.add_argument("--pool", default="", help="pool.csv (default: beside --csv)")
    ap.add_argument("--expect-rows", type=int, required=True)
    ap.add_argument("--expect-ifaces", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=500000)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--out", default="", help="write the combined snapshot here")
    ap.add_argument("--replace", action="store_true",
                    help="swap the snapshot in atomically after every check passes")
    a = ap.parse_args()
    pool_csv = a.pool or os.path.join(os.path.dirname(os.path.abspath(a.csv)), "pool.csv")
    out = a.out or a.csv + ".merged"
    if os.path.abspath(out) == os.path.abspath(a.csv):
        raise SystemExit("STOP: --out must not be the canonical csv; the swap is a separate step")

    seeds = {str(a.seed0 + i) for i in range(a.seeds)}
    samples = {str(i) for i in range(a.nsample)}
    pool = {f"{r['pdb']}_{r['chain1']}{r['chain2']}"
            for r in csv.DictReader(open(pool_csv))}

    # ---- the canonical file, and its own cleanliness -------------------------------------------
    live = read_rows(a.csv)
    seen = collections.Counter(key_of(r) for r in live)
    dup = {k: n for k, n in seen.items() if n > 1}
    print(f"canonical {a.csv}: {len(live)} row(s), {len(seen)} distinct {'+'.join(KEY)}")
    problems = []
    if dup:
        problems.append(f"the canonical file already holds {len(dup)} duplicated key(s) "
                        f"(e.g. {sorted(dup)[:3]}); run dedupe_results.py --replace first")

    # ---- the recovered rows --------------------------------------------------------------------
    src, rec = {}, []
    for p in a.rows:
        rows = read_rows(p)
        print(f"recovered {p}: {len(rows)} row(s)")
        for r in rows:
            k = key_of(r)
            if k in src:
                problems.append(f"{k} appears in both {src[k]} and {p}; each key must have one "
                                f"source, or the row is appended twice")
                continue
            src[k] = p
            rec.append(r)

    bad = collections.Counter()
    for r in rec:
        if not finite(r["dockq"]) or not finite(r["iptm"]):
            bad["non-finite dockq or iptm (folder.open_csv would skip the row)"] += 1
        if (r["arm"], r["donor_pose"], r["tmpl_chains"]) != (ARM, DONOR, TMPL):
            bad[f"provenance is not ({ARM},{DONOR},{TMPL!r})"] += 1
        if r["seed"] not in seeds:
            bad["seed outside the expected range"] += 1
        if r["sample"] not in samples:
            bad["sample outside 0..nsample-1"] += 1
        if r["iface"] not in pool:
            bad["interface is not in pool.csv"] += 1
    for why, n in bad.items():
        problems.append(f"{n} recovered row(s): {why}")

    already = [k for k in src if k in seen]
    if already:
        problems.append(f"{len(already)} recovered key(s) are already in the canonical file "
                        f"(e.g. {sorted(already)[:3]})")

    per = collections.Counter(r["iface"] for r in rec)
    want_each = a.seeds * a.nsample
    short = {i: n for i, n in per.items() if n != want_each}
    if short:
        problems.append(f"{len(short)} interface(s) do not have exactly {want_each} rows: "
                        f"{sorted(short.items())[:5]}")
    if len(rec) != a.expect_rows:
        problems.append(f"{len(rec)} recovered row(s), expected {a.expect_rows}")
    if len(per) != a.expect_ifaces:
        problems.append(f"{len(per)} recovered interface(s), expected {a.expect_ifaces}")

    if problems:
        print("\nMERGE REFUSED:")
        for p in problems:
            print(f"  - {p}")
        return 1

    # ---- build the combined snapshot -----------------------------------------------------------
    combined = live + rec
    ck = collections.Counter(key_of(r) for r in combined)
    if len(ck) != len(combined):
        print(f"\nMERGE REFUSED: the combined file would hold {len(combined)} rows but only "
              f"{len(ck)} distinct keys")
        return 1
    tmp = out + ".tmp"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CANON, lineterminator="\n")
        w.writeheader()
        w.writerows(combined)
    os.replace(tmp, out)
    print(f"\nALL CHECKS PASSED")
    print(f"  {len(live)} canonical + {len(rec)} recovered = {len(combined)} row(s), all distinct")
    print(f"  {len(per)} interface(s), {want_each} row(s) each")
    print(f"  wrote the combined snapshot to {out}")

    if not a.replace:
        print("  --replace not given, so the canonical file is untouched")
        return 0

    # ---- the swap ------------------------------------------------------------------------------
    # WHICH SESSIONS COULD WRITE *THIS* CSV, derived the same way dedupe_results.py derives it, so
    # the two cannot disagree about who a writer is. A fold whose csv lives under another fold root
    # is not a writer of this one; anything whose root cannot be determined is treated as one.
    WRITERS = {"training": ("w1fold", "w2_"), "validation": ("w1val",)}
    parts = os.path.abspath(a.csv).split(os.sep)
    fold = next((p for p in parts if p in WRITERS), None)
    prefixes = WRITERS.get(fold) or tuple(p for v in WRITERS.values() for p in v)
    sess = [ln.split(":")[0] for ln in
            subprocess.run(["tmux", "ls"], capture_output=True, text=True).stdout.splitlines()
            if ln.startswith(prefixes)]
    if sess:
        raise SystemExit(f"STOP: {sess} could still be writing {a.csv}. The snapshot is at "
                         f"{out}; re-run --replace once they have exited.")
    lp = a.csv + ".rescore.lock"
    lk = os.open(lp, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"STOP: another writer holds {lp}. The snapshot is at {out}.")
    # THE FILE MUST STILL BE THE ONE THAT WAS READ. The lock excludes a rescore from here on, but
    # not one that appended while the checks above were running, and swapping then would silently
    # drop its rows.
    now = sum(1 for _ in csv.DictReader(open(a.csv)))
    if now != len(live):
        raise SystemExit(f"STOP: {a.csv} held {len(live)} row(s) when this snapshot was built and "
                         f"holds {now} now. The snapshot at {out} would lose the difference.")
    bak = a.csv + ".pre_recovered_merge"
    shutil.copy2(a.csv, bak)
    os.replace(out, a.csv)
    print(f"  replaced {a.csv} atomically; previous content at {bak}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
