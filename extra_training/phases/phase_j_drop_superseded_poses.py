"""Remove the pose directories that no trunk was computed from, restoring one dir per (pdb, seed).

WHY. The invariant the pipeline was designed around is ONE finished pose directory per (pdb, seed).
watch_foldbench_then_training.sh put each refold in a fresh root precisely so a new trunk could
never be paired with an old pose, and build_cache refuses a seed with two finished directories
rather than guessing. The merges into validation/fold broke that invariant: their collision test was
keyed on the <seed>-<runid> path, and a refold's runid is always new, so it never fired. 1,600 of
5,300 units ended up with two finished directories across 16 structures, and the evaluator read
whichever one the filesystem happened to list first -- right 95% of the time by luck, and fatal on
the other 812. Restoring the invariant is the fix; resolving the ambiguity at read time is a guard.

WHAT GOES, AND WHY IT IS SAFE. For each seed with more than one finished directory, the cached
trunk's ci_coordinate was written by exactly ONE of the runs: that directory's cifs match it to
0.0000 A while the others sit 0.026-3.118 A away, two to three orders of magnitude outside the
tolerance. The matching directory is KEPT. The others are superseded -- the trunks of those folds
were deleted in the 381 GB reclaim, so no trunk exists for them and they cannot serve the confidence
head, and their score rows have been replaced in results.csv by the refold's. Folds do not
reproduce, so this is permanent: what goes is the original pose cifs and their ost_json DockQ.

WHAT IT REFUSES. Evidence is recomputed here rather than read from a list, because a list written
earlier can go stale between writing and acting. A seed whose trunk singles out NEITHER directory or
BOTH is left untouched and reported -- that is the ambiguous case the resolver refuses, and it must
not be resolved by deleting one side. A seed with a single directory is never touched. Nothing is
removed without --apply, and the surviving directory is re-checked against the trunk BEFORE its
rival is removed, so a seed is never left with no readable poses.
"""
import argparse, collections, glob, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_j_build_cache as BC


def survey(fold, tol, margin):
    """(keep, drop, unclear) for every seed in `fold` with more than one finished pose directory."""
    tdir = os.path.join(fold, "trunks")
    keep, drop, unclear = {}, [], []
    for pdb in sorted(os.listdir(tdir)):
        if not os.path.isdir(os.path.join(tdir, pdb)):
            continue
        for p in sorted(glob.glob(os.path.join(tdir, pdb, "*.npz"))):
            seed = os.path.basename(p).split("_")[-1].split(".")[0]
            dirs = BC.seed_dirs_all(fold, pdb, seed)
            if len(dirs) < 2:
                continue
            _z, _f, poses = BC.load_trunk(p)
            good = []
            for d in dirs:
                try:
                    BC.sample_indices(poses, BC.sample_cifs(d, pdb), tol, margin)
                    good.append(d)
                except Exception:
                    pass
            if len(good) != 1:
                unclear.append((pdb, seed, len(dirs), len(good)))
                continue
            keep[(pdb, seed)] = (good[0], p)
            drop.extend(d for d in dirs if d != good[0])
    return keep, drop, unclear


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True, help="the fold root holding trunks/ and poses/")
    ap.add_argument("--coord-tol", type=float, default=0.01)
    ap.add_argument("--coord-margin", type=float, default=0.5)
    ap.add_argument("--apply", action="store_true", help="without this, nothing is removed")
    ap.add_argument("--list-out", default="", help="write the superseded directories to a file")
    a = ap.parse_args()

    if not os.path.isdir(os.path.join(a.fold, "trunks")):
        sys.exit(f"no trunks/ under {a.fold}")

    keep, drop, unclear = survey(a.fold, a.coord_tol, a.coord_margin)
    per = collections.Counter(d.split(os.sep)[-2] for d in drop)
    size = 0
    for d in drop:
        for r, _dn, fn in os.walk(d):
            for f in fn:
                try:
                    size += os.path.getsize(os.path.join(r, f))
                except OSError:
                    pass
    print(f"ambiguous seeds, trunk resolved : {len(keep)}")
    print(f"superseded directories         : {len(drop)} across {len(per)} structure(s), "
          f"{size / 2**30:.1f} GiB")
    for k, n in sorted(per.items()):
        print(f"    {k}: {n}")
    if unclear:
        print(f"LEFT UNTOUCHED, trunk resolves to neither or both: {len(unclear)}")
        for u in unclear[:10]:
            print(f"    {u[0]}/{u[1]}: {u[2]} dir(s), {u[3]} match")
    if a.list_out:
        with open(a.list_out, "w") as fh:
            for d in sorted(drop):
                fh.write(d + "\n")
        print(f"list written to {a.list_out}")

    if not a.apply:
        print("\ndry run, nothing removed. Re-run with --apply.")
        return

    # The survivor is re-checked against its trunk immediately before its rival goes, so a seed is
    # never left with no readable poses on the strength of a measurement taken minutes earlier.
    by_seed = collections.defaultdict(list)
    for d in drop:
        by_seed[(d.split(os.sep)[-2], os.path.basename(d).rsplit("-", 1)[0])].append(d)

    gone = skipped = 0
    for (pdb, seed), losers in sorted(by_seed.items()):
        survivor, trunk = keep[(pdb, seed)]
        _z, _f, poses = BC.load_trunk(trunk)
        try:
            BC.sample_indices(poses, BC.sample_cifs(survivor, pdb), a.coord_tol, a.coord_margin)
        except Exception as e:
            print(f"  SKIP {pdb}/{seed}: the survivor no longer joins its trunk "
                  f"({type(e).__name__}: {str(e)[:70]})")
            skipped += len(losers)
            continue
        for d in losers:
            shutil.rmtree(d)
            gone += 1
    print(f"removed {gone} superseded pose directory(ies)"
          + (f", skipped {skipped}" if skipped else ""))

    # RE-ASSERT THE INVARIANT FROM DISK rather than trusting the loop above. What should remain is
    # the seeds this run deliberately left alone: the ambiguous ones, plus any whose survivor
    # stopped joining its trunk between the survey and the removal.
    tdir = os.path.join(a.fold, "trunks")
    left = []
    for pdb in sorted(os.listdir(tdir)):
        if not os.path.isdir(os.path.join(tdir, pdb)):
            continue
        for p in glob.glob(os.path.join(tdir, pdb, "*.npz")):
            seed = os.path.basename(p).split("_")[-1].split(".")[0]
            if len(BC.seed_dirs_all(a.fold, pdb, seed)) > 1:
                left.append(f"{pdb}/{seed}")
    expect = len(unclear) + len([k for k, v in by_seed.items() if any(os.path.isdir(d)
                                                                     for d in v)])
    print(f"seeds still holding more than one finished pose directory: {len(left)} "
          f"(expected {expect}: {len(unclear)} ambiguous + {expect - len(unclear)} skipped)")
    if len(left) != expect:
        sys.exit(f"STOP: expected {expect}, found {len(left)}, e.g. {left[:5]}")
    print("the one-directory-per-seed invariant holds for every seed this run resolved")


if __name__ == "__main__":
    main()
