"""Falsifiability harness for build_cache.resolve_seed_dir.

WHY THIS EXISTS. A refold writes a SECOND finished pose directory for a seed it already folded, so
1,600 of the 5,300 validation units have two directories that both hold a run.json. seed_dir's
"exactly one finished directory" cannot choose between them, and `poses/index.tsv` names the
superseded run (measured wrong on 5 of 5 spot-checked 22gi seeds, because the original fold wrote
the index and the refold never updated it). The evaluator used to take glob's first hit, which was
right only by luck -- it joined 95 of 22gi's 100 seeds and failed the other 5, and a failure there
is fatal to the evaluator's gate.

resolve_seed_dir uses the cached trunk as the arbiter: `ci_coordinate` was written by exactly one
run, so exactly one directory's cifs join to it under sample_indices' identity contract. The
guarantee that matters is NOT that it finds the right directory -- it is that it REFUSES rather
than guesses when the evidence does not single one out. That is what this harness tests, by
building fold roots of symlinks whose answer is known in advance:

  ONE matching directory          -> must resolve, and to that directory
  TWO directories that both match -> must refuse (a duplicated run is not a resolved one)
  ONE directory, the wrong run    -> must refuse (never fall back to the only candidate)
  NO directory at all             -> must refuse
  A directory with no run.json    -> invisible, so the remaining state decides

It builds nothing but symlinks under a scratch root and never writes inside the fold, so it is
safe to run against live data. Exits non-zero if any case behaves differently.
"""
import argparse, os, shutil, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_j_build_cache as BC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="validation/fold")
    ap.add_argument("--pdb", default="22gi", help="a structure with two finished dirs per seed")
    ap.add_argument("--seed", default="400004")
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--coord-tol", type=float, default=0.01)
    ap.add_argument("--coord-margin", type=float, default=0.5)
    a = ap.parse_args()

    trunk = os.path.join(a.fold, "trunks", a.pdb, f"free_{a.seed}.npz")
    if not os.path.exists(trunk):
        sys.exit(f"no trunk at {trunk}")
    _z, _f, poses = BC.load_trunk(trunk)

    # Which of the real directories is the trunk's own? Measured, not assumed -- the harness needs
    # a known-true and a known-false directory and must not take either on faith.
    real = BC.seed_dirs_all(a.fold, a.pdb, a.seed)
    true_d, false_d = None, None
    for d in real:
        try:
            BC.sample_indices(poses, BC.sample_cifs(d, a.pdb), a.coord_tol, a.coord_margin)
            true_d = d
        except Exception:
            false_d = false_d or d
    if true_d is None or false_d is None:
        sys.exit(f"{a.pdb}/{a.seed} needs one matching and one non-matching dir, got "
                 f"{len(real)}: match={true_d} other={false_d}. Pick another --pdb/--seed.")

    root = os.path.join(a.scratch, "resolve_selftest")
    shutil.rmtree(root, ignore_errors=True)
    pd = os.path.join(root, "poses", a.pdb)
    os.makedirs(pd)

    def link(seed, src, name="r"):
        os.symlink(os.path.abspath(src), os.path.join(pd, f"{seed}-{name}"))

    link("1", true_d, "true")                                   # one matching
    link("2", true_d, "copyA"); link("2", true_d, "copyB")       # two that both match
    link("3", false_d, "wrong")                                 # one, wrong run
    link("5", true_d, "true"); link("5", false_d, "wrong")       # the real-world pair
    os.makedirs(os.path.join(pd, "6-norunjson"))                 # no run.json -> invisible

    cases = [("1", "ONE matching dir", "resolve"),
             ("2", "TWO dirs, BOTH match", "refuse"),
             ("3", "ONE dir, the WRONG run", "refuse"),
             ("4", "NO dir at all", "refuse"),
             ("5", "one true + one wrong (the real case)", "resolve"),
             ("6", "only a dir with no run.json", "refuse")]
    bad = 0
    for seed, desc, want in cases:
        try:
            d, ks = BC.resolve_seed_dir(root, a.pdb, seed, poses, a.coord_tol, a.coord_margin)
            got, msg = "resolve", f"-> {os.path.basename(d)} ks={[int(k) for k in ks]}"
            if want == "resolve" and os.path.realpath(d) != os.path.realpath(true_d):
                got, msg = "wrong-dir", f"resolved to {os.path.basename(d)}, not the trunk's own"
        except Exception as e:
            got, msg = "refuse", f"{type(e).__name__}: {str(e)[:88]}"
        ok = got == want
        bad += 0 if ok else 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {desc:38s} want={want:8s} got={got:9s} {msg}")

    print(f"\n{len(cases) - bad} of {len(cases)} case(s) behaved as designed"
          f"{'' if not bad else ' -- RESOLVER IS UNSAFE'}")
    shutil.rmtree(root, ignore_errors=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
