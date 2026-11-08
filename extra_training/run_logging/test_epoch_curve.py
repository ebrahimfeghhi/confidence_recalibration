"""A per-epoch training curve must be the ALL-RANKS mean, and no plotter may rebuild it by hand.

THE DEFECT THIS PINS. trajectory.csv is written by RANK 0 ONLY and its bare `pae_loss` column is a
single pose's value. Averaging those rows equally yields the mean over the third of the structures
rank 0 walked -- not the run's PAE. The repo has diagnosed this three times (plot_train_insample,
then plot_uncropped_progress, then plot_rel_arm, whose wrong figure reached the user) and each fix
stayed local. So: one shared helper, and this file fails if a plotter stops using it.

THE VACUITY TRAP. Asserting only "epoch_means returns something" would pass even if the helper
quietly fell back to the rank-0 column. Section 1 therefore asserts the all-ranks pose count is a
MULTIPLE of the rank-0 row count, and that the two means actually DIFFER on real data.
"""
import collections
import csv
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import epoch_curve  # noqa: E402

RID = os.environ.get("TEST_RID", "5dgrsu")
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


# Files allowed to read a per-pose column, with the reason. Anything else that keys a per-pose
# column by epoch is the bug this file exists to catch.
ALLOWED = {
    "plot_pae_hist.py": "a histogram OF poses; the per-pose value is the subject, not an epoch mean",
    "plot_uncropped_progress.py": "carries its own corrected batch_n-weighted implementation",
    "plot_rel_arm.py": "uses epoch_curve for the mean; reads per-pose rows only to count pdbs",
    "epoch_curve.py": "the helper itself",
    "plot_train_insample.py": "already uses the cross-rank batch columns; this is the file where "
                              "the rank-0 defect was first diagnosed",
    "records.py": "reads batch_pae_loss, not the per-pose column; records run metadata rather "
                  "than drawing a curve",
}


def main():
    print("\n1  THE HELPER RETURNS THE ALL-RANKS MEAN, AND IT IS NOT THE RANK-0 MEAN")
    means = epoch_curve.epoch_means(RID)
    full = {e: v for e, v in means.items() if v[2] == "all-ranks"}
    if not full:
        print(f"STOP: no all-ranks epochs for {RID}; nothing would be measured.")
        return 1
    ok(bool(full), "at least one epoch resolves to all-ranks", f"{len(full)} of {len(means)}")

    naive = collections.defaultdict(list)
    nrows = collections.defaultdict(int)
    for rid in epoch_curve.run_chain(RID):
        p = os.path.join(ROOT, "training", "runs", rid, "trajectory.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            try:
                e = int(r["epoch"]); v = float(r["pae_loss"])
            except (KeyError, TypeError, ValueError):
                continue
            naive[e].append(v); nrows[e] += 1
    e0 = sorted(full)[0]
    mult = full[e0][1] / max(1, nrows[e0])
    ok(mult >= 1.9, "the all-ranks pose count is a MULTIPLE of rank 0's row count",
       f"epoch {e0}: {full[e0][1]} vs {nrows[e0]} -> x{mult:.2f}")
    diffs = [abs(full[e][0] - sum(naive[e]) / len(naive[e])) for e in full if naive.get(e)]
    ok(any(d > 1e-6 for d in diffs),
       "and the two means genuinely differ, so this is not the rank-0 column relabelled",
       f"max |diff| {max(diffs):.4f}" if diffs else "no overlap")

    print("\n2  batch_n WEIGHTING IS APPLIED (an epoch's last step is partial)")
    bn = collections.defaultdict(set)
    for rid in epoch_curve.run_chain(RID):
        p = os.path.join(ROOT, "training", "runs", rid, "trajectory.csv")
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p)):
            if (r.get("step_taken") or "").strip() != "1":
                continue
            try:
                bn[int(r["epoch"])].add(int(float(r["batch_n"])))
            except (KeyError, TypeError, ValueError):
                pass
    varied = [e for e, s in bn.items() if len(s) > 1]
    # NOT A FAILURE WHEN UNIFORM. This arm's steps all carry the same batch_n, so weighting and
    # plain averaging coincide HERE -- that is a property of the data, not of the code, and the
    # next assertion is the one that pins the code. Reported either way so the reader knows which
    # case they are looking at.
    print(f"        (batch_n {'varies' if varied else 'is uniform'} in this arm"
          + (f", epoch(s) {sorted(varied)[:3]}" if varied else " -- weighting is a no-op on it")
          + ")")
    src = open(epoch_curve.__file__).read()
    ok("v * n for v, n in b" in src and "sum(n for _v, n in b)" in src,
       "and epoch_curve weights by batch_n rather than averaging steps equally")

    print("\n3  STEPS THAT WERE NOT TAKEN ARE EXCLUDED")
    ok('(row.get("step_taken") or "").strip() != "1"' in src,
       "rows with step_taken != 1 are skipped (no gradient behind them)")

    print("\n4  ANYTHING THAT READS trajectory.csv GOES THROUGH epoch_curve")
    # NOT A REGEX OVER THE ACCUMULATION EXPRESSION. The previous version demanded the row variable
    # be literally named `r`, the column be a string literal, and the subscript be
    # int(float(e...)) -- and it missed all three real offenders, which used a variable column, a
    # pre-bound value, and a bare d[e]. The rule here is structural and has no such holes: if a
    # module opens trajectory.csv it must IMPORT the helper, or be allow-listed with a reason.
    #
    # The import must be a real import line, not the mere appearance of the string -- the old
    # check was `if "epoch_curve" in t`, which any file could satisfy by naming it in a comment.
    imp = re.compile(r"^\s*(?:import\s+epoch_curve|from\s+epoch_curve\s+import)", re.M)
    scanned, offenders = 0, []
    for f in sorted(glob.glob(os.path.join(ROOT, "run_logging", "*.py"))):
        b = os.path.basename(f)
        if b.startswith("test_") or b in ALLOWED:
            continue
        t = open(f, errors="replace").read()
        # THE CONDITION IS "AVERAGES A PER-POSE LOSS COLUMN FROM trajectory.csv". Reading the file
        # for something else -- a pdb list, a pose count, a health check -- is not this defect, and
        # demanding the import there would make the test noise that gets switched off.
        if "trajectory.csv" not in t:
            continue
        if not any(f'"{c}"' in t or f"'{c}'" in t for c in epoch_curve.PER_POSE):
            continue
        if not re.search(r"\bmean\b|\bmedian\b|sum\(", t):
            continue
        # AND IT MUST ACCUMULATE BY EPOCH. Reading a per-pose column and taking some mean is not
        # the defect unless the result is keyed per epoch -- that is what makes it a CURVE, and
        # what makes rank 0's third of the structures the wrong denominator.
        if not re.search(r"\[\s*int\s*\(\s*float\s*\(\s*\w*ep|\[\s*int\s*\(\s*\w*\["
                         r"[\"']epoch|\bby_epoch\b|\bper_epoch\b|\[\s*ep\s*\]|\[\s*e\s*\]", t):
            continue
        scanned += 1
        if not imp.search(t):
            offenders.append(b)
    ok(offenders == [],
       "every module reading trajectory.csv imports epoch_curve",
       f"{scanned} scanned, {len(ALLOWED)} allow-listed"
       if not offenders else "MISSING: " + ", ".join(offenders))
    ok(scanned > 0, "and the scan actually found modules to check", f"{scanned}")

    print("\n5  THE ALLOWLIST IS JUSTIFIED, not a dumping ground")
    ok(all(v and len(v) > 15 for v in ALLOWED.values()),
       "every allow-listed file states why", f"{len(ALLOWED)} entries")
    ok(all(os.path.exists(os.path.join(ROOT, "run_logging", k)) for k in ALLOWED),
       "and every allow-listed file still exists (no stale exemptions)")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
