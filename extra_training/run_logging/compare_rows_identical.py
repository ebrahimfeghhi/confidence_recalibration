#!/usr/bin/env python
"""Do two eval passes produce the same per-pose scores AND the same SELECTIONS?

WHAT THIS GATES. --crop-predicted-margin must score one pose per forward pass instead of five.
If splitting the pose dimension changed a score by itself, every cropped number downstream would
be measuring the split rather than the crop -- so the uncropped split is run first and compared
against the batched reference, and the cropped passes do not run unless this passes.

EXACT IS NOT THE RIGHT BAR, AND NEITHER IS "CLOSE ENOUGH". Changing the batch composition changes
the order of floating-point reductions on the GPU, so bit-identity is not expected. What IS
expected is agreement at roughly float32 round-off. The default tolerance is 1e-5 on a score in
[0,1]; the max observed difference is always printed, so a pass at 9e-6 and a pass at 1e-12 are
distinguishable rather than both reading as "OK".

COVERAGE IS CHECKED, NOT ASSUMED. A pass that scored half the poses and agreed on all of them is
not a match; every (iface, seed, sample) key present in either side must be present in both.

    python -I run_logging/compare_rows_identical.py <dir_a> <dir_b> [--tol 1e-5]
"""
import argparse
import csv
import glob
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("a")
ap.add_argument("b")
ap.add_argument("--tol", type=float, default=1e-5)
ap.add_argument("--cols", default="iptm,iptm_mean")
g = ap.parse_args()


def load(d):
    fs = sorted(glob.glob(os.path.join(d, "rows_b*_s*.csv")))
    if not fs:
        print(f"  MISSING: no rows_b*_s*.csv under {d}")
        return None
    out = {}
    for f in fs:
        for r in csv.DictReader(open(f)):
            out[(r["iface"], r["seed"], r["sample"])] = r
    return out


A, B = load(g.a), load(g.b)
if A is None or B is None:
    sys.exit(2)

ka, kb = set(A), set(B)
only_a, only_b = ka - kb, kb - ka
print(f"  A {g.a}: {len(A)} pose(s)")
print(f"  B {g.b}: {len(B)} pose(s)")
if only_a or only_b:
    print(f"  COVERAGE MISMATCH: {len(only_a)} only in A, {len(only_b)} only in B")
    for k in list(only_a)[:3] + list(only_b)[:3]:
        print(f"      {k}")
    sys.exit(3)

bad = 0
worst = {}
for col in [c for c in g.cols.split(",") if c]:
    mx, mxk = 0.0, None
    for k in ka:
        va, vb = A[k].get(col, ""), B[k].get(col, "")
        if va in ("", None) or vb in ("", None):
            print(f"  COLUMN {col} missing on {k}")
            bad += 1
            continue
        d = abs(float(va) - float(vb))
        if d > mx:
            mx, mxk = d, k
    worst[col] = (mx, mxk)
    over = sum(1 for k in ka
               if A[k].get(col) not in ("", None) and B[k].get(col) not in ("", None)
               and abs(float(A[k][col]) - float(B[k][col])) > g.tol)
    print(f"  {col:<10} max |A-B| = {mx:.3e} at {mxk}   over tol({g.tol:g}): {over}")
    bad += over

# ---- THE DECISION, NOT JUST THE NUMBERS --------------------------------------------------
# A difference below the tolerance can still flip an argmax: two poses 1e-7 apart swap places on
# a 1e-6 perturbation, and selected DockQ then changes while every value "matches". So the
# selected pose is compared directly, under aggregate's own tie-break (phase_k_eval.py:406 --
# max by (score, -seed, -sample)), for BOTH selectors.
def selected(T, col):
    by = {}
    for (iface, sd, sm), r in T.items():
        v = r.get(col)
        if v in ("", None):
            continue
        by.setdefault(iface, []).append((float(v), -int(sd), -int(sm), sd, sm))
    return {i: max(v)[3:] for i, v in by.items()}


flipped = 0
for col in [c for c in g.cols.split(",") if c]:
    sa, sb = selected(A, col), selected(B, col)
    ifs = set(sa) | set(sb)
    diff = [i for i in ifs if sa.get(i) != sb.get(i)]
    print(f"  {col:<10} selected pose differs on {len(diff)} of {len(ifs)} interface(s)"
          + (f"   e.g. {[(i, sa.get(i), sb.get(i)) for i in diff[:3]]}" if diff else ""))
    flipped += len(diff)

if bad or flipped:
    print(f"\n  MISMATCH: {bad} value(s) differ by more than {g.tol:g}; "
          f"{flipped} selected-pose flip(s). The two passes are NOT scoring the same thing.")
    sys.exit(1)
print(f"\n  MATCH: every pose agrees within {g.tol:g} on {g.cols}, and BOTH selectors pick the "
      f"same pose on every interface. Splitting the pose dimension changes neither a score nor a "
      f"decision.")
