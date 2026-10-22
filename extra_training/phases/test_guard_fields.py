#!/usr/bin/env python
"""The resume guard's three dicts must stay in step.

Run: python -I phases/test_guard_fields.py

WHY THIS EXISTS. phase_k_train keeps three parallel dicts:

  COMPARISON (`_mine`)  what this run's settings are, checked against the bundle on a resume
  DEFAULTS (`_fallback`) what to assume for a bundle written BEFORE a field existed
  SAVED                  what gets written into the bundle

THE INVARIANT THAT MATTERS is COMPARISON is a subset of SAVED. A field compared but not saved is
not guarded at all: the next resume finds it absent from the bundle, falls back to the default, and
a changed objective passes silently. That is exactly what happened to pose_perblock_share, l2_sp,
rank_loss, alpha_rank and rank_weight_by_gap -- all five were in the comparison and none in the
bundle, found by review of 04503a9 rather than by the code.

DEFAULTS IS DELIBERATELY NOT REQUIRED TO MATCH. A field that predates the migration mechanism has
no default and needs none, because every bundle ever written already carries it. What IS checked is
the other direction: a default for a field no longer in the comparison is dead, and dead defaults
are how a renamed field stops being guarded while looking guarded.
"""
import ast
import os
import sys

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase_k_train.py")
MARKER = "rank_loss"


def keys_of(node):
    return [k.value if isinstance(k, ast.Constant) and isinstance(k.value, str) else None
            for k in node.keys]


def main():
    tree = ast.parse(open(SRC).read())
    found = sorted((n.lineno, keys_of(n), n) for n in ast.walk(tree)
                   if isinstance(n, ast.Dict) and MARKER in keys_of(n))
    print(f"{SRC}")
    print(f"  dicts carrying {MARKER!r}: lines {[ln for ln, _, _ in found]}")
    if len(found) != 3:
        print(f"  FAIL  expected exactly 3 (comparison, defaults, saved), got {len(found)}. "
              f"If one was added deliberately, update this test; if one vanished, a guard field "
              f"stopped round-tripping.")
        return 1
    if any(None in ks for _, ks, _ in found):
        print("  FAIL  a non-literal key makes the guard uncheckable statically.")
        return 1

    # DEFAULTS IS THE ALL-CONSTANT ONE. The comparison and saved dicts both read a.<flag>, so
    # this identifies the three by structure rather than by line number.
    def literal(v):
        # A NEGATIVE NUMBER IS A UnaryOp, not a Constant -- "rank_tau": -1.0 parses as USub over
        # 1.0, which made the first version of this detector find zero candidates.
        return (isinstance(v, ast.Constant)
                or (isinstance(v, ast.UnaryOp) and isinstance(v.operand, ast.Constant)))

    const_only = [i for i, (_, _, n) in enumerate(found)
                  if all(literal(v) for v in n.values)]
    if len(const_only) != 1:
        print(f"  FAIL  could not identify the defaults dict by structure "
              f"({len(const_only)} all-constant candidates).")
        return 1
    di = const_only[0]
    others = [i for i in range(3) if i != di]
    # Of the remaining two, the COMPARISON is the one in the same function as the defaults, so it
    # is the nearer in line number.
    ci = min(others, key=lambda i: abs(found[i][0] - found[di][0]))
    si = [i for i in others if i != ci][0]
    comp, dflt, saved = set(found[ci][1]), set(found[di][1]), set(found[si][1])
    print(f"  comparison line {found[ci][0]} ({len(comp)} field(s))")
    print(f"  defaults   line {found[di][0]} ({len(dflt)} field(s))")
    print(f"  saved      line {found[si][0]} ({len(saved)} field(s))")

    rc = 0
    # THE OBJECTIVE AND ARCHITECTURE FIELDS. Provenance keys legitimately differ between the
    # dicts; these are the ones whose silent drift changes what the run IS.
    watched = {k for k in comp | dflt | saved
               if k.startswith(("rank_", "alpha_", "pose_perblock", "lora", "l2_sp",
                                "train_only", "frame_direction", "pae_"))}
    miss_saved = sorted(k for k in watched & comp if k not in saved)
    if miss_saved:
        rc = 1
        print(f"  FAIL  compared but NOT SAVED (unguarded on the next resume): {miss_saved}")
    dead = sorted(k for k in watched & dflt if k not in comp)
    if dead:
        rc = 1
        print(f"  FAIL  a default exists for field(s) absent from the comparison: {dead}")
    if rc == 0:
        no_default = sorted(k for k in watched & comp if k not in dflt)
        print(f"  PASS  all {len(watched & comp)} watched field(s) are saved, and no default is dead.")
        if no_default:
            print(f"        (no migration default, which is correct for fields that predate the "
                  f"mechanism: {no_default})")
    return rc


if __name__ == "__main__":
    sys.exit(main())
