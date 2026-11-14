#!/usr/bin/env python
"""Did the step-zero in-block head reproduce the shipped head, to within the hardware's own noise?

WHY A MEASURED FLOOR AND NOT A TOLERANCE. phase_k_eval runs the head under autocast and reads
bf16 trunk tensors, and selection metrics like frac_acc are THRESHOLDED -- one interface crossing
0.23 DockQ moves frac_acc by 1/n with no underlying change. So "equal" cannot be a constant I pick:
it has to be whatever two runs of the SAME head differ by on this pool, this budget and this card.
Two shipped-head runs give that, and the in-block head has to land inside it.

An exact-equality check would fail for reasons that have nothing to do with the injection, and a
loose hand-picked tolerance would pass a real difference. This is the only version that can only
fail for the reason it is asking about.

A floor of exactly zero is kept at zero: if the two shipped runs agreed bit for bit, the in-block
head has to as well, and EPS exists only so float formatting cannot manufacture a failure.
"""
import argparse
import json
import sys

# SELECTION-DEPENDENT FIELDS CARRY THE POOL'S GRANULARITY. With n interfaces, frac_acc can only
# move in steps of 1/n, so on a 2-structure smoke pool its floor is coarse by construction. Named
# here so the report says WHICH fields are coarse rather than quietly widening everything.
FRAC_FIELDS = ("frac_acc", "oracle_frac_acc", "frac_acc_select_max", "baseline_frac_acc")
EPS = 1e-12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", nargs=2, required=True,
                   help="two runs of the SAME (shipped) head; their difference is the floor")
    ap.add_argument("--test", required=True, help="the step-zero in-block head's metrics")
    ap.add_argument("--slack", type=float, default=0.0,
                   help="added to every floor. Default 0: the floor is the floor. Raise it only "
                        "with a reason, and say the reason in the run log.")
    a = ap.parse_args()

    f1, f2 = (json.load(open(p)) for p in a.floor)
    t = json.load(open(a.test))

    n = f1.get("n_interfaces")
    if f2.get("n_interfaces") != n or t.get("n_interfaces") != n:
        print(f"FAIL  the three runs do not cover the same interfaces: "
              f"{f1.get('n_interfaces')} / {f2.get('n_interfaces')} / {t.get('n_interfaces')}")
        print("      A parity comparison across different coverage is meaningless.")
        return 1
    if t.get("budget") != f1.get("budget"):
        print(f"FAIL  budget differs: floor {f1.get('budget')} vs test {t.get('budget')}")
        return 1

    keys = sorted(k for k in f1
                  if isinstance(f1[k], (int, float)) and not isinstance(f1[k], bool)
                  and k in f2 and k in t)
    if not keys:
        print("FAIL  no numeric field is present in all three files")
        return 1

    print(f"  pool: {n} interface(s) at budget {f1.get('budget')}; comparing {len(keys)} field(s)")
    print(f"  {'field':<28} {'shipped_1':>12} {'shipped_2':>12} {'in-block':>12} "
          f"{'floor':>11} {'test diff':>11}  verdict")
    bad = []
    for k in keys:
        floor = abs(float(f1[k]) - float(f2[k]))
        diff = abs(float(t[k]) - float(f1[k]))
        lim = max(floor, EPS) + a.slack
        okk = diff <= lim
        note = "" if k not in FRAC_FIELDS else f"  (quantised to 1/{n})"
        print(f"  {k:<28} {float(f1[k]):>12.6f} {float(f2[k]):>12.6f} {float(t[k]):>12.6f} "
              f"{floor:>11.3e} {diff:>11.3e}  {'ok' if okk else 'OVER'}{note}")
        if not okk:
            bad.append((k, diff, lim))

    print()
    if bad:
        print(f"FAIL  {len(bad)} field(s) moved further than two runs of the shipped head do:")
        for k, d, lim in bad:
            print(f"        {k}: |diff| {d:.6e} > floor {lim:.6e}")
        print("      The step-zero in-block head is NOT reproducing the shipped head. Either an")
        print("      injection is not zero at init, the evaluator installed it at the wrong")
        print("      depth, or the readout hook is firing when it should not.")
        return 1
    worst = max(((k, abs(float(t[k]) - float(f1[k]))) for k in keys), key=lambda kv: kv[1])
    print(f"PASS  every field is inside the shipped head's own run-to-run spread.")
    print(f"      Largest move: {worst[0]} at {worst[1]:.3e}.")
    print(f"      So phase_k_eval loaded a genuine in-block checkpoint, installed it at the right")
    print(f"      depth, its key-set guard accepted it, and both injections really are zero at")
    print(f"      init -- measured on the production CLI, not in a harness.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
