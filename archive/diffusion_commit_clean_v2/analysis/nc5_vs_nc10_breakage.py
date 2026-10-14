#!/usr/bin/env python
"""Does 10 recycles lower the solved-interface BREAKAGE RATE, or did it only fix the 9 it was aimed at?

WHY THIS EXISTS. The aggregate table counts broken solved interfaces but not WHICH ones, so a
carried-over breakage reads identically to a new one. Claude misreported phase 2's first 3 breakages
as new when all 3 were the phase-1 interfaces that had already failed to recover. The cell that
actually answers the question is `ok -> BROKE`: solved interfaces the nc5 arm KEPT and the nc10 arm
LOST. Everything else is either unchanged or the repair nc10 was built for.

THE SELECTION TRAP THIS GUARDS AGAINST. The 9 phase-1 interfaces were chosen BECAUSE nc5 broke them,
so they can only improve and must never be used to estimate nc10's breakage rate. They are reported
separately and excluded from the "fresh" rate.

POWER. nc5 broke 9/75 solved interfaces (12%). At that rate a fresh sample of n needs to reach
roughly 40 before 0 breakages is meaningfully different from 12% -- below that, "0 of 20" is what
both arms look like regardless. The script prints the fresh n so the claim is never made too early.

    ./nc5_vs_nc10_breakage.py
"""
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/home/ebrahim/tts_abag")

import compare_arms as C  # noqa: E402
from score_experiment import ACCEPTABLE, classes_at, load_scored, pool_by_seed  # noqa: E402

FB = "/home/ebrahim/tts_abag/experimental_data/protenix/foldbench"
# Selected because the nc5 arm broke them; phase 1 re-folded exactly these at 10 recycles.
PHASE1 = {"7yxu_AB", "7yxu_AC", "8fdd_AC", "8fdd_BC", "8jyr_AB",
          "8jyr_AC", "8urf_AC", "8ux6_DF", "8zp2_AB"}
FRESH_N_FOR_POWER = 40


def acceptable(arm, iface, unfinished):
    """out: True/False if the interface is scored and complete in this arm, else None."""
    if iface not in arm or iface in unfinished:
        return None
    poses = [(x[0], x[1]) for x in arm[iface]]
    return C.top(poses) >= ACCEPTABLE if poses else None


def main():
    ds, _, _ = C.arm_defaults("foldbench")
    pool, first = pool_by_seed(50, str(ds.pool_csv))
    cls = classes_at(pool, first)
    t5, t10 = C.tmpl_by_donor(f"{FB}/selftmpl_mask_on"), C.tmpl_by_donor(f"{FB}/selftmpl_mask_on_nc10")
    _, u5 = load_scored(f"{FB}/selftmpl_mask_on")
    _, u10 = load_scored(f"{FB}/selftmpl_mask_on_nc10")

    rows = []
    for iface in sorted(i for i, c in cls.items() if c == "solved"):
        a, b = acceptable(t5, iface, u5), acceptable(t10, iface, u10)
        if a is None or b is None:
            continue
        rows.append((iface, a, b))
    if not rows:
        raise SystemExit("no solved interface is scored in both arms yet")

    tt = collections.Counter((a, b) for _, a, b in rows)
    label = {(True, True): "ok -> ok", (True, False): "ok -> BROKE",
             (False, True): "broken -> fixed", (False, False): "broken -> broken"}
    print(f"solved interfaces scored in BOTH arms: {len(rows)}\n")
    print("| nc5 -> nc10 | n |")
    print("|:---|---:|")
    for key in [(True, True), (False, True), (False, False), (True, False)]:
        print(f"| {label[key]} | {tt.get(key, 0)} |")

    new_breaks = [i for i, a, b in rows if a and not b]
    print(f"\n  NEW breakages (nc5 kept, nc10 lost): {len(new_breaks)}"
          + ("" if not new_breaks else "  -- " + ", ".join(new_breaks)))
    carried = [i for i, a, b in rows if not a and not b]
    print(f"  carried over (broken in both): {len(carried)}"
          + ("" if not carried else "  -- " + ", ".join(carried)))

    fresh = [(i, a, b) for i, a, b in rows if i not in PHASE1]
    f5 = sum(1 for _, a, _ in fresh if not a)
    f10 = sum(1 for _, _, b in fresh if not b)
    print(f"\n  EXCLUDING the {len(rows) - len(fresh)} phase-1 interfaces (selected because nc5 broke"
          f" them), fresh n={len(fresh)}:")
    print(f"    broken at nc5 : {f5}/{len(fresh)}")
    print(f"    broken at nc10: {f10}/{len(fresh)}")
    if len(fresh) < FRESH_N_FOR_POWER:
        print(f"    *** fresh n={len(fresh)} < {FRESH_N_FOR_POWER}: nc5's own rate was 9/75 (12%), which"
              f" predicts ~{0.12 * len(fresh):.1f} breakages here.")
        print("    *** Too few to separate the arms. Do NOT claim nc10 lowers the breakage rate yet.")
    else:
        print(f"    fresh n={len(fresh)} is enough to read: nc5 {100 * f5 / len(fresh):.1f}% vs "
              f"nc10 {100 * f10 / len(fresh):.1f}%")


if __name__ == "__main__":
    main()
