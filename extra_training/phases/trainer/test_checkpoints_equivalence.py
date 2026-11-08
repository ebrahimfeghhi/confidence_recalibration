"""compare_guard() must accept the resumes the ORIGINAL guard accepted.

GROUND TRUTH IS THE BUNDLES THEMSELVES. Every resume_ep<N>.pt on disk was written by a run that
started, and most were then consumed by a run that CONTINUED from them -- the original guard let
them through. So comparing each bundle's recorded guard against the config of the run that
consumed it must yield no drift. A bundle this code reports as a fork, which the original
continued from, is a regression.

The negative direction is covered by mutation: flip one recorded field and the comparison must
report exactly that field, so a function that returned {} unconditionally would fail.
"""
import glob
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainer.checkpoints import _PRE_CROP, compare_guard  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RUNS = os.path.join(ROOT, "training", "runs")
FAIL = []


def ok(cond, name, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))
    if not cond:
        FAIL.append(name)


def main():
    bundles = sorted(glob.glob(os.path.join(RUNS, "*", "resume_ep*.pt")))
    print(f"\n1  EVERY REAL BUNDLE STILL READS AS A CONTINUATION ({len(bundles)} found)")
    if not bundles:
        print("STOP: no resume bundles on disk; nothing would be measured.")
        return 1
    checked = forks = 0
    for b in bundles[-12:]:                       # the most recent dozen is plenty
        rid = os.path.basename(os.path.dirname(b))
        aj = os.path.join(os.path.dirname(b), "args.json")
        if not os.path.exists(aj):
            continue
        try:
            g = (torch.load(b, map_location="cpu", weights_only=False) or {}).get("guard", {})
        except Exception as e:                    # noqa: BLE001
            ok(False, f"{rid}/{os.path.basename(b)} loads", f"{type(e).__name__}")
            continue
        if not g:
            continue
        side = json.load(open(aj))
        # "mine" is that run's own recorded config, i.e. exactly what it resumed with
        mine = {k: side[k] for k in g if k in side}
        bad, from_args, unknown, migrated = compare_guard(mine, g, side)
        checked += 1
        if bad:
            forks += 1
            ok(False, f"{rid}/{os.path.basename(b)}: continuation", f"{len(bad)} drift: {list(bad)[:3]}")
    ok(checked > 0, "bundles with a guard were actually compared", f"{checked} checked")
    ok(forks == 0, "none of them reports drift against its own run's config",
       f"{checked} continuations" if not forks else f"{forks} spurious fork(s)")

    print("\n2  AND IT IS NOT ACCEPTING EVERYTHING (mutation)")
    b = bundles[-1]
    g = (torch.load(b, map_location="cpu", weights_only=False) or {}).get("guard", {})
    side = json.load(open(os.path.join(os.path.dirname(b), "args.json")))
    mine = {k: side[k] for k in g if k in side}
    ok(bool(mine), "the last bundle yields comparable fields", f"{len(mine)} fields")
    k0 = next(iter(mine))
    tampered = dict(mine)
    tampered[k0] = "TAMPERED" if not isinstance(mine[k0], (int, float)) else mine[k0] + 1
    bad, *_ = compare_guard(tampered, g, side)
    ok(list(bad) == [k0], f"changing {k0} is reported, and only {k0}", f"{list(bad)}")

    print("\n3  THE THREE SOURCES BEHAVE AS DOCUMENTED")
    ok(compare_guard({"seed": 0}, {"seed": 0}, {}) == ({}, [], [], []),
       "a field in the bundle and equal -> no finding")
    ok(compare_guard({"seed": 0}, {}, {"seed": 0})[1] == ["seed"],
       "absent from the bundle but in args.json -> matched there, reported")
    ok(compare_guard({"crop_tokens": ""}, {}, {})[3] == ["crop_tokens"],
       "a field postdating the bundle -> migrated to the pre-crop default")
    ok(compare_guard({"zzz": 1}, {}, {})[2] == ["zzz"],
       "recorded nowhere -> reported as NOT compared, never silently passed")
    ok(compare_guard({"crop_tokens": "/x.npz"}, {}, {})[0] != {},
       "and enabling a post-dating feature IS drift, as the comment requires")
    ok(len(_PRE_CROP) >= 40, "the migration table is intact", f"{len(_PRE_CROP)} fields")

    print()
    if FAIL:
        print(f"  {len(FAIL)} FAILURE(S): {FAIL}")
        return 1
    print("  all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
