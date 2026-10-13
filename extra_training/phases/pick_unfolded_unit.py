#!/usr/bin/env python
"""Print the first (pdb, seed) in the expected range that has no run.json, for the smoke test.

Exits non-zero with a message when every expected unit is already folded -- in which case there is
nothing to smoke and the caller must not write SMOKE_OK, because assertions that count artifacts
across the whole fold directory would pass on data an earlier run or a migration produced.
"""
import argparse
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--seeds", type=int, required=True)
    ap.add_argument("--pdbs", nargs="+", required=True)
    a = ap.parse_args()
    for pdb in a.pdbs:
        done = set()
        d = os.path.join(a.fold, "poses", pdb)
        if os.path.isdir(d):
            for sub in os.listdir(d):
                if os.path.exists(os.path.join(d, sub, "run.json")):
                    done.add(sub.split("-")[0])
        for i in range(a.seeds):
            if str(a.seed0 + i) not in done:
                print(f"{pdb} {a.seed0 + i}")
                return 0
    print(f"every expected unit of {len(a.pdbs)} structure(s) x {a.seeds} seeds from {a.seed0} is "
          f"already folded; there is nothing for a smoke test to prove", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
