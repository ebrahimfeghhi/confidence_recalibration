#!/usr/bin/env python
"""Release claims whose owning worker is dead and whose unit was never fully scored.

    ./reconcile_stale_claims.py <experiment-dir> [--dataset foldbench]

Prints one line: how many were released and, if any, the (pdb, tag, seed) of each, so a watcher can
log it and a human can see which folds went missing.

WHY THIS EXISTS AS A CHECK. A worker takes a claim before folding and releases it only on failure, so
a worker killed mid-fold leaves one behind -- and a claimed-but-unscored unit is skipped by every
worker forever, silently shrinking the sweep. `folder.py` reconciles at startup, but that is not
enough in practice: on 2026-10-06 the reconcile ran while the just-killed processes were still
shutting down, `claim_is_live` saw their pids alive and spared three claims on 8fbw, and that complex
could never reach 22/22 -- so the decoy row of the comparison sat unchanged for over an hour while the
GPUs looked perfectly healthy. An identical strand cost 4 folds in the 2026-09-25 arm.

SAFE TO RUN AGAINST A LIVE SWEEP: a claim whose owner process is still alive is never touched, and a
claim from another host cannot be checked so it is left alone.

RELEASING IS NOT THE WHOLE FIX. `folder.py` reads its queue once at startup, so a worker that has
already walked past the released unit will not come back for it. The released fold is picked up at the
next worker (re)start -- which the watcher does on its own when a session dies. That is why this is
reported as an ESCALATE and not treated as resolved: if the count keeps climbing, a worker needs a
restart.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/ebrahim/tts_abag")

import folder  # noqa: E402
import paths   # noqa: E402


def main():
    if len(sys.argv) < 2:
        raise SystemExit("usage: reconcile_stale_claims.py <experiment-dir> [--dataset NAME]")
    exp = sys.argv[1].rstrip("/")
    dataset = "foldbench"
    if "--dataset" in sys.argv:
        dataset = sys.argv[sys.argv.index("--dataset") + 1]
    cdir, csvp = f"{exp}/claims", f"{exp}/results.csv"
    if not os.path.isdir(cdir) or not os.path.exists(csvp):
        print("stale claims: 0 (no claims dir or no results.csv)")
        return
    before = set(os.listdir(cdir))
    ifaces = folder.interfaces_by_pdb(str(paths.DATASETS[dataset].pool_csv))
    folder.reconcile_claims(cdir, csvp, ifaces)
    freed = sorted(before - set(os.listdir(cdir)))
    if freed:
        print(f"stale claims RELEASED: {len(freed)} -- " + ", ".join(freed[:12])
              + (" ..." if len(freed) > 12 else ""))
    else:
        print("stale claims: 0")


if __name__ == "__main__":
    main()
