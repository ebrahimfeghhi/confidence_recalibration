#!/usr/bin/env python
"""Score the poses the hook TEMPLATED ON, so we can check we templated on the right thing.

The hook decodes at recycle 10 and feeds those poses back. Until now only the fold's FINAL output was
scored, which means the input to the intervention was never measured -- if the mid-loop decode were
producing garbage (wrong atom order, unit mismatch, a half-built structure) the run would still
complete and score, and we would be comparing a continuation against an unknown starting point.

THE VALIDATION. A decode after 10 recycles should closely resemble a plain 10-recycle fold of the same
seed, which is exactly what the pool contains (bf16, N_cycle=10, same seed numbers). So scoring the
mid-loop poses and comparing them against the pool's free poses for that seed tells us whether the
hook is handing back real structures. Large disagreement means the decode is wrong, not that
templating did something.

ipTM IS NOT AVAILABLE for these poses. The confidence head does not run at the mid-loop decode --
sample_diffusion returns coordinates only -- so there is no summary_confidence json and row.py emits
an empty ipTM field. DockQ needs only coordinates and ground truth, so it is real. The pool's ipTM at
the same seed is the closest proxy, and it comes from a different run.

    score_hook_poses.py --root <inmem/nN> --csv <out.csv> [--jobs 6]
"""
import argparse
import collections
import csv
import glob
import os
import statistics as st
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit")

import paths            # noqa: E402
from fmt import table   # noqa: E402

DC = str(paths.CODE_DIR)
PY = str(paths.PROTENIX_PYTHON)


def units(root, pool_ifaces):
    """(pose dir, pdb, seed, interface, arm) for every mid-loop decode found under root.

    THE ARM COMES FROM THE CYCLE. A fold decodes at several cycles now, and cycle_no is 0-based, so
    c9 is the state after 10 recycles -> "at10", c10 -> "at11", c11 -> "at12". Labelling every decode
    "at10" (which an earlier version of this file did) collides on the csv's (iface, seed, arm) key:
    the first cycle written wins and the rest are silently dropped as duplicates, so three recycle
    counts collapse into one column that claims to be the baseline.
    """
    out = []
    for pd in sorted(glob.glob(f"{root}/free/*/*/*/_hook/c*/poses")):
        parts = pd.split("/")
        pdb, seed, cyc = parts[-6], parts[-5], parts[-2]
        if not glob.glob(f"{pd}/*_sample_*.cif"):
            continue
        try:
            arm = f"at{int(cyc.lstrip('c')) + 1}"
        except ValueError:
            continue
        for iface in sorted(pool_ifaces.get(pdb, ())):
            out.append((pd, pdb, seed, iface, arm))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--arm", default="at10", help="(unused: the arm is derived per cycle)")
    A = ap.parse_args()

    pool_ifaces = collections.defaultdict(set)
    for r in csv.DictReader(open(str(paths.PXM2025_POOL_CSV))):
        pool_ifaces[r["pdb"]].add(f"{r['pdb']}_{r['chain1']}{r['chain2']}")

    todo = units(A.root, pool_ifaces)
    done = set()
    if os.path.exists(A.csv):
        done = {(r["iface"], r["seed"], r["arm"]) for r in csv.DictReader(open(A.csv))}
    else:
        with open(A.csv, "w") as fh:
            fh.write("iface,seed,arm,donor_pose,sample,dockq,iptm\n")
    todo = [t for t in todo if (t[3], t[2], t[4]) not in done]
    print(f"  mid-loop pose sets to score: {len(todo)}")

    _, reff, scorer = paths.dataset("pxm2025")

    def one(t):
        pd, pdb, seed, iface, arm = t
        r = subprocess.run([PY, f"{DC}/row.py", "--dir", pd, "--iface", iface, "--seed", seed,
                            "--arm", arm, "--donor", "-1", "--ref", reff(pdb),
                            "--scorer", scorer], cwd=DC, capture_output=True, text=True)
        return r.stdout

    with ThreadPoolExecutor(max_workers=A.jobs) as ex:
        for out in ex.map(one, todo):
            if out.strip():
                with open(A.csv, "a") as fh:
                    fh.write(out)

    # --- the validation: mid-loop decode vs the pool's own 10-recycle fold of that seed -----------
    mid = collections.defaultdict(list)
    for r in csv.DictReader(open(A.csv)):
        if r["arm"] == "at10":
            mid[(r["iface"], r["seed"])].append(float(r["dockq"]))
    pool = collections.defaultdict(list)
    for r in csv.DictReader(open(str(paths.PXM2025_POOL_CSV))):
        pool[(f"{r['pdb']}_{r['chain1']}{r['chain2']}", r["seed"])].append(float(r["dockq"]))
    rows = []
    for k in sorted(mid):
        if k not in pool:
            continue
        m, p = sorted(mid[k], reverse=True), sorted(pool[k], reverse=True)
        rows.append([k[0], k[1], len(m), st.mean(m), st.mean(p), abs(st.mean(m) - st.mean(p)),
                     max(m), max(p)])
    if rows:
        print("\n  MID-LOOP DECODE vs THE POOL'S PLAIN 10-RECYCLE FOLD (same seed, both bf16)")
        print(table(["interface", "seed", "poses", "mid mean", "pool mean", "|diff|",
                     "mid best", "pool best"], rows[:20], align="l l r r r r r r",
                    dec={"mid mean": 3, "pool mean": 3, "|diff|": 3, "mid best": 3,
                         "pool best": 3}))
        d = [r[5] for r in rows]
        print(f"\n  n={len(rows)}  mean |diff| of the pose-set mean: {st.mean(d):.3f}  "
              f"max {max(d):.3f}")
        print("  Large values would mean the mid-loop decode is NOT producing what a 10-recycle "
              "fold produces, i.e. we are templating on something unintended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
