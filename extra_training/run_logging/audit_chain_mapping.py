#!/usr/bin/env python
"""Audit: does the STATIC chain_roles mapping agree with the PER-POSE OST mapping?

WHY. The true-iTM labels in <fold>/itm_labels were built through
phase_k_eval.interface_chain_pairs, which reads chain_roles.csv -- one mapping per structure.
phase_k_eval itself does NOT use that at scoring time: it calls ost_chain_pairs per pose, because
chain_roles has been measured wrong for chemically identical chains. itm_ceiling.py says so in its
own header and accepted the risk, because a ceiling ESTIMATE tolerates ranking noise.

Those labels are now training supervision, where a swapped pair does not add noise -- it trains
the model toward another interface's target. 298 of 368 structures have more than two pooled
chains, which is where a swap is possible at all. This measures how often it actually happens,
per pose, on the TRAINING fold, so the decision to train on them or to exclude interfaces is made
from a number.

WHAT IT COMPARES, per (structure, seed, pose): the set of (iface, asym_i, asym_j) triples from
chain_roles against the set OST wrote for that pose. Three outcomes are counted separately:
  AGREE        -- identical sets
  DISAGREE     -- both resolved and they differ: the label would be attached to the wrong pair
  UNRESOLVABLE -- OST has no mapping for that pose, so there is nothing to compare

CPU only. Reads ost_json/<k>.json and chain_roles.csv; no model, no GPU.

Usage:  python -I run_logging/audit_chain_mapping.py [--workers 9] [--limit 0]
"""
import argparse
import collections
import csv
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import provenance as PROV                                             # noqa: E402
from phase_k_eval import (interface_chain_pairs, ost_chain_pairs,     # noqa: E402
                          pool_ref_pairs)
import phase_j_build_cache as BC                                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=9)
ap.add_argument("--shard", type=int, default=-1)
ap.add_argument("--pool", default="training/full_train/pool_winnable.csv")
ap.add_argument("--roles", default="training/fold/chain_roles.csv")
ap.add_argument("--fold", default="training/fold")
ap.add_argument("--seeds", default="500000-500009")
ap.add_argument("--nsample", type=int, default=5)
ap.add_argument("--out", default="training/runs/chain_mapping_audit")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--report-only", action="store_true")
a = ap.parse_args()
OUT = os.path.join(ROOT, a.out)
LO, HI = (int(x) for x in a.seeds.split("-"))


def run_shard(k):
    static, _unres = interface_chain_pairs(os.path.join(ROOT, a.pool),
                                           os.path.join(ROOT, a.roles))
    refs = pool_ref_pairs(os.path.join(ROOT, a.pool))
    pdbs = sorted(static)
    units = [(p, s) for p in pdbs for s in range(LO, HI + 1)]
    mine = units[k::a.workers]
    if a.limit:
        mine = mine[:a.limit]
    rows, t0 = [], time.time()
    for n, (pdb, sd) in enumerate(mine):
        # seed_dir, NOT resolve_seed_dir. The latter disambiguates a refolded seed by comparing
        # pose COORDINATES, which would mean loading a trunk per unit for an audit that only needs
        # to read ost_json. Where a refold left two finished directories seed_dir raises, and that
        # unit is recorded as unresolvable rather than guessed at.
        try:
            ud = BC.seed_dir(os.path.join(ROOT, a.fold), pdb, sd)
        except Exception as _e:                             # noqa: BLE001
            rows.append((pdb, sd, -1, "no_unit_dir", type(_e).__name__, ""))
            continue
        if not ud:
            rows.append((pdb, sd, -1, "no_unit_dir", "", ""))
            continue
        exp = sorted(static[pdb])
        for i in range(a.nsample):
            got, why = ost_chain_pairs(ud, pdb, i, refs.get(pdb) or [])
            if got is None:
                rows.append((pdb, sd, i, "unresolvable", why[:60], ""))
                continue
            if sorted(got) == exp:
                rows.append((pdb, sd, i, "agree", "", ""))
            else:
                rows.append((pdb, sd, i, "disagree",
                             ";".join(f"{a_}:{b_},{c_}" for a_, b_, c_ in exp),
                             ";".join(f"{a_}:{b_},{c_}" for a_, b_, c_ in sorted(got))))
        if n and n % 50 == 0:
            el = time.time() - t0
            print(f"  shard {k}: {n}/{len(mine)} units, {el:.0f}s "
                  f"(eta {el/n*(len(mine)-n)/60:.1f} min)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    f = os.path.join(OUT, f"audit_s{k}.csv")
    with open(f + ".tmp", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "seed", "sample", "verdict", "static", "ost"])
        w.writerows(rows)
    os.replace(f + ".tmp", f)
    print(f"  shard {k}: {len(rows)} row(s) -> {os.path.relpath(f, ROOT)}", flush=True)
    return 0


def report():
    import glob
    rows = []
    for f in sorted(glob.glob(os.path.join(OUT, "audit_s*.csv"))):
        rows += list(csv.DictReader(open(f)))
    if not rows:
        sys.exit(f"  no audit_s*.csv under {a.out}")
    v = collections.Counter(r["verdict"] for r in rows)
    n = len(rows)
    print(f"\n  CHAIN-MAPPING AUDIT -- {n} (structure, seed, pose) check(s)")
    for k_, c in v.most_common():
        print(f"    {k_:>14}: {c:>6}  ({100*c/n:.2f}%)")
    bad = [r for r in rows if r["verdict"] == "disagree"]
    bad_pdbs = collections.Counter(r["pdb"] for r in bad)
    print(f"\n  structures with ANY disagreement: {len(bad_pdbs)}")
    if bad_pdbs:
        print(f"    worst: {bad_pdbs.most_common(8)}")
        ex = os.path.join(OUT, "exclude_pdbs.txt")
        with open(ex, "w") as fh:
            fh.write("\n".join(sorted(bad_pdbs)) + "\n")
        print(f"    -> {os.path.relpath(ex, ROOT)} ({len(bad_pdbs)} structure(s)) for "
              f"--manifest-skip")
    unres = collections.Counter(r["pdb"] for r in rows if r["verdict"] == "unresolvable")
    if unres:
        print(f"  structures where OST could not be read at all: {len(unres)} "
              f"(these are NOT evidence of agreement)")
    print(f"\n  VERDICT: {'labels are safe to train on as built' if not bad_pdbs else 'EXCLUDE the listed structures or rebuild the labels through the per-pose OST mapping'}")


if __name__ == "__main__":
    if a.report_only:
        report()
    elif a.shard >= 0:
        sys.exit(run_shard(a.shard))
    else:
        t0 = time.time()
        ps = [subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__),
                                "--shard", str(k), "--workers", str(a.workers),
                                "--pool", a.pool, "--roles", a.roles, "--fold", a.fold,
                                "--seeds", a.seeds, "--out", a.out]
                               + (["--limit", str(a.limit)] if a.limit else []))
              for k in range(a.workers)]
        rc = [p.wait() for p in ps]
        print(f"\n  {a.workers} worker(s) in {(time.time()-t0)/60:.1f} min, rc={rc}")
        if any(rc):
            sys.exit(f"STOP: shard(s) failed with {rc}")
        PROV.write(OUT, ROOT, inputs=[a.pool, a.roles], outputs_glob="audit_s*.csv",
                   extra={"seeds": a.seeds, "nsample": a.nsample})
        report()
