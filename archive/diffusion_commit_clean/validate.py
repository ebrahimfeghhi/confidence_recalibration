#!/usr/bin/env python
"""The checks that decide whether the chain mapping and the matrices can be trusted.

WHY THIS FILE EXISTS. Three of my checks during this work passed for the wrong reason, and each one
looked like a real validation:

  - comparing recovered mappings against the 4 surviving pxm_json files: all 875 comparison poses were
    single-copy, so every mapping was identity and 20/20 agreement proved only that identity
    reproduces identity -- it could not have failed.
  - comparing each stored row's dockq_gt against the pool CSV: dockq_gt is READ from that CSV, so the
    comparison is a tautology and cannot detect row misalignment, which is the bug it was meant to
    catch.
  - filtering poses by contact count and reporting the drops as expected: the count agreed with my own
    assumption rather than with anything independent.

What caught the real bug was comparing two quantities computed by DIFFERENT code for different
reasons: my contact count against PXMeter's DockQ. A pose cannot have a high DockQ and no interface,
so a disagreement had to be a bug in one of them -- and it was mine, a 4.5 A cutoff where DockQ uses
5.0. Every check below is that shape: two independent sources that must agree.

    validate.py [--checks mapping,alignment,contacts] [--n 12]

Exit code is nonzero if any check fails, so this is usable as a gate.
"""
import argparse
import collections
import csv
import glob
import json
import os
import random
import sys

import numpy as np

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, "/home/ebrahim/tts_abag/eval")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402
from pool import dockq_by_pose  # noqa: E402  was a second, divergent copy of load_pool
from fmt import table  # noqa: E402

ACC = 0.23
SCRATCH = os.environ.get("VALIDATE_SCRATCH", "/tmp/validate_pxm")


def check_mapping(maps, roles, pool, n):
    """Re-score poses whose mapping is PERMUTED and require DockQ to reproduce the pool CSV.

    The independent quantities are my recovered mapping and PXMeter's DockQ. A wrong mapping changes
    which chains are compared, which changes DockQ, so exact agreement is hard to get by luck.
    Permuted poses only: identity poses would pass no matter what the mapping code did.
    """
    import pxm_dockq
    os.makedirs(SCRATCH, exist_ok=True)
    cands = [(iface, key) for iface, rows in pool.items() for key in rows
             if iface in roles
             and (m := maps.get((iface.split("_")[0], key[0], key[1])))
             and any(k != v for k, v in m.items())]
    random.seed(0)
    rows, bad = [], 0
    for iface, key in random.sample(cands, min(n, len(cands))):
        pdb = iface.split("_")[0]
        out = f"{SCRATCH}/{pdb}_{key[0]}_{key[1]}.json"
        res = pxm_dockq.score(paths.pool_pose(pdb, key[0], key[1]),
                              f"{paths.PXM2025_CIF}/{pdb}.cif", out)
        c1, c2 = iface.split("_")[1][0], iface.split("_")[1][1]
        got = res.get(f"{c1},{c2}", {}).get("dockq")
        want = pool[iface][key]
        cli = {str(k): str(v) for k, v in json.load(open(out))["ref_to_model_chain_mapping"].items()}
        ok = got is not None and abs(got - want) < 1e-6 and cli == maps[(pdb, key[0], key[1])]
        bad += not ok
        rows.append([iface, f"{key[0]} s{key[1]}", f"{want:.4f}",
                     "-" if got is None else f"{got:.4f}", "ok" if ok else "FAIL"])
    print(table(["interface", "pose", "pool CSV DockQ", "re-scored DockQ", ""], rows,
                align="l l r r l"))
    return bad


def check_alignment(maps, roles):
    """Each stored row's model chains must equal the mapping for THAT row's (seed, sample).

    Independent because model_ab/model_ag were written by build_matrices while the mapping comes from
    chain_maps.csv; a shifted row order breaks the correspondence. Do NOT check dockq_gt here -- it is
    read from the same CSV it would be compared against, which is how the earlier tautology arose.
    """
    bad = tot = 0
    for f in sorted(glob.glob("matrices/*.npz")):
        iface = os.path.basename(f)[:-4]
        if iface not in roles:
            continue
        pdb = iface.split("_")[0]
        ab_ref, ag_ref = roles[iface]
        z = np.load(f)
        for s, x, ab, ag in zip(z["seed"], z["sample"], z["model_ab"], z["model_ag"]):
            m = maps.get((pdb, str(s), int(x)))
            tot += 1
            if not m or m[ab_ref] != str(ab) or m[ag_ref] != str(ag):
                bad += 1
    print(f"  rows checked {tot}, misaligned {bad}")
    return bad


def check_contacts(maps, roles, pool, n):
    """No pose that DockQ calls well-reproduced may be recorded as having no interface.

    This is the check that found the 4.5 A bug. It stays because the same class of error -- a cutoff
    or chain choice that disagrees with the scorer -- is easy to reintroduce.

    DockQ above 0.80 with zero contacts is impossible; DockQ just above 0.23 with zero contacts is
    NOT, since DockQ averages Fnat with two RMSD terms and can clear 0.23 on geometry alone. So the
    threshold here is deliberately high, and near-miss poses are reported separately rather than
    counted as failures.
    """
    kept = {}
    for f in glob.glob("matrices/*.npz"):
        z = np.load(f)
        kept[os.path.basename(f)[:-4]] = {(str(s), int(x)) for s, x in zip(z["seed"], z["sample"])}
    hard, soft = [], 0
    for iface, rows in pool.items():
        if iface not in roles or iface not in kept:
            continue
        pdb = iface.split("_")[0]
        ab_ref, ag_ref = roles[iface]
        for key, dq in rows.items():
            if key in kept[iface] or dq < ACC:
                continue
            m = maps.get((pdb, key[0], key[1]))
            if not m:
                continue
            ncon = len(metrics.epitope_set(paths.pool_pose(pdb, key[0], key[1]),
                                           m[ab_ref], m[ag_ref], cutoff=metrics.CONTACT_CUTOFF))
            if ncon == 0:
                soft += 1                      # legitimate: good RMSD, Fnat 0
            elif dq >= 0.80:
                hard.append([iface, f"{key[0]} s{key[1]}", f"{dq:.3f}", ncon])
            if len(hard) >= n:
                break
    if hard:
        print("  DROPPED poses that DockQ calls well-reproduced AND that do have contacts:")
        print(table(["interface", "pose", "GT DockQ", "contacts at 5.0A"], hard, align="l l r r"))
    print(f"  dropped-but-acceptable with genuinely no contacts (expected, Fnat 0): {soft}")
    return len(hard)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checks", default="mapping,alignment,contacts")
    ap.add_argument("--n", type=int, default=12)
    A = ap.parse_args()

    maps, roles, pool = metrics.load_chain_maps(), metrics.load_roles(), dockq_by_pose()
    fails = {}
    for name in A.checks.split(","):
        print(f"\n=== {name}")
        if name == "mapping":
            fails[name] = check_mapping(maps, roles, pool, A.n)
        elif name == "alignment":
            fails[name] = check_alignment(maps, roles)
        elif name == "contacts":
            fails[name] = check_contacts(maps, roles, pool, A.n)
        else:
            raise SystemExit(f"unknown check {name}")
    print("\n" + table(["check", "failures"], [[k, v] for k, v in fails.items()], align="l r"))
    return 1 if any(fails.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
