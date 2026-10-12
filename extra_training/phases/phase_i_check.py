#!/usr/bin/env python
"""Phase I's verification and score recovery. Called by phase_i_fold_validation.sh.

WHY THIS IS PYTHON AND NOT SHELL. The shell version of both routines was wrong in ways shell makes
easy: it read pool.csv with `IFS=, read`, which keeps the \\r of a CRLF file in the last field and
produced interfaces like "6mnq_AB\\r" that scorer.py rejects outright; and its verification let a
crashed subprocess fall back to zero failures. Both needed a real CSV reader and a real exit status.

    verify   exact coverage of expected (pdb, seed, interface, sample), finite scores, readable
             caches. Exit 1 on any shortfall.
    rescore  score only what is missing, with the dataset's own scorer, without duplicating rows.

COMPLETION MEANS SCORED AND CACHED, NOT FOLDED. folder.py writes run.json before scoring, so counting
run.json files announces success over units that have no interface scores. And a run.json is only
evidence for the unit it names: globbing poses/<pdb>/<seed>-*/run.json without reading it let an
unrelated run compensate for a missing expected one.
"""
import argparse
import collections
import csv
import glob
import json
import math
import os
import subprocess
import sys


def read_pool(path):
    """in: pool.csv.  out: {pdb: {iface}} -- via the csv module, so CRLF cannot leak into a name"""
    out = collections.defaultdict(set)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            pdb = r["pdb"].strip()
            out[pdb].add(f"{pdb}_{r['chain1'].strip()}{r['chain2'].strip()}")
    return out


def read_results(path):
    """in: results.csv.  out: {(iface, seed): {sample: (dockq, iptm)}}, and the header's field names"""
    have = collections.defaultdict(dict)
    if not os.path.exists(path):
        return have
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                have[(r["iface"].strip(), r["seed"].strip())][r["sample"].strip()] = (
                    r.get("dockq", ""), r.get("iptm", ""))
            except KeyError:
                continue
    return have


def finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def expected_units(jobs, suffix, seeds, seed0):
    pdbs = sorted(os.path.basename(f)[:-len(suffix)] for f in glob.glob(os.path.join(jobs, "*" + suffix)))
    return pdbs, [(p, str(seed0 + i)) for p in pdbs for i in range(seeds)]


def run_dir_for(E, pdb, seed):
    """out: the run directory whose run.json NAMES this (pdb, seed), or None

    Reading it matters: a directory merely matching poses/<pdb>/<seed>-* could belong to another run,
    and globbing alone let an unrelated run.json stand in for a missing expected unit.
    """
    for d in sorted(glob.glob(os.path.join(E, "poses", pdb, f"{seed}-*"))):
        rj = os.path.join(d, "run.json")
        if not os.path.exists(rj):
            continue
        try:
            with open(rj) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        if str(m.get("pdb")) == pdb and str(m.get("seed")) == str(seed):
            return d
    return None


def check_npz(path, nsample):
    """out: "" if the cache is usable for an offline replay, else why not"""
    if not os.path.exists(path):
        return "absent"
    try:
        import numpy as np
        d = np.load(path, allow_pickle=True)
        # THE SAME LIST THE SMOKE TEST ASSERTS. If verify required less than smoke, a cache that
        # smoke would have rejected could pass the final check on 1,540 units.
        need = ["z_trunk", "s_trunk", "ci_s_inputs", "ci_s_trunk_f32", "ci_coordinate",
                "ci_contact_probs", "ci_N_cycle", "ci_z_dtype", "ci_s_dtype",
                "ci_s_inputs_dtype", "rng_py", "rng_torch"] + [f"ci_feat_{k}" for k in (
                    "distogram_rep_atom_mask", "atom_to_token_idx", "atom_to_tokatom_idx",
                    "asym_id", "has_frame", "is_ligand")]
        missing = [k for k in need if k not in d.files]
        if missing:
            return f"missing {missing[:4]}"
        sh = [x for x in d["ci_coordinate"].shape]
        while len(sh) > 1 and sh[0] == 1:
            sh.pop(0)
        if not sh or sh[0] != nsample:
            return f"ci_coordinate holds {sh[0] if sh else 0} poses, expected {nsample}"
        if str(d["dtype"]) != "bf16":
            return f"dtype {d['dtype']}, expected bf16"
    except Exception as e:                                      # noqa: BLE001
        return f"unreadable: {type(e).__name__}"
    return ""


def verify(a):
    pool = read_pool(os.path.join(a.e, "pool.csv"))
    have = read_results(os.path.join(a.e, "results.csv"))
    pdbs, units = expected_units(a.jobs, a.suffix, a.seeds, a.seed0)
    if not pdbs:
        print(f"FAIL no job files matching *{a.suffix} in {a.jobs}")
        return 1

    unfolded, no_rows, bad_scores, bad_cache = [], [], [], []
    for pdb, seed in units:
        d = run_dir_for(a.e, pdb, seed)
        if d is None:
            unfolded.append((pdb, seed))
            continue
        # EXACT SAMPLE COVERAGE, not "a row exists". One of nsample samples scored is not a scored
        # unit, and a blank or NaN score is not a score.
        for ifc in sorted(pool.get(pdb, ())):
            got = have.get((ifc, seed), {})
            miss = [str(k) for k in range(a.nsample) if str(k) not in got]
            if miss:
                no_rows.append((ifc, seed, f"samples {miss}"))
            for k, (dq, ip) in got.items():
                if not finite(dq) or not finite(ip):
                    bad_scores.append((ifc, seed, k, dq, ip))
        why = check_npz(os.path.join(a.e, "trunks", pdb, f"free_{seed}.npz"), a.nsample)
        if why:
            bad_cache.append((pdb, seed, why))

    print(f"expected units {len(units)} over {len(pdbs)} structures; "
          f"interfaces per structure {sum(len(v) for v in pool.values())} total")
    print(f"  unfolded units                     : {len(unfolded)}")
    print(f"  folded units missing sample rows   : {len(no_rows)}")
    print(f"  rows with a non-finite score       : {len(bad_scores)}")
    print(f"  unusable or absent trunk caches    : {len(bad_cache)}")
    for lbl, items in (("unfolded", unfolded), ("missing rows", no_rows),
                       ("non-finite", bad_scores), ("cache", bad_cache)):
        for x in items[:3]:
            print(f"     {lbl}: {x}")
    rc = 1 if (unfolded or no_rows or bad_scores or bad_cache) else 0
    print("COMPLETE: every expected unit folded, every interface scored on every sample, "
          "every cache replayable" if rc == 0 else "INCOMPLETE")
    return rc


def rescore(a):
    """score only the (interface, sample) rows that are missing, with the dataset's own scorer"""
    pool = read_pool(os.path.join(a.e, "pool.csv"))
    res = os.path.join(a.e, "results.csv")
    have = read_results(res)
    pdbs, units = expected_units(a.jobs, a.suffix, a.seeds, a.seed0)
    added = skipped = 0
    for pdb, seed in units:
        d = run_dir_for(a.e, pdb, seed)
        if d is None:
            continue
        for ifc in sorted(pool.get(pdb, ())):
            got = have.get((ifc, seed), {})
            miss = {str(k) for k in range(a.nsample)} - set(got)
            if not miss:
                continue
            # THE DATASET'S OWN SCORER. Recovery must not change the method: sabdab_val scores with
            # 'ost' and an earlier version of this hardcoded 'pxm', which would have mixed two DockQ
            # implementations in one results.csv.
            cmd = [a.py, os.path.join(a.code, "scorer.py"), "--dir", d, "--iface", ifc,
                   "--seed", seed, "--arm", "free", "--donor", "-1", "--chains", "",
                   "--ref", os.path.join(a.refs, f"{pdb}.cif"), "--scorer", a.scorer]
            try:
                out = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT {ifc} {seed}")
                continue
            if out.returncode != 0:
                print(f"  FAILED {ifc} {seed}: {out.stderr.strip()[:160]}")
                continue
            # ONLY THE MISSING SAMPLES. scorer.py emits a row per pose, so appending all of them
            # would duplicate the ones already there and double-weight that seed in every average.
            keep = []
            for line in out.stdout.splitlines():
                f = line.split(",")
                if len(f) >= 8 and f[5].strip() in miss:
                    keep.append(line)
                elif len(f) >= 8:
                    skipped += 1
            if keep:
                with open(res, "a") as fh:
                    for line in keep:
                        fh.write(line + "\n")
                added += len(keep)
                print(f"  {ifc} {seed}: added samples {[l.split(',')[5] for l in keep]}")
    print(f"rescore added {added} row(s); left {skipped} existing row(s) untouched")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "rescore"])
    ap.add_argument("--e", required=True, help="the val_fold directory")
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--suffix", default="-final-updated.json")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=400000)
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--scorer", required=True, help="the dataset's scorer; recovery must not change it")
    ap.add_argument("--py", required=True)
    ap.add_argument("--code", required=True)
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    sys.exit(verify(a) if a.mode == "verify" else rescore(a))


if __name__ == "__main__":
    main()
