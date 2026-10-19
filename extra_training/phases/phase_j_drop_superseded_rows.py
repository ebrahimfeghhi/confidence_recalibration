#!/usr/bin/env python
"""Remove the score rows left behind by the 2026-10-15 refold, restoring one row per pose.

phase_j_drop_superseded_poses.py restored the on-disk invariant -- one finished pose directory per
(pdb, seed) -- by moving the superseded directories into _superseded_poses_<date>/. It did not
clean results.csv: the refold's rows were appended and the originals were never removed, so the
file holds 59,145 rows for 50,490 distinct (iface, seed, sample) keys. 8,655 keys carry two.

WHY IT MATTERS, AND WHY NOTHING PRODUCED BY THE PIPELINE IS WRONG. The evaluator never reads a
DockQ out of results.csv by key: it joins pose to sample by COORDINATE IDENTITY against the
cached trunk (resolve_seed_dir, and the pose->sample join in phase_k_eval.py), so it always reads
the surviving directory. Every number the evaluator has reported is unaffected. What the
duplication breaks is ad-hoc analysis that reads results.csv directly -- an `oracle` is a max over
a pose's rows, and a max over a stale row reads too high. Computing the training-fold oracle this
way gave 0.5401 where resolving the duplicates gives a different number, and 111 interfaces carry
a disagreement over 0.1 DockQ with a worst case of 0.944.

HOW A ROW IS IDENTIFIED AS SUPERSEDED. Not by recency -- results.csv has no timestamp column, and
row order is not reliable because phase_j_merge_results.py merged several files. Instead the
quarantined directories still hold their own ost_json, so the superseded DockQ is readable
directly: ost_json/<sample>.json carries `dockq` and `dockq_interfaces`, where each interface is
[ref_c1, ref_c2, model_c1, model_c2] and results.csv's iface name is the two REFERENCE chains.
That is evidence, not inference.

WHAT IT REFUSES. A duplicated key where NEITHER row matches the quarantined value, or where BOTH
do, is left untouched and reported. Both cases mean the evidence does not single out a row, and
guessing is what this script exists to avoid. It also refuses to drop a row from a key that is not
duplicated: if the quarantined value is the only row present, that row is the one the pipeline is
using and removing it would delete a real score.

Writes to --out. Replacing results.csv needs --in-place, which keeps a timestamped backup first.
"""
import argparse
import collections
import csv
import glob
import json
import os
import shutil
import statistics
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def superseded_map(quarantine):
    """{(pdb, iface, seed, sample): dockq} read from the quarantined directories' own ost_json."""
    out, dirs, files = {}, 0, 0
    for pdb in sorted(os.listdir(quarantine)):
        pdir = os.path.join(quarantine, pdb)
        if not os.path.isdir(pdir):
            continue
        for sd in sorted(os.listdir(pdir)):
            seed = sd.split("-")[0]
            if not seed.isdigit():
                continue
            dirs += 1
            for jf in sorted(glob.glob(os.path.join(pdir, sd, "ost_json", "*.json"))):
                sample = os.path.splitext(os.path.basename(jf))[0]
                if not sample.isdigit():
                    continue
                try:
                    d = json.load(open(jf))
                except Exception as e:
                    print(f"  (unreadable {jf}: {type(e).__name__}: {e})")
                    continue
                files += 1
                ifaces, dq = d.get("dockq_interfaces") or [], d.get("dockq") or []
                if len(ifaces) != len(dq):
                    print(f"  (skipping {jf}: {len(ifaces)} interface(s) and {len(dq)} dockq)")
                    continue
                for ent, v in zip(ifaces, dq):
                    if len(ent) < 2 or v is None:
                        continue
                    # THE REFERENCE CHAINS NAME THE INTERFACE, which is what results.csv uses.
                    # ent is [ref_c1, ref_c2, model_c1, model_c2] and the model chains differ
                    # whenever chain_mapping is not the identity.
                    out[(pdb, f"{pdb}_{ent[0]}{ent[1]}", int(seed), int(sample))] = float(v)
    print(f"quarantine: {dirs} seed dir(s), {files} ost_json file(s), {len(out)} superseded "
          f"(iface, seed, sample) score(s)")
    return out


def superseded_iptm(quarantine):
    """{(pdb, seed, sample): {rounded chain_pair_iptm values}} from the quarantined predictions.

    SECOND LINE OF EVIDENCE, for the keys where both rows carry the same DockQ. results.csv's
    `iptm` column is the chain-pair ipTM -- verified: 11zw_AC seed 500000 sample 2 reads 0.7802
    and that run's chain_pair_iptm holds 0.7802159786224365. Matched as a SET rather than by chain
    index, because the index order would have to be reconciled with chain_mapping and the set is
    enough: a row whose ipTM appears among the superseded run's pair values, when the other row's
    does not, is the superseded row.
    """
    out, files = {}, 0
    for pdb in sorted(os.listdir(quarantine)):
        pdir = os.path.join(quarantine, pdb)
        if not os.path.isdir(pdir):
            continue
        for sd in sorted(os.listdir(pdir)):
            seed = sd.split("-")[0]
            if not seed.isdigit():
                continue
            for jf in glob.glob(os.path.join(pdir, sd, "predictions",
                                             "*_summary_confidence_sample_*.json")):
                tail = os.path.splitext(os.path.basename(jf))[0].rsplit("_", 1)[-1]
                if not tail.isdigit():
                    continue
                try:
                    d = json.load(open(jf))
                except Exception:
                    continue
                files += 1
                vals = set()
                for row in (d.get("chain_pair_iptm") or []):
                    for v in row:
                        if v:
                            vals.add(round(float(v), 4))
                if vals:
                    out[(pdb, int(seed), int(tail))] = vals
    print(f"quarantine: {files} summary_confidence file(s), {len(out)} (pdb, seed, sample) "
          f"ipTM set(s)")
    return out


def oracle(rows):
    """(mean oracle DockQ, frac acceptable) over interfaces, one max per pose then per interface."""
    per_pose = collections.defaultdict(list)
    for r in rows:
        per_pose[(r["iface"], r["seed"], r["sample"])].append(float(r["dockq"]))
    per_if = collections.defaultdict(list)
    for (iface, _s, _p), v in per_pose.items():
        per_if[iface].append(max(v))
    best = [max(v) for v in per_if.values()]
    return statistics.mean(best), sum(1 for x in best if x >= 0.23) / len(best), len(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(ROOT, "training/fold/results.csv"))
    ap.add_argument("--quarantine",
                    default=os.path.join(ROOT, "training/fold/_superseded_poses_2026-10-15"))
    ap.add_argument("--out", default="", help="default <results>.dedup.csv")
    ap.add_argument("--in-place", action="store_true",
                    help="replace --results with the cleaned file, keeping a timestamped backup")
    ap.add_argument("--seed-lo", type=int, default=0, help="restrict the oracle report to a seed range")
    ap.add_argument("--seed-hi", type=int, default=10 ** 9)
    a = ap.parse_args()
    out = a.out or a.results + ".dedup.csv"

    sup = superseded_map(a.quarantine)
    with open(a.results) as fh:
        rd = csv.DictReader(fh)
        cols, rows = rd.fieldnames, list(rd)
    print(f"results: {len(rows)} row(s), {len(cols)} column(s)")

    by_key = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_key[(r["iface"], int(r["seed"]), int(r["sample"]))].append(i)
    dup = {k: v for k, v in by_key.items() if len(v) > 1}
    print(f"duplicated keys: {len(dup)} of {len(by_key)}")

    sup_ip = superseded_iptm(a.quarantine)
    drop, no_match, both_match, not_dup_hit, by_iptm = set(), [], [], 0, 0
    for key, idxs in sorted(dup.items()):
        iface, seed, sample = key
        pdb = iface.split("_")[0]
        want = sup.get((pdb, iface, seed, sample))
        if want is None:
            no_match.append((key, "no quarantined score"))
            continue
        # MATCHED AT THE FILE'S OWN PRECISION. results.csv stores dockq to 4 dp, the ost_json to
        # 3; comparing floats exactly would match nothing.
        hits = [i for i in idxs if abs(float(rows[i]["dockq"]) - want) < 5e-4]
        if len(hits) == 1:
            drop.add(hits[0])
        elif not hits:
            no_match.append((key, f"quarantined {want} matches neither "
                                  f"{[rows[i]['dockq'] for i in idxs]}"))
        else:
            # SAME DockQ ON BOTH ROWS -- all 3,068 of these, measured. Fall through to the ipTM
            # evidence, which distinguishes them where DockQ cannot. Note the oracle is a DockQ
            # max and is therefore already exact for these keys either way; this pass is for
            # anything that reads the ipTM column, such as phase_k_eval's self-check.
            ips = sup_ip.get((pdb, seed, sample))
            ip_hits = ([i for i in hits if round(float(rows[i]["iptm"]), 4) in ips]
                       if ips else [])
            if len(ip_hits) == 1:
                drop.add(ip_hits[0])
                by_iptm += 1
            else:
                both_match.append((key, f"quarantined dockq {want} matches {len(hits)} row(s) "
                                        f"{[rows[i]['dockq'] for i in hits]}; ipTM singles out "
                                        f"{len(ip_hits)}"))

    # A QUARANTINED SCORE FOR A KEY THAT IS NOT DUPLICATED is the only row present, which means it
    # is the row the pipeline is using. Counted and left alone rather than dropped.
    for k in sup:
        kk = (k[1], k[2], k[3])
        if kk in by_key and len(by_key[kk]) == 1:
            not_dup_hit += 1

    print(f"\nresolved: {len(drop)} row(s) to drop ({len(drop) - by_iptm} by DockQ, "
          f"{by_iptm} by ipTM where both rows shared a DockQ)")
    print(f"refused (evidence does not single out a row): {len(no_match)} unmatched, "
          f"{len(both_match)} ambiguous")
    for key, why in (no_match[:5] + both_match[:5]):
        print(f"  {key[0]} seed {key[1]} sample {key[2]}: {why}")
    if len(no_match) + len(both_match) > 10:
        print(f"  ... and {len(no_match) + len(both_match) - 10} more")
    print(f"quarantined scores whose key has only one row (left alone): {not_dup_hit}")

    keep = [r for i, r in enumerate(rows) if i not in drop]
    sel = lambda rs: [r for r in rs
                      if a.seed_lo <= int(r["seed"]) <= a.seed_hi]
    b_mean, b_frac, b_n = oracle(sel(rows))
    a_mean, a_frac, a_n = oracle(sel(keep))
    print(f"\noracle over seeds {a.seed_lo}-{a.seed_hi}, {b_n} interface(s):")
    print(f"  before  mean {b_mean:.4f}  frac_acc {b_frac:.4f}")
    print(f"  after   mean {a_mean:.4f}  frac_acc {a_frac:.4f}   ({a_n} interface(s))")
    if a_n != b_n:
        raise SystemExit(f"STOP: dropping rows changed the interface count {b_n} -> {a_n}; some "
                         f"interface lost every row, which this script must never do.")

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(keep)
    print(f"\nwrote {len(keep)} row(s) to {out}")

    still = sum(1 for k, v in collections.Counter(
        (r["iface"], r["seed"], r["sample"]) for r in keep).items() if v > 1)
    print(f"keys still duplicated in the output: {still}")

    if a.in_place:
        bak = f"{a.results}.bak.{time.strftime('%Y%m%d-%H%M%S')}"
        shutil.copy2(a.results, bak)
        shutil.move(out, a.results)
        print(f"backed up to {bak} and replaced {a.results}")


if __name__ == "__main__":
    main()
