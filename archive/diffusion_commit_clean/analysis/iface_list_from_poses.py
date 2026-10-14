#!/usr/bin/env python
"""Can the interface LIST be recovered from the predictions, instead of from the crystal?

THE LEAK. Selection sorts poses per interface, so it needs to know which antibody-antigen chain pairs
ARE interfaces. We read that from the reference roles file. Measured: on 27 of 131 complexes the
reference list is a strict SUBSET of all antibody x antigen pairs -- 113 extra interfaces across 314 --
so stoichiometry cannot substitute for it. On 9k94, 5 antibodies and 5 antigens give 25 possible pairs
and only 5 are real. The crystal is telling us which pairs touch.

THE CANDIDATE REPLACEMENT. Which pairs touch is visible in the predictions themselves: take the pairs
in contact in at least one pose. No native structure required.

WHAT IS AND IS NOT ASSUMED. Chain ROLE is assumed known -- which entity is the antibody and which is
the antigen comes from the input you submitted, not from the answer. Chain PAIRING is what must be
recovered. So this maps reference ab/ag chains to model chains to learn roles, then throws the pairing
away and re-derives it from contacts.

TWO WAYS TO FAIL, reported separately because they cost different things:
  missed      a reference interface that no pose ever forms. Unrecoverable: we would never select or
              evaluate it, so a real interface silently leaves the benchmark.
  spurious    a pair contacting in some pose but absent from the reference. Cheap but not free: it adds
              an interface list that round-robin must share the 50-donor budget with.

    iface_list_from_poses.py [--poses 10] [--jobs 6]
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, "/home/ebrahim/tts_abag")
sys.path.insert(0, "/home/ebrahim/tts_abag/diffusion_commit_clean/metrics")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import metrics  # noqa: E402
import paths  # noqa: E402
from fmt import table  # noqa: E402


def one(task):
    """Contacting (ab, ag) model-chain pairs over a few poses of one complex."""
    pdb, poses, abs_, ags = task
    seen = set()
    for seed, samp in poses:
        p = paths.pool_pose(pdb, seed, samp)
        if not os.path.exists(p):
            continue
        for a in abs_:
            for g in ags:
                try:
                    if len(metrics.epitope_set(p, a, g, cutoff=metrics.CONTACT_CUTOFF)) > 0:
                        seen.add((a, g))
                except Exception:
                    pass
    return pdb, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=6)
    A = ap.parse_args()

    roles = {}
    for r in csv.DictReader(open(str(paths.PXM2025_CHAIN_ROLES))):
        roles.setdefault(r["pdb"], set()).add((r["ab_chain"], r["ag_chain"]))
    maps = metrics.load_chain_maps()

    pool = collections.defaultdict(set)
    with open(str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            pool[r["pdb"]].add((r["seed"], int(r["sample"])))

    tasks, ref_model = [], {}
    for pdb, pairs in sorted(roles.items()):
        keys = sorted(pool.get(pdb, ()))[:A.poses]
        if not keys:
            continue
        # roles in MODEL chain labels, via any pose's mapping; pairing is deliberately discarded
        m = None
        for k in keys:
            m = maps.get((pdb, k[0], k[1]))
            if m:
                break
        if not m:
            continue
        abs_ = {m[a] for a, _ in pairs if a in m}
        ags = {m[g] for _, g in pairs if g in m}
        if not abs_ or not ags:
            continue
        ref_model[pdb] = {(m[a], m[g]) for a, g in pairs if a in m and g in m}
        tasks.append((pdb, keys, sorted(abs_), sorted(ags)))

    import multiprocessing as mp
    got = {}
    with mp.Pool(A.jobs) as p:
        for i, (pdb, seen) in enumerate(p.imap_unordered(one, tasks), 1):
            got[pdb] = seen
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)}", flush=True)

    agg = collections.Counter()
    nmiss = nspur = nref = 0
    worst = []
    for pdb, ref in ref_model.items():
        if pdb not in got:
            continue
        seen = got[pdb]
        miss, spur = ref - seen, seen - ref
        nref += len(ref)
        nmiss += len(miss)
        nspur += len(spur)
        agg["exact match" if not miss and not spur else
            ("missed a real interface" if miss else "extra pairs only")] += 1
        if miss or spur:
            worst.append([pdb, len(ref), len(seen), len(miss), len(spur)])
    print()
    print(table(["", "complexes"], [[k, v] for k, v in agg.most_common()], align="l r"))
    print(f"\n  reference interfaces: {nref}   missed: {nmiss}   spurious: {nspur}"
          f"   (poses inspected per complex: {A.poses})")
    worst.sort(key=lambda r: (-r[3], -r[4]))
    if worst:
        print()
        print(table(["pdb", "reference interfaces", "found from poses", "missed", "spurious"],
                    worst[:12], align="l r r r r"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
