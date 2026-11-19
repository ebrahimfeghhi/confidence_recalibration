#!/usr/bin/env python
"""Build the MIXED-DEPTH seed manifest that lets the whole training pool be scored in one pass.

WHAT THIS EXISTS FOR. The in-sample scoring path could only ever score a uniform-depth sub-pool.
training/combined/pool.csv holds 1,688 interfaces over 853 structures, and their folded depth is
not uniform -- so `--budget 15` over the whole pool stopped on 5,516 of 12,795 intended units with
"A metric labelled 'budget 15' must cover 15 seeds for every structure", which was the guard being
right. The stopgap was training/fulltrain15/pool.csv: the 947 interfaces that happen to be 15-deep,
scored as "the training set". This writes the per-structure seed set instead, and phase_k_eval
--mixed-budget consumes it.

DEPTH IS 15 OR 1, NOT 15 OR 10. results.csv holds DockQ for 10 seeds of every shallow interface,
and counting THAT is how the pool looks like a 15/10 split (947 interfaces at 15 seeds, 741 at 10).
But the evaluator recomputes ipTM from the cached TRUNK, and for those 394 structures only one
trunk was ever kept -- training/fold/trunks/10gh/ holds free_500000.npz alone while
results.csv has 10gh_AB at seeds 500000-500009. A seed with DockQ and no trunk cannot be scored, so
the scoreable depth of those structures is 1 (5 candidates), not 10 (50). The manifest written here
is the intersection, which is why it must be built rather than read off either source.

DISCOVERED FROM THE FOLD, AND EVERY SEED IT NAMES IS PROVED SCOREABLE BEFORE IT IS WRITTEN. The
other manifests in this repo are DECLARED (a trim algorithm chose the seeds and recorded them). No
such declaration exists for the training fold's depth: train_val_splits/cache_train_structures_
extended.csv's `seeds` column says 437 structures carry 15 and 415 carry one, while the trunks on
disk say 459 and 394 -- 22 structures have 15 trunks and 15 seeds of DockQ while the training
manifest declares one, because that column describes what the TRAINER samples, not what was folded.
So this file discovers the depth, and compensates by requiring, per seed it keeps:
    - the trunk exists at <fold>/trunks/<pdb>/free_<seed>.npz, and
    - every interface of that structure has a DockQ row at that seed for all --nsample samples.
A seed that fails either test is EXCLUDED and counted, never kept at partial depth: that is the
same key-set contract phase_k_eval would enforce at the end of a 7,279-unit pass, applied here for
free. The declared column is recorded per structure as `declared_depth` and the disagreement count
is printed, so the discovery can be audited against it.

THE OUTPUT IS DOUBLE-LOCKED AGAINST A UNIFORM PASS PICKING IT UP. It declares mixed:true and no
`k`, which phase_k_eval refuses unless --mixed-budget is passed -- and refuses --mixed-budget
without. That matters because the trainer auto-forwards --seed-manifest for any pool directory
holding a seed_manifest.json (phase_k_train.py:8404), so this file is written to its OWN directory
(training/fulltrain_mixed/) and not beside a pool anything trains on.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FREE = re.compile(r"^free_(\d+)\.npz$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=f"{ROOT}/training/combined/pool.csv")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold")
    ap.add_argument("--results", default=f"{ROOT}/training/fold/results.csv")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/fulltrain_mixed")
    ap.add_argument("--seed-base", type=int, default=500000)
    ap.add_argument("--nsample", type=int, default=5,
                    help="samples per (structure, seed), as the fold wrote them. A seed is kept "
                         "only if every interface of the structure has all of them scored -- the "
                         "same cross product phase_k_eval's coverage contract enforces.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written and write nothing. The depth histogram is "
                         "the thing worth reading before a 7,279-unit GPU pass is launched.")
    a = ap.parse_args()

    # ---- the pool, by structure
    ifaces = collections.defaultdict(list)
    with open(a.pool) as fh:
        for r in csv.DictReader(fh):
            pdb = r["pdb"].strip()
            ifaces[pdb].append(f"{pdb}_{r['chain1'].strip()}{r['chain2'].strip()}")
    if not ifaces:
        sys.exit(f"STOP: {a.pool} yielded no interfaces.")
    n_iface = sum(len(v) for v in ifaces.values())
    print(f"pool: {n_iface} interface(s) over {len(ifaces)} structure(s) from {a.pool}")

    # ---- join 3, the DockQ keys
    scored = collections.defaultdict(set)          # iface -> {(seed, sample)}
    with open(a.results) as fh:
        for r in csv.DictReader(fh):
            # PARSED BEFORE IT IS COUNTED, and all four fields. phase_k_eval reads results.csv
            # with the same try/except around the casts and keeps only rows where every one of
            # them parses, so a seed kept here on the strength of a row with an empty dockq
            # would pass this builder and fail the eval's coverage contract hours later.
            try:
                key = (int(r["seed"]), int(r["sample"]))
                float(r["dockq"])
                float(r["iptm"])
            except (TypeError, ValueError, KeyError):
                continue
            scored[r["iface"]].add(key)
    print(f"results.csv: {sum(len(v) for v in scored.values())} scored (iface, seed, sample) "
          f"row(s) over {len(scored)} interface(s)")

    # ---- the declared depth, for audit only
    declared = {}
    _dec = f"{ROOT}/train_val_splits/cache_train_structures_extended.csv"
    if os.path.exists(_dec):
        with open(_dec) as fh:
            for r in csv.DictReader(fh):
                # A RANGE COUNTS AS MANY SEEDS. The column holds either '500000-500014' or a single
                # '500000'; splitting it on commas and counting the pieces reports every 15-seed
                # structure as 1-seed, which is how this pool first looked uniform.
                _sd = (r.get("seeds") or "").strip()
                n = 0
                for part in _sd.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        lo, hi = part.split("-", 1)
                        try:
                            n += int(hi) - int(lo) + 1
                        except ValueError:
                            n += 1
                    else:
                        n += 1
                declared[r.get("pdb", "").strip()] = n

    # ---- per structure: the seeds that can actually be scored
    trunks = os.path.join(a.fold, "trunks")
    structures, excluded = {}, collections.Counter()
    no_trunk_dir = []
    for pdb in sorted(ifaces):
        d = os.path.join(trunks, pdb)
        if not os.path.isdir(d):
            no_trunk_dir.append(pdb)
            continue
        # A STRICT FILENAME AND A RANGE FILTER. int(f[5:-4]) raises on an odd name like
        # free_500000_tmp.npz, and a stray trunk below --seed-base would be written into the
        # manifest and then rejected by phase_k_eval's own seed_range check at the start of the
        # pass. Both are loud rather than silent, which is why they are filtered here and
        # counted rather than left to fail later.
        have = sorted(int(m.group(1)) for m in
                      (_FREE.match(f) for f in os.listdir(d)) if m
                      and int(m.group(1)) >= a.seed_base)
        keep = []
        for sd in have:
            miss = [(i, k) for i in ifaces[pdb] for k in range(a.nsample)
                    if (sd, k) not in scored.get(i, ())]
            if miss:
                excluded[f"{pdb}/{sd}"] = len(miss)
                continue
            keep.append(sd)
        if not keep:
            no_trunk_dir.append(pdb)
            continue
        structures[pdb] = {"seeds": keep, "n_seeds": len(keep),
                           "interfaces": sorted(ifaces[pdb]),
                           "trunks_on_disk": len(have),
                           "declared_depth": declared.get(pdb)}
    # A STRUCTURE WITH NOTHING SCOREABLE IS A HARD STOP, NOT A QUIET OMISSION. The manifest is the
    # only statement of per-structure depth under --mixed-budget, and phase_k_eval refuses any pool
    # structure the manifest does not name -- so dropping one here would turn into a refusal at the
    # start of the pass with no explanation of what was dropped or why.
    if no_trunk_dir:
        sys.exit(f"STOP: {len(no_trunk_dir)} pool structure(s) have no scoreable seed at all "
                 f"(no trunk, or no seed whose every interface x sample has DockQ): "
                 f"{no_trunk_dir[:8]}. Either fold/score them or remove them from {a.pool}; a "
                 f"manifest that silently omits them would shrink the pool without saying so.")
    if excluded:
        print(f"  EXCLUDED {len(excluded)} (structure, seed) pair(s) that HAVE a trunk but are "
              f"short of {a.nsample} sample(s) of DockQ on some interface, e.g. "
              + ", ".join(f"{k}:{v} missing" for k, v in list(excluded.items())[:6]))

    depth_hist = collections.Counter(v["n_seeds"] for v in structures.values())
    iface_hist = collections.Counter(v["n_seeds"] for v in structures.values()
                                     for _ in v["interfaces"])
    n_units = sum(v["n_seeds"] for v in structures.values())
    maxd = max(v["n_seeds"] for v in structures.values())
    mind = min(v["n_seeds"] for v in structures.values())
    print(f"  structures by depth: "
          + ", ".join(f"{c} at {d} seed(s)" for d, c in sorted(depth_hist.items(), reverse=True)))
    print(f"  interfaces by depth: "
          + ", ".join(f"{c} at {d} seed(s) ({d * a.nsample} candidates)"
                      for d, c in sorted(iface_hist.items(), reverse=True)))
    print(f"  ONE PASS = {n_units} unit(s) (sum of per-structure depths), against "
          f"{len(structures) * maxd} if the deepest depth were claimed for all of them")
    _dis = {p: (v["declared_depth"], v["n_seeds"]) for p, v in structures.items()
            if v["declared_depth"] is not None and v["declared_depth"] != v["n_seeds"]}
    print(f"  the training manifest's declared `seeds` column disagrees with the fold on "
          f"{len(_dis)} structure(s)" + (f", e.g. {list(_dis.items())[:4]} (declared, scoreable)"
                                         if _dis else "")
          + " -- recorded per structure as declared_depth; the FOLD is what can be scored.")
    if maxd == mind:
        sys.exit(f"STOP: every structure in {a.pool} carries {maxd} seed(s), so this pool does not "
                 f"need a mixed manifest -- score it at --budget {maxd} with no manifest at all. "
                 f"Writing mixed:true here would disable the uniform-depth assertion for nothing.")

    man = {
        "mixed": True,
        "k": None,
        "algo": "scoreable-depth-v1",
        "note": ("per-structure depth = the seeds whose TRUNK exists under <fold>/trunks AND "
                 "whose every interface x sample carries DockQ in results.csv. Requires "
                 "phase_k_eval --mixed-budget; a uniform pass refuses it by design."),
        "pool": os.path.relpath(os.path.abspath(a.pool), ROOT),
        "fold": os.path.relpath(os.path.abspath(a.fold), ROOT),
        "results_csv": os.path.relpath(os.path.abspath(a.results), ROOT),
        "nsample": a.nsample,
        "seed_range": [a.seed_base, max(max(v["seeds"]) for v in structures.values())],
        "max_depth": maxd,
        "min_depth": mind,
        "n_structures": len(structures),
        "n_interfaces": sum(len(v["interfaces"]) for v in structures.values()),
        "n_units": n_units,
        "depth_hist_structures": {str(d): c for d, c in sorted(depth_hist.items(), reverse=True)},
        "depth_hist_interfaces": {str(d): c for d, c in sorted(iface_hist.items(), reverse=True)},
        "structures": structures,
    }
    # THE DIGEST COVERS THE SEED SETS AND NOTHING ELSE, so it identifies the measurement rather
    # than the comment above it: phase_k_eval copies it into every metric json, and two manifests
    # with the same depths but different seeds must not share one.
    man["digest"] = hashlib.sha1(
        json.dumps({p: v["seeds"] for p, v in sorted(structures.items())},
                   sort_keys=True).encode()).hexdigest()[:12]
    out = os.path.join(a.out_dir, "seed_manifest.json")
    if a.dry_run:
        print(f"  --dry-run: would write {out} (digest {man['digest']})")
        return 0
    os.makedirs(a.out_dir, exist_ok=True)
    tmp = out + f".{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(man, fh, indent=1)
    os.replace(tmp, out)
    print(f"  wrote {out} (digest {man['digest']})")
    # THE POOL IS COPIED IN BESIDE IT, not referenced across directories: --seed-manifest and
    # --pool must describe the same interface set or the eval refuses, and the one way that has
    # gone wrong in this repo is a cache or a roles table shared between two pools and left stale.
    pool_out = os.path.join(a.out_dir, "pool.csv")
    with open(a.pool) as src, open(pool_out + ".tmp", "w") as dst:
        dst.write(src.read())
    os.replace(pool_out + ".tmp", pool_out)
    print(f"  wrote {pool_out} ({n_iface} interface(s), a byte copy of {a.pool})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
