#!/usr/bin/env python
"""OST's per-pose interface -> (asym_i, asym_j) mapping, persisted for the dataset. CPU only.

WHY THIS FILE EXISTS, AND WHY INFERRING IT WAS WRONG. An ABAG-Rank example is one INTERFACE, and
an interface is a chain pair. `results.csv`'s iface column -- `7zra_AC` -- names REFERENCE chain
letters, while the dump's asym_i/asym_j name MODEL chains. The map between them is not a constant:
phase_k_eval.ost_chain_pairs reads it from `ost_json/<k>.json` for EACH POSE, because "the first
antigen is not determined by sequence -- OST decides it per pose, by whichever assignment scores
best". That evaluator's own output records OST disagreeing with the static chain_roles table on
145 poses across 2 structures, one of them 7zra -- the structure whose ipTM join already failed
once on exactly this.

dataset.py originally recovered the pair by assuming the interface's letters indexed the sorted
asym ids alphabetically. That is the same assumption, reintroduced, and it would have attached the
right DockQ label to the wrong chain pair's features on precisely the poses that are hardest.

So the mapping is read from OST, through the same coordinate-identity join the PAE dump used to
decide which sample a pose is, and written beside the dump as <pdb>/_chainmap.json:

    {"<seed>_<sample>": {"<iface_id>": [asym_i, asym_j], ...}, ...}

A pose whose mapping OST cannot resolve is recorded as such rather than guessed, and the dataset
drops it.
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))


def _fingerprint(fold, dump, pdb):
    """Identity of the trunks and pose files this complex's mapping was derived from."""
    parts = []
    for f in sorted(glob.glob(os.path.join(fold, "trunks", pdb, "free_*.npz"))) + \
             sorted(f for f in glob.glob(os.path.join(dump, pdb, "*.npz"))
                    if not os.path.basename(f).startswith("_")):
        try:
            st = os.stat(f)
            parts.append(f"{os.path.basename(f)}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{os.path.basename(f)}:MISSING")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="the --dump-pae directory")
    ap.add_argument("--fold", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    a = ap.parse_args()

    import phase_k_eval as EV
    import phase_j_build_cache as BC

    refs = EV.pool_ref_pairs(os.path.join(ROOT, a.pool))
    dump = os.path.join(ROOT, a.dump)
    pdbs = sorted(d for d in os.listdir(dump) if os.path.isdir(os.path.join(dump, d)))
    if a.nshards > 1:
        pdbs = pdbs[a.shard::a.nshards]

    n_ok = n_bad = 0
    for pdb in pdbs:
        out_f = os.path.join(dump, pdb, "_chainmap.json")
        # A STALE MAP IS WORSE THAN NO MAP: it attaches an interface's DockQ label to the wrong
        # model-chain pair, which is the 7zra failure class wearing a valid-looking file. The
        # inputs are fingerprinted -- the trunks the coordinates came from and the pose files
        # whose samples were mapped -- so a refold or a re-export regenerates rather than reuses.
        # Caught in review.
        fp = _fingerprint(os.path.join(ROOT, a.fold), dump, pdb)
        if os.path.exists(out_f):
            try:
                if json.load(open(out_f)).get("_source") == fp:
                    continue
            except (ValueError, OSError):
                pass
            print(f"  REGENERATE {pdb}: chain map does not match its inputs", flush=True)
        poses = sorted(f for f in glob.glob(os.path.join(dump, pdb, "*.npz"))
                       if not os.path.basename(f).startswith("_"))
        if not poses:
            continue
        # group by seed: resolve_seed_dir works per (pdb, seed) and needs that unit's coordinates
        by_seed = collections.defaultdict(list)
        for f in poses:
            sd, smp = os.path.basename(f)[:-4].split("_")
            by_seed[int(sd)].append(int(smp))
        out = {}
        for sd, samples in sorted(by_seed.items()):
            tr = os.path.join(ROOT, a.fold, "trunks", pdb, f"free_{sd}.npz")
            if not os.path.exists(tr):
                n_bad += len(samples)
                continue
            px = np.load(tr)["ci_coordinate"]
            try:
                unit_dir, _ks = BC.resolve_seed_dir(os.path.join(ROOT, a.fold), pdb, sd, px,
                                                    0.01, 0.5)
            except Exception as e:                # noqa: BLE001
                for smp in samples:
                    out[f"{sd}_{smp}"] = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
                n_bad += len(samples)
                continue
            for smp in samples:
                pairs, err = EV.ost_chain_pairs(unit_dir, pdb, smp, refs[pdb])
                if pairs is None:
                    out[f"{sd}_{smp}"] = {"error": err}
                    n_bad += 1
                else:
                    out[f"{sd}_{smp}"] = {ifc: [int(i), int(j)] for ifc, i, j in pairs}
                    n_ok += 1
        tmp = out_f + ".tmp"
        out["_source"] = fp
        with open(tmp, "w") as fh:
            json.dump(out, fh)
        os.replace(tmp, out_f)
    print(f"  shard {a.shard}: {n_ok} pose mapping(s) resolved, {n_bad} unresolved")
    return 0


if __name__ == "__main__":
    sys.exit(main())
