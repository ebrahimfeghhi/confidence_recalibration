#!/usr/bin/env python
"""Read the index files: which poses exist, and which chains of a pose play which role.

SUMMARY
    The csvs the pipeline keeps re-reading, in one place so they are parsed the same way everywhere.

    items_for_interface() is the one that does more than read: it turns an interface plus a list of
    poses into the (pose, antibody chain, antigen chain) triples the rest of the code works on. Every
    step takes ONE CHAIN PAIR PER POSE, never a fixed pair for a whole set -- on a complex with several
    copies of an entity, different poses bind different copies, so a fixed pair compares different
    molecules without failing.

    NO DEFAULT PATHS for maps and roles. They used to default to the CRYSTAL files, and a caller that
    forgot to pass the anchor frame silently selected in the crystal's. Selection passes the anchor
    frame explicitly; only evaluation reads the crystal-scored pool (load_pool).

INPUTS
    none -- functions take paths as arguments

OUTPUTS
    none -- a library

EXAMPLE
    import pool_index
    maps = pool_index.load_chain_maps("exp/chain_frame/chain_maps_anchor.csv")
    roles = pool_index.load_roles("exp/interfaces.csv")
    items, dropped = pool_index.items_for_interface("8ror_BA", keys, maps, roles)
    pool_index.pair_iptm_at(pose, "B", "A")

FUNCTIONS
    load_pool():           the crystal-scored pool csv -> {interface: [(seed, sample, dockq, pair_iptm)]}
                           -- EVALUATION ONLY
    load_chain_maps():     chain-map csv -> {(pdb, seed, sample): {anchor chain: pose chain}}
    load_roles():          interfaces csv -> {"<pdb>_<c1><c2>": (antibody chain, antigen chain)}
    items_for_interface(): an interface and its poses -> [(pose path, antibody chain, antigen chain)]
    pair_iptm_at():        a pose and two of its chains -> the model's chain_pair_iptm for that pair
"""
import collections
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402


def load_pool(pool_csv=None):
    """in: the crystal-scored pool csv.  out: {interface: [(seed, sample, dockq, pair_iptm), ...]}

    EVALUATION ONLY: dockq is against the crystal, and pair_iptm was read at crystal-matched chains, so
    each (confidence, correctness) pair is about one reference interface.
    """
    pool = collections.defaultdict(list)
    with open(pool_csv or str(paths.PXM2025_POOL_CSV)) as fh:
        for r in csv.DictReader(fh):
            pool[f"{r['pdb']}_{r['chain1']}{r['chain2']}"].append(
                (r["seed"], int(r["sample"]), float(r["dockq"]), float(r["pair_iptm"])))
    return pool


def load_chain_maps(path):
    """in: a chain-map csv.  out: {(pdb, seed, sample): {anchor chain: pose chain}}

    A blank mapping means it failed for that pose, and the pose is left out.
    """
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["chain_mapping"]:
                out[(r["pdb"], r["seed"], int(r["sample"]))] = json.loads(r["chain_mapping"])
    return out


def load_roles(path):
    """in: an interfaces csv.  out: {"<pdb>_<c1><c2>": (antibody chain, antigen chain)}"""
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            out[f"{r['pdb']}_{r['chain_1']}{r['chain_2']}"] = (r["ab_chain"], r["ag_chain"])
    return out


def items_for_interface(iface, seeds_samples, maps, roles, free_dir=None):
    """in: an interface, its (seed, sample) list, a chain map, roles, the free pool (default pxm2025's).
    out: ([(pose path, antibody chain, antigen chain)], poses dropped for a missing mapping)

    Input order is kept. A pose whose mapping is missing, or does not cover both chains, is dropped and
    reported rather than falling back to the anchor letters.
    """
    pdb = iface.split("_")[0]
    ab_ref, ag_ref = roles[iface]
    items, dropped = [], []
    for seed, sample in seeds_samples:
        m = maps.get((pdb, str(seed), int(sample)))
        if not m or ab_ref not in m or ag_ref not in m:
            dropped.append((seed, sample))
            continue
        items.append((paths.pool_pose(pdb, seed, sample, free_dir), m[ab_ref], m[ag_ref]))
    return items, dropped


def pair_iptm_at(pose, ab, ag):
    """in: a pose cif path and two of ITS chain names.  out: chain_pair_iptm[ab][ag] from its json

    The json's matrix rows are unlabelled and follow the cif's chain order (verified 800/800 against an
    independent code path). Read at the chains actually being compared.
    """
    import gemmi
    d, base = os.path.split(pose)
    js = os.path.join(d, base.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json"))
    idx = {c.name: n for n, c in enumerate(gemmi.read_structure(pose)[0])}
    M = json.load(open(js))["chain_pair_iptm"]
    if len(M) != len(idx):
        raise ValueError(f"{pose}: {len(M)} ipTM rows but {len(idx)} chains; order cannot be trusted")
    return float(M[idx[ab]][idx[ag]])
