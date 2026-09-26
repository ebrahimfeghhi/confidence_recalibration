#!/usr/bin/env python
"""Read the index files: which poses exist, and which chains of a pose play which role.

SUMMARY
    Three csvs the pipeline keeps re-reading, in one place so they are parsed the same way everywhere.
    There used to be two different load_pool definitions that had drifted apart.

    items_for_interface() is the one that does more than read: it turns an interface plus a list of
    poses into the (pose, antibody chain, antigen chain) triples the rest of the code works on. Every
    step takes ONE CHAIN PAIR PER POSE, never a fixed pair for a whole set -- on a complex with several
    copies of an entity, different poses bind different copies, so a fixed pair compares different
    molecules without failing.

INPUTS
    none -- functions take paths as arguments, defaulting to the locations in paths.py

OUTPUTS
    none -- a library

FUNCTIONS
    load_pool():           pool csv -> {interface: [(seed, sample, dockq_gt, pair_iptm), ...]}
    load_chain_maps():     chain-map csv -> {(pdb, seed, sample): {reference_chain: pose_chain}}
    load_roles():          roles csv -> {"<pdb>_<c1><c2>": (antibody chain, antigen chain)}
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


def _pool_rows(pool_csv, extra=None):
    for src in [pool_csv] + ([extra] if extra else []):
        with open(src) as fh:
            yield from csv.DictReader(fh)


def load_pool(pool_csv=None, extra=None):
    """in: pool csv, and optionally a second csv of predicted-only interfaces.
    out: {interface: [(seed, sample, dockq_gt, pair_iptm), ...]}

    dockq is NaN where the column is empty, which is how a predicted-only interface is recorded: it has
    no reference counterpart, so writing 0.0 would read as a very bad interface rather than an
    unscoreable one.
    """
    pool = collections.defaultdict(list)
    for r in _pool_rows(pool_csv or str(paths.PXM2025_POOL_CSV), extra):
        dq = float(r["dockq"]) if r["dockq"] not in ("", None) else float("nan")
        pool[f"{r['pdb']}_{r['chain1']}{r['chain2']}"].append(
            (r["seed"], int(r["sample"]), dq, float(r["pair_iptm"])))
    return pool


def load_chain_maps(path=None):
    """in: a chain-map csv.  out: {(pdb, seed, sample): {reference_chain: pose_chain}}

    Default is the crystal-frame map. Pass anchor_relabel.py's output to work in the anchor frame.
    A blank mapping means it failed for that pose, and the pose is left out.
    """
    out = {}
    with open(path or f"{paths.PXM2025_DIR}/chain_maps.csv") as fh:
        for r in csv.DictReader(fh):
            if r["chain_mapping"]:
                out[(r["pdb"], r["seed"], int(r["sample"]))] = json.loads(r["chain_mapping"])
    return out


def load_roles(path=None):
    """in: a roles csv.  out: {"<pdb>_<c1><c2>": (antibody chain, antigen chain)}

    Keyed on chain_1/chain_2 to match the pool csv's interface key, not on the `interface` column,
    which already contains the pdb -- using it yields "8w5d_8w5d_A_B" and every lookup misses.
    """
    out = {}
    with open(path or str(paths.PXM2025_CHAIN_ROLES)) as fh:
        for r in csv.DictReader(fh):
            out[f"{r['pdb']}_{r['chain_1']}{r['chain_2']}"] = (r["ab_chain"], r["ag_chain"])
    return out


def items_for_interface(iface, seeds_samples, maps, roles):
    """in: an interface, its (seed, sample) list, a chain map and roles.
    out: ([(pose path, antibody chain, antigen chain)], poses dropped for a missing mapping)

    A pose whose mapping is missing, or does not cover both chains, is dropped and reported rather than
    falling back to the reference letters -- that fallback is the bug this plumbing exists to prevent.
    """
    pdb = iface.split("_")[0]
    ab_ref, ag_ref = roles[iface]
    items, dropped = [], []
    for seed, sample in seeds_samples:
        m = maps.get((pdb, str(seed), int(sample)))
        if not m or ab_ref not in m or ag_ref not in m:
            dropped.append((seed, sample))
            continue
        items.append((paths.pool_pose(pdb, seed, sample), m[ab_ref], m[ag_ref]))
    return items, dropped


def pair_iptm_at(pose, ab, ag):
    """in: a pose cif path and two of ITS chain names.  out: chain_pair_iptm[ab][ag] from its json

    The json's matrix rows are unlabelled and follow the cif's chain order (verified 800/800 against
    an independent code path, see fetch_spurious_interface_iptm.py). Read at the chains actually being
    compared: the pool csv's pair_iptm was read at the chains a CRYSTAL mapping chose, which on
    duplicate-copy complexes is a different copy for ~25% of poses.
    """
    import gemmi
    d, base = os.path.split(pose)
    js = os.path.join(d, base.replace("_sample_", "_summary_confidence_sample_").replace(".cif", ".json"))
    idx = {c.name: n for n, c in enumerate(gemmi.read_structure(pose)[0])}
    M = json.load(open(js))["chain_pair_iptm"]
    if len(M) != len(idx):
        raise ValueError(f"{pose}: {len(M)} ipTM rows but {len(idx)} chains; order cannot be trusted")
    return float(M[idx[ab]][idx[ag]])
