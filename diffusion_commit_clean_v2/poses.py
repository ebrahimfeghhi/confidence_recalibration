#!/usr/bin/env python
"""Shared lookups: which poses exist, which chains to compare, and the contacts between them.

SUMMARY
    Imported by the other steps rather than run. Four things they all need:

      the pool index    what poses exist and what each scored
      the chain mapping which chains of a pose play which role, per pose
      the roles         which chain is the antibody and which the antigen, per interface
      contacts          which antigen residues the antibody touches in one pose

    Every function takes ONE CHAIN PAIR PER POSE, never a fixed pair for a whole set. On a complex with
    several copies of an entity, different poses bind different copies, so a fixed pair silently
    compares different molecules.

INPUTS
    none -- functions take their paths as arguments, defaulting to the locations in paths.py

OUTPUTS
    none -- a library

FUNCTIONS
    load_pool():           pool csv -> {interface: [(seed, sample, dockq_gt, pair_iptm), ...]}
    dockq_by_pose():       the same, keyed for lookup -> {interface: {(seed, sample): dockq_gt}}
    load_chain_maps():     chain-map csv -> {(pdb, seed, sample): {reference_chain: pose_chain}}
    load_roles():          roles csv -> {"<pdb>_<c1><c2>": (antibody chain, antigen chain)}
    items_for_interface(): an interface and its poses -> [(pose path, antibody chain, antigen chain)]
    pose_chains():         one pose cif -> {chain: (coordinates, residue index)}
    epitope_set():         one pose and one chain pair -> the antigen residues in contact
"""
import collections
import csv
import json
import os
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths  # noqa: E402


# DockQ's own heavy-atom contact distance. Used for "does this pose have an interface at all", so it
# must match the scorer: at 4.5 A this dropped ~1,110 poses DockQ scores as real interfaces, 162 of
# them above 0.80, because a ring assembly's second interface is a ~4.7 A glance.
CONTACT_CUTOFF = 5.0


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


def dockq_by_pose(pool_csv=None, extra=None):
    """in: same as load_pool.  out: {interface: {(seed, sample): dockq_gt}} for lookup."""
    out = collections.defaultdict(dict)
    for iface, rows in load_pool(pool_csv, extra).items():
        for s, x, dq, _ in rows:
            out[iface][(s, x)] = dq
    return out


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


def pose_chains(path):
    """in: a pose cif.  out: {label_asym_id: (coordinates, residue index)} for polymer atoms

    Reads label_asym_id explicitly rather than trusting gemmi's chain naming: the two disagree on
    several complexes, and a lookup built on the wrong one compares the wrong chains without failing.
    """
    block = gemmi.cif.read(path).sole_block()
    cols = [list(block.find_values("_atom_site." + c)) for c in
            ("label_asym_id", "label_seq_id", "Cartn_x", "Cartn_y", "Cartn_z", "group_PDB")]
    D = collections.defaultdict(lambda: ([], []))
    for a, s, x, y, z, g in zip(*cols):
        if g != "ATOM":
            continue
        D[a][0].append([float(x), float(y), float(z)])
        D[a][1].append(s)
    if not D:
        raise ValueError(f"{path}: no polymer atoms parsed")
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in D.items()}


def epitope_set(path, ab, ag, cutoff=CONTACT_CUTOFF):
    """in: a pose, its antibody chain, its antigen chain.
    out: the antigen residues with a heavy atom within cutoff of the antibody
    """
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, _ = D[ab]
    XB, IB = D[ag]
    hit = {j for js in cKDTree(XA).query_ball_tree(cKDTree(XB), r=cutoff) for j in js}
    return {IB[j] for j in hit}
