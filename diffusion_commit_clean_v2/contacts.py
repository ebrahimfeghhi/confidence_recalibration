#!/usr/bin/env python
"""Decide whether two chains of one pose are in contact.

SUMMARY
    The cutoff is DockQ's own heavy-atom contact distance, so "has an interface" means the same thing
    here as it does to the scorer. At 4.5 A this dropped ~1,110 poses DockQ scores as real interfaces,
    162 of them above 0.80, because a ring assembly's second interface is a ~4.7 A glance.

    Callers ask only whether an interface exists: the anchor step lists each pose's antibody-antigen
    pairs in contact, and the DockQ step drops poses with no contact before building a matrix.

INPUTS
    none -- functions take a pose path and two chain names

OUTPUTS
    none -- a library

EXAMPLE
    import contacts
    contacts.has_contact(pose, "A", "C")                      # True / False
    contacts.contacting(pose, [("A", "C"), ("A", "D")])       # the pairs in contact

FUNCTIONS
    pose_chains():  one pose cif -> {chain: (coordinates, residue index)}
    has_contact():  one pose and two chains -> whether any heavy atoms are within the cutoff
    contacting():   one pose and many chain pairs -> the pairs in contact, parsing the pose once
"""
import collections
import sys

import gemmi
import numpy as np
from scipy.spatial import cKDTree

# DockQ's heavy-atom contact distance
CONTACT_CUTOFF = 5.0


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


def has_contact(path, ab, ag, cutoff=CONTACT_CUTOFF):
    """in: a pose, its antibody chain, its antigen chain.
    out: True if any heavy atom of one is within cutoff of the other

    Stops at the first pair found. Both callers only ask whether an interface exists at all: one drops
    poses with no contacts before building a similarity matrix, the other decides whether a chain pair
    counts as an interface.
    """
    D = pose_chains(path)
    for c in (ab, ag):
        if c not in D:
            raise KeyError(f"{path}: chain {c} absent (have {sorted(D)})")
    XA, _ = D[ab]
    XB, _ = D[ag]
    return bool(cKDTree(XA).query_ball_point(XB, r=cutoff, return_length=True).any())


def contacting(path, pairs, cutoff=CONTACT_CUTOFF):
    """in: a pose and [(chain, chain)].  out: the pairs with any heavy atoms within cutoff

    One parse and one tree per chain for all pairs; has_contact() re-reads the cif on every call.
    """
    D = pose_chains(path)
    trees = {}
    out = []
    for a, b in pairs:
        if a not in D or b not in D:
            raise KeyError(f"{path}: chain {a if a not in D else b} absent (have {sorted(D)})")
        if a not in trees:
            trees[a] = cKDTree(D[a][0])
        if trees[a].query_ball_point(D[b][0], r=cutoff, return_length=True).any():
            out.append((a, b))
    return out
