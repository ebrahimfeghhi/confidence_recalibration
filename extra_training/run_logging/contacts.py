#!/usr/bin/env python
"""Heavy-atom contact between two chains of a reference, measured off atom_site.

WHY A MODULE AND NOT A ONE-OFF. "These chains do not touch" was an inference everywhere it
appeared -- the scorer skipped a pose whose interface OST did not report, and the verifier counted
that as a result rather than a gap, on no evidence but OST's silence. It happened to be true for
all 550 interface-samples that carried the label, but nothing checked, and the same assumption is
made in three places. One measurement, imported by all of them, is what keeps the next one honest.

NAMESPACE. Chain letters here are label_asym_id, which is what pool.csv, the scorer and OST all
use. auth names are absent from most of these references and collapse copies that label names keep
apart (3wfc_BC: 194 contacts by label, 1,906 by auth), so measuring in auth space answers a
different question. Pass space="auth" only to compare the two deliberately.

CUTOFF. 5.0 A between heavy atoms. Calibrated against what the scorer accepts rather than chosen:
6i3z_BC is scored by OST on 5 contact pairs whose closest approach is 4.27 A, and the eleven pairs
with no score are 6.33 A apart at nearest. Any cutoff in between separates them identically.
"""
import os

import gemmi
import numpy as np

CUTOFF = 5.0
_CACHE = {}


def chain_coords(path, space="label"):
    """out: {chain name: Nx3 heavy-atom coordinates}, cached per (path, space)

    Read from atom_site rather than through gemmi.Structure, because the Structure's chain names
    depend on how it was set up and the whole point is to control which namespace is used.
    """
    key = (os.path.abspath(str(path)), space)
    if key in _CACHE:
        return _CACHE[key]
    col = "label_asym_id" if space == "label" else "auth_asym_id"
    blk = gemmi.cif.read(str(path)).sole_block()
    tab = blk.find("_atom_site.", [col, "Cartn_x", "Cartn_y", "Cartn_z", "type_symbol"])
    acc = {}
    for r in tab:
        if r[4].strip('"') == "H":
            continue
        acc.setdefault(r[0], []).append((float(r[1]), float(r[2]), float(r[3])))
    out = {c: np.asarray(v) for c, v in acc.items()}
    _CACHE[key] = out
    return out


def contacts(path, c1, c2, cutoff=CUTOFF, space="label"):
    """in: a reference cif and two chain names.  out: (n atom pairs within cutoff, min distance)

    Returns (None, None) when either chain is absent, which is a different statement from zero
    contacts and must not be reported as "they do not touch".
    """
    ch = chain_coords(path, space)
    p, q = ch.get(c1), ch.get(c2)
    if p is None or q is None or not len(p) or not len(q):
        return None, None
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    return int((d <= cutoff).sum()), float(d.min())


def touching(path, c1, c2, cutoff=CUTOFF):
    """out: True / False / None (cannot tell: no such file, or a chain is absent)"""
    if not os.path.exists(str(path)):
        return None
    n, _ = contacts(path, c1, c2, cutoff)
    return None if n is None else n > 0
