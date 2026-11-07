#!/usr/bin/env python
"""Do the two chains of an interface actually TOUCH? Measured, not inferred.

WHY THIS EXISTS. scorer.py skips a pose whose interface OST did not report, with the comment "this
pose has no such interface -- a real condition, not an error", and phase_i_check counts those as
"scored, no contact at that interface, which is a result not a fault". Both statements are
INFERENCES from OST's silence. Asked how we know the chains do not touch, the honest answer was
that we did not: 550 interface-samples carried that label on no evidence beyond the absence of a
DockQ entry.

The pattern made the inference doubtful. Those 550 are eleven interfaces, each missing in all ten
seeds and all five poses -- 11 x 50. A pose-by-pose geometric accident does not land on every pose
of one interface and no pose of its three siblings. Something about the REFERENCE decides it, so
the reference is where to look.

WHAT IT MEASURES. Heavy-atom contacts within --cutoff between the two chains, in the reference
mmCIF, counted directly from atom_site rather than through OST. Hydrogens are excluded because
references rarely carry them and a cutoff tuned on heavy atoms means nothing with them included.

IN BOTH CHAIN NAMESPACES, ALWAYS. An mmCIF carries label_asym_id and auth_asym_id, and they
disagree often enough to have already cost this project a scorer retry path that could never fire
(_identity_chain_map built auth-name pairs for an OST that works in label space). A contact count
is meaningless without saying which namespace its chain letters came from, so this prints both and
says whether they agree.

CONTROLS ARE NOT OPTIONAL. --control adds the interfaces of the same structures that DID score, so
the measurement is read against pairs known to be real rather than against an assumption about
what a contacting interface looks like.

Usage:
  python -I run_logging/interface_contacts.py --pairs 3rhw_DK 1qfw_AF --control
  python -I run_logging/interface_contacts.py --pairs-file /tmp/pairs.txt --cutoff 5.0
"""
import argparse
import os
import sys

import gemmi
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--refs", default="training/refs")
ap.add_argument("--pool", default="training/fold/pool.csv")
ap.add_argument("--pairs", nargs="*", default=[], help="<pdb>_<C1><C2>")
ap.add_argument("--pairs-file", default="")
ap.add_argument("--cutoff", type=float, default=5.0)
ap.add_argument("--control", action="store_true",
                help="also measure every OTHER pool interface of the same structures")
a = ap.parse_args()

pairs = list(a.pairs)
if a.pairs_file:
    pairs += [ln.split()[0] for ln in open(a.pairs_file) if ln.strip()
              and not ln.startswith("#")]
if not pairs:
    raise SystemExit("nothing to measure: pass --pairs or --pairs-file")


def chains_of(path):
    """out: {"label": {name: Nx3 heavy-atom coords}, "auth": {...}}

    Read straight from atom_site. gemmi's Structure gives one namespace depending on how it was
    set up; the point here is to see both without that choice being made for us.
    """
    doc = gemmi.cif.read(str(path))
    blk = doc.sole_block()
    cols = ["label_asym_id", "auth_asym_id", "Cartn_x", "Cartn_y", "Cartn_z", "type_symbol"]
    tab = blk.find("_atom_site.", cols)
    out = {"label": {}, "auth": {}}
    for r in tab:
        if r[5].strip('"') == "H":
            continue
        xyz = (float(r[2]), float(r[3]), float(r[4]))
        out["label"].setdefault(r[0], []).append(xyz)
        out["auth"].setdefault(r[1], []).append(xyz)
    return {k: {c: np.asarray(v) for c, v in d.items()} for k, d in out.items()}


def contacts(p, q, cutoff):
    """out: (n pairs within cutoff, min distance) -- brute force, chains are a few thousand atoms"""
    if p is None or q is None or not len(p) or not len(q):
        return None, None
    d = np.linalg.norm(p[:, None, :] - q[None, :, :], axis=-1)
    return int((d <= cutoff).sum()), float(d.min())


if a.control:
    import csv
    want = {p.split("_")[0] for p in pairs}
    extra = []
    for r in csv.DictReader(open(a.pool)):
        if r["pdb"] in want:
            nm = f"{r['pdb']}_{r['chain1']}{r['chain2']}"
            if nm not in pairs:
                extra.append(nm)
    pairs = pairs + sorted(set(extra))

cache = {}
print(f"heavy-atom contacts within {a.cutoff} A, from the reference mmCIF's atom_site")
print(f"{'interface':12s} {'label-space':>22s} {'auth-space':>22s}  verdict")
rows = []
for nm in pairs:
    pdb, ch = nm.split("_")[0], nm.split("_")[-1]
    c1, c2 = ch[0], ch[1]
    path = os.path.join(a.refs, f"{pdb}.cif")
    if not os.path.exists(path):
        print(f"{nm:12s} {'no reference cif':>22s}")
        continue
    if pdb not in cache:
        cache[pdb] = chains_of(path)
    d = cache[pdb]
    cell = {}
    for space in ("label", "auth"):
        n, mn = contacts(d[space].get(c1), d[space].get(c2), a.cutoff)
        cell[space] = (n, mn)
    def fmt(t):
        n, mn = t
        if n is None:
            return "chain absent"
        return f"{n:5d} pairs, min {mn:5.2f}"
    ln, _ = cell["label"]
    an, _ = cell["auth"]
    best = max(x for x in (ln or 0, an or 0))
    verdict = "TOUCHING" if best else ("no contact" if (ln is not None or an is not None)
                                       else "chain absent in both")
    print(f"{nm:12s} {fmt(cell['label']):>22s} {fmt(cell['auth']):>22s}  {verdict}")
    rows.append((nm, best))

print()
touch = [nm for nm, b in rows if b]
print(f"{len(touch)} of {len(rows)} measured interface(s) have at least one heavy-atom contact")
