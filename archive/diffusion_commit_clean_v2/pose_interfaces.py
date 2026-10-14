#!/usr/bin/env python
"""Decide which antibody-antigen chain pairs count as interfaces, from contacts in the poses alone.

SUMMARY
    Runs after anchor_relabel.py. Later steps sort poses per interface, which requires knowing which
    pairs to sort for; the input cannot say (9k94 has 5 antibody and 5 antigen chains, 25 possible
    pairings). The rule:

        an antibody-antigen pair of ANCHOR chains is an interface if those two chains come within
        5.0 A in AT LEAST ONE pose

    The cutoff is the heavy-atom contact distance DockQ uses. There is deliberately no "must appear in
    k poses" rule, which would be a knob. Which chains are antibody and antigen is known input
    (anchor_relabel.py); contacts come from the poses, computed by anchor_relabel.py in each pose's own
    letters and translated here through the anchor mapping. Every interface is named in anchor
    letters, "<pdb>_<antibody chain><antigen chain>".

INPUTS
    --anchor-dir: anchor_relabel.py's output directory. Required.
    --out:        the interfaces csv to write. Required.

OUTPUTS
    <out>: pdb, chain_1, chain_2, ab_chain, ag_chain, poses_in_contact -- one row per interface, with
           chain_1 = ab_chain and chain_2 = ag_chain. The roles columns make it readable by
           pool_index.load_roles(), which every later step uses.

EXAMPLE
    pose_interfaces.py --anchor-dir exp/chain_frame --out exp/interfaces.csv

FUNCTIONS
    main(): translate every pose's contacts into anchor letters and count them
"""
import argparse
import collections
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pool_index  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor-dir", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    maps = pool_index.load_chain_maps(f"{A.anchor_dir}/chain_maps_anchor.csv")
    role = {(r["pdb"], r["chain"]): r["role"]
            for r in csv.DictReader(open(f"{A.anchor_dir}/anchor_chains.csv"))}
    seen, unmapped = collections.Counter(), 0
    for r in csv.DictReader(open(f"{A.anchor_dir}/pose_contacts.csv")):
        key = (r["pdb"], r["seed"], int(r["sample"]))
        m = maps.get(key)
        inv = {v: k for k, v in m.items()} if m else {}
        a, g = inv.get(r["ab_chain"]), inv.get(r["ag_chain"])
        if a is None or g is None:
            unmapped += 1
            continue
        # a mapping pairs like with like, so an antibody chain lands on an anchor antibody chain
        assert role[(r["pdb"], a)] == "antibody" and role[(r["pdb"], g)] == "antigen", \
            f"{key}: mapping sent {r['ab_chain']}-{r['ag_chain']} to {a}-{g}"
        seen[(r["pdb"], a, g)] += 1
    if any(len(a) != 1 or len(g) != 1 for _, a, g in seen):
        raise SystemExit("multi-letter chain names: '<pdb>_<ab><ag>' would be ambiguous")

    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    with open(A.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["pdb", "chain_1", "chain_2", "ab_chain", "ag_chain", "poses_in_contact"])
        for (p, a, g), n in sorted(seen.items()):
            w.writerow([p, a, g, a, g, n])
    print(f"  {len(seen)} interfaces over {len({p for p, _, _ in seen})} complexes "
          f"({unmapped} pose contacts on an unmapped chain skipped)  -> {A.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
