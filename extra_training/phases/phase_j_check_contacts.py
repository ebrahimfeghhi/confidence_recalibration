#!/usr/bin/env python
"""PHASE J -- does each selected antigen chain actually touch the selected antibody?

WHY. phase_j0_select_structures.py picks chains by SEQUENCE, and sequence cannot tell two copies of
an antigen apart. A crystal with several copies can therefore hand the label an antigen that sits
elsewhere in the lattice and never contacts the antibody we folded -- the label would describe a
complex that does not exist, while every sequence check passes.

WHAT IS MEASURED. On the written job cif, the minimum heavy-atom distance between the antibody chains
(roles H, L) and each antigen chain (roles ag*). An antigen with no atom within --cutoff of the
antibody is reported: either the wrong copy was chosen, or the entry genuinely has a non-contacting
antigen, and the two are told apart by looking at whether ANOTHER copy in the entry would contact.

This reads the job cifs and the entry cifs only. It folds nothing and changes nothing.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import phase_j_job_cifs as J          # noqa: E402


def min_dist(st, a_chains, b_chain, cutoff):
    """Smallest heavy-atom distance between any of `a_chains` and `b_chain`, capped at a search box."""
    import gemmi
    ns = gemmi.NeighborSearch(st, max(cutoff, 5.0)).populate()
    best = 1e9
    want = set(a_chains)
    for res in st[0][b_chain]:
        for atom in res:
            if atom.element == gemmi.Element("H"):
                continue
            for m in ns.find_atoms(atom.pos, '\0', radius=cutoff):
                cra = m.to_cra(st[0])
                if cra.chain.name in want and cra.atom.element != gemmi.Element("H"):
                    d = cra.atom.pos.dist(atom.pos)
                    best = min(best, d)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=f"{ROOT}/results/cache_train_structures.csv")
    ap.add_argument("--splits", default=f"{ROOT}/data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--job-cifs", default=f"{ROOT}/train_prep/job_cifs")
    ap.add_argument("--src-dir", default=f"{ROOT}/data/rcsb_training_cifs")
    ap.add_argument("--cutoff", type=float, default=5.0, help="heavy-atom contact distance, A")
    ap.add_argument("--out", default=f"{ROOT}/train_prep/antigen_contacts.csv")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    import gemmi
    sel = list(csv.DictReader(open(a.selection)))
    src = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}
    if a.limit:
        sel = sel[:a.limit]
    rows, bad = [], []
    for n, r in enumerate(sel, 1):
        row = src[r["instance"]]
        ent = gemmi.read_structure(os.path.join(a.src_dir, f"{r['pdb']}.cif"))
        ent.setup_entities()
        pairs, _ = J.job_chains(row, J.chain_seqs(ent))
        # the job cif renames chains A, B, C... in the order of `pairs`
        roles = [p[0] for p in pairs]
        names = [chr(ord("A") + i) for i in range(len(pairs))]
        job = gemmi.read_structure(os.path.join(a.job_cifs, f"{r['pdb']}.cif"))
        job.setup_entities()
        ab = [nm for role, nm in zip(roles, names) if role in ("H", "L")]
        if not ab:
            bad.append((r["pdb"], "no antibody chain in the job cif"))
            continue
        for role, nm in zip(roles, names):
            if not role.startswith("ag"):
                continue
            d = min_dist(job, ab, nm, 12.0)
            rows.append({"pdb": r["pdb"], "role": role, "chain": nm,
                         "min_dist_to_antibody": round(d, 2) if d < 1e8 else "",
                         "contacts": int(d <= a.cutoff)})
        if n % 100 == 0:
            print(f"  {n}/{len(sel)}")
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    nc = [x for x in rows if not x["contacts"]]
    print(f"{len(rows)} antigen chain(s) over {len({x['pdb'] for x in rows})} structures")
    print(f"  contacting the antibody within {a.cutoff} A: {len(rows)-len(nc)}")
    print(f"  NOT contacting: {len(nc)}")
    for x in nc[:15]:
        print(f"    {x['pdb']} {x['role']} chain {x['chain']}: nearest {x['min_dist_to_antibody']} A")
    print(f"wrote {a.out}")
    for p, why in bad:
        print(f"  SKIPPED {p}: {why}")


if __name__ == "__main__":
    main()
