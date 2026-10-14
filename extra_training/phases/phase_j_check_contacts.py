#!/usr/bin/env python
"""PHASE J -- does each selected antigen chain actually touch the selected antibody?

WHY. phase_j0_select_structures.py picks chains by SEQUENCE, and sequence cannot tell two copies of
an antigen apart. A crystal with several copies can therefore hand the label an antigen that sits
elsewhere in the lattice and never contacts the antibody we folded -- the label would describe a
complex that does not exist, while every sequence check passes.

WHAT IS MEASURED. On the ENTRY coordinates, between the chains phase_j_job_cifs resolves for this
instance: the minimum heavy-atom distance between the antibody chains (roles H, L) and each antigen
chain (roles ag*). An antigen with no atom within --cutoff of the antibody is reported, together with
whether that chain came from SAbDab2's name or from sequence matching -- which is what tells a wrong
copy apart from an entry whose antigen genuinely sits just outside the cutoff.

MEASURED ON THE ENTRY, NOT ON THE WRITTEN JOB CIF. The job cif renames chains A, B, C... in role
order, so reading it back gave distances for the chains the file was written with while the roles
printed were the ones just recomputed -- after a resolution fix the two disagreed and the report
still showed the old distance (9djw: chain changed, 37.28 A unchanged). The entry holds the same
coordinates and keeps the chain names, so it answers the question directly; `--job-cifs` is still
read, to assert the file on disk was written from the chains being reported.

This reads the entry cifs and the job cifs only. It folds nothing and changes nothing.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import phase_j_job_cifs as J          # noqa: E402


def min_dist(st, a_chains, b_chain):
    """Smallest heavy-atom distance between any of `a_chains` and `b_chain`, within the deposited
    coordinates only.

    BRUTE FORCE, NOT NeighborSearch. gemmi's neighbour search returns SYMMETRY IMAGES, and
    `cra.atom.pos` is the original atom's position, not the image's -- so a contact made through a
    crystal neighbour was reported as the distance to its far-away original. That produced 118 A for
    7udh where the true separation is 20.8 A, and a handful of impossible-looking 100+ A readings.
    The question here is whether the antigen we selected touches the antibody we selected IN THE
    ASYMMETRIC UNIT, so symmetry mates are exactly what must NOT count.
    """
    import numpy as np
    def coords(name):
        return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in st[0][name] for a in r
                         if a.element.name != "H"], dtype=float)
    b = coords(b_chain)
    if not len(b):
        return float("nan")
    best = float("inf")
    for nm in a_chains:
        a = coords(nm)
        if not len(a):
            continue
        # chunked, so a 5,000 x 5,000 pair never materialises as one array
        for i in range(0, len(b), 512):
            blk = b[i:i + 512]
            d = np.sqrt(((blk[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min()
            best = min(best, float(d))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--splits", default=f"{ROOT}/SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--job-cifs", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--src-dir", default=f"{ROOT}/SAbDab2_data/rcsb_training_cifs")
    ap.add_argument("--cutoff", type=float, default=5.0, help="heavy-atom contact distance, A")
    ap.add_argument("--out", default=f"{ROOT}/training/prep/antigen_contacts.csv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated pdb ids; diagnose a subset")
    a = ap.parse_args()

    import gemmi
    sel = list(csv.DictReader(open(a.selection)))
    src = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}
    if a.only:
        keep = {x.strip().lower() for x in a.only.split(",") if x.strip()}
        sel = [r for r in sel if r["pdb"].lower() in keep]
    if a.limit:
        sel = sel[:a.limit]
    # role -> src chain as actually written, from write_job_cif's own log
    wrote = {}
    up = f"{ROOT}/training/prep/unplaced_residues.csv"
    if os.path.exists(up):
        for x in csv.DictReader(open(up)):
            wrote.setdefault(x["pdb"], {})[x["role"]] = x["src_chain"]
    rows, bad, stale = [], [], []
    for n, r in enumerate(sel, 1):
        row = src[r["instance"]]
        ent = gemmi.read_structure(os.path.join(a.src_dir, f"{r['pdb']}.cif"))
        ent.setup_entities()
        pairs, _ = J.job_chains(row, J.chain_seqs(ent), ent)
        # the job cif renames chains A, B, C... in the order of `pairs`
        roles = [p[0] for p in pairs]
        srcs = [p[1] for p in pairs]
        names = [chr(ord("A") + i) for i in range(len(pairs))]
        named = J.role_names(row, J.chain_seqs(ent))
        # THE JOB CIF MUST HAVE BEEN WRITTEN FROM THESE CHAINS. write_job_cif records the
        # role -> (src chain, out chain) mapping it used in unplaced_residues.csv; if it disagrees
        # with what we just resolved the file on disk is stale and the report would describe chains
        # that were never written.
        w = wrote.get(r["pdb"])
        if w is not None and w != {role: sc for role, sc in zip(roles, srcs)}:
            stale.append((r["pdb"], w, {role: sc for role, sc in zip(roles, srcs)}))
        ab = [nm for role, nm in zip(roles, srcs) if role in ("H", "L")]
        if not ab:
            bad.append((r["pdb"], "no antibody chain resolved"))
            continue
        for role, nm, sc in zip(roles, names, srcs):
            if not role.startswith("ag"):
                continue
            d = min_dist(ent, ab, sc)
            rows.append({"pdb": r["pdb"], "role": role, "chain": nm, "src_chain": sc,
                         "sabdab_chain": named.get(role, ""),
                         "by_name": int(named.get(role, "") == sc),
                         "min_dist_to_antibody": round(d, 2) if d == d and d < 1e8 else "",
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
    for x in sorted(nc, key=lambda x: -(x["min_dist_to_antibody"] or 0)):
        tag = "by name" if x["by_name"] else f"BY SEQUENCE (sabdab names {x['sabdab_chain'] or '-'})"
        print(f"    {x['pdb']} {x['role']} src chain {x['src_chain']}: "
              f"nearest {x['min_dist_to_antibody']} A -- {tag}")
    print(f"wrote {a.out}")
    for p, why in bad:
        print(f"  SKIPPED {p}: {why}")
    if stale:
        print(f"  STALE job cif for {len(stale)} structure(s) -- rerun phase_j_job_cifs.py")
        for p, w, now in stale[:10]:
            print(f"    {p}: written {w}, resolved {now}")


if __name__ == "__main__":
    main()
