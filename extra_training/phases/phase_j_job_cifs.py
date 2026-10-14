#!/usr/bin/env python
"""PHASE J -- write one mmCIF per training JOB: the antibody and its polypeptide antigen, nothing else.

WHY. The training cache pairs a POSE with a LABEL, and the two must describe the same system. Our
folds are the antibody plus its antigen -- median 644 tokens. Protenix's training dataset, handed a
whole RCSB entry, builds the whole deposited assembly: median 1,095 tokens, up to 12,879, and more
than twice the job for 205 of 500 structures. A label built from the assembly cannot be compared to a
pose of the job: different atoms, different ordering, different molecules.

Protenix's own answer to that mismatch is to CROP the assembly to 768 tokens, which is what FT3 does.
We do not take it, because the head is deployed on whole antibody-antigen complexes and the
validation cache it will be measured against is uncropped -- training on crops and deploying on whole
complexes is the distribution gap this project exists to close, and we are not short of memory at
644 tokens the way they are at PDB scale.

So instead the system is cut down to the job before the bioassembly is built: keep the auth chains
SAbDab2 names as Hchain, Lchain and the polypeptide entries of agchains, drop everything else, and
write a single-assembly cif that Protenix's parser accepts.

WHAT IS DROPPED, deliberately: other copies of the same complex in the asymmetric unit, unrelated
chains, and every non-polypeptide antigen (sugar, hapten, ion) -- the eligibility filter already
requires a polypeptide antigen, and the fold does not include the others either.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import phase_a_define_train_set as A          # noqa: E402

POLY = A.POLY


def job_chains(row):
    """The auth chain names the fold actually contains, in H, L, antigen order."""
    out = []
    for k in ("Hchain", "Lchain"):
        v = (row[k] or "").strip()
        if v:
            out.append(v)
    names = [c.strip() for c in str(row["agchains"]).split("/")]
    types = str(row["agtypes"]).split("/")
    if len(names) != len(types):
        raise SystemExit(f"STOP: {row['PDB_ID']} has {len(types)} antigen types but {len(names)} "
                         "antigen chain names; ag<i> is positional in both.")
    out += [c for c, t in zip(names, types) if c and t in POLY]
    return out


def write_job_cif(src_cif, keep, out_cif):
    """Keep `keep` auth chains and write a one-assembly mmCIF. Returns (n_chains, n_residues)."""
    import gemmi
    st = gemmi.read_structure(src_cif)
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    # POLYMER ONLY. The fold's job json carries sequences and nothing else, so the pose has no ions,
    # no ligands and no waters. Left in, they appear in the label as atoms the pose cannot have --
    # a first test kept 8wnp's ion chains I and J and the indices csv duly listed intra_ions rows.
    st.remove_ligands_and_waters()
    # ONE MODEL. An NMR entry carries many; the fold predicts one structure, and a label built from
    # several superposed models would not correspond to it.
    while len(st) > 1:
        del st[1]
    have = {ch.name for ch in st[0]}
    missing = [c for c in keep if c not in have]
    if missing:
        return None, f"chains {missing} absent from the cif (present: {sorted(have)})"
    for ch in [c.name for c in st[0]]:
        if ch not in keep:
            st[0].remove_chain(ch)
    st.setup_entities()
    # A SINGLE ASSEMBLY naming exactly the kept chains. Protenix's parser reads
    # pdbx_struct_assembly first (parser.py:170) and raises without it; SAbDab2's own trimmed cifs
    # lack the block, which is why they cannot be used here.
    # gemmi's Assembly.generators and Gen.operators have no setters, so the containers are mutated
    # in place; Gen.chains is a plain list property and must be ASSIGNED (appending to the value it
    # returns is appending to a copy).
    asm = gemmi.Assembly("1")
    gen = gemmi.Assembly.Gen()
    gen.chains = [ch.name for ch in st[0]]
    gen.operators.append(gemmi.Assembly.Operator())
    asm.generators.append(gen)
    asm.author_determined = True
    asm.oligomeric_details = f"{len(st[0])}-mer"
    st.assemblies.clear()
    st.assemblies.append(asm)
    nres = sum(len(ch) for ch in st[0])
    os.makedirs(os.path.dirname(out_cif), exist_ok=True)
    st.make_mmcif_document().write_file(out_cif)
    return (len(st[0]), nres), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=f"{ROOT}/results/cache_train_structures.csv")
    ap.add_argument("--splits", default=f"{ROOT}/data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--src-dir", default=f"{ROOT}/data/rcsb_training_cifs")
    ap.add_argument("--out-dir", default=f"{ROOT}/train_prep/job_cifs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--skipped", default=f"{ROOT}/train_prep/skipped_job_cifs.csv",
                    help="structures whose chains cannot be resolved, with the reason. A failure "
                         "not already listed here is a STOP.")
    a = ap.parse_args()

    sel = list(csv.DictReader(open(a.selection)))
    src = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}
    if a.limit:
        sel = sel[:a.limit]
    ok, bad = 0, []
    for r in sel:
        row = src[r["instance"]]
        keep = job_chains(row)
        out = os.path.join(a.out_dir, f"{r['pdb']}.cif")
        info, why = write_job_cif(os.path.join(a.src_dir, f"{r['pdb']}.cif"), keep, out)
        if why:
            bad.append((r["pdb"], r["instance"], why))
            continue
        ok += 1
    print(f"wrote {ok} job cif(s) to {a.out_dir}")
    # SKIP WITH A RECORDED REASON. SAbDab2 names some antibody chains with a trailing index --
    # E1/E2, H1/H2, A1/C2 -- where the cif has a single auth chain E, H, A. Those are single-chain
    # constructs whose VH and VL SAbDab2 splits into segments, so the crystal has one chain where
    # our job json has two sequences. Resolving that mapping is possible but it is a different
    # problem from building a cache, and getting it subtly wrong puts the wrong atoms in the label.
    known = {}
    if os.path.exists(a.skipped):
        known = {r["pdb"]: r.get("reason", "") for r in csv.DictReader(open(a.skipped))}
    new = [(p, i, w) for p, i, w in bad if p not in known]
    if new:
        for p, i, w in new[:10]:
            print(f"  FAILED {p} ({i}): {w}")
        raise SystemExit(f"STOP: {len(new)} job cif(s) failed and are not in {a.skipped}")
    if bad:
        print(f"{len(bad)} structure(s) skipped with a recorded reason")


if __name__ == "__main__":
    main()
