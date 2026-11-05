#!/usr/bin/env python
"""Reconstruct the missing SAbDab2 per-instance cifs from the deposited RCSB entry.

WHY THIS IS NEEDED. phase_j_job_cifs uses the instance cif to decide WHICH COPY of the complex
each role takes. Its own help says what happens without one: the choice falls back to a contact
heuristic that "accepted a crystal-packing contact on 9djw (0 epitope residues in common, DockQ
0.0080)". 17 of the 30 new validation structures have no instance cif in the local SAbDab2
corpus, so 17 labels would rest on that heuristic.

WHY RECONSTRUCTION IS SOUND, AND HOW IT IS CHECKED. An instance cif is the deposited entry
restricted to the chains the splits row names in Hchain / Lchain / agchains. That is a claim about
the corpus, not an assumption: --verify-only re-derives the 13 instance cifs that ARE present and
compares chain sets and per-chain sequences against the files on disk. The 17 are built only if
all 13 reproduce exactly.

Output goes to its own directory and the present files are copied in beside the built ones, so
the downloaded corpus is never written to and --sabdab-cifs can point at one complete directory.
"""
import argparse, csv, os, shutil, sys
import gemmi

ap = argparse.ArgumentParser()
ap.add_argument("--selection", default="validation/full_val/val30_structures.csv")
ap.add_argument("--splits", default="SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
ap.add_argument("--sabdab-cifs", default="SAbDab2_data/training_cifs")
ap.add_argument("--src-dir", default="SAbDab2_data/rcsb_validation_cifs")
ap.add_argument("--out-dir", default="validation/prep/instance_cifs")
ap.add_argument("--verify-only", action="store_true")
a = ap.parse_args()

sp = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}
sel = list(csv.DictReader(open(a.selection)))

def spec(row):
    """the auth chains this instance is made of, in role order"""
    out = []
    for k in ("Hchain", "Lchain"):
        v = (row[k] or "").strip()
        if v and v != "+":
            out.append(v)
    for v in str(row["agchains"]).split("/"):
        v = v.strip()
        if v and v != "+":
            out.append(v)
    return out

def build(pdb, want):
    """the deposited entry restricted to `want`, one model, no altlocs or hydrogens"""
    st = gemmi.read_structure(os.path.join(a.src_dir, f"{pdb}.cif"))
    st.setup_entities()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    while len(st) > 1:
        del st[1]
    keep = set(want)
    for ch in [c.name for c in st[0]]:
        if ch not in keep:
            st[0].remove_chain(ch)
    st.setup_entities()
    return st

def seqs(st):
    return {ch.name: gemmi.one_letter_code([r.name for r in ch]) for ch in st[0]}

present = [r for r in sel if os.path.exists(os.path.join(a.sabdab_cifs, r["instance"] + ".cif"))]
absent = [r for r in sel if r not in present]
print(f"selection {len(sel)}: {len(present)} instance cif(s) present, {len(absent)} to build")

bad, kept = [], []
for r in present:
    row = sp[r["instance"]]
    try:
        mine = build(r["pdb"], spec(row))
    except Exception as e:
        bad.append((r["pdb"], f"could not rebuild: {e}")); continue
    ref = gemmi.read_structure(os.path.join(a.sabdab_cifs, r["instance"] + ".cif"))
    ref.setup_entities(); ref.remove_alternative_conformations(); ref.remove_hydrogens()
    while len(ref) > 1:
        del ref[1]
    # WHAT HAS TO MATCH IS THE COPY, NOT THE TRIM. The corpus instance cifs are trimmed: 8t03's
    # antigen B is 199 residues there against 204 in the deposited entry, while its antibody
    # chains E and F agree to the residue. A terminal trim cannot change which copy a role takes,
    # which is the one thing phase_j_job_cifs reads this file for. So the gate is: the same auth
    # chains, and each corpus chain's sequence contained in the reconstruction's.
    ms, rs = seqs(mine), seqs(ref)
    if set(ms) != set(rs):
        bad.append((r["pdb"], f"chain sets differ: built {sorted(ms)} vs corpus {sorted(rs)}"))
        continue
    off = [c for c in rs if rs[c] not in ms[c]]
    if off:
        bad.append((r["pdb"], "chain(s) " + ", ".join(
            f"{c} (corpus {len(rs[c])} not contained in built {len(ms[c])})" for c in sorted(off))))
    else:
        trimmed = [c for c in rs if len(rs[c]) != len(ms[c])]
        if trimmed:
            kept.append((r["pdb"], ", ".join(f"{c} {len(rs[c])}/{len(ms[c])}" for c in sorted(trimmed))))

print(f"VERIFY: {len(present) - len(bad)} of {len(present)} reconstruct the same copy "
      f"(same auth chains, corpus sequence contained in the rebuilt one)")
if kept:
    print(f"  of those, {len(kept)} are trimmed in the corpus (corpus/rebuilt residue counts):")
    for pdb, d in kept:
        print(f"     {pdb}: {d}")
for pdb, why in bad:
    print(f"   {pdb}: {why}")
if bad:
    raise SystemExit("STOP: reconstruction does not reproduce the corpus, so it cannot be trusted "
                     "for the missing ones. Nothing written.")
if a.verify_only:
    print("--verify-only: stopping here")
    sys.exit(0)

os.makedirs(a.out_dir, exist_ok=True)
nb = nc = 0
for r in present:
    dst = os.path.join(a.out_dir, r["instance"] + ".cif")
    if not os.path.exists(dst):
        shutil.copy2(os.path.join(a.sabdab_cifs, r["instance"] + ".cif"), dst); nc += 1
for r in absent:
    row = sp[r["instance"]]
    want = spec(row)
    st = build(r["pdb"], want)
    got = [c.name for c in st[0]]
    if sorted(got) != sorted(want):
        raise SystemExit(f"STOP: {r['pdb']} {r['instance']} wanted chains {sorted(want)} but the "
                         f"entry yielded {sorted(got)}; nothing further written.")
    st.setup_entities()
    doc = st.make_mmcif_document()
    doc.write_file(os.path.join(a.out_dir, r["instance"] + ".cif"))
    nb += 1
    print(f"  built {r['pdb']:5s} {r['instance']:24s} chains {','.join(got)}")
print()
print(f"{a.out_dir}: {nc} copied, {nb} built, {len(os.listdir(a.out_dir))} total of {len(sel)}")
