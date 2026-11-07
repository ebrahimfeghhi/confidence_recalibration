#!/usr/bin/env python
"""Extract missing SAbDab2 per-instance cifs from the release archive already on disk.

WHY THIS EXISTS. SAbDab2_data/training_cifs is a PARTIAL extraction: 4,903 of the archive's
15,810 instance cifs, covering 3,053 of 8,796 PDB entries. Nothing about the missing ones is
special -- they were simply never unpacked. 17 of the 30 Wave 1 validation structures fell outside
that extraction, which cost them their pool rows (build_wave1_pool reads auth -> label_asym_id
from this file) and their DockQ reference (phase_g stages the same file as validation/refs/<pdb>.cif).

The archive holds all 15,810, so this is a local extraction, not a download.

VERIFIED, NOT ASSUMED: every extracted cif's chain set must equal the chains the splits row names
in Hchain / Lchain / agchains, or it is not installed. An existing file is never overwritten --
the corpus is the authority for what was already there.
"""
import argparse, csv, os, shutil, subprocess, sys, tempfile
import gemmi

ap = argparse.ArgumentParser()
ap.add_argument("--selection", required=True, help="csv with pdb,instance columns")
ap.add_argument("--splits", default="SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
ap.add_argument("--archive", default="SAbDab2_data/zenodo_splits/splits_v0.2.0.tar.gz")
ap.add_argument("--member-dir", default="splits_final")
ap.add_argument("--corpus", default="SAbDab2_data/training_cifs")
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

sel = list(csv.DictReader(open(a.selection)))
need = [r for r in sel if not os.path.exists(os.path.join(a.corpus, r["instance"] + ".cif"))]
print(f"{len(sel)} selected; {len(need)} instance cif(s) missing from {a.corpus}")
if not need:
    print("nothing to extract")
    sys.exit(0)

sp = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}


def spec(row):
    out = set()
    for k in ("Hchain", "Lchain"):
        v = (row[k] or "").strip()
        if v and v != "+":
            out.add(v)
    for v in str(row["agchains"]).split("/"):
        v = v.strip()
        if v and v != "+":
            out.add(v)
    return out


tmp = tempfile.mkdtemp(prefix="inst_cif_")
try:
    members = [f"{a.member_dir}/{r['instance']}.cif" for r in need]
    lst = os.path.join(tmp, "members.txt")
    open(lst, "w").write("\n".join(members) + "\n")
    r = subprocess.run(["tar", "-xzf", a.archive, "-C", tmp, "-T", lst],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr.strip()[:500])
        raise SystemExit(f"STOP: extraction failed ({r.returncode}); nothing installed")

    ok, bad = [], []
    for row in need:
        src = os.path.join(tmp, a.member_dir, row["instance"] + ".cif")
        if not os.path.exists(src):
            bad.append((row["pdb"], "not in the archive")); continue
        st = gemmi.read_structure(src)
        st.setup_entities()
        got = {c.name for c in st[0]}
        want = spec(sp[row["instance"]])
        if got != want:
            bad.append((row["pdb"], f"chains {sorted(got)} != splits {sorted(want)}")); continue
        ok.append((row["pdb"], row["instance"], src))

    print(f"  verified against the splits chain spec: {len(ok)} of {len(need)}")
    for pdb, why in bad:
        print(f"  PROBLEM {pdb}: {why}")
    if bad:
        raise SystemExit("STOP: nothing installed; every extracted cif must match the splits row")
    if not a.apply:
        print("report only; pass --apply to install")
        sys.exit(0)
    n = 0
    for pdb, instance, src in ok:
        dst = os.path.join(a.corpus, instance + ".cif")
        if os.path.exists(dst):            # never overwrite what the corpus already had
            continue
        shutil.copy2(src, dst)
        os.chmod(dst, 0o664)
        n += 1
    print(f"installed {n} instance cif(s); {a.corpus} now holds {len(os.listdir(a.corpus))} files")
finally:
    shutil.rmtree(tmp, ignore_errors=True)
