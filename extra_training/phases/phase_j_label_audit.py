"""Independent label audit: our RCSB-derived training labels vs SAbDab2's curated coordinates.

WHY AN EXTERNAL REFERENCE. The labels in `training/prep/job_cifs/` are built by phase_j_job_cifs
from the deposited RCSB entries, and the chain-to-role mapping is INFERRED -- by sequence, with
contact checks and tie-breaks. Nothing in that path is checked against a second opinion. Every audit
written so far has compared our data to quantities derived by the same code, and four of those
checks were wrong in the same way: auth chain names used where label_asym_id was meant, resolved
positions pooled across chain copies, a symmetric difflib ratio applied to a partial chain, and
gemmi's remove_ligands_and_waters deleting peptide antigens. An external file does not share those
assumptions.

SAbDab2's instance cifs are that external file. They are NOT the same artifact as our labels:

    training/refs/<pdb>.cif        md5-identical to SAbDab2's instance cif. The DockQ reference.
    training/prep/job_cifs/<pdb>.cif   built from rcsb_training_cifs. The TRAINING LABEL.

For 10fj the label carries chains of 220, 220 and 172 residues -- full Fab plus antigen -- while
SAbDab2 carries 120, 113 and 172, the variable domains plus the antigen. Same deposit, independently
processed, overlapping over the variable region and the antigen.

RESIDUE CORRESPONDENCE COMES FROM SEQUENCE ALIGNMENT, NOT FROM label_seq. The two files number
independently: for 10fj's heavy chain SAbDab2 runs 20..139 while the label runs 1..220, a constant
offset of 18, and comparing at equal label_seq matched 8 of 120 residue identities. Aligning the
sequences instead gives an exact 120-residue block. An offset is not a defect; assuming a shared
index is.

WHAT IS COMPARED. Only atoms present in BOTH files, matched by (chain role, aligned residue, atom
name).
The constant domains SAbDab2 deliberately crops are excluded by construction, since they are absent
from its files -- that is a coverage limit, not a mismatch, and is reported as coverage rather than
as error. If the coordinate frames differ, ONE rigid transform is fitted over the whole complex and
applied to all chains together; never a separate fit per chain, which would mask a chain that has
been swapped for another copy.

KNOWN LIMITATION: SINGLE-CHAIN CONSTRUCTS PAIR 2-TO-1 AND ARE NOT AUDITED. SAbDab2 keeps an scFv as
ONE fused chain while our label splits it into two -- 7tzh is sabdab chain C at 230 residues against
label chains A:125 and B:108, and 8v8f and 9cn2 sum EXACTLY to the fused length. The pairing here is
one-to-one, so those antibody roles come back unpaired and are reported as AMBIGUOUS CHAIN or "too
few shared atoms" rather than as verified. Their ANTIGEN roles still audit exactly. Normalising the
sequence-coverage fallback by the label chain instead of the SAbDab2 chain was tried and made
matters worse -- it left 7tzh and 8v8f unpaired anyway and regressed 9nr6 -- so the limitation is
recorded rather than papered over. This is the same class as open review item J1/J3, the 9 split
single-chain constructs. Auditing them needs an explicit 2-to-1 mode, which is not written.

A PASS REQUIRES EVERY EXPECTED ROLE, AND SUBSTITUTIONS ARE COUNTED EXPLICITLY. Two holes were found
in review of c8137d8 and are closed here:

  * A ROLE THAT COULD NOT BE PAIRED USED TO BE INVISIBLE. Validation's 9nr6 was reported `ok` with
    BOTH antibody chains unpaired -- only its antigen had been compared -- because an unpaired role
    was noted in the coverage string and never reached the verdict. A structure now passes only if
    every role in `chain_roles` was actually audited; otherwise the verdict is INCOMPLETE and says
    which roles were skipped.
  * THE RESIDUE-IDENTITY CHECK WAS UNFALSIFIABLE. It walked `get_matching_blocks()`, which by
    definition returns only EQUAL blocks, so a substitution fell outside every block, never entered
    the correspondence, and could not be reported. "0 residue mismatches over 2.03M atoms" was
    therefore not evidence of anything. The walk is now over `get_opcodes()`, so `replace` is
    reported as a SUBSTITUTION and `delete`/`insert` as residues present in only one file.

AND PARTIAL AGREEMENT IS NOT A PASS. Counting rows with rmsd < 0.01 overstated coverage: 496 of 501
training rows were coordinate-exact but only 481 were complete passes, the other 15 being exact over
an incomplete set of roles. The report carries `roles_expected`, `roles_audited` and `verdict`, and
only `verdict == ok` means fully verified.

WHAT COUNTS AS A FAILURE. Coordinates should agree to file-rounding precision, because both files
hold the same deposited atoms. Residue identity must match exactly. A large RMSD, a large maximum
deviation, or a residue-identity mismatch means the mapping is wrong somewhere and is reported for
inspection rather than silently tolerated or silently dropped.
"""
import argparse, collections, csv, glob, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _one(resname):
    import gemmi
    t = gemmi.find_tabulated_residue(resname)
    return t.one_letter_code.upper() if t else "X"


def load(path, key="subchain"):
    """{chain key: {label_seq: (resname, {atom name: (x,y,z)})}}.

    THE TWO FILES ARE KEYED DIFFERENTLY AND chain_roles NAMES BOTH. The label cif written by
    phase_j_job_cifs is in model order with chain NAMES A, B, C -- gemmi reports its subchains as
    `Ap`, `Bp`, `Cp`, so keying the label on subchain matches nothing and the first run of this
    audit reported 0 shared atoms for every structure. chain_roles' `model_chain` is the label's
    chain name and its `ref_chain` is SAbDab2's label_asym_id, so each side is looked up with its
    own identifier instead of assuming the two agree.
    """
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_alternative_conformations()
    out = collections.defaultdict(dict)
    for ch in st[0]:
        for r in ch:
            if r.label_seq is None:
                continue
            k = (r.subchain or ch.name) if key == "subchain" else ch.name
            atoms = {}
            for a in r:
                # FIRST OCCURRENCE WINS, so an altloc that survived cannot silently replace an atom
                if a.name not in atoms:
                    atoms[a.name] = (a.pos.x, a.pos.y, a.pos.z)
            out[k][int(r.label_seq)] = (r.name, atoms)
    return out


def kabsch(P, Q):
    """ONE rigid transform fitted over every matched atom of the complex. Returns (rmsd, maxdev)."""
    n = len(P)
    if n < 3:
        return None, None
    cp = [sum(c[i] for c in P) / n for i in range(3)]
    cq = [sum(c[i] for c in Q) / n for i in range(3)]
    p = [[c[i] - cp[i] for i in range(3)] for c in P]
    q = [[c[i] - cq[i] for i in range(3)] for c in Q]
    H = [[sum(p[k][i] * q[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    try:
        import numpy as np
        U, S, Vt = np.linalg.svd(np.array(H))
        d = 1.0 if np.linalg.det(Vt.T @ U.T) > 0 else -1.0
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        pa = np.array(p); qa = np.array(q)
        diff = (pa @ R.T) - qa
        per = np.sqrt((diff ** 2).sum(axis=1))
        return float(np.sqrt((per ** 2).mean())), float(per.max())
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["training", "validation"], required=True)
    ap.add_argument("--selection", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--sabdab", default=f"{ROOT}/SAbDab2_data/training_cifs")
    ap.add_argument("--roles", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    sel = a.selection or (f"{ROOT}/train_val_splits/cache_train_structures.csv" if a.which == "training"
                          else f"{ROOT}/train_val_splits/val_split_proposal.csv")
    labels = a.labels or f"{ROOT}/{a.which}/prep/job_cifs"
    rolesf = a.roles or f"{ROOT}/{a.which}/fold/chain_roles.csv"

    inst = {}
    for r in csv.DictReader(open(sel)):
        if r.get("pdb") and r.get("instance"):
            inst[r["pdb"]] = r["instance"]
    roles = collections.defaultdict(dict)
    for r in csv.DictReader(open(rolesf)):
        roles[r["pdb"]][r["role"]] = r
    pdbs = sorted(inst)
    if a.limit:
        pdbs = pdbs[:a.limit]

    rows = []
    for pdb in pdbs:
        rec = dict(pdb=pdb, instance=inst[pdb])
        lab_p = os.path.join(labels, f"{pdb}.cif")
        sab_p = os.path.join(a.sabdab, f"{inst[pdb]}.cif")
        if not os.path.exists(lab_p):
            rec["verdict"] = "no label cif"; rows.append(rec); continue
        if not os.path.exists(sab_p):
            rec["verdict"] = "no sabdab cif"; rows.append(rec); continue
        try:
            lab, sab = load(lab_p, key="name"), load(sab_p, key="subchain")
        except Exception as e:
            rec["verdict"] = f"read error: {type(e).__name__}"; rows.append(rec); continue

        # role -> (label chain key, sabdab chain key). The label is keyed by its own label_asym_id,
        # which chain_roles records per role; SAbDab2's file is keyed by ITS label_asym_id, so the
        # roles are paired through chain_roles rather than by assuming the names agree.
        P, Q, pairs, ambig, idmis, cov = [], [], [], [], [], []
        subs, only_s, only_l = [], 0, 0
        audited = set()
        expected = sorted(roles.get(pdb, {}))
        for role, rr in sorted(roles.get(pdb, {}).items()):
            lk = rr.get("model_chain")       # the label's chain NAME
            sk = rr.get("ref_chain")         # SAbDab2's label_asym_id
            how = "named"
            if sk not in sab:
                # chain_roles records SAbDab2's label id, but for a single-chain construct it can
                # carry a copy index (B1/B2 for the two domains of one chain) and for some entries
                # the id simply does not appear. Fall back to the chain whose sequence matches the
                # label chain best, and RECORD that it was paired by sequence rather than by name,
                # so a sequence-paired row is never mistaken for a named one.
                cand = None
                if lk in lab:
                    lseq0 = "".join(_one(lab[lk][x][0]) for x in sorted(lab[lk]))
                    import difflib as _dl
                    for k2, v2 in sab.items():
                        s2 = "".join(_one(v2[x][0]) for x in sorted(v2))
                        m2 = sum(b.size for b in _dl.SequenceMatcher(None, s2, lseq0,
                                 autojunk=False).get_matching_blocks())
                        c2 = m2 / max(1, len(s2))
                        if cand is None or c2 > cand[0]:
                            cand = (c2, k2)
                if cand and cand[0] >= 0.95:
                    sk, how = cand[1], "by-seq"
            if not lk or not sk or lk not in lab or sk not in sab:
                cov.append(f"{role}:unpaired({lk}/{sk})"); continue
            lres, sres = lab[lk], sab[sk]
            lp, sp = sorted(lres), sorted(sres)
            lseq = "".join(_one(lres[p][0]) for p in lp)
            sseq = "".join(_one(sres[p][0]) for p in sp)
            import difflib
            sm = difflib.SequenceMatcher(None, sseq, lseq, autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size]
            matched = sum(b.size for b in blocks)
            if not matched or matched / len(sseq) < 0.95:
                ambig.append(f"{role}:{lk}/{sk}@{matched}/{len(sseq)}"); continue
            # OPCODES, NOT MATCHING BLOCKS. `equal` gives the correspondence to compare; `replace`
            # is a substitution and must be reported rather than skipped; `delete` and `insert` are
            # residues present in only one of the two files.
            corr = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        corr.append((sp[i1 + k], lp[j1 + k]))
                elif tag == "replace":
                    n = min(i2 - i1, j2 - j1)
                    for k in range(n):
                        subs.append(f"{role}:{sp[i1+k]}->{lp[j1+k]}:"
                                    f"{sres[sp[i1+k]][0]}!={lres[lp[j1+k]][0]}")
                    only_s += max(0, (i2 - i1) - n)
                    only_l += max(0, (j2 - j1) - n)
                elif tag == "delete":
                    only_s += i2 - i1
                elif tag == "insert":
                    only_l += j2 - j1
            audited.add(role)
            pairs.append(f"{role}={lk}/{sk}" + ("" if how == "named" else "~seq"))
            for sp_i, lp_i in corr:
                if sres[sp_i][0] != lres[lp_i][0]:
                    idmis.append(f"{role}:{sp_i}->{lp_i}:{sres[sp_i][0]}!={lres[lp_i][0]}")
                    continue
                for an, xyz in sres[sp_i][1].items():
                    if an in lres[lp_i][1]:
                        Q.append(xyz); P.append(lres[lp_i][1][an])
            cov.append(f"{role}:{matched}/{len(sseq)}")

        rmsd, maxdev = kabsch(P, Q) if len(P) >= 3 else (None, None)
        skipped = [r for r in expected if r not in audited]
        rec.update(roles_expected=len(expected), roles_audited=len(audited),
                   roles_skipped=";".join(skipped), n_subs=len(subs), subs=";".join(subs[:6]),
                   only_in_sabdab=only_s, only_in_label=only_l,
                   matched_atoms=len(P), pairs=";".join(pairs), coverage=";".join(cov),
                   rmsd="" if rmsd is None else f"{rmsd:.4f}",
                   maxdev="" if maxdev is None else f"{maxdev:.4f}",
                   n_resid_mismatch=len(idmis), resid_mismatch=";".join(idmis[:6]),
                   ambiguous=";".join(ambig))
        # A PASS REQUIRES EVERY ROLE. An unpaired role used to vanish into the coverage string.
        if skipped:
            rec["verdict"] = f"INCOMPLETE ({len(skipped)} role(s) not audited)"
        elif len(P) < 3:
            rec["verdict"] = "too few shared atoms"
        elif subs:
            rec["verdict"] = "SUBSTITUTION"
        elif idmis:
            rec["verdict"] = "RESIDUE IDENTITY MISMATCH"
        elif ambig:
            rec["verdict"] = "AMBIGUOUS CHAIN"
        elif rmsd is not None and rmsd > 0.10:
            rec["verdict"] = "COORD MISMATCH"
        else:
            rec["verdict"] = "ok"
        rows.append(rec)

    cols = ["pdb", "instance", "verdict", "roles_expected", "roles_audited", "roles_skipped",
            "matched_atoms", "rmsd", "maxdev", "n_subs", "subs", "only_in_sabdab", "only_in_label",
            "n_resid_mismatch", "pairs", "coverage", "resid_mismatch", "ambiguous"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    c = collections.Counter(r.get("verdict") for r in rows)
    print(f"audited {len(rows)} structure(s) -> {a.out}")
    for k, v in c.most_common():
        print(f"  {k:<28} {v}")
    bad = [r for r in rows if r.get("verdict") not in ("ok",)]
    for r in bad[:10]:
        print(f"    {r['pdb']}: {r.get('verdict')}  rmsd={r.get('rmsd')} max={r.get('maxdev')} "
              f"atoms={r.get('matched_atoms')} {r.get('resid_mismatch','')[:60]}")


if __name__ == "__main__":
    main()
