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
applied to all chains together, and THAT pooled fit is what the verdict uses -- a per-chain fit
alone would mask a chain swapped for another copy. A per-chain fit is also reported, in
`per_chain_rmsd`, precisely to tell the two cases apart: when every chain fits at ~0 but the pooled
fit does not, the atoms are right and the COPIES are mixed (8dpl: chains 0.000/0.020/0.020, pooled
28.5 A). Without that column a large pooled RMSD cannot be distinguished from wrong coordinates.

THE HARD PART IS MATCHING CHAIN TO CHAIN, NOT THE CHEMISTRY. Both files hold the same molecule,
but they divide it into chains differently, in three ways, and every structure this audit failed to
verify before 2026-10-15 failed for one of them:

  1. THEY CUT THE ANTIBODY DIFFERENTLY. An scFv is one physical chain whose heavy and light domains
     are joined by a linker, and each source chose independently whether to call that one chain or
     two -- in BOTH directions. 32nz: our label A=122 + B=115, SAbDab2 A=238 fused. 10fj: our label
     A=220 fused, SAbDab2 B=120 + C=113.
  2. CHAIN NAMES DO NOT LINE UP. chain_roles sometimes records an id absent from SAbDab2's file, or
     one carrying a copy suffix (B1/B2 for the two domains of one chain), so pairing by name fails.
  3. CRYSTALS OFTEN HOLD TWO COPIES OF THE COMPLEX. Our label can take the antibody from copy 1 and
     SAbDab2's from copy 2. Every atom is correct; they sit in different places.

HOW THE PAIRING IS DECIDED. Jointly for the whole structure, then by concatenation -- two steps,
neither of which introduces a tolerance:

  1. ONE ASSIGNMENT, NOT A CHOICE PER ROLE. Containment of each label chain in each SAbDab2 chain is
     scored, and a maximum-weight one-to-one matching is solved over the whole structure
     (scipy.optimize.linear_sum_assignment). Pairing roles independently let one role take a chain
     another needed: on 9vb4 the H role scored best against the ANTIGEN chain C (33/351) and claimed
     it, so both antibody roles went unaudited while their own chains sat unused. A role left with no
     chain of its own -- which is what having more roles than chains MEANS -- may then share one,
     and the named chain breaks ties so identical copies keep the pairing chain_roles recorded.
  2. CONCATENATION, NOT SLICING. Roles are compared GROUPED BY REFERENCE CHAIN. When two label
     chains share one, the claim being made is that they are its pieces, so they are concatenated
     and the result compared against it with the same matcher, the same acceptance test and the same
     substitution accounting as any one-to-one chain. Order comes from where each chain's longest
     match falls in the reference, because VH-then-VL and VL-then-VH both occur.

WHY NOT SLICE THE FUSED CHAIN INSTEAD. That was built first and abandoned. Comparing one role
against the whole fused chain needs the other domain excluded, and the two domains are homologous,
so it required a register tolerance to discard spurious blocks -- a constant with no principled
value, which had to be hand-tuned per structure (30 passed 29tj and failed 32nz at 122 matched over
a 144-wide span; 8 passed 32nz). Forcing the window to the label chain's length instead invented
substitutions at the boundary, reporting 29tj H 142-144 ASP/ILE/VAL against ALA/ALA/SER. Both
failures came from the slice, not the data. Concatenation needs no such constant.

A LABEL CHAIN MAY LEGITIMATELY HOLD MORE THAN THE REFERENCE. SAbDab2 is the reference, so coverage
is measured over ITS chain: the claim is that every reference residue appears in our label with the
same identity and the same coordinates. Residues our label has in ADDITION are reported as
only_in_label, never as a failure. Requiring our side to be covered too was tried and is wrong,
because the fusion runs both ways -- on 10fj our fused chain A (220) rightly carries more than
SAbDab2's B (120) -- and that requirement rejected 230 structures whose reference residues all
matched exactly.

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
import argparse, collections, csv, difflib, glob, math, os, sys

# THE ONE THRESHOLD IN THIS FILE, used for both pairing and acceptance so there is a single number
# to argue about. A chain pairs with, and is accepted against, a reference chain when this fraction
# of the REFERENCE chain is matched.
#
# IT IS NOT LOAD-BEARING, AND THAT IS MEASURED RATHER THAN ASSERTED. Over the 1,517 training and 140
# validation chain groups audited on 2026-10-15 the LOWEST accepted fraction is 0.9871 (training) and
# 0.9848 (validation); exactly one group in each falls below 0.99 and none below 0.97. Nothing sits
# between 0.95 and 0.98, so any cutoff in that range produces identical verdicts, and no structure
# passes by sitting just above the line. Re-run with --coverage-hist to re-measure this.
#
# OPEN QUESTION FOR REVIEW: a reference chain that is not FULLY accounted for could be a hard failure
# with the shortfall itemised, rather than a fraction at all. That would remove the last threshold
# from this file. The five structures that fail today fail on coordinates, not coverage, so the
# choice changes nothing now -- but it is a design decision, not something to settle by tuning.
MIN_COVER = 0.95

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
    """ONE rigid transform fitted over every matched atom of the complex.

    Returns (rmsd, maxdev, n_over_1A): the last counts atoms still more than 1 A from their partner
    after the fit. A mismatch of a few side chains and a mismatch of a whole domain give similar
    maxdev, and only the count separates them -- 8d53 reports maxdev 10.17 A, which means nothing
    until one knows whether it is 3 atoms or 3,000.
    """
    n = len(P)
    if n < 3:
        return None, None, None
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
        return (float(np.sqrt((per ** 2).mean())), float(per.max()),
                int((per > 1.0).sum()))
    except Exception:
        return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["training", "validation"], required=True)
    ap.add_argument("--selection", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--sabdab", default=f"{ROOT}/SAbDab2_data/training_cifs")
    ap.add_argument("--roles", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--debug-pdb", default="",
                    help="print why each chain group of this pdb was rejected (matched, "
                         "denominator, lengths), which is how the pairing bugs above were found.")
    ap.add_argument("--coverage-hist", action="store_true",
                    help="print the distribution of matched/denominator over every audited chain "
                         "group, which is how MIN_COVER is shown not to be load-bearing. Re-measure "
                         "this rather than trusting the number quoted beside MIN_COVER.")
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
        per_group = []       # per-chain rmsd/maxdev/atoms-over-1A, separating copy mixing from bad atoms
        subs, only_s, only_l = [], 0, 0
        audited = set()
        expected = sorted(roles.get(pdb, {}))

        # ---------------- PAIRING PRE-PASS: AN ASSIGNMENT, NOT A PER-ROLE GUESS ----------------
        # Pairing each role independently with its best-matching chain lets one role take a chain
        # another role needs. Measured on 9vb4, the H role scored best against the ANTIGEN chain C
        # (33/351) and claimed it, so both antibody roles ended up unaudited while their own chains
        # sat unused. The pairing is therefore solved jointly: maximise total containment over a
        # one-to-one matching first, which is the normal case, and only then let a role that has no
        # chain left SHARE one. That order is what makes sharing mean something -- a chain is reused
        # only when the structure genuinely has more roles than SAbDab2 has chains, which is the
        # scFv case, where SAbDab2 fuses VH and VL into one chain and our label splits them.
        _rl = sorted(roles.get(pdb, {}).items())
        seqs_l, named = {}, {}
        for _role, _rr in _rl:
            _lk = _rr.get("model_chain")
            named[_role] = _rr.get("ref_chain")
            if _lk in lab:
                seqs_l[_role] = (_lk, "".join(_one(lab[_lk][x][0]) for x in sorted(lab[_lk])))
        seqs_s = {k: "".join(_one(v[x][0]) for x in sorted(v)) for k, v in sab.items()}

        def _contain(ls, ss):
            # fraction of the LABEL chain found in the SAbDab2 chain. Asymmetric on purpose: the
            # question is whether this label chain is a segment of that chain, so normalising by the
            # SAbDab2 length would score each half of a fused scFv at ~0.5 and reject both.
            _sm = difflib.SequenceMatcher(None, ss, ls, autojunk=False)
            return sum(b.size for b in _sm.get_matching_blocks()) / max(1, len(ls))

        roles_o, chains_o = list(seqs_l), list(seqs_s)
        score = {}
        for _r in roles_o:
            for _s in chains_o:
                # the named chain wins ties, so identical copies keep the pairing chain_roles
                # recorded and nothing that passes today is re-paired by a coin flip
                score[(_r, _s)] = _contain(seqs_l[_r][1], seqs_s[_s]) + (1e-6 if named.get(_r) == _s else 0.0)
        assign = {}
        if roles_o and chains_o:
            import numpy as _np
            from scipy.optimize import linear_sum_assignment as _lsa
            _ri, _ci = _lsa(_np.array([[-score[(r, c)] for c in chains_o] for r in roles_o]))
            for _i, _j in zip(_ri, _ci):
                if score[(roles_o[_i], chains_o[_j])] >= MIN_COVER:
                    assign[roles_o[_i]] = chains_o[_j]
        for _r in roles_o:                       # roles with no chain of their own may share one
            if _r in assign:
                continue
            _b = max(chains_o, key=lambda c: score[(_r, c)]) if chains_o else None
            # below threshold, keep the chain chain_roles NAMED so the rejection is reported against
            # the chain actually claimed rather than against whichever one happened to score highest
            assign[_r] = _b if (_b is not None and score[(_r, _b)] >= MIN_COVER) else named.get(_r)
        shared = collections.Counter(v for v in assign.values() if v)

        # ------------- COMPARISON, GROUPED BY REFERENCE CHAIN -------------
        # ONE GROUP PER SAbDab2 CHAIN, NOT ONE PER ROLE. An scFv is one physical chain whose heavy
        # and light domains are joined by a linker, and the two sources disagree about whether to
        # call that one chain or two -- in BOTH directions. Measured here: on 32nz our label splits
        # it (A=122, B=115) and SAbDab2 fuses it (A=238); on 10fj our label fuses it (A=220) and
        # SAbDab2 splits it (B=120, C=113). A role-by-role comparison has nothing to align in the
        # first case and leaves both antibody roles unaudited.
        #
        # Grouping states the claim directly and needs NO new tolerance or threshold: if these label
        # chains are the pieces of that reference chain, then CONCATENATING them reproduces it. The
        # concatenation is then compared with the same matcher, the same acceptance test and the same
        # substitution accounting as any one-to-one chain, so the fused case is not a special path
        # with its own constants. An earlier attempt did slice the fused chain to a window instead,
        # which required a register tolerance to reject homologous blocks in the other domain; the
        # tolerance had to be hand-tuned per structure (30 passed 29tj but failed 32nz) and is gone.
        groups = collections.defaultdict(list)
        for role, rr in _rl:
            lk = rr.get("model_chain")                      # the label's chain NAME
            sk = assign.get(role) or rr.get("ref_chain")    # SAbDab2's label_asym_id
            if not lk or not sk or lk not in lab or sk not in sab:
                cov.append(f"{role}:unpaired({lk}/{sk})"); continue
            groups[sk].append((role, lk, sk == rr.get("ref_chain")))

        for sk, members in sorted(groups.items()):
            sres = sab[sk]
            sp = sorted(sres)
            sseq = "".join(_one(sres[x][0]) for x in sp)
            # Concatenation order comes from the data, not from the role names: each label chain is
            # placed where its longest match falls in the reference chain. VH-then-VL and VL-then-VH
            # both occur, so assuming an order would silently mis-pair whichever convention differs.
            def _anchor(lk):
                _ls = "".join(_one(lab[lk][x][0]) for x in sorted(lab[lk]))
                return difflib.SequenceMatcher(None, sseq, _ls, autojunk=False) \
                        .find_longest_match(0, len(sseq), 0, len(_ls)).a
            members = sorted(members, key=lambda m: _anchor(m[1]))

            lp, owner = [], []          # lp: (label chain, residue key); owner: the role it belongs to
            for role, lk, _named in members:
                for x in sorted(lab[lk]):
                    lp.append((lk, x)); owner.append(role)
            lseq = "".join(_one(lab[lk_][x][0]) for lk_, x in lp)

            sm = difflib.SequenceMatcher(None, sseq, lseq, autojunk=False)
            matched = sum(b.size for b in sm.get_matching_blocks() if b.size)
            # SAbDab2 IS THE REFERENCE, so the denominator is ITS chain. The audit's claim is that
            # every reference residue appears in our label with the same identity and the same
            # coordinates; residues our label has in ADDITION are reported as only_in_label rather
            # than as a failure. Requiring our side to be covered too is wrong and was measured to
            # be: on 10fj our fused chain A (220) legitimately carries more than SAbDab2's B (120),
            # and that requirement rejected 230 structures whose reference residues all matched.
            denom = len(sseq)
            ok_cov = bool(matched) and matched / denom >= MIN_COVER
            if not ok_cov:
                if a.debug_pdb == pdb:
                    print(f"  DBG {pdb} {'+'.join(m[0] for m in members)} "
                          f"lk={'+'.join(m[1] for m in members)} sk={sk} "
                          f"matched={matched} denom={denom} len_l={len(lseq)}", file=sys.stderr)
                ambig.append(f"{'+'.join(m[0] for m in members)}:"
                             f"{'+'.join(m[1] for m in members)}/{sk}@{matched}/{denom}")
                continue

            # OPCODES, NOT MATCHING BLOCKS. `equal` gives the correspondence to compare; `replace`
            # is a substitution and must be reported rather than skipped; `delete` and `insert` are
            # residues present in only one of the two files.
            corr = []
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "equal":
                    for k in range(i2 - i1):
                        corr.append((sp[i1 + k], lp[j1 + k], owner[j1 + k]))
                elif tag == "replace":
                    n = min(i2 - i1, j2 - j1)
                    for k in range(n):
                        _s, _l = sp[i1 + k], lp[j1 + k]
                        subs.append(f"{owner[j1+k]}:{_s}->{_l[0]}{_l[1]}:"
                                    f"{sres[_s][0]}!={lab[_l[0]][_l[1]][0]}")
                    only_s += max(0, (i2 - i1) - n)
                    only_l += max(0, (j2 - j1) - n)
                elif tag == "delete":
                    only_s += i2 - i1
                elif tag == "insert":
                    only_l += j2 - j1

            for role, lk, _named in members:
                audited.add(role)
                pairs.append(f"{role}={lk}/{sk}" + ("" if _named else "~seq"))
            cov.append(f"{'+'.join(m[0] for m in members)}:{matched}/{denom}")
            _gP, _gQ = [], []
            for sp_i, lp_i, role in corr:
                _lres = lab[lp_i[0]][lp_i[1]]
                if sres[sp_i][0] != _lres[0]:
                    idmis.append(f"{role}:{sp_i}->{lp_i[1]}:{sres[sp_i][0]}!={_lres[0]}")
                    continue
                for an, xyz in sres[sp_i][1].items():
                    if an in _lres[1]:
                        Q.append(xyz); P.append(_lres[1][an])
                        _gQ.append(xyz); _gP.append(_lres[1][an])
            # EACH GROUP SUPERPOSED ON ITS OWN, as well as all of them together below. The two
            # answers differ when the pairing is right chain-by-chain but mixes COPIES: a crystal
            # with two copies of the complex lets our label's antibody pair to copy 1 and its
            # antigen to copy 2, and each chain then superposes exactly while the pooled fit does
            # not. Without this column a large pooled RMSD cannot be told apart from wrong atoms.
            _gr, _gm, _gn = kabsch(_gP, _gQ) if len(_gP) >= 3 else (None, None, None)
            per_group.append(f"{'+'.join(m[0] for m in members)}:"
                             + ("" if _gr is None
                                else f"{_gr:.3f}/{_gm:.3f}/{_gn}of{len(_gP)}"))

        rmsd, maxdev, n_over = kabsch(P, Q) if len(P) >= 3 else (None, None, None)
        skipped = [r for r in expected if r not in audited]
        rec.update(roles_expected=len(expected), roles_audited=len(audited),
                   roles_skipped=";".join(skipped), n_subs=len(subs), subs=";".join(subs[:6]),
                   only_in_sabdab=only_s, only_in_label=only_l,
                   shared_chains=";".join(f"{k}x{v}" for k, v in sorted(shared.items()) if v > 1),
                   matched_atoms=len(P), atoms_over_1A="" if n_over is None else n_over,
                   pairs=";".join(pairs), coverage=";".join(cov),
                   per_chain_rmsd=";".join(per_group),
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
            "n_resid_mismatch", "atoms_over_1A", "shared_chains", "per_chain_rmsd",
            "pairs", "coverage", "resid_mismatch", "ambiguous"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    c = collections.Counter(r.get("verdict") for r in rows)
    if a.coverage_hist:
        fr = []
        for r in rows:
            for g in (r.get("coverage") or "").split(";"):
                if ":" in g and "/" in g and "unpaired" not in g:
                    num, den = g.split(":")[1].split("/")
                    fr.append(int(num) / int(den))
        fr.sort()
        if fr:
            print(f"coverage over {len(fr)} chain group(s): min={fr[0]:.4f} "
                  f"below_0.99={sum(1 for x in fr if x < 0.99)} "
                  f"below_0.97={sum(1 for x in fr if x < 0.97)} "
                  f"below_MIN_COVER={sum(1 for x in fr if x < MIN_COVER)}")
            print(f"  lowest 8: {[round(x, 4) for x in fr[:8]]}")
            print(f"  MIN_COVER={MIN_COVER}; a cutoff anywhere below min above changes no verdict")
    print(f"audited {len(rows)} structure(s) -> {a.out}")
    for k, v in c.most_common():
        print(f"  {k:<28} {v}")
    bad = [r for r in rows if r.get("verdict") not in ("ok",)]
    for r in bad[:10]:
        print(f"    {r['pdb']}: {r.get('verdict')}  rmsd={r.get('rmsd')} max={r.get('maxdev')} "
              f"atoms={r.get('matched_atoms')} {r.get('resid_mismatch','')[:60]}")


if __name__ == "__main__":
    main()
