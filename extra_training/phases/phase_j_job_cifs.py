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
import collections
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import phase_a_define_train_set as A          # noqa: E402

POLY = A.POLY


def chain_seqs(st):
    import gemmi
    return {ch.name: gemmi.one_letter_code([r.name for r in ch]).upper() for ch in st[0]}


def assign_chains(roles, seqs, floor=0.85):
    """in: [(role, sequence)] and {chain: sequence}.  out: ([(role, chain)], [unresolved roles])

    ONE CHAIN PER ROLE, assigned greedily best-first over DISTINCT chains. A homodimeric antigen
    gives two roles the same sequence, so both score identically against the same chain; keeping only
    one chain silently halved the system (8wnp: 1,143 expected tokens, 792 kept, an entire antigen
    copy gone). Each role takes its best chain that no other role has taken.

    SCORED BY COVERAGE OF THE QUERY, not identity over the shorter sequence. Over-the-shorter lets a
    tiny chain score 1.0 against a long query whenever it is a subsequence: 9vb4's 354-residue
    antigen matched a 6-residue chain at 1.0, and the label would have carried six residues where the
    pose has 354. Coverage of the query gives 6/354 and rejects it. The reverse asymmetry is normal
    and must NOT be penalised -- SAbDab2's Hseq is the variable domain while the entry's chain is the
    whole heavy chain (9p4c: 211 against 560), so a symmetric length floor wrongly rejected 25
    full-length antibodies.

    BY SEQUENCE, NOT BY NAME, because SAbDab2's chain names disagree with the entry's in four ways:
    a trailing copy index (A1/C2, B2/B1, D1/D2), numeric antigen chains, two-character names on large
    assemblies, and labels like J000. Name resolution failed on 32 of 501; sequence places all 501.
    """
    import difflib
    scored = []
    for ri, (role, seq) in enumerate(roles):
        for c, q in seqs.items():
            if not q or not seq:
                continue
            m = difflib.SequenceMatcher(None, seq, q, autojunk=False)
            blocks = [b for b in m.get_matching_blocks() if b.size]
            r = sum(b.size for b in blocks) / len(seq)
            if r >= floor and blocks:
                # THE ALIGNED SPAN, so the chain can be trimmed to what the fold actually contains.
                # SAbDab2's Hseq is the variable domain; the entry's chain is the whole heavy chain,
                # so keeping the chain whole put 1,187 residues in 9p4c's label against 444 in the
                # pose. j is the index into the CHAIN.
                lo = min(b.b for b in blocks)
                hi = max(b.b + b.size for b in blocks)
                # A SPAN CANNOT BE MUCH LONGER THAN THE QUERY. First-to-last block is right for a
                # long antigen with disordered gaps, but for a short peptide two three-residue
                # fragments can match at opposite ends of a long chain and the span swallows it:
                # 9pkv's 9-residue peptide took 244 residues of chain A. When the span runs away,
                # fall back to the single longest block, which is contiguous by construction.
                if hi - lo > 1.5 * len(seq) + 10:
                    big = max(blocks, key=lambda b: b.size)
                    lo, hi = big.b, big.b + big.size
                    r = big.size / len(seq)
                    if r < floor:
                        continue
                scored.append((r, ri, c, lo, hi))
    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    # A CHAIN MAY SERVE TWO ROLES IF THE SPANS DO NOT OVERLAP. Single-chain constructs put VH and VL
    # on one chain -- SAbDab2 names them H1/H2, D1/D2, A2/A1 -- so the heavy and light sequences
    # align to disjoint stretches of the same chain. Forbidding reuse outright left 9 of 501 with no
    # chain for L. Overlapping spans are still refused: that would be two roles claiming the same
    # residues, which is a mis-assignment rather than a construct.
    taken_role, taken, out = set(), collections.defaultdict(list), {}
    for r, ri, c, lo, hi in scored:
        if ri in taken_role:
            continue
        if any(not (hi <= plo or plhi <= lo) for plo, plhi in taken[c]):
            continue
        taken_role.add(ri); taken[c].append((lo, hi)); out[ri] = (c, lo, hi)
    pairs = [(roles[i][0], *out[i]) for i in range(len(roles)) if i in out]
    bad = [roles[i][0] for i in range(len(roles)) if i not in out]
    return pairs, bad


def _heavy(ch):
    import numpy as np
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch for a in r
                     if a.element.name != "H"], dtype=float)


def _min_dist(A, B):
    import numpy as np
    if not len(A) or not len(B):
        return float("inf")
    m = float("inf")
    for i in range(0, len(A), 512):
        m = min(m, float(np.sqrt(((A[i:i + 512, None, :] - B[None, :, :]) ** 2).sum(-1)).min()))
    return m


def prefer_contacting(pairs, st, seqs, cutoff=5.0, ident_floor=0.9):
    """Re-point each antigen role at a copy that actually TOUCHES the chosen antibody.

    THE SEQUENCE CANNOT TELL TWO COPIES APART. A crystal routinely holds several copies of the
    complex, so picking chains by sequence with an alphabetical tie-break can take the antibody from
    one copy and the antigen from another: same sequences, different assembly, no contact. Measured
    over the 501 selected structures, 71 ended up with NO antigen chain within 5 A of the antibody,
    and in all 25 of those sampled another copy of that antigen DID contact. The label would have
    described a complex that does not exist in that crystal while every sequence check passed.

    Only the CHAIN changes, never the sequence, so the fold's job json and every pose already
    produced are unaffected -- this is a label-side correction.
    """
    ab = [c for role, c, _lo, _hi in pairs if role in ("H", "L")]
    if not ab:
        return pairs, []
    abX = [_heavy(st[0][c]) for c in ab]
    taken = {c for _r, c, _l, _h in pairs}
    out, moved = [], []
    for role, c, lo, hi in pairs:
        if not role.startswith("ag"):
            out.append((role, c, lo, hi))
            continue
        if min(_min_dist(_heavy(st[0][c]), X) for X in abX) <= cutoff:
            out.append((role, c, lo, hi))
            continue
        import difflib
        q = seqs[c]
        best = None
        for cand, cq in seqs.items():
            if cand in taken or cand in ab:
                continue
            sm = difflib.SequenceMatcher(None, q, cq, autojunk=False)
            ident = sum(b.size for b in sm.get_matching_blocks()) / max(1, min(len(q), len(cq)))
            if ident < ident_floor:
                continue
            d = min(_min_dist(_heavy(st[0][cand]), X) for X in abX)
            if d <= cutoff and (best is None or d < best[0]):
                best = (d, cand)
        if best is None:
            out.append((role, c, lo, hi))
            continue
        # re-align the replacement, since its span is its own
        import difflib as _dl
        sm = _dl.SequenceMatcher(None, q, seqs[best[1]], autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        nlo, nhi = min(b.b for b in blocks), max(b.b + b.size for b in blocks)
        taken.discard(c); taken.add(best[1])
        out.append((role, best[1], nlo, nhi))
        moved.append((role, c, best[1], round(best[0], 2)))
    return out, moved


def named_chains(row, seqs):
    """SAbDab2's OWN chain assignment for this instance, when every name resolves in the entry.

    THIS IS TRIED FIRST, and it is the correct source. An INSTANCE is a specific antibody-antigen
    pairing, and Hchain / Lchain / agchains name the chains of THAT pairing -- so they contact by
    construction. Resolving by sequence instead cannot tell two copies of a complex apart, and a
    crystal routinely holds several: picking the antibody from one copy and the antigen from another
    gives identical sequences, a different assembly, and no contact at all.

    Measured, which is why this function exists: resolving everything by sequence left 71 of 501
    structures with NO antigen chain within 5 A of the antibody. For 67 of those, SAbDab2's own named
    chains contact at 2.3-2.8 A. The names were right and were being discarded -- sequence matching
    was adopted globally to rescue 32 structures whose names do not resolve, and it took the other
    469 with it.

    Returns None when any name is absent, so the caller falls back to sequence resolution.
    """
    out = []
    for role, key in (("H", "Hchain"), ("L", "Lchain")):
        v = (row[key] or "").strip()
        if v:
            out.append((role, v))
    types = str(row["agtypes"]).split("/")
    names = [c.strip() for c in str(row["agchains"]).split("/")]
    if len(types) != len(names):
        return None
    out += [(f"ag{i}", c) for i, (c, t) in enumerate(zip(names, types)) if c and t in POLY]
    if not out or any(c not in seqs for _r, c in out):
        return None
    if len({c for _r, c in out}) != len(out):
        return None            # two roles on one named chain: let sequence resolution split it
    return out


def spans_for(named, row, seqs):
    """Align each role's EXPECTED sequence to its named chain, giving (role, chain, lo, hi)."""
    import difflib
    exp = {"H": (row["Hseq"] or "").strip().upper(), "L": (row["Lseq"] or "").strip().upper()}
    types = str(row["agtypes"]).split("/")
    e = str(row["agexpectedseqs"]).split("/")
    r_ = str(row["agresolvedseqs"]).split("/")
    for i, t in enumerate(types):
        if t in POLY:
            for arr in (r_, e):
                q = arr[i].strip().upper() if i < len(arr) else ""
                if q:
                    exp[f"ag{i}"] = q
                    break
    out = []
    for role, c in named:
        q = exp.get(role, "")
        if not q:
            return None
        sm = difflib.SequenceMatcher(None, q, seqs[c], autojunk=False)
        blocks = [b for b in sm.get_matching_blocks() if b.size]
        if not blocks or sum(b.size for b in blocks) / len(q) < 0.85:
            return None        # the named chain does not carry this role's sequence
        lo, hi = min(b.b for b in blocks), max(b.b + b.size for b in blocks)
        if hi - lo > 1.5 * len(q) + 10:
            big = max(blocks, key=lambda b: b.size)
            lo, hi = big.b, big.b + big.size
        out.append((role, c, lo, hi))
    return out


def job_chains(row, seqs):
    """in: a splits row and {chain: sequence} from the entry.
    out: ([(role, chain, lo, hi)], [unresolved roles]) -- the chains the fold actually contains.

    NAMES FIRST, sequence as the fallback. See named_chains() for why."""
    nm = named_chains(row, seqs)
    if nm:
        sp = spans_for(nm, row, seqs)
        if sp:
            return sp, []
    want = [("H", (row["Hseq"] or "").strip().upper()),
            ("L", (row["Lseq"] or "").strip().upper())]
    types = str(row["agtypes"]).split("/")
    exp = str(row["agexpectedseqs"]).split("/")
    rsv = str(row["agresolvedseqs"]).split("/")
    if len(types) != len(exp):
        raise SystemExit(f"STOP: {row['PDB_ID']} has {len(types)} antigen types but {len(exp)} "
                         "antigen sequences; ag<i> is positional in both.")
    for i, t in enumerate(types):
        if t not in POLY:
            continue
        # RESOLVED FIRST, then expected: the crystal holds what was ordered, not the construct.
        for arr in (rsv, exp):
            q = arr[i].strip().upper() if i < len(arr) else ""
            if q:
                want.append((f"ag{i}", q))
                break
    return assign_chains([(k, q) for k, q in want if q], seqs)


def write_job_cif(st, keep, expected, out_cif, max_unplaced_frac=0.15):
    """in: the entry structure, [(role, chain, lo, hi)], {role: expected sequence}, output path.

    THE LABEL'S FRAME IS THE FOLD'S EXPECTED SEQUENCE. Each output chain gets its own entity whose
    full_sequence is the sequence the fold was given for that role, and every kept residue is placed
    at its position in that sequence. Unresolved positions then simply have no residue, which is how
    Protenix's pipeline masks them -- the label covers what the crystal ordered, expressed in the
    coordinates the pose uses.

    EXPLICIT ENTITIES AND SUBCHAINS, not whatever setup_entities() leaves behind. Trimming and
    splitting mutate a structure in ways gemmi does not repair: splitting a single-chain construct
    produced two auth chains sharing one subchain AND one entity whose full_sequence was the
    259-residue fusion (9vb4, chains D and C both on subchain F, entity 6). Two auth chains under one
    label_asym is a wrong chain decomposition, and every chain-pair quantity the confidence head
    produces depends on it. Ordinary chains happened to come out right; accident is not an invariant.
    """
    import difflib
    import gemmi
    out = gemmi.Structure()
    out.name = st.name
    out.spacegroup_hm = st.spacegroup_hm
    out.cell = st.cell
    out.resolution = st.resolution
    model = gemmi.Model("1")
    names = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
    ents, chains, report = [], [], []
    for n, (role, cname, lo, hi) in enumerate(keep):
        exp = expected[role].upper()
        src = st[0][cname]
        res = [src[i].clone() for i in range(lo, hi)]
        obs = gemmi.one_letter_code([r.name for r in res]).upper()
        # POSITION EVERY KEPT RESIDUE IN THE EXPECTED SEQUENCE. One alignment per output chain, and
        # anything that cannot be placed is reported rather than renumbered silently.
        sm = difflib.SequenceMatcher(None, exp, obs, autojunk=False)
        pos = {}
        for b in sm.get_matching_blocks():
            for k in range(b.size):
                pos[b.b + k] = b.a + k
        # UNPLACED RESIDUES ARE REPORTED, AND A LARGE LOSS FAILS. An observed residue that does not
        # align to the fold's sequence cannot belong in the label -- the label's frame IS that
        # sequence -- so dropping it is correct wherever it sits. Two causes, both measured:
        #   - TERMINAL OVERHANG: SAbDab2's Hseq is the variable domain while the crystal chain runs
        #     into the constant region. 9p1m observes 215 residues against an expected 193, the extra
        #     22 being CH1 (...SVFPLAPSGTAALGC...).
        #   - SHORT INSERTIONS at the variable/constant junction, where the crystallised construct
        #     differs from the expected sequence: 8fg0 observes SVFPLAPS(SK)GTAALG, 9hud (AQT).
        # Neither is in the folded system. What was wrong before is that this happened SILENTLY --
        # 17 of 501 structures lost residues, up to 49. Now every loss is written to --unplaced and a
        # chain losing more than --max-unplaced-frac of its observed residues stops the phase, since
        # that is no longer a junction artefact.
        unplaced = [i for i in range(len(res)) if i not in pos]
        if unplaced and len(unplaced) > max_unplaced_frac * len(res):
            return None, (f"{role}: {len(unplaced)} of {len(res)} observed residues do not align to "
                          f"the fold's sequence ({100*len(unplaced)/len(res):.0f}%, limit "
                          f"{100*max_unplaced_frac:.0f}%) -- too much to be a junction artefact")
        new_name = next(names)
        ch = gemmi.Chain(new_name)
        for i, r in enumerate(res):
            if i not in pos:
                continue
            r.subchain = f"{new_name}p"
            r.label_seq = pos[i] + 1
            r.het_flag = "A"
            ch.add_residue(r)
        model.add_chain(ch)
        ent = gemmi.Entity(str(n + 1))
        ent.entity_type = gemmi.EntityType.Polymer
        ent.polymer_type = gemmi.PolymerType.PeptideL
        ent.subchains = [f"{new_name}p"]
        ent.full_sequence = gemmi.expand_one_letter_sequence(exp, gemmi.ResidueKind.AA)
        ents.append(ent)
        chains.append(new_name)
        report.append((role, cname, new_name, len(ch), len(exp), len(unplaced)))
    out.add_model(model)
    out.entities = gemmi.EntityList()
    for e in ents:
        out.entities.append(e)
    asm = gemmi.Assembly("1")
    gen = gemmi.Assembly.Gen()
    gen.chains = chains
    gen.operators.append(gemmi.Assembly.Operator())
    asm.generators.append(gen)
    asm.author_determined = True
    asm.oligomeric_details = f"{len(chains)}-mer"
    out.assemblies.append(asm)
    os.makedirs(os.path.dirname(out_cif), exist_ok=True)
    out.make_mmcif_document().write_file(out_cif)
    return report, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=f"{ROOT}/results/cache_train_structures.csv")
    ap.add_argument("--splits", default=f"{ROOT}/data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--src-dir", default=f"{ROOT}/data/rcsb_training_cifs")
    ap.add_argument("--out-dir", default=f"{ROOT}/train_prep/job_cifs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--unplaced", default=f"{ROOT}/train_prep/unplaced_residues.csv",
                    help="per-chain record of observed residues dropped because they are not in the "
                         "fold's sequence; written every run, reviewed, never silent")
    ap.add_argument("--max-unplaced-frac", type=float, default=0.15,
                    help="a chain losing more than this fraction of its observed residues STOPS")
    ap.add_argument("--jobs-dir", default=f"{ROOT}/train_jobs",
                    help="the fold's job jsons; the written cif's sequences are asserted against them")
    ap.add_argument("--skipped", default=f"{ROOT}/train_prep/skipped_job_cifs.csv",
                    help="structures whose chains cannot be resolved, with the reason. A failure "
                         "not already listed here is a STOP.")
    a = ap.parse_args()

    sel = list(csv.DictReader(open(a.selection)))
    src = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}
    if a.limit:
        sel = sel[:a.limit]
    import gemmi
    ok, bad, unplaced_rows = 0, [], []
    for r in sel:
        row = src[r["instance"]]
        st = gemmi.read_structure(os.path.join(a.src_dir, f"{r['pdb']}.cif"))
        st.setup_entities()
        st.remove_alternative_conformations()
        st.remove_hydrogens()
        # ONE MODEL. An NMR entry carries many; the fold predicts one structure.
        while len(st) > 1:
            del st[1]
        pairs, unresolved = job_chains(row, chain_seqs(st))
        if unresolved:
            bad.append((r["pdb"], r["instance"],
                        f"no chain matches the {unresolved} sequence(s) at 85% identity"))
            continue
        # THE EXPECTED SEQUENCE PER ROLE is what the fold was given, so it is the frame the label
        # must be expressed in -- NOT agresolvedseqs, which is only what the crystal ordered.
        exp_of = {"H": (row["Hseq"] or "").strip().upper(),
                  "L": (row["Lseq"] or "").strip().upper()}
        _types = str(row["agtypes"]).split("/")
        _exp = str(row["agexpectedseqs"]).split("/")
        for i2, t2 in enumerate(_types):
            if t2 in POLY and i2 < len(_exp) and _exp[i2].strip():
                exp_of[f"ag{i2}"] = _exp[i2].strip().upper()
        out = os.path.join(a.out_dir, f"{r['pdb']}.cif")
        expected = exp_of
        info, why = write_job_cif(st, pairs, exp_of, out, a.max_unplaced_frac)
        if info:
            for role, src_c, out_c, kept, expn, unp in info:
                unplaced_rows.append({"pdb": r["pdb"], "role": role, "src_chain": src_c,
                                      "out_chain": out_c, "kept": kept, "expected": expn,
                                      "unplaced": unp})
        if why:
            bad.append((r["pdb"], r["instance"], why))
            continue
        # ASSERT AGAINST THE FOLD INPUT. The label's frame has to be the sequences the fold was
        # actually given, so the entity sequences of the written cif are compared to the job json
        # phase G produced. Anything else means the label describes a different molecule.
        # READ THE WRITTEN FILE BACK. Comparing the expected-sequence dict we just built proves
        # nothing about what landed on disk; and skipping the check when the job json is absent made
        # the only assertion optional. Both are now mandatory and read the output.
        jj = os.path.join(a.jobs_dir, f"{r['pdb']}-final-updated.json")
        if not os.path.exists(jj):
            bad.append((r["pdb"], r["instance"],
                        f"no job json at {jj}; the label cannot be checked against the fold input"))
            continue
        want = sorted((sq["proteinChain"]["sequence"] or "").strip().upper()
                      for task in json.load(open(jj)) for sq in task["sequences"]
                      if "proteinChain" in sq)
        back = gemmi.read_structure(out)
        back.setup_entities()
        got = sorted(gemmi.one_letter_code(list(e.full_sequence)).upper()
                     for e in back.entities
                     for _ in e.subchains if e.entity_type == gemmi.EntityType.Polymer)
        if want != got:
            bad.append((r["pdb"], r["instance"],
                        f"the written cif's entity sequences do not match the job json: "
                        f"{len(got)} vs {len(want)} chains, lengths {[len(x) for x in got]} "
                        f"vs {[len(x) for x in want]}"))
            continue
        # and every residue must sit at a position its entity sequence actually has
        for ch in back[0]:
            ent = next((e for e in back.entities if ch[0].subchain in e.subchains), None)
            if ent is None:
                bad.append((r["pdb"], r["instance"], f"chain {ch.name} has no entity"))
                break
            n = len(ent.full_sequence)
            off = [rr.label_seq for rr in ch if not (1 <= (rr.label_seq or 0) <= n)]
            if off:
                bad.append((r["pdb"], r["instance"],
                            f"chain {ch.name} has {len(off)} residue(s) outside its entity "
                            f"sequence of {n}: label_seq {off[:5]}"))
                break
        else:
            ok += 1
        continue

    with open(a.unplaced, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "role", "src_chain", "out_chain", "kept",
                                           "expected", "unplaced"], lineterminator="\n")
        w.writeheader()
        w.writerows(unplaced_rows)
    nz = [r for r in unplaced_rows if r["unplaced"]]
    print(f"wrote {ok} job cif(s) to {a.out_dir}")
    print(f"dropped residues not in the fold's sequence: {sum(r['unplaced'] for r in nz)} across "
          f"{len({r['pdb'] for r in nz})} structure(s) -> {a.unplaced}")
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
