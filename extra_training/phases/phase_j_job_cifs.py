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


def job_chains(row, seqs):
    """in: a splits row and {chain: sequence} from the entry.
    out: ([(role, chain)], [unresolved roles]) -- the chains the fold actually contains."""
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


def write_job_cif(st, keep, out_cif):
    """Keep `keep` auth chains and write a one-assembly mmCIF. Returns (n_chains, n_residues)."""
    import gemmi
    # POLYMER ONLY. The fold's job json carries sequences and nothing else, so the pose has no ions,
    # no ligands and no waters. Left in, they appear in the label as atoms the pose cannot have --
    # a first test kept 8wnp's ion chains I and J and the indices csv duly listed intra_ions rows.
    st.remove_ligands_and_waters()
    # ONE MODEL. An NMR entry carries many; the fold predicts one structure, and a label built from
    # several superposed models would not correspond to it.
    by_chain = collections.defaultdict(list)
    for c, lo, hi in keep:
        by_chain[c].append((lo, hi))
    for ch in [c.name for c in st[0]]:
        if ch not in by_chain:
            st[0].remove_chain(ch)
    # ONE CHAIN PER SPAN. The fold's job json lists the heavy and light sequences as separate
    # proteinChain entries, so a single-chain construct must be SPLIT here or the label would carry
    # one chain where the pose has two -- which changes asym_id, the frames, and every chain-pair
    # quantity the confidence head produces.
    used = {c.name for c in st[0]}
    for cname, spans in sorted(by_chain.items()):
        src_ch = st[0][cname]
        resid = [r.clone() for r in src_ch]
        spans = sorted(spans)
        for n, (lo, hi) in enumerate(spans):
            if n == 0:
                for i in range(len(src_ch) - 1, -1, -1):
                    if not (lo <= i < hi):
                        del src_ch[i]
                continue
            new = next(x for x in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                       if x not in used)
            used.add(new)
            st[0].add_chain(gemmi.Chain(new))
            for i in range(lo, hi):
                st[0][new].add_residue(resid[i])
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
    import gemmi
    ok, bad = 0, []
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
        keep = [(c, lo, hi) for _role, c, lo, hi in pairs]
        out = os.path.join(a.out_dir, f"{r['pdb']}.cif")
        info, why = write_job_cif(st, keep, out)
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
