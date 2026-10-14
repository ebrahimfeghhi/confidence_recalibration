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
    """{chain: one-letter sequence}, with SELENOMETHIONINE spelled M.

    MEASURED, NOT ASSUMED, AND NARROW ON PURPOSE. `gemmi.one_letter_code` (0.7.5) returns X for every
    modified residue; 27 residues over 10 of the 501 structures render that way (PCA 8, SEP 9, TPO 5,
    MSE 3, DAL 1, MEA 1). Resolving ALL of them to their parent residue made things WORSE, because
    SAbDab2 is not consistent either:

      | structure | SAbDab2's sequence says | the entry's residue |
      |---|---|---|
      | 8rwf ag0 pos 88 | M | MSE |
      | 9p4c H pos 0, 9og1 H pos 0 | X | PCA |

    So mapping every modified residue to its parent fixed 8rwf (coverage 0.994 -> 1.000, one
    selenomethionine recovered) and BROKE five structures whose N-terminal pyroglutamate SAbDab2
    spells X: the chain said E, the query said X, the span started one residue late and 40 atoms were
    lost across the set. MSE alone is the case where the two conventions disagree in the direction
    that costs us a residue -- it is the one modified residue sequence tooling normally normalises --
    so MSE alone is mapped. Everything else keeps gemmi's X, which is what SAbDab2 writes.
    """
    return {ch.name: "".join(res_code(r.name) for r in ch) for ch in st[0]}


def res_code(name):
    """One letter for one residue name. ONE DEFINITION, used by every alignment in this file.

    It was two: chain_seqs spelled MSE as M while write_job_cif's per-chain alignment called
    gemmi.one_letter_code directly and spelled it X. So 8rwf's chain aligned to SAbDab2's sequence at
    coverage 1.000 and then lost that same residue when the kept chain was positioned in the fold's
    expected sequence -- a residue dropped between two alignments that disagreed about one letter.
    """
    import gemmi
    return "M" if name == "MSE" else gemmi.one_letter_code([name]).upper()


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


def polymer_subchains(st):
    """{subchain id} of every POLYMER entity, so polymer membership is read from the file's own
    entity records rather than guessed from `het_flag`.

    WHY NOT het_flag. Excluding every HETATM residue also excludes MODIFIED POLYMER residues --
    selenomethionine is the common one, and this project already had to unify MSE handling once.
    Keeping only `not is_water()` goes the other way and admits calcium, sugars and cryoprotectant:
    7stz chain C read 565 "observed residues" that way, of which 25 were CA, BMA, MAN, NAG, PG4 and
    EDO, and its 4.68 A "contact" with the antibody was a water. Entity type is the only
    unambiguous answer.
    """
    import gemmi
    out = set()
    for e in st.entities:
        if e.entity_type == gemmi.EntityType.Polymer:
            out.update(e.subchains)
    return out


def _heavy(ch, polymer=None):
    """Heavy-atom coordinates of a chain's POLYMER residues.

    `polymer` is the subchain set from polymer_subchains(); without it every non-hydrogen atom is
    taken, which is what let solvent and ions count as contacts.
    """
    import numpy as np
    return np.array([[a.pos.x, a.pos.y, a.pos.z] for r in ch
                     if polymer is None or r.subchain in polymer
                     for a in r if a.element.name != "H"], dtype=float)


def _min_dist(A, B):
    import numpy as np
    if not len(A) or not len(B):
        return float("inf")
    m = float("inf")
    for i in range(0, len(A), 512):
        m = min(m, float(np.sqrt(((A[i:i + 512, None, :] - B[None, :, :]) ** 2).sum(-1)).min()))
    return m


def pick_contacting_copies(pairs, row, seqs, st, cutoff=5.0, ident_floor=0.9):
    """Re-point the SEQUENCE-RESOLVED roles at a copy of the complex that actually touches.

    THE SEQUENCE CANNOT TELL TWO COPIES APART. A crystal routinely holds several copies of the same
    complex, so a role resolved by sequence with an alphabetical tie-break can take the antibody from
    one copy and the antigen from another: same sequences, different assembly, no contact. 9djw is
    the measured case -- SAbDab2 names two-character chains (`AG`, `AE`) that the entry does not
    have, so BOTH roles fell to sequence, the antigen matched exactly one chain and the heavy chain
    matched three, and the pair that came out sat 37.3 A apart.

    ROLES RESOLVED BY NAME ARE NEVER MOVED. SAbDab2's own assignment is authoritative (see
    named_chains): an INSTANCE is a specific pairing and its named chains contact by construction.
    Only chains that sequence matching guessed are candidates for replacement, so this cannot undo
    the names-first fix.

    HOW. Fixed = the named chains. Each free role's candidates are the chains at >= `ident_floor`
    identity to the sequence it resolved to, ordered so the current pick comes first; roles are
    filled fewest-candidates-first, each taking the candidate closest to what is already placed. With
    one candidate the role anchors the rest, which is exactly 9djw. Greedy, not exhaustive: 9dc3 has
    60 candidates for each of two roles and the product is not enumerable, while any contacting copy
    pair is an equally valid label.

    Only the CHAIN changes, never the sequence, so the fold's job json and every pose already
    produced are unaffected -- this is a label-side correction.
    """
    names = role_names(row, seqs)
    free = [t for t in pairs if names.get(t[0]) != t[1]]
    if not free:
        return pairs, []
    ab = [c for role, c, _lo, _hi in pairs if role in ("H", "L")]
    ags = [c for role, c, _lo, _hi in pairs if role.startswith("ag")]
    if not ab or not ags:
        return pairs, []
    # NOTHING TO FIX unless an antigen is out of contact. A structure whose sequence-resolved pick
    # already touches is left exactly as it is.
    hv = {}
    _poly = polymer_subchains(st)
    def heavy(c):
        if c not in hv:
            hv[c] = _heavy(st[0][c], _poly)
        return hv[c]
    if all(min(_min_dist(heavy(g), heavy(x)) for x in ab) <= cutoff for g in ags):
        return pairs, []

    import difflib
    fixed = {t[0]: t for t in pairs if names.get(t[0]) == t[1]}
    cands = {}
    for role, c, lo, hi in free:
        q = seqs[c]
        out = [c]
        for cand, cq in seqs.items():
            if cand == c or not cq:
                continue
            sm = difflib.SequenceMatcher(None, q, cq, autojunk=False)
            if sum(b.size for b in sm.get_matching_blocks()) / max(1, min(len(q), len(cq))) >= ident_floor:
                out.append(cand)
        cands[role] = out
    placed = dict(fixed)
    for role, c, lo, hi in sorted(free, key=lambda t: len(cands[t[0]])):
        taken = {t[1] for t in placed.values()}
        pool = [x for x in cands[role] if x not in taken] or [c]
        if len(pool) == 1 or not placed:
            best = pool[0]
        else:
            ref = [heavy(t[1]) for t in placed.values()]
            best = min(pool, key=lambda x: min(_min_dist(heavy(x), R) for R in ref))
        # RE-ALIGN through the same matcher the first pass used, so the span is this chain's own.
        sp = assign_chains([(role, dict(role_sequences(row))[role])], {best: seqs[best]})[0]
        placed[role] = sp[0] if sp else (role, best, lo, hi)
    order = [k for k, _q in role_sequences(row)]
    out = sorted(placed.values(), key=lambda t: order.index(t[0]))
    moved = [(t[0], o[1], t[1]) for o, t in zip(sorted(pairs, key=lambda t: order.index(t[0])), out)
             if o[1] != t[1]]
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


def role_sequences(row):
    """[(role, the sequence SAbDab2 gives for it)] for H, L and every polypeptide antigen."""
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
    return [(k, q) for k, q in want if q]


def role_names(row, seqs=None):
    """{role: the chain SAbDab2 names for it}, for the roles it names.

    WITH `seqs`, A NAME THE ENTRY DOES NOT HAVE IS RETRIED WITHOUT ITS TRAILING COPY INDEX.
    SAbDab2 numbers the two domains of a single-chain construct `B1`/`B2`, `A1`/`A2`, `D1`/`D2` --
    both are chain `B`, `A`, `D` in the entry, carrying VH and VL on disjoint stretches. Taking those
    names literally left both roles unresolved, so sequence matching placed them, and with five
    copies of the scFv in the entry (8h3c) it could take VH from one copy and VL from another. The
    index is only stripped when the full name is absent and the stripped one is present, so a
    genuinely numeric antigen chain (`4`, `12`) is untouched.
    """
    out = {}
    for role, key in (("H", "Hchain"), ("L", "Lchain")):
        v = (row[key] or "").strip()
        if v:
            out[role] = v
    types = str(row["agtypes"]).split("/")
    names = [c.strip() for c in str(row["agchains"]).split("/")]
    if len(types) == len(names):
        for i, (c, t) in enumerate(zip(names, types)):
            if c and t in POLY:
                out[f"ag{i}"] = c
    if seqs is not None:
        for role, v in list(out.items()):
            if v not in seqs and len(v) > 1 and v[-1].isdigit() and v[:-1] in seqs:
                out[role] = v[:-1]
    return out


def job_chains(row, seqs, st=None):
    """in: a splits row, {chain: sequence} from the entry, and optionally the entry structure.
    out: ([(role, chain, lo, hi)], [unresolved roles]) -- the chains the fold actually contains.

    PER ROLE, names first. SAbDab2's own chain assignment is the correct source (see named_chains);
    sequence resolution is the fallback for a role whose name does not resolve.

    THE FALLBACK IS PER ROLE, NOT ALL-OR-NOTHING. A single unresolvable name -- usually an antigen
    the entry numbers differently -- used to send EVERY role back to sequence matching, so a heavy
    chain that SAbDab2 had named correctly could be reassigned to another copy. The roles whose names
    do resolve are kept and their chains are reserved; only the rest go to sequence.
    """
    want = role_sequences(row)
    names = role_names(row, seqs)
    resolved, spans, leftover = [], collections.defaultdict(list), []
    for role, q in want:
        c = names.get(role)
        if c is None or c not in seqs:
            leftover.append((role, q))
            continue
        sp = spans_for([(role, c)], row, seqs)
        if not sp:
            leftover.append((role, q))      # the named chain does not carry this role's sequence
            continue
        _r, _c, lo, hi = sp[0]
        # TWO ROLES MAY SHARE A CHAIN IF THEIR SPANS ARE DISJOINT -- an scFv puts VH and VL on one
        # chain. Overlapping spans are two roles claiming the same residues, which is a mistake.
        if any(not (hi <= plo or phi <= lo) for plo, phi in spans[c]):
            leftover.append((role, q))
            continue
        resolved.append(sp[0])
        spans[c].append((lo, hi))
    taken = {c for c in spans}
    if not leftover:
        return resolved, []
    # SEQUENCE ONLY FOR WHAT IS LEFT, over the chains the named roles did not claim.
    rest = {c: q for c, q in seqs.items() if c not in taken}
    more, bad = assign_chains(leftover, rest)
    out = resolved + more
    # keep the caller's role order stable
    order = [k for k, _q in want]
    out.sort(key=lambda t: order.index(t[0]))
    # WITH COORDINATES, make the sequence-resolved roles come from ONE copy of the complex.
    if st is not None and not bad:
        out, _moved = pick_contacting_copies(out, row, seqs, st)
    return out, bad


def native_polymer(st):
    """{chain name: (entity sequence, {native position -> residue index in the chain})}.

    The native position is `label_seq_id`: the residue's index in its ENTITY sequence. It is the only
    numbering that survives unresolved density -- auth numbering skips, restarts and carries
    insertion codes -- so it is what every placement here is keyed on.
    """
    import gemmi
    sub2seq = {}
    for e in st.entities:
        if e.entity_type == gemmi.EntityType.Polymer:
            seq = gemmi.one_letter_code(list(e.full_sequence)).upper()
            for sb in e.subchains:
                sub2seq[sb] = seq
    out = {}
    for ch in st[0]:
        seq, pos = None, {}
        for i, r in enumerate(ch):
            if r.subchain not in sub2seq:
                continue                     # ions, sugars, solvent: not polymer
            seq = seq or sub2seq[r.subchain]
            if r.label_seq is not None and r.label_seq > 0:
                pos[int(r.label_seq)] = i
        if seq:
            out[ch.name] = (seq, pos)
    return out


def place_by_native_position(native_seq, expected, native_of_index, span):
    """observed residue index -> position in `expected`, via NATIVE POLYMER POSITIONS.

    WHY NOT LETTERS. The previous version aligned the OBSERVED residue letters against `expected`.
    Letters are not unique -- there are twenty -- so a residue can land in a plausible slot while
    belonging somewhere else: 8h3c's heavy role needed 117 residues, native 119-133 are unresolved,
    the span reached past the gap and took auth 134, and the matcher placed that THR at expected
    position 117 because the real 117 is absent. It sits with the LIGHT domain, 32 A away.

    WHY THIS IS SOUND. The entity's full sequence and `expected` are both COMPLETE -- neither has
    holes punched in it by unresolved density -- so aligning those two is a different and far more
    reliable problem than aligning what happens to be observed. That gives native -> expected once,
    after which each observed residue's own `label_seq` names its slot with no guessing, an
    unresolved native position simply has no residue and becomes a gap rather than an invitation to
    pull one in from next door, and a residue whose native position falls outside the span of this
    role cannot be placed at all.

    Returns (pos, outside, missing_native) where `pos` is observed index -> 0-based expected
    position, `outside` are observed indices whose native position is not in this role's mapped
    span, and `missing_native` are native positions inside the span that `expected` does not
    contain -- the internal omissions a corrected query has to restore.
    """
    import difflib
    sm = difflib.SequenceMatcher(None, native_seq, expected, autojunk=False)
    n2e = {}
    for b in sm.get_matching_blocks():
        for k in range(b.size):
            n2e[b.a + k + 1] = b.b + k          # 1-based native -> 0-based expected
    if not n2e:
        return {}, list(range(len(native_of_index))), []
    lo_n, hi_n = min(n2e), max(n2e)
    pos, outside = {}, []
    for idx, np_ in native_of_index.items():
        if span is not None and not (span[0] <= idx < span[1]):
            continue
        e = n2e.get(np_)
        if e is None:
            outside.append(idx)
        else:
            pos[idx] = e
    missing_native = [n for n in range(lo_n + 1, hi_n) if n not in n2e]
    return pos, outside, missing_native


def drop_geometrically_impossible(res, pos, max_ca=4.5):
    """Remove placements that the geometry says cannot be that residue.

    WHY A LETTER MATCH IS NOT A PLACEMENT. The alignment above positions each observed residue by
    SEQUENCE, which is right, but a residue from past the domain boundary can still align: it only
    has to carry the same amino acid as the expected position, and the expected position itself is
    often unresolved. 8h3c is the clean case -- splitting its single-chain construct gave the heavy
    domain auth residues 2..118, then the span reached past the unresolved stretch 119..133 and took
    auth 134, a THR, which the matcher placed at expected position 117 because the real residue 117
    is absent. Residue 134 sits with the LIGHT domain, 32 A away.

    WHAT THAT COST. Protenix then applies AF3's filter (filter.py:146): two residues whose label_seq
    differ by one and whose CA atoms are more than 10 A apart make the chain physically impossible,
    so the WHOLE CHAIN is deleted. 104 of 501 structures lost a chain from the label this way and
    were refused by the cache builder; 34 more carried a 4.5-10 A break, which the filter passes, so
    they would have trained against a wrong residue SILENTLY. The same parser on the original RCSB
    entry drops no chain, so the break is ours, not upstream.

    THE THRESHOLD IS MEASURED, NOT CHOSEN. Over all 501 job cifs, 308,955 of 310,599 consecutive
    CA-CA distances (99.47%) fall in 3.5-4.0 A, the physical value; the 4.0-4.5 A band holds SIX
    pairs. 4.5 A sits in an empty valley, far from anything real.

    The offender is removed, not renumbered, and is counted as unplaced so the caller's
    --max-unplaced-frac stop applies. Greedy by violation count, because one bad residue can break
    continuity on both sides.
    """
    def ca(i):
        for a in res[i]:
            if a.name == "CA":
                return a.pos
        return None

    dropped = []
    while True:
        placed = sorted(((pos[i], i) for i in pos), key=lambda t: t[0])
        viol = collections.Counter()
        for (s1, i1), (s2, i2) in zip(placed, placed[1:]):
            if s2 - s1 != 1:
                continue
            a, b = ca(i1), ca(i2)
            if a is None or b is None:
                continue
            if a.dist(b) > max_ca:
                viol[i1] += 1
                viol[i2] += 1
        if not viol:
            return dropped
        # the worst offender; a tie goes to the one nearer a terminus, which is where a span
        # overruns its domain
        n = len(placed)
        rank = {i: min(k, n - 1 - k) for k, (_s, i) in enumerate(placed)}
        worst = min(viol, key=lambda i: (-viol[i], rank[i]))
        del pos[worst]
        dropped.append(worst)


def write_job_cif(st, keep, expected, out_cif, max_unplaced_frac=0.15,
                  resolution=None, method=None):
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
    ents, chains, report, native_gaps = [], [], [], []
    for n, (role, cname, lo, hi) in enumerate(keep):
        exp = expected[role].upper()
        src = st[0][cname]
        res = [src[i].clone() for i in range(lo, hi)]
        # POSITION EVERY KEPT RESIDUE THROUGH ITS NATIVE POLYMER POSITION, not by letter. See
        # place_by_native_position for why the letter alignment was unsound and what it cost.
        _nat = native_polymer(st).get(cname)
        if _nat is None:
            return None, f"{role}: chain {cname} has no polymer entity in the deposited file"
        _nseq, _npos = _nat
        # observed index within this role's span -> its native position
        _idx2nat = {i - lo: np_ for np_, i in _npos.items() if lo <= i < hi}
        pos, _outside, _missing_native = place_by_native_position(
            _nseq, exp, _idx2nat, span=(0, len(res)))
        # CLASSIFY EACH MISSING RUN; DO NOT RESTORE ANYTHING AUTOMATICALLY. Two different things
        # produce a native position the query lacks, and they call for opposite actions:
        #
        #   a GENUINE OMISSION -- a real piece of this domain that the query dropped because it was
        #     unresolved. 11zw's heavy query omits native 159-165, SKSTSGG. It must be restored, and
        #     until it is, the fold joins two residues 10.78 A apart as neighbours.
        #   an INTER-DOMAIN LINKER -- the synthetic tether of a single-chain construct, which belongs
        #     to NEITHER role. 8h3c's heavy query "omits" native 119-133, GGGGSGGGGSGGGGS. Restoring
        #     that would splice a linker into an antibody domain.
        #
        # A FIRST ATTEMPT AT THIS WAS WRONG and is worth recording: I tried "the run lies between the
        # two roles' spans". It does not. The linker is UNRESOLVED, so it appears in neither span,
        # and 8h3c's heavy span runs native 5-151 with the linker sitting inside it. The spans are
        # contiguous at 151|152 and the test found nothing.
        #
        # So the run is CLASSIFIED and REPORTED, not acted on. `kind` is a judgement a reviewer can
        # overrule from the csv, and nothing is spliced into a sequence on the strength of it.
        def _runs(xs):
            out = []
            for x in sorted(xs):
                if out and x == out[-1][1] + 1:
                    out[-1][1] = x
                else:
                    out.append([x, x])
            return [(a2, b2) for a2, b2 in out]

        def _classify(a2, b2):
            seg = _nseq[a2 - 1:b2]
            gs = sum(1 for ch2 in seg if ch2 in "GS")
            shared = any(c2 == cname and r2 != role for (r2, c2, _l2, _h2) in keep)
            if shared and len(seg) >= 5 and gs / len(seg) >= 0.7:
                return "linker"          # GS-rich, on a chain carrying another role: a tether
            if gs / max(1, len(seg)) >= 0.8 and len(seg) >= 8:
                return "linker?"         # GS-rich but the chain carries one role: flag, do not act
            return "omission"

        # RECORDED, NOT INFERRED. The native positions this role's query omits are what a corrected
        # query must restore, and they are reported per role rather than rediscovered later.
        for _a, _b in _runs(_missing_native):
            native_gaps.append(dict(
                pdb=st.name, role=role, src_chain=cname, native_from=_a, native_to=_b,
                n_missing=_b - _a + 1, sequence=_nseq[_a - 1:_b], kind=_classify(_a, _b),
                roles_on_chain=sum(1 for (_r2, c2, _l2, _h2) in keep if c2 == cname),
                outside_span=len(_outside), expected_len=len(exp), native_len=len(_nseq)))
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
        # GEOMETRY VETOES A LETTER MATCH. See drop_geometrically_impossible: a residue from past the
        # domain boundary can align by sequence and still be 32 A from its neighbour, which costs
        # the whole chain downstream.
        misplaced = drop_geometrically_impossible(res, pos)
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
        report.append((role, cname, new_name, len(ch), len(exp), len(unplaced),
                       len(misplaced)))
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
    doc = out.make_mmcif_document()
    # THE RESOLUTION, OR THE CONFIDENCE LOSS IS SILENTLY ZERO. The job cif is built from scratch and
    # carried none of the entry's experimental metadata, so Protenix's parser fell through to its
    # -1.0 default (data/core/parser.py:182-208) and every prepared row read resolution -1.0.
    # loss.py:1735-1752 then compares -1.0 against loss.resolution [0.1, 4.0], sets
    # has_valid_resolution to 0, and ZEROES plddt / pde / pae / resolved -- the four terms FT3
    # trains. The loss would have been exactly 0.000000 on every structure in the set, with no error
    # anywhere. Found by phase_j_replay.py.
    #
    # The item depends on the method, which is what the parser looks for in order: refine for
    # crystallography, em_3d_reconstruction for cryo-EM. The value is SAbDab2's own, not invented;
    # when there is none the field is left out and the parser's -1.0 stands, which is Protenix's own
    # handling of a method without a resolution (phase_j_build_cache.py then refuses the unit rather
    # than training it at zero loss).
    if resolution is not None:
        try:
            r = float(resolution)
        except (TypeError, ValueError):
            r = -1.0
        if r > 0:
            blk = doc.sole_block()
            if str(method).upper() == "EM":
                blk.set_pair("_em_3d_reconstruction.resolution", f"{r:.2f}")
                blk.set_pair("_em_3d_reconstruction.entry_id", out.name or "job")
            else:
                blk.set_pair("_refine.ls_d_res_high", f"{r:.2f}")
                blk.set_pair("_refine.entry_id", out.name or "job")
    doc.write_file(out_cif)
    return (report, native_gaps), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default=f"{ROOT}/train_val_splits/cache_train_structures.csv")
    ap.add_argument("--splits", default=f"{ROOT}/SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--src-dir", default=f"{ROOT}/SAbDab2_data/rcsb_training_cifs")
    ap.add_argument("--out-dir", default=f"{ROOT}/training/prep/job_cifs")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated pdb ids, for testing one path "
                                              "without rewriting every job cif")
    ap.add_argument("--native-gaps", default=f"{ROOT}/training/prep/native_gaps.csv",
                    help="roles whose folded query omits internal native residues -- the refold list")
    ap.add_argument("--unplaced", default=f"{ROOT}/training/prep/unplaced_residues.csv",
                    help="per-chain record of observed residues dropped because they are not in the "
                         "fold's sequence; written every run, reviewed, never silent")
    ap.add_argument("--max-unplaced-frac", type=float, default=0.15,
                    help="a chain losing more than this fraction of its observed residues STOPS")
    ap.add_argument("--jobs-dir", default=f"{ROOT}/training/jobs",
                    help="the fold's job jsons; the written cif's sequences are asserted against them")
    ap.add_argument("--skipped", default=f"{ROOT}/training/prep/skipped_job_cifs.csv",
                    help="structures whose chains cannot be resolved, with the reason. A failure "
                         "not already listed here is a STOP.")
    a = ap.parse_args()

    sel = list(csv.DictReader(open(a.selection)))
    src = {r["INSTANCE"]: r for r in csv.DictReader(open(a.splits))}
    if a.only:
        keep_ids = {x.strip().lower() for x in a.only.split(",") if x.strip()}
        sel = [r for r in sel if r["pdb"].lower() in keep_ids]
        if not sel:
            raise SystemExit(f"STOP: --only {a.only} matched no row of {a.selection}")
    if a.limit:
        sel = sel[:a.limit]
    import gemmi
    ok, bad, unplaced_rows = 0, [], []
    # NATIVE POSITIONS A ROLE'S QUERY OMITS, one row per role that has any. THIS IS THE REFOLD LIST:
    # a query missing an INTERNAL native stretch folded a molecule whose residues are covalently
    # adjacent when they are not, and no label-side repair can add those tokens back.
    native_gap_rows = []
    for r in sel:
        row = src[r["instance"]]
        st = gemmi.read_structure(os.path.join(a.src_dir, f"{r['pdb']}.cif"))
        st.setup_entities()
        st.remove_alternative_conformations()
        st.remove_hydrogens()
        # ONE MODEL. An NMR entry carries many; the fold predicts one structure.
        while len(st) > 1:
            del st[1]
        pairs, unresolved = job_chains(row, chain_seqs(st), st)
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
        info, why = write_job_cif(st, pairs, exp_of, out, a.max_unplaced_frac,
                                  resolution=row.get("resolution"), method=row.get("method"))
        if info:
            rep, gaps = info
            for role, src_c, out_c, kept, expn, unp, mis in rep:
                unplaced_rows.append({"pdb": r["pdb"], "role": role, "src_chain": src_c,
                                      "out_chain": out_c, "kept": kept, "expected": expn,
                                      "unplaced": unp, "misplaced": mis})
            for g in gaps:
                g["pdb"] = r["pdb"]
                if g["n_missing"]:
                    native_gap_rows.append(g)
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
        # THE CHAIN DECOMPOSITION, CHECKED ON THE FILE AS WRITTEN. Review round 5's J1: splitting a
        # single-chain construct left its two auth chains sharing ONE subchain and one fused entity,
        # so two roles sat under one label_asym and every chain-pair quantity the confidence head
        # produces was computed over the wrong decomposition. 18 of 501 structures are such
        # constructs (two roles on one source chain), so this is not a rare path.
        #
        # READ WITHOUT setup_entities(). It MERGES entities that share a sequence, which is exactly
        # what a homodimeric antigen has -- 8dpl and 8f6x each carry their antigen twice -- so a
        # check run after it reported those two as sharing an entity when the file on disk gives them
        # their own (Cp -> 3, Dp -> 4). One entity per identical sequence would in fact be correct
        # mmCIF; two auth chains under one SUBCHAIN is the defect, and that is what is checked here.
        raw = gemmi.read_structure(out)
        sub2chain = collections.defaultdict(set)
        for ch in raw[0]:
            for rr in ch:
                sub2chain[rr.subchain].add(ch.name)
        shared = {sb: sorted(c) for sb, c in sub2chain.items() if len(c) > 1}
        multi = {ch.name: sorted({rr.subchain for rr in ch}) for ch in raw[0]
                 if len({rr.subchain for rr in ch}) != 1}
        in_two = collections.Counter(sb for e in raw.entities for sb in e.subchains)
        dup = {sb: n for sb, n in in_two.items() if n > 1}
        if shared or multi or dup:
            bad.append((r["pdb"], r["instance"],
                        f"the written cif's chain decomposition is wrong: subchains shared by two "
                        f"auth chains {shared}, chains without exactly one subchain {multi}, "
                        f"subchains claimed by two entities {dup}"))
            continue
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
                                           "expected", "unplaced", "misplaced"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(unplaced_rows)

    # THE REFOLD LIST. Written every run, so it cannot drift from the cifs this run produced.
    with open(a.native_gaps, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "role", "src_chain", "native_from", "native_to",
                                           "n_missing", "sequence", "kind", "roles_on_chain",
                                           "outside_span", "expected_len", "native_len"],
                           lineterminator="\n")
        w.writeheader(); w.writerows(native_gap_rows)
    import collections as _c
    _k = _c.Counter(g["kind"] for g in native_gap_rows)
    _om = [g for g in native_gap_rows if g["kind"] == "omission"]
    print(f"internal native runs missing from a folded query: {len(native_gap_rows)} over "
          f"{len({g['pdb'] for g in native_gap_rows})} structure(s) -> {a.native_gaps}")
    for _kk, _vv in _k.most_common():
        print(f"   {_vv:5d}  {_kk}")
    print(f"  GENUINE OMISSIONS -> the refold list: {len(_om)} run(s) over "
          f"{len({g['pdb'] for g in _om})} structure(s), {sum(g['n_missing'] for g in _om)} residues")
    nz = [r for r in unplaced_rows if r["unplaced"]]
    mz = [r for r in unplaced_rows if r.get("misplaced")]
    print(f"wrote {ok} job cif(s) to {a.out_dir}")
    print(f"dropped residues not in the fold's sequence: {sum(r['unplaced'] for r in nz)} across "
          f"{len({r['pdb'] for r in nz})} structure(s) -> {a.unplaced}")
    # THE GEOMETRIC VETO, COUNTED SEPARATELY. These aligned by sequence and were removed because the
    # geometry says they are not that residue; see drop_geometrically_impossible.
    print(f"of those, vetoed by geometry: {sum(r['misplaced'] for r in mz)} residue(s) across "
          f"{len({r['pdb'] for r in mz})} structure(s)")
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
