"""Recover interfaces OST dropped because it left a target chain unmapped.

WHY. `ost/mol/alg/scoring.py:1121` admits a target interface only
`if i[0] in flat_mapping and i[1] in flat_mapping`, so an interface disappears whenever either of
its target chains is absent from OST's own chain mapping. On four structures that costs a real
interface: 9vaw_AB (ag0-H, 185 contacts at 5 A), 9vb4_AD (ag0-L, 164), 9q5v_EJ (ag1-H, 123),
9q5m_EH (ag2-H, 81). For 9vb4 the mapping covers target A, B and C and omits target D, the
111-residue light chain, although model chain B matches it at identity 1.00.

WHY `-c` ALONE CANNOT FIX IT. OST validates a custom mapping against its own chemical grouping and
refuses the correct map outright:

    RuntimeError: Chem group mismatch in custom mapping: target chain "D" groups with the following
    chemically equivalent target chains: ['B', 'D'] but model chain "B" maps to the following
    target chains: ['C']

For 9vb4, model chain B is the 111-residue LIGHT chain and target C is the 119-residue HEAVY chain,
and OST considers them chemically equivalent -- they are homologous immunoglobulin V domains. Target
C is already taken by the real heavy chain, so model B has nowhere to go and target D, the light
chain, is left unmapped. This confirms the chem-grouping hypothesis from the diagnosis, in OST's own
words, and it means the grouping cannot be overridden from the command line.

WHAT THIS DOES INSTEAD. It isolates the two chains of the wanted interface -- in both reference and
model -- and scores that pair on its own. With one chain per chemical group the grouping is trivial,
the mapping comes out right, and DockQ is computed on exactly the two chains it is defined over.
For 9vb4_AD that returns chain_mapping {A: C, D: B} and DockQ 0.894, where the full-complex run
returned nothing at all. The parser is untouched throughout -- it was wrongly accused of a
column-order bug and is correct: OST 2.8.0 documents `(trg_ch1, trg_ch2, mdl_ch1, mdl_ch2)`.

THE MAP IS VALIDATED, NOT ASSUMED. Each reference chain must have exactly one model chain above
`--min-ident`, and that model chain must not be the best match for any other reference chain.
Anything ambiguous is refused rather than guessed: forcing a wrong map would silently attribute one
interface's DockQ to another, which is the failure this whole investigation was about.

REFERENCE NAMES ARE label_asym_id, NOT auth. OST reports 9vb4's reference chains as A, B, C, D while
the file's auth names are A, B, D1, D2 -- `C` is auth `D1`. `-c` keys must therefore be label ids.
(`ost_dockq._identity_chain_map` builds its retry map from gemmi's `ch.name`, which is the auth name,
so that fallback map is wrong for any structure where the two differ. Not changed here.)

AND THE RESULT IS CHECKED AGAINST THE INTENDED PHYSICAL PAIR. A recovered row is accepted only when
the model chains OST reports for it are the ones the roles say they should be, so a number can never
be filed under an interface it does not belong to.
"""
import argparse, collections, csv, json, os, sys, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import ost_dockq                                                    # noqa: E402

AA = set("ACDEFGHIKLMNPQRSTVWY")


def chain_seqs(path, by_label):
    """{chain name: sequence}. by_label keys on label_asym_id (what OST calls the chain)."""
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    out = collections.defaultdict(str)
    for ch in st[0]:
        for res in ch:
            t = gemmi.find_tabulated_residue(res.name)
            c = t.one_letter_code.upper() if t else "X"
            if c in AA:
                out[res.subchain if by_label else ch.name] += c
    return {k: v for k, v in out.items() if v}


def build_map(ref_seqs, mdl_seqs, min_ident):
    """[(ref, mdl, ident)] one-to-one, or ([], reason) if anything is ambiguous."""
    scores = {}
    for rc, rs in ref_seqs.items():
        for mc, ms in mdl_seqs.items():
            scores[(rc, mc)] = difflib.SequenceMatcher(None, rs, ms, autojunk=False).ratio()
    pairs, used, bad = [], set(), []
    for rc in sorted(ref_seqs):
        ranked = sorted(((scores[(rc, mc)], mc) for mc in mdl_seqs), reverse=True)
        if not ranked or ranked[0][0] < min_ident:
            bad.append(f"{rc}: best model match only {ranked[0][0]:.2f}" if ranked else f"{rc}: none")
            continue
        top, mc = ranked[0]
        runner = ranked[1][0] if len(ranked) > 1 else 0.0
        if runner >= min_ident and top - runner < 0.05:
            bad.append(f"{rc}: ambiguous, {ranked[0][1]}={top:.2f} vs {ranked[1][1]}={runner:.2f}")
            continue
        if mc in used:
            bad.append(f"{rc}: model chain {mc} already taken")
            continue
        used.add(mc)
        pairs.append((rc, mc, top))
    return pairs, bad


def write_subset(src, keep, out, by_label):
    """Write a cif holding only `keep`, preserving chain names so -c and the parser still line up."""
    import gemmi
    st = gemmi.read_structure(str(src))
    st.setup_entities()
    nm = gemmi.Structure()
    nm.name, nm.spacegroup_hm, nm.cell = st.name, st.spacegroup_hm, st.cell
    mod = gemmi.Model("1")
    for ch in st[0]:
        nc = gemmi.Chain(ch.name)
        used = False
        for res in ch:
            if (res.subchain if by_label else ch.name) in keep:
                nc.add_residue(res); used = True
        if used:
            mod.add_chain(nc)
    nm.add_model(mod)
    nm.setup_entities()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    nm.make_mmcif_document().write_file(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True,
                    help="pdb:iface,... e.g. 9vb4:9vb4_AD,9vaw:9vaw_AB")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold_regen")
    ap.add_argument("--refs", default=f"{ROOT}/training/refs")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles.csv")
    ap.add_argument("--seed", default="500000")
    ap.add_argument("--min-ident", type=float, default=0.80)
    ap.add_argument("--out", default="")
    ap.add_argument("--scratch", default="/tmp/rescore_unmapped")
    a = ap.parse_args()

    roles = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.roles)):
        roles[r["pdb"]][r["ref_chain"]] = (r["role"], r["model_chain"])

    want = collections.defaultdict(list)
    for tok in a.targets.split(","):
        pdb, ifc = tok.split(":")
        want[pdb].append(ifc)

    import glob
    rows = []
    for pdb, ifaces in sorted(want.items()):
        ref = os.path.join(a.refs, f"{pdb}.cif")
        pd = sorted(glob.glob(f"{a.fold}/poses/{pdb}/{a.seed}-*"))
        pd = [d for d in pd if os.path.exists(os.path.join(d, "run.json"))]
        if not pd or not os.path.exists(ref):
            print(f"  {pdb}: no pose dir or no reference -- skipped"); continue
        mdl = sorted(glob.glob(f"{pd[0]}/predictions/*_sample_0.cif"))
        if not mdl:
            print(f"  {pdb}: no sample_0 cif -- skipped"); continue

        rs = chain_seqs(ref, by_label=True)       # label ids: what OST calls the chain
        ms = chain_seqs(mdl[0], by_label=False)
        pairs, bad = build_map(rs, ms, a.min_ident)
        print(f"=== {pdb} seed {a.seed}")
        print(f"  reference chains (label): " +
              ", ".join(f"{k}={len(v)}" for k, v in sorted(rs.items())))
        print(f"  model chains:             " +
              ", ".join(f"{k}={len(v)}" for k, v in sorted(ms.items())))
        for rc, mc, ident in pairs:
            ro = roles[pdb].get(rc, ("?", "?"))
            flag = "" if ro[1] in ("?", mc) else f"   <-- chain_roles says model {ro[1]}"
            print(f"    map {rc} -> {mc}  ident {ident:.2f}  role {ro[0]}{flag}")
        for b in bad:
            print(f"    REFUSED {b}")
        if bad:
            print(f"  {pdb}: map incomplete, NOT rescored (a forced wrong map would mis-attribute)")
            continue

        cm = {rc: mc for rc, mc, _ in pairs}
        conf = glob.glob(f"{pd[0]}/predictions/*summary_confidence_sample_0.json")
        for ifc in ifaces:
            c1, c2 = ifc.split("_", 1)[1][0], ifc.split("_", 1)[1][1]
            if c1 not in cm or c2 not in cm:
                print(f"    {ifc}: one of its target chains is not in the validated map"); continue
            # ISOLATE THE PAIR. One chain per chemical group, so the grouping cannot collapse H onto
            # L and the mapping comes out right. DockQ is defined over exactly these two chains.
            rsub = f"{a.scratch}/{pdb}_{ifc}_ref.cif"
            msub = f"{a.scratch}/{pdb}_{ifc}_mdl.cif"
            write_subset(ref, {c1, c2}, rsub, by_label=True)
            write_subset(mdl[0], {cm[c1], cm[c2]}, msub, by_label=False)
            out = f"{a.scratch}/{pdb}_{ifc}.json"
            if not ost_dockq.invoke(msub, rsub, out):
                print(f"    {ifc}: OST failed on the isolated pair"); continue
            res = ost_dockq.parse(out, conf_json=conf[0] if conf else None)
            v = ost_dockq.interface(res, c1, c2)
            if not v or v.get("dockq") is None:
                print(f"    {ifc}: STILL MISSING"); continue
            # CHECK THE PHYSICAL PAIR, not just that a number came back
            got = tuple(v.get("model_chains") or ())
            exp = tuple(roles[pdb].get(c, ("?", "?"))[1] for c in (c1, c2))
            r1 = roles[pdb].get(c1, ("?",))[0]; r2 = roles[pdb].get(c2, ("?",))[0]
            good = set(got) == set(exp)
            print(f"    {ifc}: dockq={v['dockq']} pair_iptm={v.get('pair_iptm')} "
                  f"roles {r1}-{r2}  model chains {got} expected {exp} "
                  f"{'OK' if good else 'MISMATCH -- rejected'}")
            if good:
                rows.append(dict(pdb=pdb, iface=ifc, seed=a.seed, roles=f"{r1}-{r2}",
                                 dockq=v["dockq"], pair_iptm=v.get("pair_iptm"),
                                 chain_map=" ".join(cm)))
    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nwrote {a.out} ({len(rows)} recovered interface(s))")


if __name__ == "__main__":
    main()
