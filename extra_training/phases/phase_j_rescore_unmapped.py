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


def contained(rs, ms):
    """out: matched residues as a fraction of the SHORTER sequence

    WHY NOT difflib's ratio. ratio divides by the COMBINED length, so a reference chain that is a
    perfect subsequence of a longer model chain scores ~0.68 rather than ~1.0. These references are
    frequently the variable domain alone -- 6lyn's reference chains are 112 and 118 residues against
    model chains of 212 and 217 -- so the correct pairing was rejected by --min-ident on 7 of the 8
    interfaces used to validate this script, while the one reference that happened to hold
    full-length chains (6yxd) mapped at 1.00 and scored correctly.

    Dividing by the shorter length asks the question that matters -- is this reference chain the
    same molecule as that model chain, as far as the reference goes -- and separates the candidates
    cleanly: on 6lyn, 6s3t, 6umg and 6gv1 every reference chain matches exactly one model chain at
    1.00 with the runner-up at 0.38 or below.

    IT IS THE LONGEST CONTIGUOUS RUN, not the sum of matching blocks. Summing lets scattered
    coincidental matches add up to a whole chain whenever that chain is short, and 4zto's
    10-residue peptide antigen GKAMYAPPIR duly scored 1.00 against a 216-residue light chain --
    tying with the real peptide and making the map ambiguous, which is why this script refused
    4zto. Against the longest single run the light chain is 0.20 and the peptide 1.00.

    The one-to-one and ambiguity guards below are unchanged and still refuse anything that is not a
    clear single winner.
    """
    sm = difflib.SequenceMatcher(None, rs, ms, autojunk=False)
    return sm.find_longest_match(0, len(rs), 0, len(ms)).size / min(len(rs), len(ms))


def build_map(ref_seqs, mdl_seqs, min_ident):
    """[(ref, mdl, ident)] one-to-one, or ([], reason) if anything is ambiguous."""
    scores = {}
    for rc, rs in ref_seqs.items():
        for mc, ms in mdl_seqs.items():
            scores[(rc, mc)] = contained(rs, ms)
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


CANON = ["iface", "seed", "arm", "donor_pose", "tmpl_chains", "sample", "dockq", "iptm"]


def canon_row(iface, seed, sample, dockq, iptm, arm="free", donor="-1", tmpl=""):
    """One row in run_model/scorer.py's exact column order and number formatting.

    THE FORMATTING IS PART OF THE CONTRACT, not cosmetic. scorer.py prints dockq as .4f and
    pair_iptm as .4f (empty when absent), and every row already in training/fold/results.csv was
    written that way. A recovered row that carried full float repr would be the one visibly
    different row in the file and would not compare equal to a rescore of the same pose.
    """
    return {"iface": iface, "seed": str(seed), "arm": arm, "donor_pose": donor,
            "tmpl_chains": tmpl, "sample": str(sample),
            "dockq": f"{float(dockq):.4f}",
            "iptm": "" if iptm is None else f"{float(iptm):.4f}"}


def existing_keys(path):
    """{(iface, seed, sample)} already in a results.csv, so a sweep cannot create a duplicate.

    A duplicate is not a cosmetic problem here: every average over a seed would double-weight that
    pose. The sweep is therefore idempotent -- re-running it adds nothing it has already added.
    """
    if not path or not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {(r["iface"], r["seed"], r["sample"]) for r in csv.DictReader(fh)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True,
                    help="pdb:iface,... e.g. 9vb4:9vb4_AD,9vaw:9vaw_AB")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold_regen")
    ap.add_argument("--refs", default=f"{ROOT}/training/refs")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles.csv")
    ap.add_argument("--seed", default="500000")
    ap.add_argument("--seed0", type=int, default=None,
                    help="first seed of a range; with --seeds this sweeps instead of using --seed")
    ap.add_argument("--seeds", type=int, default=1, help="how many consecutive seeds from --seed0")
    ap.add_argument("--nsample", type=int, default=1, help="diffusion samples per pose dir")
    ap.add_argument("--skip-existing", default="",
                    help="a results.csv whose (iface,seed,sample) rows are already scored")
    ap.add_argument("--rows-out", default="",
                    help="candidate rows in scorer.py's canonical column order, ready to append")
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
    seeds = ([str(a.seed0 + i) for i in range(a.seeds)] if a.seed0 is not None else [str(a.seed)])
    done = existing_keys(a.skip_existing)
    print(f"sweep: {len(want)} structure(s), seeds {seeds[0]}..{seeds[-1]}, "
          f"samples 0..{a.nsample - 1}; {len(done)} (iface,seed,sample) already scored")

    rows, canon, refused, failed, skipped = [], [], [], [], 0
    for pdb, ifaces in sorted(want.items()):
        ref = os.path.join(a.refs, f"{pdb}.cif")
        if not os.path.exists(ref):
            print(f"  {pdb}: no reference -- skipped"); continue
        # THE ISOLATED REFERENCE IS THE SAME FOR EVERY POSE, so it is written once per interface
        # rather than once per pose. 850 poses would otherwise re-derive the same 17 files.
        refsub = {}
        for ifc in ifaces:
            c1, c2 = ifc.split("_", 1)[1][0], ifc.split("_", 1)[1][1]
            p = f"{a.scratch}/{pdb}_{ifc}_ref.cif"
            write_subset(ref, {c1, c2}, p, by_label=True)
            refsub[ifc] = p

        for seed in seeds:
            pd = sorted(glob.glob(f"{a.fold}/poses/{pdb}/{seed}-*"))
            pd = [d for d in pd if os.path.exists(os.path.join(d, "run.json"))]
            if not pd:
                print(f"  {pdb} seed {seed}: no pose dir -- skipped")
                failed.append((pdb, seed, "*", "no pose dir"))
                continue
            for k in range(a.nsample):
                mdl = sorted(glob.glob(f"{pd[0]}/predictions/*_sample_{k}.cif"))
                if not mdl:
                    failed.append((pdb, seed, str(k), "no sample cif"))
                    continue
                todo = [i for i in ifaces if (i, seed, str(k)) not in done]
                if not todo:
                    skipped += len(ifaces); continue

                # THE MAP IS REBUILT PER POSE, not once per structure. Model chain naming is job
                # order and is stable in practice, but a map carried across poses would be an
                # assumption, and the whole bug this recovers from was an assumption about chain
                # names. Re-deriving it costs a sequence read against a multi-second OST call.
                rs = chain_seqs(ref, by_label=True)   # label ids: what OST calls the chain
                ms = chain_seqs(mdl[0], by_label=False)
                pairs, bad = build_map(rs, ms, a.min_ident)
                if bad:
                    for b in bad:
                        refused.append((pdb, seed, str(k), b))
                    continue
                cm = {rc: mc for rc, mc, _ in pairs}
                conf = glob.glob(f"{pd[0]}/predictions/*summary_confidence_sample_{k}.json")

                for ifc in todo:
                    c1, c2 = ifc.split("_", 1)[1][0], ifc.split("_", 1)[1][1]
                    if c1 not in cm or c2 not in cm:
                        failed.append((pdb, seed, str(k), f"{ifc}: chain outside the validated map"))
                        continue
                    # ISOLATE THE PAIR. One chain per chemical group, so the grouping cannot
                    # collapse H onto L; DockQ is computed over exactly the two chains it is
                    # defined over.
                    msub = f"{a.scratch}/{pdb}_{ifc}_{seed}_{k}_mdl.cif"
                    write_subset(mdl[0], {cm[c1], cm[c2]}, msub, by_label=False)
                    out = f"{a.scratch}/{pdb}_{ifc}_{seed}_{k}.json"
                    if not ost_dockq.invoke(msub, refsub[ifc], out):
                        failed.append((pdb, seed, str(k), f"{ifc}: OST failed on the isolated pair"))
                        continue
                    res = ost_dockq.parse(out, conf_json=conf[0] if conf else None)
                    v = ost_dockq.interface(res, c1, c2)
                    if not v or v.get("dockq") is None:
                        failed.append((pdb, seed, str(k), f"{ifc}: still missing"))
                        continue
                    # CHECK THE PHYSICAL PAIR, not merely that a number came back
                    got = tuple(v.get("model_chains") or ())
                    exp = tuple(roles[pdb].get(c, ("?", "?"))[1] for c in (c1, c2))
                    r1 = roles[pdb].get(c1, ("?",))[0]; r2 = roles[pdb].get(c2, ("?",))[0]
                    if set(got) != set(exp):
                        failed.append((pdb, seed, str(k),
                                       f"{ifc}: model chains {got} but roles expect {exp}"))
                        continue
                    rows.append(dict(pdb=pdb, iface=ifc, seed=seed, sample=str(k),
                                     roles=f"{r1}-{r2}", dockq=v["dockq"],
                                     pair_iptm=v.get("pair_iptm"), chain_map=" ".join(cm)))
                    canon.append(canon_row(ifc, seed, k, v["dockq"], v.get("pair_iptm")))
                    os.remove(msub)
            print(f"  {pdb} seed {seed}: {len(rows)} row(s) so far", flush=True)

    print(f"\nrecovered {len(rows)} row(s); {skipped} already present; "
          f"{len(refused)} refused map(s); {len(failed)} failure(s)")
    for lbl, items in (("REFUSED", refused), ("FAILED", failed)):
        for pdb, seed, k, why in items[:40]:
            print(f"  {lbl} {pdb} seed {seed} sample {k}: {why}")
        if len(items) > 40:
            print(f"  ... and {len(items) - 40} more {lbl.lower()}")

    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {a.out} ({len(rows)} recovered pose-interface(s))")
    if a.rows_out and canon:
        with open(a.rows_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CANON)
            w.writeheader()
            for r in canon:
                w.writerow(r)
        print(f"wrote {a.rows_out} ({len(canon)} row(s), canonical column order)")
    return 1 if (failed or refused) else 0


if __name__ == "__main__":
    sys.exit(main())
