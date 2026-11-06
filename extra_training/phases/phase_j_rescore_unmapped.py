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
import argparse, collections, csv, json, math, os, sys, difflib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import ost_dockq                                                    # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "run_logging"))
import pose_sets                                                    # noqa: E402

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
    # WRITTEN THEN RENAMED, so no reader can ever see a half-written cif. Raised in review: the
    # shared reference subset is rewritten per structure, and two concurrent invocations sharding a
    # seed range over the same --scratch would truncate it while the other's OST process reads it.
    # A failed read is loud, but a partially flushed yet PARSEABLE cif -- a chain short some
    # residues -- yields a valid, silently low DockQ for a real interface. os.replace is atomic
    # within a filesystem, so a reader sees either the whole old file or the whole new one.
    tmp = f"{out}.{os.getpid()}.tmp"
    nm.make_mmcif_document().write_file(tmp)
    os.replace(tmp, out)


CANON = ["iface", "seed", "arm", "donor_pose", "tmpl_chains", "sample", "dockq", "iptm"]
ARM, DONOR, TMPL = "free", "-1", ""      # every queued training unit; checked against queue.tsv


def free_provenance(run_dir):
    """out: "" when run.json describes an untemplated free fold, else why it does not.

    Separate from pose_sets.provenance_ok on purpose. That resolver is shared with the drain gate
    and the rescorer, which judge whether a unit's poses EXIST and are replayable; tightening it
    would change what those two call recoverable. This asks the narrower question only this tool
    needs: may a row be written claiming donor_pose="-1" and tmpl_chains=""?
    """
    rj = os.path.join(run_dir, "run.json")
    try:
        with open(rj) as fh:
            m = json.load(fh)
    except (OSError, ValueError) as e:
        return f"unreadable run.json ({type(e).__name__})"
    if m.get("templated") not in (False, None):
        return f"templated={m.get('templated')!r}"
    if m.get("templates"):
        return f"templates={m.get('templates')!r}"
    if m.get("template_namespace"):
        return f"template_namespace={m.get('template_namespace')!r}"
    if str(m.get("kind", ARM)) != ARM:
        return f"kind={m.get('kind')!r}, not {ARM!r}"
    return ""


def ok_number(x):
    """True only for a value that run_model/folder.py's open_csv would accept as a score.

    It does `float(r["dockq"]), float(r["iptm"])` inside a try and skips the row on ValueError or
    on a non-finite result, so None, "", nan and inf all mean "this sample is still to do". A row
    this tool emits with such a value would look present in the csv and be invisible to the
    pipeline -- the worst of both states.
    """
    if x is None:
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def canon_row(iface, seed, sample, dockq, iptm, arm=ARM, donor=DONOR, tmpl=TMPL):
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

    IT REFUSES A PATH IT CANNOT READ rather than returning an empty set. Raised in review: a typo
    or a relative path run from the wrong cwd would have disabled the whole duplicate guard
    silently, re-emitting every row and doubling them on append -- and the one signal, "0 already
    scored", is also what a genuinely fresh sweep prints, so nothing distinguishes the two. The
    guard that protects the data must not be the one that degrades quietly.
    """
    if not path:
        return set()
    if not os.path.exists(path):
        raise SystemExit(f"--skip-existing {path}: no such file. Refusing to sweep without the "
                         f"duplicate guard; pass an existing results.csv or omit the flag.")
    with open(path) as fh:
        return {(r["iface"], r["seed"], r["sample"]) for r in csv.DictReader(fh)}


def parse_targets(spec):
    """{pdb: [iface]} from 'pdb:iface,...', refusing anything that could mis-attribute a row.

    THE INTERFACE MUST BELONG TO THE pdb. Raised in review, and it is the worst failure this tool
    could have: `--targets 9vb4:9vaw_AB` reads 9vb4's reference, poses, chain map and roles -- every
    guard passes, because they all consult 9vb4 -- and files 9vb4's DockQ under the real interface
    name 9vaw_AB. Nothing downstream could detect it, and the canonical row has no pdb column to
    leave a trace in. --targets is assembled programmatically from a pdb list and an iface list, so
    any zip or sort misalignment produces exactly this token shape.
    """
    want = collections.OrderedDict()
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.count(":") != 1:
            raise SystemExit(f"--targets {tok!r}: expected exactly one ':' as in 9vb4:9vb4_AD")
        pdb, ifc = (s.strip() for s in tok.split(":"))
        if not ifc.startswith(f"{pdb}_"):
            raise SystemExit(f"--targets {tok!r}: interface {ifc} does not belong to {pdb}. "
                             f"Scoring it would file {pdb}'s DockQ under another structure's name.")
        chains = ifc.split("_", 1)[1]
        if len(chains) != 2:
            raise SystemExit(f"--targets {tok!r}: expected <pdb>_<C1><C2> with single-letter chains")
        if chains != "".join(sorted(chains)):
            # pool.csv writes sorted((c1, c2)), so <pdb>_DA names the same interface as <pdb>_AD
            # while matching neither its existing rows nor the skip guard -- it would be scored
            # again and appended under a second name for one physical interface.
            raise SystemExit(f"--targets {tok!r}: chains must be in sorted order "
                             f"({pdb}_{''.join(sorted(chains))}), as pool.csv writes them")
        if ifc not in want.setdefault(pdb, []):
            want[pdb].append(ifc)
    return want


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
    ap.add_argument("--route", choices=("isolated", "full"), default="isolated",
                    help="isolated = pair isolation (OST dropped the interface); "
                         "full = score the whole complex, letting OST map and the repaired "
                         "sequence retry act as its fallback (use when isolation itself fails)")
    ap.add_argument("--min-ident", type=float, default=0.80)
    ap.add_argument("--out", default="")
    ap.add_argument("--scratch", default="/tmp/rescore_unmapped")
    a = ap.parse_args()

    roles = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.roles)):
        roles[r["pdb"]][r["ref_chain"]] = (r["role"], r["model_chain"])

    want = parse_targets(a.targets)

    import glob
    # --seeds WITHOUT --seed0 WAS A SILENT NO-OP. Raised in review: `--seed 500000 --seeds 10` swept
    # exactly one seed and the summary looked like a clean, complete run, with nothing reporting the
    # nine seeds never touched. The two spellings are now mutually exclusive and stated as such.
    if a.seeds < 1:
        raise SystemExit("--seeds must be at least 1")
    if a.seeds > 1 and a.seed0 is None:
        raise SystemExit("--seeds > 1 needs --seed0; --seed sets a single seed only")
    seeds = ([str(a.seed0 + i) for i in range(a.seeds)] if a.seed0 is not None else [str(a.seed)])
    if a.nsample < 1:
        raise SystemExit("--nsample must be at least 1")
    done = existing_keys(a.skip_existing)
    print(f"sweep [{a.route}]: {len(want)} structure(s), seeds {seeds[0]}..{seeds[-1]}, "
          f"samples 0..{a.nsample - 1}; {len(done)} (iface,seed,sample) already scored")

    # failed  = something the caller must act on (OST failed, attribution wrong, reference absent)
    # absent  = a pose that was never folded, which is information, not a fault
    # refused = an ambiguous chain map, which is this tool working correctly
    rows, canon, refused, failed, absent, warned = [], [], [], [], [], []
    emitted, skipped = set(), 0
    for pdb, ifaces in sorted(want.items()):
        ref = os.path.join(a.refs, f"{pdb}.cif")
        if not os.path.exists(ref):
            # RECORDED, NOT JUST PRINTED. Raised in review: this branch used to `continue` with no
            # entry anywhere, so every pose of that structure vanished and the run still reported
            # "0 failure(s)" and exited 0 -- work that should have been scored, gone with a clean
            # bill of health.
            failed.append((pdb, "*", "*", f"no reference at {ref}"))
            continue
        # THE ISOLATED REFERENCE IS THE SAME FOR EVERY POSE, so it is written once per interface
        # rather than once per pose. 850 poses would otherwise re-derive the same 17 files.
        refsub = {}
        for ifc in (ifaces if a.route == "isolated" else []):
            c1, c2 = ifc.split("_", 1)[1][0], ifc.split("_", 1)[1][1]
            # PID IN THE NAME, because this is the one scratch path with no pose in it. The model
            # subset and the OST json both carry seed and sample, so they are unique per pose; the
            # shared reference is not, and concurrent shards of one seed range would collide on it.
            p = f"{a.scratch}/{pdb}_{ifc}_ref.{os.getpid()}.cif"
            write_subset(ref, {c1, c2}, p, by_label=True)
            refsub[ifc] = p

        for seed in seeds:
            # PROVENANCE IS THE SHARED DEFINITION, NOT A GLOB. Raised in review: matching
            # poses/<pdb>/<seed>-* and checking only that run.json EXISTS is a weaker notion of
            # saved-pose identity than the drain gate and the rescorer already use, while this tool
            # hardcodes arm="free" and donor_pose="-1" into every row it writes. A run belonging to
            # a templated arm would be relabelled as an untemplated pose, silently. pose_sets reads
            # run.json and requires it to NAME this pdb, seed and arm; it requires all nsample cifs
            # with a parseable summary json carrying chain_pair_iptm, which is where iptm comes
            # from; and it refuses rather than picks when two complete runs exist.
            status, run_dir, detail = pose_sets.classify(f"{a.fold}/poses", pdb, seed,
                                                         a.nsample, arm=ARM)
            if status == pose_sets.NONE:
                # A SEED THAT WAS NEVER FOLDED IS NOT A FAULT. Raised in review: counting it as a
                # failure made `--seeds 100` against 10 folded seeds report 90 failures and exit
                # non-zero on a sweep that did everything it could.
                absent.append((pdb, seed, "*", f"never folded ({detail})"))
                continue
            if status != pose_sets.COMPLETE:
                failed.append((pdb, seed, "*", f"{status}: {detail}"))
                continue
            # THE ARM LABEL IS NOT THE WHOLE PROVENANCE. Raised in review: classify checks
            # run.json's arm, and every row written here carries donor_pose="-1" and
            # tmpl_chains="" on the strength of it -- so a run labelled free that was actually
            # folded against donor templates would be recorded as an untemplated pose. run.json
            # also states `templated` and `templates`, and for a free fold both must be empty.
            # `templates_found` is NOT checked: a free fold still runs the structural template
            # search, and 4 found templates is its normal output, not a donor.
            why = free_provenance(run_dir)
            if why:
                failed.append((pdb, seed, "*", f"run.json is not an untemplated free fold: {why}"))
                continue
            samples = pose_sets.run_samples(run_dir)
            for k in range(a.nsample):
                todo = [i for i in ifaces if (i, seed, str(k)) not in done
                        and (i, seed, str(k)) not in emitted]
                skipped += len(ifaces) - len(todo)
                if not todo:
                    # CHECKED BEFORE THE POSE. Raised in review: a missing sample cif was recorded
                    # as a failure even when every interface of that pose was already in
                    # results.csv, so re-running with too large an --nsample reported failures for
                    # work that was complete.
                    continue
                # classify() already guaranteed every sample 0..nsample-1 has a cif and a usable
                # summary json, so a miss here is a contradiction worth failing on, not skipping.
                if k not in samples or not samples[k][2]:
                    failed.append((pdb, seed, str(k), "pose passed classify but has no usable "
                                                      "cif/summary -- contradiction"))
                    continue
                mdl = [samples[k][0]]
                conf = [samples[k][1]]

                cm = {}
                if a.route == "isolated":
                    # THE MAP IS REBUILT PER POSE, not once per structure. Model chain naming is
                    # job order and is stable in practice, but a map carried across poses would be
                    # an assumption, and the whole bug this recovers from was an assumption about
                    # chain names. A sequence read costs nothing against a multi-second OST call.
                    rs = chain_seqs(ref, by_label=True)   # label ids: what OST calls the chain
                    ms = chain_seqs(mdl[0], by_label=False)
                    pairs, bad = build_map(rs, ms, a.min_ident)
                    if bad:
                        for b in bad:
                            refused.append((pdb, seed, str(k), b))
                        continue
                    cm = {rc: mc for rc, mc, _ in pairs}
                else:
                    # THE FULL-COMPLEX ROUTE IS THE PRODUCTION SCORER'S OWN CALL. run_model/
                    # scorer.py does exactly `mod.score(cif, ref, out, conf_json=...)` and then
                    # ost_dockq.interface(res, c1, c2); the only difference here is that the json
                    # goes to scratch instead of into the pose directory, which keeps the cached
                    # FAILURE records that document the original bug. OST's own chain mapping
                    # decides first, and the repaired sequence map is reached only as its
                    # fallback. That distinction was stated wrongly once and matters: this route
                    # does not force a map on the first attempt, but the fallback does supply one,
                    # exactly as the production scorer does. What it must never do is override a
                    # mapping OST succeeded with -- forcing one has been measured to turn a DockQ
                    # near 0.9 into one near 0.01 on identical-copy structures.
                    out = f"{a.scratch}/{pdb}_full_{seed}_{k}.json"
                    if os.path.exists(out):
                        os.remove(out)
                    res_full = ost_dockq.score(mdl[0], ref, out, conf_json=conf[0])

                for ifc in todo:
                    c1, c2 = ifc.split("_", 1)[1][0], ifc.split("_", 1)[1][1]
                    if a.route == "isolated":
                        if c1 not in cm or c2 not in cm:
                            failed.append((pdb, seed, str(k),
                                           f"{ifc}: chain outside the validated map"))
                            continue
                        # ISOLATE THE PAIR. One chain per chemical group, so the grouping cannot
                        # collapse H onto L; DockQ is computed over exactly the two chains it is
                        # defined over.
                        msub = f"{a.scratch}/{pdb}_{ifc}_{seed}_{k}_mdl.cif"
                        write_subset(mdl[0], {cm[c1], cm[c2]}, msub, by_label=False)
                        out = f"{a.scratch}/{pdb}_{ifc}_{seed}_{k}.json"
                        # REMOVED FIRST. ost_dockq.invoke has no skip-if-exists and judges success
                        # by `returncode == 0 and usable(out)`, and its own docstring records that
                        # OST can exit 0 having written nothing -- in which case a PREVIOUS run's
                        # json at this path would be accepted as this run's result. The path
                        # carries pdb, iface, seed and sample but not --fold, so a stale result
                        # could even come from a different fold directory.
                        if os.path.exists(out):
                            os.remove(out)
                        if not ost_dockq.invoke(msub, refsub[ifc], out):
                            failed.append((pdb, seed, str(k),
                                           f"{ifc}: OST failed on the isolated pair"))
                            continue
                        res = ost_dockq.parse(out, conf_json=conf[0])
                    else:
                        res = res_full
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
                    # A ROW WITHOUT A FINITE iptm IS NOT A SCORE, SO IT IS A FAILURE. Raised in
                    # review, and confirmed in run_model/folder.py: open_csv does
                    # `float(r["dockq"]), float(r["iptm"])` and skips the row when either does not
                    # parse or is not finite. Emitting a blank iptm would therefore write a row the
                    # rest of the pipeline counts as UNSCORED -- the gap would survive the recovery
                    # while looking filled in the csv. Warning about it was not enough.
                    if not ok_number(v.get("pair_iptm")):
                        failed.append((pdb, seed, str(k),
                                       f"{ifc}: pair_iptm {v.get('pair_iptm')!r} is not finite; "
                                       f"folder.open_csv would treat this row as unscored"))
                        continue
                    if not ok_number(v.get("dockq")):
                        failed.append((pdb, seed, str(k),
                                       f"{ifc}: dockq {v.get('dockq')!r} is not finite"))
                        continue
                    rows.append(dict(pdb=pdb, iface=ifc, seed=seed, sample=str(k),
                                     roles=f"{r1}-{r2}", dockq=v["dockq"],
                                     pair_iptm=v.get("pair_iptm"), chain_map=" ".join(cm)))
                    canon.append(canon_row(ifc, seed, k, v["dockq"], v.get("pair_iptm")))
                    emitted.add((ifc, seed, str(k)))
                    if a.route == "isolated":
                        os.remove(msub)
            print(f"  {pdb} seed {seed}: {len(rows)} row(s) so far", flush=True)

    print(f"\nrecovered {len(rows)} row(s); {skipped} already present; "
          f"{len(refused)} refused map(s); {len(failed)} failure(s); "
          f"{len(absent)} never folded; {len(warned)} warning(s)")
    for lbl, items in (("REFUSED", refused), ("FAILED", failed),
                       ("NEVER FOLDED", absent), ("WARNING", warned)):
        for pdb, seed, k, why in items[:40]:
            print(f"  {lbl} {pdb} seed {seed} sample {k}: {why}")
        if len(items) > 40:
            print(f"  ... and {len(items) - 40} more {lbl.lower()}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["pdb", "iface", "seed", "sample", "roles",
                                               "dockq", "pair_iptm", "chain_map"])
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"wrote {a.out} ({len(rows)} recovered pose-interface(s))")
    if a.rows_out:
        # WRITTEN EVEN WHEN EMPTY, header and all. Raised in review: skipping the write left the
        # PREVIOUS run's file in place, so a caller that sweeps and then appends --rows-out would
        # append the earlier run's rows a second time once the sweep had nothing new to add.
        with open(a.rows_out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CANON)
            w.writeheader()
            for r in canon:
                w.writerow(r)
        print(f"wrote {a.rows_out} ({len(canon)} row(s), canonical column order)")

    # THE EXIT CODE KEYS ON WHAT THE CALLER MUST ACT ON. Raised in review: returning 1 for any
    # refusal meant a wholly successful sweep failed, because refusing an ambiguous map is this
    # tool working as designed (4 of 26 structures tried hold identical copies). A caller chaining
    # `sweep && append` was handed good rows and told not to use them.
    if failed:
        return 1
    if not canon and not skipped:
        return 2          # nothing recovered and nothing already present: the sweep did no work
    return 0


if __name__ == "__main__":
    sys.exit(main())
