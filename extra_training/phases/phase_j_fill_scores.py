"""Fill the scoring gaps OST's chain mapper left, by scoring each interface's chain pair alone.

WHY THESE FOUR ARE MISSING. `ost/mol/alg/scoring.py:1121` admits a target interface only if both its
chains are in OST's mapping, and OST's chemical grouping cannot separate the homologous heavy and
light immunoglobulin domains, so one target chain is left unmapped and every interface touching it
disappears. Scoring the pair in isolation puts one chain per chemical group and the mapping is then
unambiguous. See docs/concern1_diagnosis.md.

TWO KINDS OF GAP, handled differently.

  1. UNAMBIGUOUS. 9vb4_AD and 9vaw_AB: each reference chain has exactly one model chain matching it,
     so the pair is determined and scored directly.

  2. AMBIGUOUS BY SYMMETRY. 9q5m_EH and 9q5v_EJ: the reference's antigen is partially resolved -- 81
     residues of a 532-residue chain -- and matches THREE identical model copies at 1.00. Sequence
     cannot say which copy it is, and `chain_roles.csv`'s choice is a positional convention rather
     than evidence. Every candidate copy is therefore scored and the BEST is taken, which is what
     symmetry-aware DockQ does for a homo-oligomer, and the winning copy is recorded.

IDENTITY IS NORMALISED BY THE SHORTER CHAIN, not `difflib.ratio`. The ratio is symmetric, so an
81-residue reference fragment against its own 532-residue model chain scores 0.26 and looks like a
non-match; it rejected both symmetric cases on the first attempt. Matched residues over the shorter
sequence gives 1.00, which is what "this fragment is part of that chain" should score.
"""
import argparse, collections, csv, difflib, glob, json, os, sys
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(ROOT)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "eval"))
import ost_dockq                                                     # noqa: E402

AA = set("ACDEFGHIKLMNPQRSTVWY")


def seqs(path, by_label):
    import gemmi
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    out = {}
    for ch in st[0]:
        for res in ch:
            t = gemmi.find_tabulated_residue(res.name)
            c = t.one_letter_code.upper() if t else "X"
            if c in AA:
                k = res.subchain if by_label else ch.name
                out[k] = out.get(k, "") + c
    return out


def cov_ident(a, b):
    """Matched residues over the SHORTER sequence. See the module docstring."""
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = sum(x.size for x in sm.get_matching_blocks())
    return m / max(1, min(len(a), len(b)))


def write_subset(src, keep, out, by_label):
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


def score_one(job):
    pdb, ifc, c1, c2, m1, m2, ref, mdl, sample, seed, scratch = job
    tag = f"{pdb}_{ifc}_{seed}_{sample}_{m1}{m2}"
    rs, ms = f"{scratch}/{tag}_r.cif", f"{scratch}/{tag}_m.cif"
    out = f"{scratch}/{tag}.json"
    try:
        write_subset(ref, {c1, c2}, rs, True)
        write_subset(mdl, {m1, m2}, ms, False)
        conf = glob.glob(os.path.join(os.path.dirname(mdl),
                                      f"*summary_confidence_sample_{sample}.json"))
        if not ost_dockq.invoke(ms, rs, out):
            return None
        res = ost_dockq.parse(out, conf_json=conf[0] if conf else None)
        v = ost_dockq.interface(res, c1, c2)
        if not v or v.get("dockq") is None:
            return None
        got = tuple(v.get("model_chains") or ())
        if set(got) != {m1, m2}:              # the pair returned must be the pair asked for
            return None
        return dict(iface=f"{pdb}_{ifc}", seed=seed, arm="free", sample=sample,
                    dockq=float(v["dockq"]), pair_iptm=v.get("pair_iptm"),
                    model_chains="+".join(got))
    except Exception:
        return None
    finally:
        for f in (rs, ms):
            try: os.remove(f)
            except OSError: pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True, help="pdb:IFACE,... e.g. 9vb4:AD,9vaw:AB")
    ap.add_argument("--fold", default=f"{ROOT}/training/fold_regen")
    ap.add_argument("--refs", default=f"{ROOT}/training/refs")
    ap.add_argument("--roles", default=f"{ROOT}/training/fold/chain_roles.csv")
    ap.add_argument("--seeds", default="500000-500009")
    ap.add_argument("--nsample", type=int, default=5)
    ap.add_argument("--min-ident", type=float, default=0.90)
    ap.add_argument("--scratch", default="/tmp/fill_scores")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    s0, s1 = (int(x) for x in a.seeds.split("-"))
    roles = collections.defaultdict(dict)
    for r in csv.DictReader(open(a.roles)):
        roles[r["pdb"]][r["ref_chain"]] = r["role"]
    os.makedirs(a.scratch, exist_ok=True)

    jobs, plan = [], []
    for tok in a.targets.split(","):
        pdb, ifc = tok.split(":")
        c1, c2 = ifc[0], ifc[1]
        ref = os.path.join(a.refs, f"{pdb}.cif")
        rs = seqs(ref, True)
        for seed in range(s0, s1 + 1):
            pds = [d for d in glob.glob(f"{a.fold}/poses/{pdb}/{seed}-*")
                   if os.path.exists(os.path.join(d, "run.json"))]
            if not pds:
                continue
            for sample in range(a.nsample):
                mdls = glob.glob(f"{pds[0]}/predictions/*_sample_{sample}.cif")
                if not mdls:
                    continue
                ms = seqs(mdls[0], False)
                cand = {}
                for rc in (c1, c2):
                    hits = [mc for mc, m in ms.items() if cov_ident(rs[rc], m) >= a.min_ident]
                    cand[rc] = hits
                if not cand[c1] or not cand[c2]:
                    continue
                for m1 in cand[c1]:
                    for m2 in cand[c2]:
                        if m1 == m2:
                            continue
                        jobs.append((pdb, ifc, c1, c2, m1, m2, ref, mdls[0], sample, seed, a.scratch))
                if seed == s0 and sample == 0:
                    plan.append((pdb, ifc, roles[pdb].get(c1, "?"), roles[pdb].get(c2, "?"),
                                 len(cand[c1]), len(cand[c2])))
    for pdb, ifc, r1, r2, n1, n2 in plan:
        amb = " AMBIGUOUS, best-over-symmetry" if n1 * n2 > 1 else ""
        print(f"  {pdb}_{ifc}  roles {r1}-{r2}  candidate model pairs: {n1}x{n2}{amb}")
    print(f"  OST calls queued: {len(jobs)}")

    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(score_one, jobs, chunksize=1):
            if r:
                rows.append(r)
    # best per (iface, seed, sample): symmetry-aware DockQ takes the best assignment
    best = {}
    for r in rows:
        k = (r["iface"], r["seed"], r["sample"])
        if k not in best or r["dockq"] > best[k]["dockq"]:
            best[k] = r
    out = sorted(best.values(), key=lambda r: (r["iface"], r["seed"], r["sample"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["iface", "seed", "arm", "sample", "dockq",
                                           "pair_iptm", "model_chains"])
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"\n  scored {len(rows)} assignment(s) -> {len(out)} best rows -> {a.out}")
    for i in sorted({r["iface"] for r in out}):
        v = [r for r in out if r["iface"] == i]
        ch = collections.Counter(r["model_chains"] for r in v)
        print(f"    {i}: {len(v)} rows, dockq {min(x['dockq'] for x in v):.3f}-"
              f"{max(x['dockq'] for x in v):.3f}, winning copies {dict(ch)}")


if __name__ == "__main__":
    main()
