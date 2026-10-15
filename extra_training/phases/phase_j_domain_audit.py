"""Stage-1 label audit: does every FOLDED query correspond to a real stretch of its deposited chain,
and for an antibody role, to the variable domain it is supposed to be?

WHY THIS EXISTS. The refold list was built by patching the stored query sequences -- restoring
native residues a query omitted. Patching is only sound if the query is otherwise a faithful,
contiguous piece of the deposited polymer. Two findings showed it is not always:

  * The role sequences come from SAbDab2 (`Hseq`, `Lseq`, `agexpectedseqs`), NOT from the mmCIF.
    8iys's deposited chain E reads ...WGQGTTLTVS-A-GGGG, while the stored heavy query reads
    ...WGQGTTLTVS-SA. The query carries a serine the deposited entity does not have. Restoring
    "missing" residues around a divergence like that produces a sequence that matches neither.

  * A single-chain construct (scFv) puts VH and VL on ONE deposited chain joined by a GS tether,
    and the splice point was being guessed from composition. That guess failed BOTH ways: 8dpl's
    excluded run 115-136 opens with TVSS, the real end of VH, while 8cz8/9ejz/8vy9/8pww/9tuy had
    pure tether (KGGG, GGG, GGGG, SGGG, SG) queued for restoration AS IF it were domain sequence.
    Refolding those would have written a worse input than the one it replaced.

So the boundary is not guessed here. It is read from HMMER against ANARCI's own germline profile
library -- the same engine and the same HMMs that ANARCI itself uses.

WHY NOT THE ANARCI PYTHON API. `anarci.run_anarci` returns None for every one of these chains on
this box: its Biopython 1.81 parse layer cannot read HMMER 3.4 domain coordinates ("'None' exist in
query_start coordinates; ignored") and it discards the hits. `hmmscan --domtblout` on the identical
ALL.hmm reports mouse_H 1-118 at E=1.1e-59. The engine is fine; only the wrapper is broken. This
reads domtblout directly and is version-proof in the way the wrapper is not.

INDEPENDENCE. The queries audited here are read from the job JSONs -- the bytes Protenix actually
folded -- and the roles from the MSA directory names embedded in them, NOT from this repo's chain
mapper. A log written by the mapper under audit would be provenance, not validation.
"""
import argparse, collections, csv, difflib, json, os, re, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from phase_j_job_cifs import native_polymer          # ONE normalisation for every alignment

# ANARCI ships the germline profiles; we drive them with hmmscan directly (see module docstring).
HMM = os.path.join(os.path.dirname(__import__("anarci").__file__), "dat", "HMMs", "ALL.hmm")

ROLE_RE = re.compile(r"_([A-Za-z0-9]+)_[0-9a-f]{6}$")   # ..._<pdb>_<role>_<hash>


def role_of_msa_path(p):
    """The role a job's MSA directory name carries: c2567_10fj_H_6e2cb7 -> H."""
    if not p:
        return None
    m = ROLE_RE.search(os.path.basename(os.path.dirname(p)))
    return m.group(1) if m else None


def job_queries(path):
    """[(role, sequence)] exactly as folded. Role from the MSA path, never from a chain mapper."""
    d = json.load(open(path))
    rec = d[0] if isinstance(d, list) else d
    out = []
    for i, ent in enumerate(rec.get("sequences", [])):
        pc = ent.get("proteinChain")
        if not pc:
            continue
        role = (role_of_msa_path(pc.get("pairedMsaPath"))
                or role_of_msa_path(pc.get("unpairedMsaPath")) or f"idx{i}")
        out.append((role, (pc.get("sequence") or "").strip().upper()))
    return rec.get("name", os.path.basename(path).split("-")[0]), out


def hmm_domains(seqs, keep_frac=0.5, max_evalue=1e-10):
    """{key: [(chain_type, from, to, evalue)]} over 1-based sequence positions.

    One hmmscan for the whole set. Hits are taken best-E-value first and a hit is kept only if it
    does not overlap an already-kept one by more than `keep_frac` of its own length, which is what
    separates the two domains of an scFv.

    THE E-VALUE CUTOFF IS LOAD-BEARING. A genuine variable domain scores around 1e-50 against these
    germline profiles; 8dpl chain C also draws a cow_H hit at 201-230, E=6.7e-05, sitting inside its
    light domain. Without a cutoff those junk partials survive the overlap filter -- they are short
    enough not to overlap anything by half their own length -- and chains came back with four, five
    and six "domains", which then made every ordinary Fab look contaminated. 1e-10 is far below any
    real hit here and far above the partials.
    """
    if not seqs:
        return {}
    with tempfile.TemporaryDirectory() as td:
        fa, tb = os.path.join(td, "q.fa"), os.path.join(td, "q.domtbl")
        with open(fa, "w") as fh:
            for k, s in seqs.items():
                fh.write(">%s\n%s\n" % (k, s))
        subprocess.run(["hmmscan", "--domtblout", tb, "-o", os.devnull, HMM, fa],
                       check=True, capture_output=True)
        hits = collections.defaultdict(list)
        for line in open(tb):
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 23:
                continue
            # domtblout: 1=target 4=query 13=domain E-value 18/19=ali from/to on the QUERY
            ev = float(f[12])
            if ev > max_evalue:
                continue
            hits[f[3]].append((ev, f[0], int(f[17]), int(f[18])))
    out = {}
    for k, hs in hits.items():
        kept = []
        for ev, tgt, lo, hi in sorted(hs):
            ln = hi - lo + 1
            if any(min(hi, b) - max(lo, a) + 1 > keep_frac * ln for _t, a, b in kept):
                continue
            kept.append((tgt.rsplit("_", 1)[-1], lo, hi))
        out[k] = [(t, a, b) for t, a, b in sorted(kept, key=lambda x: x[1])]
    return out


def align(query, native):
    """Map every QUERY residue to the native position it came from, then measure.

    A SPAN IS NOT min/max OVER THE OPCODES. difflib gets `delete` blocks for free, so a 118-residue
    heavy query against a 243-residue scFv chain "spans" 1-243 and scores identity 1.0 while
    actually being VH plus two residues borrowed from the far side of the tether. 8iys is exactly
    that: its stored heavy query ends ...TTLTVS-SA, the deposited chain reads ...TTLTVS-A-GGGG, and
    the extra serine matches a serine INSIDE the linker while the alanine matches the one that opens
    ADIVMTQ -- the light domain. Both residues are real, both are in order, and neither belongs to
    VH. Only a per-residue position map exposes that.

    Positions are 1-based native `label_seq`, the only numbering that survives unresolved density.
    `get_opcodes`, not `get_matching_blocks`: matching blocks skip mismatches, which would report a
    substitution as a deletion plus an insertion and count it as a missing residue.
    """
    sm = difflib.SequenceMatcher(None, native, query, autojunk=False)
    n_of_q = [None] * len(query)      # query index -> native position (1-based), None if unmatched
    eq = subs = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            eq += i2 - i1
            for k in range(j2 - j1):
                n_of_q[j1 + k] = i1 + k + 1
        elif tag == "replace" and (i2 - i1) == (j2 - j1):
            subs += i2 - i1           # same length: residue-for-residue substitution, still placed
            for k in range(j2 - j1):
                n_of_q[j1 + k] = i1 + k + 1
    placed = [x for x in n_of_q if x is not None]
    lo = min(placed) if placed else 0
    hi = max(placed) if placed else 0
    used = set(placed)
    # native residues sitting INSIDE the matched span that no query residue claims: the omissions
    gaps = [n for n in range(lo, hi + 1) if n not in used] if placed else []
    return dict(eq=eq, subs=subs, lo=lo, hi=hi, n_of_q=n_of_q, used=used,
                gaps=gaps, n_unplaced=sum(1 for x in n_of_q if x is None),
                ident=eq / max(len(query), 1),
                contig=(len(placed) == hi - lo + 1) if placed else False)


def to_runs(ns):
    """[3,4,5,9] -> [(3,5),(9,9)]"""
    out = []
    for n in sorted(ns):
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return [tuple(x) for x in out]


def runs(xs):
    return ";".join("%d-%d" % (a, b) for a, b in xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-dir", default=f"{ROOT}/training/jobs")
    ap.add_argument("--src-dir", default=f"{ROOT}/SAbDab2_data/rcsb_training_cifs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="", help="comma-separated pdb ids, for a focused re-check")
    a = ap.parse_args()

    import gemmi
    only = {x.strip().lower() for x in a.only.split(",") if x.strip()}
    jobs = sorted(f for f in os.listdir(a.jobs_dir) if f.endswith(".json"))
    rows, natives = [], {}

    for jf in jobs:
        pdb, qs = job_queries(os.path.join(a.jobs_dir, jf))
        pdb = pdb.lower()
        if only and pdb not in only:
            continue
        cif = os.path.join(a.src_dir, pdb + ".cif")
        if not os.path.exists(cif):
            rows.append(dict(pdb=pdb, role="*", verdict="no_cif")); continue
        st = gemmi.read_structure(cif)
        st.setup_entities()
        nat = {c: s for c, (s, _p) in native_polymer(st).items() if s}
        for role, q in qs:
            if not q:
                continue
            # the chain this query actually came from: best identity, not a name lookup
            best, bk = None, None
            for c, s in nat.items():
                r = align(q, s)
                if best is None or r["ident"] > best["ident"]:
                    best, bk = r, c
            if best is None:
                rows.append(dict(pdb=pdb, role=role, verdict="no_polymer")); continue
            natives[(pdb, bk)] = nat[bk]
            rows.append(dict(pdb=pdb, role=role, chain=bk, qlen=len(q),
                             native_len=len(nat[bk]), ident=round(best["ident"], 4),
                             subs=best["subs"], span_lo=best["lo"], span_hi=best["hi"],
                             contig=int(best["contig"]),
                             dels=runs(to_runs(best["gaps"])),
                             n_del=len(best["gaps"]), n_ins=best["n_unplaced"], q=q,
                             # the CONTIGUOUS native stretch this role covers: the corrected query
                             # for an internal omission is exactly this, holes filled in place.
                             corrected=nat[bk][best["lo"] - 1:best["hi"]] if best["lo"] else "",
                             _best=best))

    # domain spans, one hmmscan for every native chain an antibody role resolved to
    want = {"%s|%s" % (p, c): s for (p, c), s in natives.items()
            if any(r.get("chain") == c and r.get("pdb") == p and r.get("role") in ("H", "L")
                   for r in rows)}
    doms = hmm_domains(want)

    for r in rows:
        if r.get("verdict"):
            continue
        key = "%s|%s" % (r["pdb"], r["chain"])
        d = doms.get(key, [])
        r["n_domains"] = len(d)
        r["domains"] = ";".join("%s:%d-%d" % x for x in d)
        if r["role"] in ("H", "L") and d:
            want_t = "H" if r["role"] == "H" else "KL"
            cand = [x for x in d if x[0] in want_t] or d
            # the domain this role's own span sits in, so an scFv's two roles pick different ones
            # the domain this role's own residues actually sit in: most query residues inside it
            def inside(x):
                return sum(1 for n in r["_best"]["used"] if x[1] <= n <= x[2])
            t, lo, hi = max(cand, key=inside)
            r["dom_type"], r["dom_lo"], r["dom_hi"] = t, lo, hi
            out = sorted(n for n in r["_best"]["used"] if not (lo <= n <= hi))
            r["outside_domain"] = len(out)
            r["outside_runs"] = runs(to_runs(out))
            miss = [n for n in range(lo, hi + 1) if n not in r["_best"]["used"]]
            r["short_of_domain"] = len(miss)
            r["short_runs"] = runs(to_runs(miss))
        v = []
        if r["n_ins"]:
            v.append("CHIMERA")                 # query holds residues the deposited entity lacks
        if not r.get("contig", 1):
            v.append("NONCONTIG")               # matched residues are in order but not adjacent
        if r.get("outside_domain"):
            v.append("CONTAMINATED")            # query reaches past the variable domain
        if r["n_del"]:
            v.append("OMITS")
        if r["subs"]:
            v.append("subs")
        r["verdict"] = "+".join(v) or "clean"

    cols = ["pdb", "role", "chain", "verdict", "qlen", "native_len", "ident", "subs",
            "span_lo", "span_hi", "n_domains", "domains", "dom_type", "dom_lo", "dom_hi",
            "outside_domain", "outside_runs", "short_of_domain", "short_runs",
            "contig", "n_del", "dels", "n_ins", "q", "corrected"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["pdb"], x["role"])):
            w.writerow(r)

    c = collections.Counter(r["verdict"] for r in rows)
    print("audited %d role(s) over %d structure(s) -> %s"
          % (len(rows), len({r["pdb"] for r in rows}), a.out))
    for k, n in c.most_common():
        print("  %-28s %d" % (k, n))
    bad = [r for r in rows if "CHIMERA" in r["verdict"] or "CONTAMINATED" in r["verdict"]]
    print("  structures needing a boundary decision: %d" % len({r["pdb"] for r in bad}))


if __name__ == "__main__":
    main()
