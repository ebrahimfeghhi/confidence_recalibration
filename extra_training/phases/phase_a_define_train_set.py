#!/usr/bin/env python
"""Drop SAbDab2 training clusters homologous to the benchmark complexes SAbDab2 does not list.

SUMMARY
    Excluding our benchmark complexes from a SAbDab2-derived training pool is normally a lookup: match
    the complex to its PDB_ID row and read ab_ag_cluster. 39 of our 227 complexes have NO row in
    SAbDab2 (verified: zero substring hits across all four csvs, no cif either, and the file spans
    depositions to 2026-07-22 so it is not recency). They have no cluster to exclude, so a training
    cluster could share an antigen with one of them and we would train on it and then evaluate on it.

    This closes the hole by sequence. For each unlisted complex it takes EVERY polymer chain, classes
    them with hmmscan, and searches the non-antibody chains against the training pool's antigens and
    the antibody chains against its antibody sequences. Any training cluster with a hit is dropped.

    WHY EVERY CHAIN, NOT THE ANNOTATED ANTIGEN. chain_roles.csv and our local reference cifs disagree
    on chain naming -- 9n6e's roles name chains E/F/G while its local cif holds A,B,C,D,H,I,L -- so the
    two cannot be cross-referenced to isolate "the antigen". Taking every non-antibody polymer chain as
    a candidate antigen sidesteps that entirely, and over-including can only drop MORE training
    clusters, which is the safe direction for an exclusion filter. Structures come from RCSB so the
    chain set is the deposited one rather than whatever our local copy happens to carry.

    WHY hmmscan AND NOT ANARCI. ANARCI is installed and its HMMs work -- raw hmmscan scores a known VH
    at 2e-53 against alpaca_H -- but its Python wrapper returns no domains here, because HMMER 3.4
    output parsed through Biopython 1.81 loses the query coordinates. So this calls hmmscan directly on
    ANARCI's own ALL.hmm: the hit name ends _H or _L and gives the chain type, and the envelope gives
    the variable-domain bounds. Same HMMs SAbDab2 used, and no hand-rolled IMGT numbering. The
    consequence is that the antibody query is the whole variable domain at --vh-id, a conservative
    stand-in for a CDR-only comparison.

    AN APPROXIMATION, NOT THEIR CRITERION. SAbDab2 ships no clustering code and documents no
    thresholds; the only hint is a cdrh3_cluster_95 column implying 95% identity. The antigen
    thresholds here are ours and must be reported as ours: --prot-id against PROTEIN-typed antigens,
    --pep-id against PEPTIDE-typed ones, since a short peptide needs much higher identity before a
    match means anything. Identity is --seq-id-mode 2 (over the shorter sequence) for the same reason.

INPUTS
    --splits:    SAbDab2 abag_split.csv. Default under external_data/zenodo_splits.
    --cifdir:    where RCSB mmCIFs are cached. Default external_data/rcsb_unlisted.
    --prot-id:   identity threshold against PROTEIN antigens. Default 0.40.
    --pep-id:    identity threshold against PEPTIDE antigens. Default 0.80.
    --vh-id:     identity threshold for the antibody variable domain. Default 0.95.
    --cov:       minimum alignment coverage. Default 0.50.
    --evalue:    hmmscan e-value below which a chain counts as an antibody domain. Default 1e-10.
    --threads:   mmseqs/hmmscan threads. Default 32, never above the 64-core budget.
    --no-fetch:  use only cifs already cached, do not download.
    --out:       write the dropped-cluster table here as csv. Optional.

OUTPUTS
    stdout: which complexes are unlisted, what chains they contributed, how many training clusters each
            search removes, the union, and the surviving pool
    --out:  one row per dropped cluster: which query hit it, in which search, at what identity

FUNCTIONS
    unlisted():         our pool complexes, split into the ones SAbDab2 lists and the ones it does not
    training_clusters(): THE TRAINING SET -- the surviving clusters and their rows. Imported by B and C
    fetch():         pdb ids -> cached RCSB mmCIF paths
    chains_of():     an mmCIF -> {chain: one-letter polymer sequence}
    classify():      chain sequences -> antibody chains (with variable-domain bounds) and the rest
    train_tables():  the candidate clusters -> their antigen seqs by type, and their antibody seqs
    search():        query/target fastas and a threshold -> {cluster: (query, identity)}
    main():          build, search, report
"""
import argparse
import collections
import csv
import os
import re
import subprocess
import sys
import tempfile
import urllib.request

import pandas as pd

# This file lives at <repo>/extra_training/phases/, so the repo root is three levels up, not two.
# paths.py stays at the repo root: 20+ files across the project import it, and it is shared
# infrastructure rather than anything specific to this pipeline.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import paths  # noqa: E402

SPLITS = f"{REPO}/extra_training/data/zenodo_splits/splits_final/abag_split.csv"
# Written by this phase's main(); read back by training_clusters() and by phases B and C.
EXCLUDED = f"{REPO}/extra_training/results/excluded_homolog_clusters.csv"
CIFDIR = f"{REPO}/extra_training/data/rcsb_unlisted"
MMSEQS = f"{REPO}/tools/mmseqs/bin/mmseqs"
ALLHMM = ("/home/ebrahim/Protenix/.venv/lib/python3.12/site-packages/anarci/dat/HMMs/ALL.hmm")
POLY = ("PROTEIN", "PEPTIDE")

# THE EVALUATION SETS, named explicitly. Never paths.DATASET_NAMES: that tuple also carries
# sabdab_val, which is DERIVED from this pool. Sweeping it in here would exclude the validation
# clusters from the pool they are drawn from, and in phase C it would file 79 validation structures
# as "benchmark" sequences. Both are silent errors -- sabdab_val has a readable pool.csv.
BENCHMARKS = ("pxm2025", "foldbench")

# ANARCI's wrapper parses hmmscan's TEXT output with Biopython's Hmmer3TextParser, which cannot read
# HMMER 3.4's format: the HSP coordinates come back None, every hit is dropped, and no domain is
# numbered. ANARCI was written against HMMER 3.1/3.2, so it gets its own copy of 3.1b2 and its own
# pressed HMMs rather than Biopython being patched or the Protenix venv being modified. Verified: with
# these two set, ANARCI returns CDR-H3 `VRRAFDSDVGFAS` for SAbDab2's first row, matching their CDRH3
# column exactly.
HMMER312 = f"{REPO}/tools/hmmer-3.1b2-linux-intel-x86_64/binaries"
ANARCI_HMMS = f"{REPO}/tools/anarci_hmms"
CDR3_IMGT = (105, 117)      # IMGT CDR3 span; reproduces SAbDab2's CDRH3 column


def unlisted(splits):
    """in: the splits csv.  out: (unlisted pdb ids, listed pdb ids, clusters our listed ones occupy)

    The cluster set is read from EVERY row a complex appears in, whatever its antigen type. Reading it
    from the polypeptide-antigen rows alone misses clusters whose only rows for that complex are apo
    (agtypes blank) -- those rows do carry a cluster id, and skipping them left 61 clusters unexcluded.
    """
    s = pd.read_csv(splits, usecols=["PDB_ID", "ab_ag_cluster"], low_memory=False)
    s["short"] = s.PDB_ID.str[-4:].str.lower()
    ours = set()
    # The exclusion asks "which clusters hold a complex we evaluate on", so it ranges over the
    # evaluation sets only -- see BENCHMARKS above for why that tuple is written out by hand.
    for ds in BENCHMARKS:
        with open(str(paths.DATASETS[ds].pool_csv)) as fh:
            ours |= {r["pdb"].lower() for r in csv.DictReader(fh)}
    hit = s[s.short.isin(ours)]
    return sorted(ours - set(hit.short)), set(hit.short), set(hit.ab_ag_cluster.dropna())


def training_clusters(splits, excluded):
    """in: the splits csv, the homology-exclusion csv.  out: (surviving cluster set, the splits rows)

    THE DEFINITION OF THE TRAINING SET, and the reason this function lives in phase A rather than in
    the phase that first needed it. Filter 1 removes the clusters holding one of our 227 benchmark
    complexes, by id lookup over every row each complex appears in. Filter 2 removes the handful that
    main() found homologous to the 39 complexes SAbDab2 does not list, read back from its csv.

    Phases B, C and G all import this rather than keeping a cluster list of their own: four copies of
    this derivation is how two tables in one paper end up disagreeing about the cluster count.
    """
    _miss, _listed, by_id = unlisted(splits)
    hom = set()
    if os.path.exists(excluded):
        with open(excluded) as fh:
            hom = {r["cluster"] for r in csv.DictReader(fh)}
    s = pd.read_csv(splits, low_memory=False,
                    usecols=["INSTANCE", "PDB_ID", "agtypes", "ab_ag_cluster", "Hseq", "Lseq",
                             "agexpectedseqs"])
    s = s[s.agtypes.apply(lambda v: any(t in POLY for t in str(v).split("/")))]
    s = s[~s.ab_ag_cluster.isin(by_id) & ~s.ab_ag_cluster.astype(str).isin(hom)]
    return set(s.ab_ag_cluster.dropna()), s


def fetch(pdbs, cifdir, allow=True):
    """in: pdb ids, the cache dir, whether downloading is allowed.  out: ({pdb: path}, failures)"""
    os.makedirs(cifdir, exist_ok=True)
    got, bad = {}, []
    for p in pdbs:
        dst = f"{cifdir}/{p}.cif"
        if not os.path.exists(dst) or os.path.getsize(dst) < 1000:
            if not allow:
                bad.append(p)
                continue
            try:
                urllib.request.urlretrieve(
                    f"https://files.rcsb.org/download/{p.upper()}.cif", dst)
            except Exception as e:                      # noqa: BLE001
                bad.append(f"{p} ({type(e).__name__})")
                continue
        got[p] = dst
    return got, bad


def chains_of(cif):
    """in: an mmCIF path.  out: {chain name: one-letter polymer sequence}

    gemmi's own polymer accessor, so the three-to-one mapping and the chain/entity bookkeeping are the
    library's rather than ours. Non-polymer chains (ligands, waters, ions) yield nothing and drop out.
    """
    import gemmi
    st = gemmi.read_structure(cif)
    st.setup_entities()
    out = {}
    for ch in st[0]:
        pol = ch.get_polymer()
        if not len(pol):
            continue
        seq = gemmi.one_letter_code(pol.extract_sequence())
        seq = re.sub(r"[^A-Za-z]", "", seq).upper()
        if len(seq) >= 5:
            out[ch.name] = seq
    return out


def classify(records, hmm, evalue, threads, workdir):
    """in: [(name, seq)], the ALL.hmm path, an e-value, threads, a workdir.
    out: ({name: (chain_type, vh seq)} antibody chains, [(name, seq)] everything else)

    hmmscan --domtblout against ANARCI's HMM set. Columns 4 (query), 7 (full-sequence E-value),
    1 (target hmm, e.g. alpaca_H) and 20..21 (envelope in QUERY coordinates) are the ones used; the
    envelope is what carves the variable domain out of a full-length chain.
    """
    fa = f"{workdir}/chains.fa"
    with open(fa, "w") as fh:
        for n, s in records:
            fh.write(f">{n}\n{s}\n")
    dom = f"{workdir}/chains.domtbl"
    subprocess.run(["hmmscan", "--domtblout", dom, "--cpu", str(threads), hmm, fa],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    best = {}
    with open(dom) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 22:
                continue
            tgt, qry, ev = f[0], f[3], float(f[6])
            if ev > evalue:
                continue
            if qry not in best or ev < best[qry][0]:
                best[qry] = (ev, tgt, int(f[19]), int(f[20]))
    seqs = dict(records)
    ab, rest = {}, []
    for n, s in records:
        if n in best:
            _ev, tgt, a, b = best[n]
            ab[n] = (tgt.split("_")[-1], s[max(0, a - 1):b])
        else:
            rest.append((n, s))
    return ab, rest


def cdrh3_of(records, hmmerpath=HMMER312, hmms=ANARCI_HMMS):
    """in: [(name, seq)] chains, the HMMER 3.1b2 bin dir, the pressed HMM dir.
    out: {name: cdrh3} for every chain ANARCI numbers as a HEAVY domain

    IMGT 105..117, which is SAbDab2's own CDRH3 definition. Light chains are skipped on purpose: the
    comparison that matters is CDR-H3, and a light-chain framework is germline enough to cross 95%
    between unrelated antibodies (measured here: heavy chains hit 0 training clusters at 95% on the
    whole variable domain, light chains hit 13 -- all of them framework artefacts).
    """
    import importlib
    m = importlib.import_module("anarci.anarci")
    m = sys.modules["anarci.anarci"]
    m.HMM_path = hmms
    out = {}
    for name, seq in records:
        try:
            _s, numbered, details, _h = m.run_anarci(
                [(name, seq)], scheme="imgt", hmmerpath=hmmerpath)
        except Exception:                               # noqa: BLE001
            continue
        if not numbered or not numbered[0]:
            continue
        for dom, det in zip(numbered[0], details[0] or []):
            if (det or {}).get("chain_type") != "H":
                continue
            lo, hi = CDR3_IMGT
            cdr = "".join(aa for (pos, _ins), aa in dom[0] if lo <= pos <= hi and aa != "-")
            if len(cdr) >= 3:
                out[name] = cdr
            break
    return out


def brute_cdr_hits(queries, targets, ident):
    """in: {name: cdrh3}, [(cluster|i, cdrh3)], an identity threshold.
    out: {cluster: (query, identity)} by direct comparison

    A 13-residue loop at 95% identity needs 13 of 13 positions, so this threshold is near-exact and a
    k-mer prefilter can drop the very hits it is meant to find. This compares every query against every
    target at equal length (and as a substring either way), which is exact at this scale -- 39 queries
    against a few thousand loops -- and is the check mmseqs is validated against.
    """
    best = {}
    for qn, q in queries.items():
        for tn, t in targets:
            cl = tn.split("|")[0]
            if not q or not t:
                continue
            if len(q) == len(t):
                same = sum(a == b for a, b in zip(q, t)) / len(q)
            elif q in t or t in q:
                same = min(len(q), len(t)) / max(len(q), len(t))
            else:
                continue
            if same >= ident and (cl not in best or same > best[cl][1]):
                best[cl] = (qn, same)
    return best


def train_tables(splits, drop_clusters):
    """in: the splits csv, the clusters already excluded by id lookup.
    out: (protein-antigen records, peptide-antigen records, antibody records, the cluster set)

    Each record is (header, seq) with the cluster id first in the header. agtypes, agresolvedseqs and
    agexpectedseqs are '/'-joined in the same chain order, so they are zipped to put each antigen chain
    under its own type. BOTH resolved and expected sequences go in: resolved has unmodelled gaps a
    homolog could hide in, and searching both is strictly more conservative.
    """
    s = pd.read_csv(splits, low_memory=False,
                    usecols=["PDB_ID", "agtypes", "ab_ag_cluster", "agresolvedseqs",
                             "agexpectedseqs", "VH_numerable_seq", "VL_numerable_seq", "Hseq",
                             "Lseq", "CDRH3"])
    s = s[~s.ab_ag_cluster.isin(drop_clusters)]
    s = s[s.agtypes.apply(lambda v: any(t in POLY for t in str(v).split("/")))]
    prot, pep, ab, cdr = [], [], [], []
    for i, r in enumerate(s.itertuples(index=False)):
        types = str(r.agtypes).split("/")
        for which in (r.agresolvedseqs, r.agexpectedseqs):
            parts = str(which).split("/")
            for t, p in zip(types, parts):
                p = p.strip().upper()
                if len(p) < 5 or p == "NAN":
                    continue
                (prot if t == "PROTEIN" else pep if t == "PEPTIDE" else []).append(
                    (f"{r.ab_ag_cluster}|{i}", p))
        for v in (r.VH_numerable_seq, r.VL_numerable_seq, r.Hseq, r.Lseq):
            if isinstance(v, str) and len(v) >= 20:
                ab.append((f"{r.ab_ag_cluster}|{i}", v.strip().upper()))
        if isinstance(r.CDRH3, str) and len(r.CDRH3.strip()) >= 3:
            cdr.append((f"{r.ab_ag_cluster}|{i}", r.CDRH3.strip().upper()))
    return prot, pep, ab, cdr, set(s.ab_ag_cluster.dropna())


def _fasta(path, records):
    with open(path, "w") as fh:
        for n, s in records:
            fh.write(f">{n}\n{s}\n")
    return path


def search(q, t, out, ident, cov, threads, label, extra=()):
    """in: query fasta, target fasta, result path, identity, coverage, threads, a label for logging.
    out: {cluster: (query, identity)} keeping each cluster's best hit

    -s 7.5 because a 40% antigen match is remote enough that default sensitivity misses it, and
    --seq-id-mode 2 so identity is over the shorter sequence -- a 15-residue peptide against a
    500-residue protein is otherwise scored as trivially dissimilar.
    """
    if os.path.getsize(q) == 0 or os.path.getsize(t) == 0:
        print(f"  {label}: skipped (empty {'query' if os.path.getsize(q)==0 else 'target'})")
        return {}
    subprocess.run([MMSEQS, "easy-search", q, t, out, f"{out}.tmp",
                    "--min-seq-id", str(ident), "-c", str(cov), "--cov-mode", "0",
                    "--seq-id-mode", "2", "-s", "7.5", "--threads", str(threads),
                    "--format-output", "query,target,fident,alnlen,evalue", "-v", "1", *extra],
                   check=True, capture_output=True, text=True)
    best = {}
    with open(out) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            cl, fid = f[1].split("|")[0], float(f[2])
            if cl not in best or fid > best[cl][1]:
                best[cl] = (f[0], fid)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--cifdir", default=CIFDIR)
    ap.add_argument("--prot-id", type=float, default=0.40)
    ap.add_argument("--pep-id", type=float, default=0.80)
    ap.add_argument("--vh-id", type=float, default=0.95)
    ap.add_argument("--cdr-id", type=float, default=0.95,
                    help="CDR-H3 identity; SAbDab2's cdrh3_cluster_95 implies 0.95")
    ap.add_argument("--cdr-cov", type=float, default=0.90,
                    help="CDR-H3 coverage; high, so a short loop must align almost end to end")
    ap.add_argument("--cov", type=float, default=0.50)
    ap.add_argument("--evalue", type=float, default=1e-10)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    miss, listed, ours_cl = unlisted(a.splits)
    print(f"our complexes: {len(listed)+len(miss)}  listed in SAbDab2 {len(listed)}  unlisted {len(miss)}")
    print(f"clusters excluded by id lookup: {len(ours_cl)}")

    got, bad = fetch(miss, a.cifdir, allow=not a.no_fetch)
    print(f"\nRCSB mmCIFs: {len(got)}/{len(miss)}" + (f"   FAILED: {bad}" if bad else ""))

    records = []
    for p, path in sorted(got.items()):
        for c, s in chains_of(path).items():
            records.append((f"{p}_{c}", s))
    print(f"polymer chains across them: {len(records)}")

    wd = tempfile.mkdtemp(prefix="exclhom_")
    abq, agq = classify(records, ALLHMM, a.evalue, a.threads, wd)
    kinds = collections.Counter(t for t, _ in abq.values())
    print(f"  antibody chains (hmmscan): {len(abq)}  types {dict(kinds)}")
    print(f"  candidate antigen chains : {len(agq)} from "
          f"{len({n.split('_')[0] for n,_ in agq})} complexes")

    heavy = [(n, v) for n, (t, v) in abq.items() if t == "H"]
    qcdr = cdrh3_of(heavy)
    print(f"  CDR-H3 recovered (ANARCI, IMGT {CDR3_IMGT[0]}-{CDR3_IMGT[1]}): "
          f"{len(qcdr)}/{len(heavy)} heavy chains, "
          f"{len({n.split('_')[0] for n in qcdr})} complexes")
    if qcdr:
        ls = sorted(len(v) for v in qcdr.values())
        print(f"    loop length: min {ls[0]} median {ls[len(ls)//2]} max {ls[-1]}")

    prot_t, pep_t, ab_t, cdr_t, train_cl = train_tables(a.splits, ours_cl)
    print(f"\ncandidate training pool: {len(train_cl)} clusters")
    print(f"  targets: {len(prot_t)} PROTEIN seqs, {len(pep_t)} PEPTIDE seqs, "
          f"{len(ab_t)} antibody seqs, {len(cdr_t)} CDRH3s")

    qa = _fasta(f"{wd}/q_ag.fa", agq)
    qv = _fasta(f"{wd}/q_vh.fa", [(n, v) for n, (_t, v) in abq.items()])
    print()
    h1 = search(qa, _fasta(f"{wd}/t_prot.fa", prot_t), f"{wd}/r_prot.m8",
                a.prot_id, a.cov, a.threads, f"antigen vs PROTEIN  >={100*a.prot_id:.0f}%")
    print(f"  antigen vs PROTEIN  >={100*a.prot_id:.0f}% id: {len(h1)} clusters hit")
    h2 = search(qa, _fasta(f"{wd}/t_pep.fa", pep_t), f"{wd}/r_pep.m8",
                a.pep_id, a.cov, a.threads, f"antigen vs PEPTIDE  >={100*a.pep_id:.0f}%")
    print(f"  antigen vs PEPTIDE  >={100*a.pep_id:.0f}% id: {len(h2)} clusters hit")
    h3 = search(qv, _fasta(f"{wd}/t_ab.fa", ab_t), f"{wd}/r_ab.m8",
                a.vh_id, a.cov, a.threads, f"variable domain     >={100*a.vh_id:.0f}%")
    print(f"  variable domain     >={100*a.vh_id:.0f}% id: {len(h3)} clusters hit  "
          f"(reported for reference; framework-driven, not used)")

    # CDR-H3 is SAbDab2's actual antibody criterion. Run it twice: mmseqs with a short-sequence kmer,
    # and a direct comparison, because at 95% on a ~15-mer the prefilter can drop the hits it is for.
    h4 = search(_fasta(f"{wd}/q_cdr.fa", sorted(qcdr.items())),
                _fasta(f"{wd}/t_cdr.fa", cdr_t), f"{wd}/r_cdr.m8",
                a.cdr_id, a.cdr_cov, a.threads, f"CDR-H3 (mmseqs)     >={100*a.cdr_id:.0f}%",
                extra=("-k", "5", "--spaced-kmer-mode", "0", "--comp-bias-corr", "0"))
    h5 = brute_cdr_hits(qcdr, cdr_t, a.cdr_id)
    print(f"  CDR-H3 mmseqs       >={100*a.cdr_id:.0f}% id: {len(h4)} clusters hit")
    print(f"  CDR-H3 brute force  >={100*a.cdr_id:.0f}% id: {len(h5)} clusters hit"
          + ("  [AGREE]" if set(h4) == set(h5) else f"  [DISAGREE: {sorted(set(h4) ^ set(h5))}]"))

    drop = (set(h1) | set(h2) | set(h4) | set(h5)) & {str(c) for c in train_cl}
    keep = {c for c in train_cl if str(c) not in drop}
    print(f"\nunion dropped: {len(drop)} clusters")
    print(f"TRAINING CLUSTERS: {len(train_cl)} -> {len(keep)}")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cluster", "search", "query", "identity"])
            for name, h in (("antigen_protein", h1), ("antigen_peptide", h2),
                            ("cdrh3_mmseqs", h4), ("cdrh3_brute", h5)):
                for cl, (q, f) in sorted(h.items()):
                    if cl in drop:
                        w.writerow([cl, name, q, f"{f:.3f}"])
        print(f"wrote {a.out}")
    print(f"workdir {wd}")


if __name__ == "__main__":
    main()
