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
    cdr_identity():  two CDR-H3 loops -> (identity over the shorter, coverage)
    cdr_hits():      CDR-H3 queries vs targets, every pair -> {cluster: (query, identity)}
    search():        query/target fastas and a threshold -> {cluster: (query, identity)}
    main():          build, search, report
"""
import argparse
import collections
import csv
import glob
import json
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
def _anarci_allhmm():
    """out: the path to ANARCI's ALL.hmm, asked of the installed package

    NOT A HARDCODED VENV PATH. The previous literal pointed into
    /home/ebrahim/Protenix/.venv/lib/python3.12/site-packages/, which breaks on a rebuilt venv and --
    worse -- silently changes which HMM set types our chains if that venv is ever repointed. Asking
    the imported package where it lives keeps the answer tied to the ANARCI actually in use. The
    literal stays only as a last resort, and either way the caller checks the file exists.
    """
    try:
        import anarci
        p = os.path.join(os.path.dirname(anarci.__file__), "dat", "HMMs", "ALL.hmm")
        if os.path.exists(p):
            return p
    except Exception:                               # noqa: BLE001
        pass
    return "/home/ebrahim/Protenix/.venv/lib/python3.12/site-packages/anarci/dat/HMMs/ALL.hmm"


ALLHMM = _anarci_allhmm()
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


def _ckey(v):
    """in: a cluster id from either the splits csv or our own csv.  out: a canonical string key

    WHY THIS EXISTS. ab_ag_cluster is int64 in today's splits file, so str() gives "1496" and matches
    the "1496" our exclusion csv stores. A single NaN anywhere in that column makes pandas read it as
    float64, str() then gives "1496.0", every homology exclusion stops matching, and the pool grows
    back from 557 to 561 with no error raised -- we would train on benchmark-homologous clusters and
    never know. Collapsing both sides through int() removes the dependence on the column's dtype.
    """
    try:
        return str(int(float(v)))
    except (TypeError, ValueError):
        return str(v)


def job_sequences():
    """out: [(name, sequence)] for EVERY protein chain the benchmark job jsons fold, any length

    proteinChain.sequence is what Protenix is given, which is the right thing to compare against for
    leakage: a chain the benchmark folds can be identical to a training antigen even when SAbDab2
    never recorded it as an antigen at all.

    NO LENGTH FILTER HERE. This used to drop chains under 20 residues, on the reasoning that a short
    query searches badly -- but dropping them means a short folded antigen is never compared to
    anything, and the exclusion is only as complete as the set it covers. Length is handled where it
    belongs instead: the caller sends long chains to the protein targets at --prot-id and short ones
    to the peptide targets at --pep-id, which is the same split the SAbDab2-side searches use.

    COVERAGE IS RETURNED, NOT ASSUMED. An unreadable or absent job json used to be a printed note, so
    an audit over 200 of 227 complexes read exactly like an audit over all of them. The caller
    compares the pdb ids actually read against each benchmark's own pool and stops on a shortfall.
    """
    out, seen, covered = [], set(), {}
    for ds in BENCHMARKS:
        jdir = str(paths.DATASETS[ds].jobs)
        got, unreadable = set(), []
        for f in sorted(glob.glob(os.path.join(jdir, "*.json"))):
            pdb = os.path.basename(f).split("-")[0].lower()
            try:
                with open(f) as fh:
                    tasks = json.load(fh)
            except (OSError, ValueError) as e:
                unreadable.append(f"{os.path.basename(f)}: {e}")
                continue
            n_chain = 0
            for t in (tasks if isinstance(tasks, list) else [tasks]):
                for k, sq in enumerate(t.get("sequences", [])):
                    c = sq.get("proteinChain")
                    if not c:
                        continue
                    q = str(c.get("sequence", "")).upper()
                    if not q:
                        continue
                    n_chain += 1
                    if q in seen:
                        continue
                    seen.add(q)
                    out.append((f"{pdb}_j{k}", q))
            if n_chain:
                got.add(pdb)
        covered[ds] = (got, unreadable)
    return out, covered


def benchmark_pdbs(ds):
    """out: the pdb ids this benchmark evaluates, from its own pool csv

    The denominator for the coverage check comes from the dataset definition rather than from a
    literal 227, so adding an interface to a pool cannot leave the audit quietly partial.
    """
    pc = getattr(paths.DATASETS[ds], "pool_csv", None)
    if not pc or not os.path.exists(str(pc)):
        return set()
    with open(str(pc), newline="") as fh:
        return {str(r.get("pdb", "")).strip().lower()
                for r in csv.DictReader(fh) if r.get("pdb")}


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
    # A MISSING CSV IS A MISSING ARTIFACT, NOT AN EMPTY RESULT. Continuing silently meant a mistyped
    # --excluded, or running any downstream phase before phase A's main() had produced the file, let
    # benchmark-homologous clusters straight into the training pool with nothing raised. A header-only
    # file is the legitimate way to say "no homologous clusters"; absence means the filter never ran.
    if not os.path.exists(excluded):
        raise SystemExit(
            f"STOP: {excluded} does not exist.\n"
            "It is written by this phase's main(). Run phase A before phases B, C and G:\n"
            f"  python {os.path.basename(__file__)} --out {excluded}\n"
            "A file with only a header is how 'no homologous clusters' is recorded; an absent file\n"
            "means the homology filter has not been applied at all.")
    with open(excluded) as fh:
        hom = {r["cluster"] for r in csv.DictReader(fh)}
    s = pd.read_csv(splits, low_memory=False,
                    usecols=["INSTANCE", "PDB_ID", "agtypes", "ab_ag_cluster", "Hseq", "Lseq",
                             "agexpectedseqs"])
    s = s[s.agtypes.apply(lambda v: any(t in POLY for t in str(v).split("/")))]
    keys = s.ab_ag_cluster.map(_ckey)
    hom_keys = {_ckey(h) for h in hom}
    # A cluster in the exclusion csv that matches nothing in the splits file means the two files have
    # drifted apart -- a regenerated splits release, or the dtype problem _ckey() documents. Either
    # way the exclusion is not being applied, which must not pass quietly.
    orphans = sorted(hom_keys - set(keys))
    if orphans:
        raise SystemExit(
            f"STOP: {len(orphans)} cluster(s) in {excluded} are absent from {splits}: {orphans}\n"
            "The homology exclusion is not being applied. Re-run phase A to regenerate the csv.")
    s = s[~s.ab_ag_cluster.isin(by_id) & ~keys.isin(hom_keys)]
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
            # DOWNLOAD TO .part AND RENAME ON SUCCESS. urlretrieve writes straight to dst, so an
            # interrupted transfer left a truncated cif on disk, and the "> 1000 bytes" test above
            # then treats it as complete forever. gemmi parses a truncated mmCIF without complaining
            # and returns a short chain set -- fewer antigen queries, and exclusions silently missed.
            # This is the filter that decides what we may train on, so the cache must only ever hold
            # whole files.
            tmp = dst + ".part"
            try:
                urllib.request.urlretrieve(
                    f"https://files.rcsb.org/download/{p.upper()}.cif", tmp)
                os.replace(tmp, dst)
            except Exception as e:                      # noqa: BLE001
                if os.path.exists(tmp):
                    os.remove(tmp)
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
    out: ({"<name>#<i>": (chain_type, domain seq)} antibody domains, [(name, seq)] everything else)

    hmmscan --domtblout against ANARCI's HMM set. Columns used: 4 query, 1 target hmm (e.g.
    alpaca_H), 13 the PER-DOMAIN i-Evalue, and 20..21 the envelope in QUERY coordinates.

    ONE ENTRY PER DOMAIN, NOT PER CHAIN. Single-chain constructs carry two variable domains -- 8r4u_B
    is 236 residues with envelopes 1-127 and 126-236, an scFv -- and keeping one best hit per chain
    threw the other domain away. Four of the 39 complexes (8r4u, 9gry, 9gs3, 9i20) consist only of
    such chains, so they contributed no heavy domain and Filter 3 never queried their CDR-H3 at all.
    It reported 0 hits for them whether or not a leak existed.

    RANKED BY THE PER-DOMAIN E-VALUE (column 13), NOT THE FULL-SEQUENCE ONE (column 7). Column 7 is a
    property of the (query, target) pair, so every domain of one chain carries the identical value --
    8r4u_B's two halves both read 5.0e-64 against human_L. Ranking on it cannot tell a VH from a VL
    even in principle, which is why these chains came back typed light while human_H hit them at
    1.9e-57. Envelopes that overlap by more than half are treated as the same domain, keeping the
    better-scoring assignment.
    """
    fa = f"{workdir}/chains.fa"
    with open(fa, "w") as fh:
        for n, s in records:
            fh.write(f">{n}\n{s}\n")
    dom = f"{workdir}/chains.domtbl"
    subprocess.run(["hmmscan", "--domtblout", dom, "--cpu", str(threads), hmm, fa],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Collect every domain hit, then keep the best assignment per distinct envelope.
    per_chain = collections.defaultdict(list)
    with open(dom) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 23:
                continue
            tgt, qry = f[0], f[3]
            ev = float(f[12])                      # i-Evalue: this domain, not the whole sequence
            if ev > evalue:
                continue
            per_chain[qry].append((ev, tgt, int(f[19]), int(f[20])))

    def _same_domain(x, y):
        """two envelopes describing the same variable domain: overlap over half the shorter one"""
        lo, hi = max(x[0], y[0]), min(x[1], y[1])
        ov = max(0, hi - lo + 1)
        return ov > 0.5 * min(x[1] - x[0] + 1, y[1] - y[0] + 1)

    ab, rest = {}, []
    for n, seq in records:
        kept = []                                  # [(ev, type, a, b)], one per distinct domain
        for ev, tgt, a, b in sorted(per_chain.get(n, [])):
            if any(_same_domain((a, b), (ka, kb)) for _e, _t, ka, kb in kept):
                continue                           # a worse hit on a domain already assigned
            kept.append((ev, tgt.split("_")[-1], a, b))
        if not kept:
            rest.append((n, seq))
            continue
        for i, (_ev, typ, a, b) in enumerate(sorted(kept, key=lambda k: k[2])):
            ab[f"{n}#{i}"] = (typ, seq[max(0, a - 1):b])
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


def cdr_identity(q, t):
    """in: two CDR-H3 loops.  out: (identity over the shorter loop, coverage)

    GAPPED, via Biopython's PairwiseAligner. The previous version slid the shorter loop along the
    longer one ungapped, which cannot see an internal insertion: a 14-residue loop against the same
    loop with one residue inserted scored identity 0.50, because after the insertion every position
    is offset by one. A gapped alignment returns 1.0, which is what "identity over the shorter
    sequence" means. The earlier claim that gap handling could not matter at a 95% threshold was
    wrong, and it was wrong in the unsafe direction: it MISSES a leak.

    Local alignment with a simple match/mismatch scheme rather than a substitution matrix: the
    question is near-identity, so a biochemical similarity score would only blur it. Identity is
    matches over the shorter loop's length; coverage is the length ratio, reported separately because
    conflating the two was the original defect here.
    """
    if not q or not t:
        return 0.0, 0.0
    from Bio import Align
    al = Align.PairwiseAligner(mode="local", match_score=1, mismatch_score=-1,
                               open_gap_score=-1, extend_gap_score=-1)
    try:
        aln = al.align(q.upper(), t.upper())[0]
    except Exception:                                       # noqa: BLE001
        return 0.0, 0.0
    a, b = str(aln[0]), str(aln[1])
    matches = sum(1 for x, y in zip(a, b) if x == y and x != "-")
    short = min(len(q), len(t))
    return matches / short, short / max(len(q), len(t))


def cdr_hits(queries, targets, ident, cov):
    """in: {name: cdrh3}, [(cluster|i, cdrh3)], an identity threshold, a coverage threshold.
    out: {cluster: (query, identity)} keeping each cluster's best hit

    EVERY PAIR, NO PREFILTER. This used to be two functions -- an mmseqs search and a direct
    comparison -- cross-checked against each other, because mmseqs indexes short shared chunks and
    can discard a near-exact match on a 14-letter string before aligning it. Running both left two
    sets of blind spots to reason about and a bug in one of them. At this scale the direct comparison
    IS the simple option: 60 query loops against a few thousand targets is a few hundred thousand
    pairs, instant in plain Python, with nothing skipped and nothing to cross-check.

    Ties go to the lowest query name, because a complex often deposits identical copies of a chain
    and the artifact has to reproduce.
    """
    best = {}
    for qn, q in queries.items():
        for tn, t in targets:
            cl = tn.split("|")[0]
            fid, c = cdr_identity(q, t)
            if fid < ident or c < cov:
                continue
            if cl not in best or (fid, best[cl][0]) > (best[cl][1], qn):
                best[cl] = (qn, fid)
    return best


def train_tables(splits, drop_clusters):
    """in: the splits csv, the clusters already excluded by id lookup.
    out: (protein-antigen records, peptide-antigen records, antibody records, the cluster set)

    THE HEADER'S CLUSTER ID GOES THROUGH _ckey(). search() and cdr_hits() parse it straight back out
    of the header, and main() intersects what they return against {_ckey(c) for c in train_cl}. With a
    raw f-string the two sides disagree the moment ab_ag_cluster is float64: the hit keys read
    "1496.0", the normalised side reads "1496", the intersection is empty, the run prints "union
    dropped: 0 clusters", and the csv is written with only a header. training_clusters() then reads an
    empty exclusion set, the orphan guard finds no orphans to complain about, and the pool comes back
    561 instead of 557 -- benchmark-homologous clusters admitted, nothing raised. Canonicalising here
    is what makes the two sides the same key space by construction.

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
                    (f"{_ckey(r.ab_ag_cluster)}|{i}", p))
        for v in (r.VH_numerable_seq, r.VL_numerable_seq, r.Hseq, r.Lseq):
            if isinstance(v, str) and len(v) >= 20:
                ab.append((f"{_ckey(r.ab_ag_cluster)}|{i}", v.strip().upper()))
        if isinstance(r.CDRH3, str) and len(r.CDRH3.strip()) >= 3:
            cdr.append((f"{_ckey(r.ab_ag_cluster)}|{i}", r.CDRH3.strip().upper()))
    return prot, pep, ab, cdr, set(s.ab_ag_cluster.dropna())


def _fasta(path, records):
    with open(path, "w") as fh:
        for n, s in records:
            fh.write(f">{n}\n{s}\n")
    return path


def search(q, t, out, ident, cov, threads, label, seq_id_mode=1, cov_mode=0):
    """in: query fasta, target fasta, result path, identity, coverage, threads, a label for logging.
    out: {cluster: (query, identity)} keeping each cluster's best hit

    -s 7.5 because a 40% antigen match is remote enough that default sensitivity misses it.

    --seq-id-mode: 1 IS "shorter", 2 IS "longer". This had it backwards. mmseqs' own help reads
    "0: alignment length  1: shorter, 2: longer sequence", and this code passed 2 while the comment
    claimed identity was taken over the shorter sequence. The consequence is the opposite of the
    stated method: a 15-residue peptide matching perfectly inside a 500-residue protein scored
    15/500 = 3% and could never reach the 40% threshold, so fragment/full-length homologs the filter
    was meant to catch were missed. Now a flag, defaulting to 1, which is what the method says.

    COVERAGE IS A SEPARATE DECISION and --cov-mode 0 defeats containment on its own: it requires the
    fraction to hold for query AND target, so the same 15-in-500 case fails at 15/500 coverage of the
    target however identity is computed. Exposed as --cov-mode so the choice is explicit rather than
    implied; see the note in the module docstring for what each mode means here.
    """
    # AN EMPTY INPUT IS A FAULT. Returning {} here would read as "this filter found nothing to
    # exclude", which is exactly what a working filter reports -- so a broken classifier or a broken
    # CDR-H3 extractor would look like a clean result. Every search in this phase has thousands of
    # targets and tens of queries on real data.
    for role, path in (("query", q), ("target", t)):
        if os.path.getsize(path) == 0:
            raise SystemExit(f"STOP: {label} has an empty {role} fasta ({path}). "
                             "An exclusion filter with no input cannot exclude anything.")
    subprocess.run([MMSEQS, "easy-search", q, t, out, f"{out}.tmp",
                    "--min-seq-id", str(ident), "-c", str(cov),
                    "--seq-id-mode", str(seq_id_mode), "--cov-mode", str(cov_mode),
                    "-s", "7.5", "--threads", str(threads),
                    "--format-output", "query,target,fident,alnlen,evalue", "-v", "1"],
                   check=True, capture_output=True, text=True)
    best = {}
    with open(out) as fh:
        for line in fh:
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            cl, fid = f[1].split("|")[0], float(f[2])
            # DETERMINISTIC TIE-BREAK. A complex often deposits several identical copies of its
            # antigen -- 8xnh chains A, B and C are the same 438-residue sequence -- so they hit the
            # same cluster at exactly the same identity. With a strict `>` the winner was whichever
            # row mmseqs emitted first, which is not stable across runs: the same input produced
            # 8xnh_B one run and 8xnh_C the next. The cluster and the identity never moved, only the
            # label, but an artifact that does not reproduce is not one to put in a paper. Lowest
            # query name wins a tie.
            # best[cl] is (query_name, identity). Reads oddly on purpose: compare
            # (this identity, STORED name) against (stored identity, THIS name), so higher identity
            # wins and an exact tie goes to the lexicographically smaller query name. Writing it the
            # obvious way round would make ties prefer the LARGER name. Do not "simplify" this.
            if cl not in best or (fid, best[cl][0]) > (best[cl][1], f[0]):
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
    ap.add_argument("--seq-id-mode", type=int, default=1, choices=[0, 1, 2],
                    help="mmseqs identity denominator: 0 alignment length, 1 SHORTER sequence, "
                         "2 longer. Default 1, which is what this filter's method states. It passed "
                         "2 (longer) while claiming shorter, which silently weakened it.")
    ap.add_argument("--cov-mode", type=int, default=1, choices=[0, 1, 2, 3, 4, 5],
                    help="mmseqs coverage mode: 0 query AND target, 1 target, 2 query, 5 shorter "
                         "against longer length. Default 1, coverage of the TARGET, so a training "
                         "antigen contained in one of our benchmark chains is caught. Mode 0 "
                         "requires the fraction of query AND target and so misses containment "
                         "whatever the identity mode: it left clusters 1035 and 1147 in the pool, "
                         "both matching benchmark complex 9ms8 at identity 0.996 and 1.000.")
    ap.add_argument("--evalue", type=float, default=1e-10)
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--min-cdr-frac", type=float, default=0.95,
                    help="fail unless CDR-H3 is recovered for this fraction of heavy chains; guards "
                         "against a silent ANARCI failure making the CDR-H3 filter vacuous")
    ap.add_argument("--no-job-antigens", action="store_true",
                    help="skip querying the antigen chains the benchmarks actually fold. Off by "
                         "default: the cluster-id exclusion is only as complete as SAbDab2's record "
                         "of what each complex contains, and that record omits folded chains.")
    ap.add_argument("--min-job-len", type=int, default=20,
                    help="length boundary for the folded-chain queries: at or above this they are compared to\n                         the PROTEIN targets at --prot-id, below it to the PEPTIDE targets at\n                         --pep-id. Nothing is dropped for being short.")
    ap.add_argument("--allow-partial-audit", action="store_true",
                    help="proceed when the folded-chain comparison does not cover every benchmark "
                         "complex. Off by default: leakage is excluded only for the complexes "
                         "actually compared, so a partial audit understates it.")
    ap.add_argument("--audit-all", action="store_true",
                    help="also compare the 188 complexes excluded by cluster id against the "
                         "surviving pool, by sequence. Report-only: it tests whether SAbDab2's "
                         "clustering actually removed what the id lookup assumes it did.")
    ap.add_argument("--no-fetch", action="store_true")
    ap.add_argument("--allow-missing-cdr-complexes", action="store_true",
                    help="proceed although some unlisted complex contributed no CDR-H3 query. Filter "
                         "3 cannot cover those complexes; off by default.")
    ap.add_argument("--allow-missing-structures", action="store_true",
                    help="write an exclusion csv even though some benchmark structure could not be "
                         "fetched. Off by default: those complexes get no homology check at all.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    miss, listed, ours_cl = unlisted(a.splits)
    print(f"our complexes: {len(listed)+len(miss)}  listed in SAbDab2 {len(listed)}  unlisted {len(miss)}")
    print(f"clusters excluded by id lookup: {len(ours_cl)}")

    got, bad = fetch(miss, a.cifdir, allow=not a.no_fetch)
    print(f"\nRCSB mmCIFs: {len(got)}/{len(miss)}" + (f"   FAILED: {bad}" if bad else ""))
    # ABORT BEFORE SEARCHING OR WRITING. A structure we could not fetch gets no homology check at
    # all, and the run still printed a cluster count and wrote an exclusion csv -- an artifact that
    # looks complete while covering only the complexes that happened to download. Partial cache
    # coverage under --no-fetch has the same effect. The exclusion must be all-or-nothing.
    if bad and not a.allow_missing_structures:
        raise SystemExit(
            f"STOP: {len(bad)} of {len(miss)} unlisted benchmark structures are unavailable: {bad}\n"
            "They would receive no homology check, and the csv written would silently cover only the\n"
            "rest. Re-run with network access, or pass --allow-missing-structures to accept a\n"
            "knowingly partial exclusion.")

    records = []
    for p, path in sorted(got.items()):
        for c, s in chains_of(path).items():
            records.append((f"{p}_{c}", s))
    print(f"polymer chains across them: {len(records)}")

    wd = tempfile.mkdtemp(prefix="exclhom_")
    abq, agq = classify(records, ALLHMM, a.evalue, a.threads, wd)
    kinds = collections.Counter(t for t, _ in abq.values())
    print(f"  antibody DOMAINS (hmmscan): {len(abq)} over "
          f"{len({k.split('#')[0] for k in abq})} chains  types {dict(kinds)}")
    print(f"  candidate antigen chains : {len(agq)} from "
          f"{len({n.split('_')[0] for n,_ in agq})} complexes")

    heavy = [(n, v) for n, (t, v) in abq.items() if t == "H"]   # n is "<pdb>_<chain>#<domain>"
    qcdr = cdrh3_of(heavy)
    print(f"  CDR-H3 recovered (ANARCI, IMGT {CDR3_IMGT[0]}-{CDR3_IMGT[1]}): "
          f"{len(qcdr)}/{len(heavy)} heavy chains, "
          f"{len({n.split('_')[0] for n in qcdr})} complexes")
    if qcdr:
        ls = sorted(len(v) for v in qcdr.values())
        print(f"    loop length: min {ls[0]} median {ls[len(ls)//2]} max {ls[-1]}")
    # THE GATE THAT MAKES FILTER 3'S RESULT MEAN SOMETHING. cdrh3_of() swallows exceptions per chain,
    # so a wrong HMM path or a missing hmmer-3.1b2 binary yields an empty qcdr -- and then this filter
    # reports "0 clusters hit", indistinguishable from the genuine answer it reports today. Filter 3
    # removing 0 clusters is quoted as evidence that filters 1 and 2 were thorough, so the recovery
    # rate has to be checked rather than printed. Observed: 60/60.
    if heavy and len(qcdr) < a.min_cdr_frac * len(heavy):
        raise SystemExit(
            f"STOP: CDR-H3 recovered for only {len(qcdr)} of {len(heavy)} heavy domains "
            f"(< {a.min_cdr_frac:.0%}). ANARCI is not numbering domains, so the CDR-H3 filter would "
            f"report 0 hits whether or not any exist. Check HMMER312={HMMER312} and "
            f"ANARCI_HMMS={ANARCI_HMMS}, then re-run.")
    # AND THE SAME CHECK ONE LEVEL UP, WHICH IS WHERE IT ACTUALLY BITES. The gate above counts
    # domains already classified heavy; it says nothing about how many of the 39 complexes
    # contributed a heavy domain at all. That count was printed and never checked -- the exact
    # pattern the gate above exists to end. It matters most here: these 39 are the complexes SAbDab2's
    # own numbering failed on, so engineered formats are overrepresented. Four of them (8r4u, 9gry,
    # 9gs3, 9i20) are single chains carrying two variable domains whose best full-sequence hit lands
    # on a light model; they contributed no CDR-H3 query, and Filter 3 reported 0 for them whether or
    # not a leak existed.
    # AN ESCAPE, DELIBERATELY. Strict equality is right today (39/39) and hard-failing is the safe
    # direction for an exclusion filter. But a future benchmark entry may genuinely carry no antibody
    # heavy chain -- a TCR complex, or a construct hmmscan cannot type -- and the answer to that
    # should not be editing this file while a pipeline is blocked.
    cdr_complexes = {n.split("_")[0] for n in qcdr}
    if len(cdr_complexes) < len(got) and not a.allow_missing_cdr_complexes:
        missing = sorted(set(got) - cdr_complexes)
        raise SystemExit(
            f"STOP: only {len(cdr_complexes)} of {len(got)} unlisted complexes contributed a CDR-H3 "
            f"query. No query means Filter 3 cannot find a leak for: {missing}\n"
            "Inspect their hmmscan domain assignment -- a single chain carrying two variable domains "
            "must yield one entry per domain, see classify().\n"
            "If a complex genuinely has no heavy chain, pass --allow-missing-cdr-complexes and say "
            "so in the methods: filter 3 cannot cover it.")

    prot_t, pep_t, ab_t, cdr_t, train_cl = train_tables(a.splits, ours_cl)
    print(f"\ncandidate training pool: {len(train_cl)} clusters")
    print(f"  targets: {len(prot_t)} PROTEIN seqs, {len(pep_t)} PEPTIDE seqs, "
          f"{len(ab_t)} antibody seqs, {len(cdr_t)} CDRH3s")

    qa = _fasta(f"{wd}/q_ag.fa", agq)
    qv = _fasta(f"{wd}/q_vh.fa", [(n, v) for n, (_t, v) in abq.items()])
    print()
    h1 = search(qa, _fasta(f"{wd}/t_prot.fa", prot_t), f"{wd}/r_prot.m8",
                a.prot_id, a.cov, a.threads, f"antigen vs PROTEIN  >={100*a.prot_id:.0f}%",
                a.seq_id_mode, a.cov_mode)
    print(f"  antigen vs PROTEIN  >={100*a.prot_id:.0f}% id: {len(h1)} clusters hit")

    # ---------------------------------------------------------------- all 227, on what they FOLD
    # THE CLUSTER-ID EXCLUSION IS ONLY AS COMPLETE AS SAbDab2's RECORD, and that record omits things
    # the benchmark folds. 8vui's single SAbDab2 row reads agtypes='ION' with agclusters=nan: it is
    # catalogued as an antibody bound to an ion and its protein chain is never recorded as an
    # antigen. Its ab_ag_cluster was therefore formed without that protein, so there is no
    # antigen-cluster link to follow to the training cluster whose antigen is identical to it. Taking
    # the clusters of the 188 listed complexes on trust left 18 such clusters in the pool, ten at
    # identity 1.000.
    #
    # So filter 2 also queries the antigen chains of ALL 227 complexes as they are actually folded --
    # proteinChain.sequence from the job jsons, classified by the same hmmscan pass, keeping the
    # non-antibody chains. This is the same criterion applied to the same question; what changes is
    # that it no longer depends on SAbDab2 having written the antigen down.
    h6, h7 = {}, {}
    if not a.no_job_antigens:
        jobseq, covered = job_sequences()
        # COVERAGE FIRST, because an exclusion is only as complete as the set it was applied to.
        # Unreadable or absent job jsons used to be a printed note, so an audit over 200 of the 227
        # complexes read exactly like an audit over all of them -- and the clusters that the missing
        # 27 would have excluded stay in the training pool.
        short = []
        for ds, (got, unreadable) in covered.items():
            want = benchmark_pdbs(ds)
            print(f"  {ds}: job jsons read for {len(got)} complexes"
                  + (f", pool lists {len(want)}" if want else ", pool csv unavailable"))
            for u in unreadable:
                print(f"     UNREADABLE {u}")
            if unreadable and not a.allow_partial_audit:
                sys.exit(f"STOP: {len(unreadable)} job json(s) in {ds} could not be read, so the "
                         "chains they fold were never compared against the training pool. Fix them "
                         "or pass --allow-partial-audit to proceed with a stated gap.")
            if want:
                miss_ds = sorted(want - got)
                if miss_ds:
                    short.append((ds, miss_ds))
        if short and not a.allow_partial_audit:
            msg = "; ".join(f"{ds}: {len(m)} missing e.g. {m[:5]}" for ds, m in short)
            sys.exit(f"STOP: the folded-chain audit does not cover every benchmark complex ({msg}). "
                     "Leakage is excluded only for the complexes actually compared, so a partial "
                     "audit understates it. Pass --allow-partial-audit to proceed with a stated gap.")
        for ds, m in short:
            print(f"  NOTE audit gap accepted for {ds}: {len(m)} complexes not compared: {m[:5]}")

        print(f"\nfolded benchmark chains from the job jsons: {len(jobseq)}")
        if jobseq:
            wdj = tempfile.mkdtemp(prefix="exclhom_jobs_")
            abj, agj = classify(jobseq, ALLHMM, a.evalue, a.threads, wdj)
            print(f"  antibody domains {len(abj)}; folded ANTIGEN chains {len(agj)}")
            # BY LENGTH, the same split the SAbDab2-side searches use. A long chain is compared to
            # the protein targets at --prot-id; a short one to the peptide targets at --pep-id,
            # because a 15-residue peptide matching perfectly inside a 500-residue protein is not
            # the same event as two proteins being the same protein.
            ag_long = [(n, q) for n, q in agj if len(q) >= a.min_job_len]
            ag_short = [(n, q) for n, q in agj if len(q) < a.min_job_len]
            print(f"  of those, {len(ag_long)} chains of {a.min_job_len}+ residues and "
                  f"{len(ag_short)} shorter")
            if ag_long:
                h6 = search(_fasta(f"{wdj}/q.fa", ag_long), _fasta(f"{wd}/t_prot.fa", prot_t),
                            f"{wdj}/r.m8", a.prot_id, a.cov, a.threads,
                            f"folded antigen vs PROTEIN >={100*a.prot_id:.0f}%",
                            a.seq_id_mode, a.cov_mode)
                print(f"  folded antigen vs PROTEIN >={100*a.prot_id:.0f}% id: {len(h6)} clusters hit")
            # THE PEPTIDE TARGETS TOO. The folded-chain search covered protein-typed training
            # antigens only, so the peptide half of the training pool was still being compared
            # against the 39 unlisted complexes alone -- a benchmark chain identical to a training
            # PEPTIDE antigen, in any of the other 188 complexes, had nothing to hit.
            qpep = ag_short + ag_long
            if qpep:
                h7 = search(_fasta(f"{wdj}/qp.fa", qpep), _fasta(f"{wd}/t_pep.fa", pep_t),
                            f"{wdj}/rp.m8", a.pep_id, a.cov, a.threads,
                            f"folded antigen vs PEPTIDE >={100*a.pep_id:.0f}%",
                            a.seq_id_mode, a.cov_mode)
                print(f"  folded antigen vs PEPTIDE >={100*a.pep_id:.0f}% id: {len(h7)} clusters hit")
    h2 = search(qa, _fasta(f"{wd}/t_pep.fa", pep_t), f"{wd}/r_pep.m8",
                a.pep_id, a.cov, a.threads, f"antigen vs PEPTIDE  >={100*a.pep_id:.0f}%",
                a.seq_id_mode, a.cov_mode)
    print(f"  antigen vs PEPTIDE  >={100*a.pep_id:.0f}% id: {len(h2)} clusters hit")
    h3 = search(qv, _fasta(f"{wd}/t_ab.fa", ab_t), f"{wd}/r_ab.m8",
                a.vh_id, a.cov, a.threads, f"variable domain     >={100*a.vh_id:.0f}%",
                a.seq_id_mode, a.cov_mode)
    print(f"  variable domain     >={100*a.vh_id:.0f}% id: {len(h3)} clusters hit  "
          f"(reported for reference; framework-driven, not used)")

    # CDR-H3 is SAbDab2's actual antibody criterion (their cdrh3_cluster_95 column implies 95%).
    # One exhaustive comparison, no search tool: see cdr_hits() for why that is the simpler choice.
    h4 = cdr_hits(qcdr, cdr_t, a.cdr_id, a.cdr_cov)
    print(f"  CDR-H3              >={100*a.cdr_id:.0f}% id, >={100*a.cdr_cov:.0f}% cov: "
          f"{len(h4)} clusters hit   ({len(qcdr)} queries x {len(cdr_t)} targets, every pair)")

    drop = (set(h1) | set(h2) | set(h4) | set(h6) | set(h7)) & {_ckey(c) for c in train_cl}
    keep = {c for c in train_cl if _ckey(c) not in drop}
    print(f"\nunion dropped: {len(drop)} clusters")
    print(f"TRAINING CLUSTERS: {len(train_cl)} -> {len(keep)}")

    # ---------------------------------------------------------------- the full leakage audit
    # THE OTHER 188 ARE EXCLUDED BY CLUSTER ID, which is SAbDab2's clustering taken on trust. This
    # checks it: every one of the 227 benchmark complexes is compared by SEQUENCE against the pool
    # that survives. A hit here means a surviving cluster shares an antigen with a complex we
    # evaluate on, i.e. the id-based exclusion missed it. REPORT-ONLY -- it changes no artifact,
    # because if it ever fires the right response is to look, not to silently drop more.
    if a.audit_all:
        say_pdbs = sorted(listed)
        got2, bad2 = fetch(say_pdbs, a.cifdir, allow=not a.no_fetch)
        print(f"\nAUDIT: all {len(listed)+len(miss)} benchmark complexes against the surviving pool")
        print(f"  listed complexes fetched: {len(got2)}/{len(say_pdbs)}"
              + (f"   unavailable: {len(bad2)}" if bad2 else ""))
        recs2 = []
        for pdb_, path in sorted(got2.items()):
            for c, q in chains_of(path).items():
                recs2.append((f"{pdb_}_{c}", q))
        wd2 = tempfile.mkdtemp(prefix="audit_")
        ab2, ag2 = classify(recs2, ALLHMM, a.evalue, a.threads, wd2)
        print(f"  chains {len(recs2)}; antibody domains {len(ab2)}; candidate antigens {len(ag2)}")
        prot2, pep2, _ab, _cdr, surv = train_tables(a.splits, ours_cl)
        surv = {c for c in surv if _ckey(c) not in drop}
        keep_rows = [(n, q) for n, q in prot2 if _ckey(n.split("|")[0]) in {_ckey(c) for c in surv}]
        h_a = search(_fasta(f"{wd2}/q.fa", ag2), _fasta(f"{wd2}/t.fa", keep_rows),
                     f"{wd2}/r.m8", a.prot_id, a.cov, a.threads,
                     f"AUDIT antigen vs surviving PROTEIN >={100*a.prot_id:.0f}%",
                     a.seq_id_mode, a.cov_mode)
        hits = {c: v for c, v in h_a.items() if c in {_ckey(x) for x in surv}}
        print(f"  surviving clusters sharing an antigen with ANY benchmark complex: {len(hits)}")
        for c, (qn, fid) in sorted(hits.items(), key=lambda kv: -kv[1][1]):
            print(f"     cluster {c} <- {qn} at {fid:.3f}")
        if not hits:
            print("  nothing: SAbDab2's cluster-id exclusion is complete on this measure")

    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["cluster", "search", "query", "identity"])
            for name, h in (("antigen_protein", h1), ("antigen_peptide", h2),
                            ("cdrh3", h4), ("folded_benchmark_antigen", h6),
                            ("folded_benchmark_peptide", h7)):
                for cl, (q, f) in sorted(h.items()):
                    if cl in drop:
                        w.writerow([cl, name, q, f"{f:.3f}"])
        print(f"wrote {a.out}")
    print(f"workdir {wd}")


if __name__ == "__main__":
    main()
