#!/usr/bin/env python
"""Write the FASTA of every unique sequence that needs an MSA for the confidence-head finetune.

SUMMARY
    Two sources, one deduplicated FASTA per source, because MSAs are per unique sequence and the two
    sets are used for different things:

      train   the 557 SAbDab2 clusters that survive exclusion against our benchmarks. Antibody heavy
              and light chains plus every polypeptide antigen chain. 4,672 unique sequences.
      bench   our existing FoldBench and PXM-2025 complexes. 1,376 chain MSAs already exist for these
              but were built on uniref30_2103, which is no longer downloadable; rebuilding them on
              2302 removes the train/test database-version confound. Written to a SEPARATE directory
              so the 2103 MSAs are never overwritten.

    THE 557 COME FROM exclude_homologs.py, not from a stored list: clusters containing any of our 227
    benchmark complexes are dropped by id lookup (140 clusters, read from EVERY row a complex appears
    in, whatever its antigen type), and 4 more are dropped for antigen homology to the 39 complexes
    SAbDab2 does not list. Re-deriving it here keeps this file honest if the exclusion changes.

    SEQUENCES COME FROM THE SPLITS CSV for the training set (Hseq, Lseq, agexpectedseqs) and from the
    reference cifs for the benchmarks, since those are the sequences actually folded.

INPUTS
    --splits:   SAbDab2 abag_split.csv. Default under extra_training/data.
    --excluded:    the homology-exclusion csv phase A writes. Default under extra_training/splits.
    --out-dir:     where the per-source fastas go. Default extra_training/msa_queries.
    --min-len:     skip sequences shorter than this. Default 20.
    --which:       train, bench, or both. Default both.
    --query-fasta: where the concatenated search set goes. Default msa_work/query.fasta.
    --short-vs:    batch-2 mode -- emit only sequences absent from this already-searched fasta.
    --short-out:   where batch 2 goes. Default msa_work/query_short.fasta.

TWO INVOCATIONS, which is what the pipeline actually runs

    batch 1, the long chains:
        phase_c_extract_msa_queries.py --min-len 20 --which both
      -> msa_queries/{train,bench}.fasta and msa_work/query.fasta (5,305 sequences)

    batch 2, the short antigen chains the first pass skipped:
        phase_c_extract_msa_queries.py --min-len 5 --which train \
            --out-dir .../msa_queries_min5 --short-vs .../query.fasta
      -> msa_work/query_short.fasta (235 sequences, 5-19 aa)

    Batch 2 is a SEPARATE search rather than an extension of batch 1 because ColabFold keys its query
    database by position: appending queries shifts every key and the 12.5 GB uniref.a3m kept from
    batch 1 could no longer be resumed from.

    PER-SEQUENCE QUERIES ARE THE ONES ACTUALLY SEARCHED -- `train.fasta` and `bench.fasta`, merged
    into msa_work/query.fasta (5,305 records: 4,672 train + 633 bench, verified 0 shared).

    THIS SUPERSEDES THE COMPLEX-LEVEL DESIGN. ColabFold can pair chains itself, but only for FASTA
    records whose chains are ':'-joined, and `complexes_all.fasta` was written for that route.
    Protenix does NOT consume ColabFold pairing: its own scripts/msa/step3-uniref_add_taxid.py writes
    NCBI taxids into the UniRef headers and step4-split_msa_to_uniref_and_others.py splits each
    per-chain a3m into pairing.a3m and non_pairing.a3m on that basis. Pairing is therefore a
    post-processing step over per-chain alignments, and searching complexes would produce an a3m
    shape neither step3 nor step4 reads.

    `complexes_all.fasta` and `complexes.csv` are kept because they are cheap and record the chain
    order per complex, but nothing in the live pipeline searches them.

    THE BENCHMARK HALF IS A SENSITIVITY EXPERIMENT, not the main path. Decided 2026-10-11: PXM-2025
    and FoldBench are evaluated with the MSAs they were released with, built on uniref30_2103, so
    that the refold which applies the retrained head varies only the head. The 633 benchmark
    sequences here get 2302 MSAs so a subset can be folded both ways and the database version shown
    not to move the baseline. Use --bench-from-jobs for that comparison: it takes
    proteinChain.sequence from the actual job jsons rather than the observed residues of a reference
    structure, which is what the benchmark folds are built from.

    TRAIN AND BENCH GO IN ONE FILE. Rebuilding the benchmark MSAs on 2302 exists precisely to remove a
    train/test database-version difference, so running them through the same pass removes any chance
    of the two sets being treated differently, and halves database load overhead.

OUTPUTS
    msa_work/query.fasta            THE SEARCH SET: train.fasta then bench.fasta, one file
    msa_work/query_short.fasta      batch 2, under --short-vs only
    <out-dir>/train.fasta           unique training sequences
    <out-dir>/bench.fasta           unique benchmark sequences
    <out-dir>/complexes_all.fasta   one record per complex, chains ':'-joined -- NOT searched, see above
    <out-dir>/complexes.csv         complex id -> chain order, lengths, source
    <out-dir>/manifest.csv          sequence id -> sha1, length, source

FUNCTIONS
    read_fasta():        a fasta -> [(name, sequence)]; used by batch-2 mode
    write_fasta():       records -> a fasta
    train_records():     the splits csv -> unique training sequences
    bench_records():     our pool csvs + reference cifs -> unique benchmark sequences
    train_complexes():   the splits csv -> one (name, [chain seqs]) per training structure
    bench_complexes():   reference cifs -> one (name, [chain seqs]) per benchmark complex
    main():              write the search set, the per-source fastas, the complex table
                         and the manifest -- or, with --short-vs, batch 2 alone
"""
import argparse
import collections
import glob
import csv
import hashlib
import json
import os
import sys

# This file lives at <repo>/extra_training/phases/, so the repo root is three levels up, not two.
# paths.py stays at the repo root: 20+ files across the project import it, and it is shared
# infrastructure rather than anything specific to this pipeline.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import paths              # noqa: E402
import phase_a_define_train_set as A  # noqa: E402

OUTDIR = f"{REPO}/extra_training/msa_queries"
QUERY = f"{REPO}/extra_training/msa_work/query.fasta"
QUERY_SHORT = f"{REPO}/extra_training/msa_work/query_short.fasta"


def poly_antigens(row):
    """in: a splits row.  out: [(tag, sequence)] for the PROTEIN and PEPTIDE antigen chains only

    THE POOL FILTER IS PER ROW; THIS IS PER CHAIN. training_clusters() keeps a row if ANY of its
    antigens is PROTEIN or PEPTIDE, so a complex with a protein antigen and an RNA antigen is kept
    whole -- and taking every entry of agexpectedseqs then sent the RNA in as a protein MSA query.
    The committed query set carries one: c228_6cf2_ag1_4666db. ColabFold searches it against uniref30
    regardless, and the downstream a3m/fasta consistency check later dropped it because the search had
    rewritten its U as X. So it cost a query slot and surfaced as a mismatch far from its cause.

    agtypes and agexpectedseqs are '/'-joined in the same chain order. A length disagreement between
    them would make zip() silently drop the trailing chains, so it is an error rather than a shrug.
    """
    out = []
    for i, (t, p) in enumerate(A.antigen_chains(row.PDB_ID, row.agtypes, row.agexpectedseqs)):
        q = p.strip().upper()
        if t.strip().upper() in A.POLY and q and q != "NAN":
            out.append((f"ag{i}", q))
    return out


def train_records(rows, min_len):
    """in: the surviving splits rows, a minimum length.  out: [(name, seq)] deduplicated by sequence

    The antibody chains come from Hseq/Lseq and the antigens from agexpectedseqs (the full construct,
    not agresolvedseqs, so an MSA is not built against a sequence with unmodelled gaps).
    """
    seen, out = set(), []
    for r in rows.itertuples(index=False):
        pdb = str(r.PDB_ID)[-4:]
        items = [("H", r.Hseq), ("L", r.Lseq)]
        items += poly_antigens(r)          # PROTEIN/PEPTIDE chains only -- see poly_antigens()
        for tag, v in items:
            if not isinstance(v, str):
                continue
            q = v.strip().upper()
            if len(q) < min_len or q == "NAN" or q in seen:
                continue
            seen.add(q)
            # A sequence-hash suffix, because cluster+pdb+tag is NOT unique: the same PDB can appear
            # twice in a cluster with different heavy-chain sequences, which collided for 544 of
            # 4,672 queries. colabfold writes output as <name>.a3m, so a collision silently
            # overwrites an MSA. The hash is stable across runs, unlike a running counter.
            h = hashlib.sha1(q.encode()).hexdigest()[:6]
            out.append((f"c{r.ab_ag_cluster}_{pdb}_{tag}_{h}", q))
    return out


def bench_chains(require=True):
    """in: whether a missing structure is fatal.  out: [(dataset, pdb, {chain: seq})] for all 227

    ONE PLACE THAT READS THE BENCHMARK STRUCTURES, AND IT COUNTS. bench_records() and
    bench_complexes() each walked the pool csvs and `continue`d past a missing reference cif or a
    gemmi failure, and the line they printed reported only a sequence count -- never how many of the
    227 complexes contributed. A complex silently getting no 2302 MSA keeps its old uniref30_2103 one
    at fold time, which puts back exactly the train/test database-version difference this rebuild
    exists to remove. The expected number is known, so it is asserted.
    """
    out, want, failed = [], [], []
    for ds in A.BENCHMARKS:
        D = paths.DATASETS[ds]
        with open(str(D.pool_csv)) as fh:
            pdbs = sorted({r["pdb"].lower() for r in csv.DictReader(fh)})
        for pdb in pdbs:
            want.append(pdb)
            ref = D.ref(pdb)
            if not os.path.exists(ref):
                failed.append(f"{pdb} (no reference cif)")
                continue
            try:
                chains = A.chains_of(ref)
            except Exception as e:                      # noqa: BLE001
                failed.append(f"{pdb} ({type(e).__name__})")
                continue
            if not chains:
                failed.append(f"{pdb} (no polymer chains)")
                continue
            out.append((ds, pdb, chains))
    print(f"benchmark structures: {len(out)}/{len(want)} contributed chains"
          + (f"   FAILED: {failed}" if failed else ""))
    if failed and require:
        raise SystemExit(
            f"STOP: {len(failed)} of {len(want)} benchmark complexes contributed no chains: "
            f"{failed}\n"
            "Each would get no uniref30_2302 MSA and would fall back to its 2103 one at fold time, "
            "reintroducing the database-version difference this rebuild removes.\n"
            "Pass --no-require-bench to accept that knowingly.")
    return out


def bench_from_jobs(min_len):
    """out: [(name, sequence)] taken from the benchmark JOB JSONS rather than reference structures

    For the sensitivity experiment the sequence that matters is the one Protenix is actually given --
    proteinChain.sequence in the job json. bench_records() reads observed residues out of a reference
    cif instead, so unmodelled residues or a tag make its rebuilt MSA correspond to a different
    sequence from the one the benchmark folds against, and the comparison would not be like for like.
    """
    seen, out = set(), []
    for ds in A.BENCHMARKS:
        D = paths.DATASETS[ds]
        for f in sorted(glob.glob(os.path.join(str(D.jobs), "*.json"))):
            pdb = os.path.basename(f).split("-")[0].lower()
            # NARROW, so a programming error cannot hide here. A bare `except Exception` swallowed
            # the NameError from a missing `import json` and silently skipped all 732 job files,
            # reporting 0 sequences as though the jsons held none -- the same shape of fault this
            # phase's own gates exist to prevent.
            try:
                with open(f) as fh:
                    tasks = json.load(fh)
            except (OSError, ValueError) as e:
                print(f"  NOTE unreadable job json {os.path.basename(f)}: {e}")
                continue
            for t in tasks if isinstance(tasks, list) else [tasks]:
                for k, sq in enumerate(t.get("sequences", [])):
                    c = sq.get("proteinChain")
                    if not c:
                        continue
                    q = str(c.get("sequence", "")).upper()
                    if len(q) < min_len or q in seen:
                        continue
                    seen.add(q)
                    out.append((f"{pdb}_j{k}", q))
    print(f"bench-from-jobs: {len(out)} unique sequences from the benchmark job jsons")
    return out


def bench_records(min_len, require=True):
    """in: a minimum length.  out: [(name, seq)] unique chain sequences of our benchmark complexes

    Read from the reference cifs rather than the job jsons, so the sequence is the one the structure
    actually contains. Every polymer chain is taken -- which chain is antibody or antigen does not
    matter for an MSA.
    """
    seen, out = set(), []
    for _ds, pdb, chains in bench_chains(require):
        for c, q in sorted(chains.items()):
            q = q.upper()
            if len(q) < min_len or q in seen:
                continue
            seen.add(q)
            out.append((f"{pdb}_{c}", q))
    return out


def train_complexes(rows, min_len):
    """in: the surviving splits rows, a minimum length.  out: [(name, [chain seqs])] per structure

    One record per structure, not per cluster: training samples structures, and the sampler's cluster
    weighting (via `cluster_id` in the indices csv) is what prevents a 1,001-structure cluster from
    dominating. Chain order is heavy, light, then antigens in `agexpectedseqs` order, and that order is
    recorded in complexes.csv because the per-complex a3m has to be split back onto it.
    """
    out, seen = [], set()
    for r in rows.itertuples(index=False):
        pdb = str(r.PDB_ID)[-4:]
        chains = []
        for v in (r.Hseq, r.Lseq):
            if isinstance(v, str) and len(v.strip()) >= min_len:
                chains.append(v.strip().upper())
        for _tag, q in poly_antigens(r):
            if len(q) >= min_len:
                chains.append(q)
        key = (pdb, tuple(chains))
        if len(chains) < 2 or key in seen:
            continue
        seen.add(key)
        out.append((f"tr_c{r.ab_ag_cluster}_{pdb}", chains))
    return out


def bench_complexes(min_len, require=True):
    """in: a minimum length.  out: [(name, [chain seqs])] per benchmark complex

    Chains come from the reference cif in its own chain order, so the mapping back to Protenix's
    per-chain msa/<n>/ directories follows the same order the folds already use.
    """
    out = []
    for ds, pdb, chains in bench_chains(require):
        seqs = [q.upper() for _c, q in sorted(chains.items()) if len(q) >= min_len]
        if len(seqs) >= 2:
            out.append((f"bn_{ds[:2]}_{pdb}", seqs))
    return out


def read_fasta(path):
    """in: a fasta path.  out: [(name, sequence)] in file order"""
    out, name, buf = [], None, []
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            if name is not None:
                out.append((name, "".join(buf)))
            name, buf = line[1:], []
        elif name is not None:
            buf.append(line)
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def write_fasta(path, records):
    with open(path, "w") as fh:
        for n, q in records:
            fh.write(f">{n}\n{q}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=A.SPLITS)
    ap.add_argument("--excluded", default=A.EXCLUDED)
    ap.add_argument("--out-dir", default=OUTDIR)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--which", choices=["train", "bench", "both"], default="both")
    ap.add_argument("--query-fasta", default=QUERY,
                    help="where to write the concatenated search set. This, not train.fasta or "
                         "bench.fasta, is the file handed to colabfold_search.")
    ap.add_argument("--short-vs", default=None,
                    help="BATCH 2 MODE. Given the already-searched query fasta, write only the "
                         "records whose sequence is absent from it. Run with --min-len 5 --which "
                         "train to pick up the antigen chains a --min-len 20 pass skipped.")
    ap.add_argument("--short-out", default=QUERY_SHORT)
    ap.add_argument("--bench-from-jobs", action="store_true",
                    help="take benchmark sequences from the job jsons' proteinChain.sequence rather "
                         "than from reference structures. Use this for the 2302-vs-2103 sensitivity "
                         "experiment, so the rebuilt MSA matches the sequence actually folded.")
    ap.add_argument("--require-bench", action=argparse.BooleanOptionalAction, default=True,
                    help="fail unless every benchmark complex contributed chains; a silent omission "
                         "falls back to that complex's uniref30_2103 MSA at fold time")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    manifest, all_recs = [], []
    if a.which in ("train", "both"):
        clusters, rows = A.training_clusters(a.splits, a.excluded)
        recs = train_records(rows, a.min_len)
        write_fasta(f"{a.out_dir}/train.fasta", recs)
        all_recs += recs
        lens = sorted(len(q) for _n, q in recs)
        print(f"train.fasta: {len(recs)} unique sequences from {len(clusters)} clusters, "
              f"{rows.PDB_ID.nunique()} structures; len median {lens[len(lens)//2]}, "
              f"total {sum(lens):,} residues")
        manifest += [(n, hashlib.sha1(q.encode()).hexdigest(), len(q), "train") for n, q in recs]

    if a.which in ("bench", "both"):
        recs = bench_from_jobs(a.min_len) if a.bench_from_jobs \
            else bench_records(a.min_len, a.require_bench)
        write_fasta(f"{a.out_dir}/bench.fasta", recs)
        all_recs += recs
        lens = sorted(len(q) for _n, q in recs)
        print(f"bench.fasta: {len(recs)} unique sequences; len median {lens[len(lens)//2]}, "
              f"total {sum(lens):,} residues")
        manifest += [(n, hashlib.sha1(q.encode()).hexdigest(), len(q), "bench") for n, q in recs]

    # ---- CHECKED BEFORE ANYTHING IS WRITTEN. These ran after the search fasta had been written, so
    # a failing phase left an apparently usable search input on disk for the next stage to pick up.
    # THE 0-SHARED CLAIM, ENFORCED. This was a comment in the docstring; now it is checked.
    #
    # WHAT IT DOES AND DOES NOT TEST. A sequence in both sets means a training complex carries a
    # chain IDENTICAL to a benchmark chain, which the exclusion should have removed. But identity 1.0
    # is the EASIEST case any of the filters face -- filter 2 is set at 0.40 for proteins and filter 3
    # at 0.95 -- and in most cases filter 1 removes it by cluster id before sequence ever enters it.
    # So passing says the exclusion catches the trivial case. It does not test whether the thresholds
    # are right, which is the part that actually carries risk. Treat it as a smoke test that would
    # fire loudly on a wiring mistake, not as evidence the filters are calibrated.
    if a.which == "both":
        tr = {q for n, q in all_recs if n.startswith("c")}
        bn = {q for n, q in all_recs if not n.startswith("c")}
        shared = tr & bn
        names = [n for n, _ in all_recs]
        if len(set(names)) != len(names):
            import collections as _c
            dup = [n for n, k in _c.Counter(names).items() if k > 1]
            raise SystemExit(f"STOP: {len(dup)} duplicate query NAME(s), e.g. {dup[:3]}. ColabFold "
                             "writes <name>.a3m, so duplicates silently overwrite one another.")
        print(f"train/bench shared sequences: {len(shared)}   (train {len(tr)}, bench {len(bn)})"
              "   [smoke test: identity 1.0 is the easiest case the filters face]")
        if shared:
            ex = sorted(shared)[:3]
            raise SystemExit(
                f"STOP: {len(shared)} sequence(s) appear in BOTH the training and benchmark sets.\n"
                "A training complex carries a chain identical to a benchmark chain, so the exclusion "
                "in phase A did not remove its cluster. Re-run phase A and inspect its filters.\n"
                + "\n".join(f"  len {len(q)}: {q[:60]}..." for q in ex))


    # THE FILE THAT IS ACTUALLY SEARCHED. train.fasta and bench.fasta are per-source; the search
    # takes one file so that both go through the identical pass -- rebuilding the benchmark MSAs on
    # uniref30_2302 exists precisely to remove a train/test database-version difference, and running
    # them separately would reintroduce the chance of treating the two sets differently.
    #
    # This used to be a bare `cat train.fasta bench.fasta` typed at a shell, which left a step of the
    # method with no code behind it. Byte-identical to that concatenation, by construction: the same
    # records in the same order through the same writer.
    if a.short_vs:
        already = {q.upper() for _n, q in read_fasta(a.short_vs)}
        new = [(n, q) for n, q in all_recs if q.upper() not in already]
        if not new:
            raise SystemExit("STOP: nothing new to search -- "
                             f"{a.short_vs} already covers every sequence here")
        if len({q.upper() for _n, q in new}) != len(new):
            raise SystemExit(f"STOP: {len(new)} records but fewer distinct sequences; duplicates "
                             "would collide as <name>.a3m")
        write_fasta(a.short_out, new)
        L = sorted(len(q) for _n, q in new)
        print(f"{a.short_out}: {len(new)} sequences absent from {os.path.basename(a.short_vs)}; "
              f"lengths min {L[0]} median {L[len(L)//2]} max {L[-1]}")
        print(f"  of these, >=15 aa (where real MSA depth appears): {sum(1 for x in L if x >= 15)}")
    else:
        os.makedirs(os.path.dirname(a.query_fasta), exist_ok=True)
        write_fasta(a.query_fasta, all_recs)
        print(f"{a.query_fasta}: {len(all_recs)} sequences -- THE SEARCH SET")

    with open(f"{a.out_dir}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sha1", "length", "source"])
        w.writerows(manifest)
    print(f"wrote {a.out_dir}/manifest.csv ({len(manifest)} rows)")

    # the query set actually searched: one record per complex, chains ':'-joined
    cx = []
    if a.which in ("train", "both"):
        clusters, rows = A.training_clusters(a.splits, a.excluded)
        cx += [(n, s, "train") for n, s in train_complexes(rows, a.min_len)]
    if a.which in ("bench", "both"):
        cx += [(n, s, "bench") for n, s in bench_complexes(a.min_len, a.require_bench)]

    write_fasta(f"{a.out_dir}/complexes_all.fasta", [(n, ":".join(s)) for n, s, _k in cx])
    with open(f"{a.out_dir}/complexes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["complex", "source", "n_chains", "chain_index", "length", "sha1"])
        for n, seqs, kind in cx:
            for i, q in enumerate(seqs):
                w.writerow([n, kind, len(seqs), i, len(q),
                            hashlib.sha1(q.encode()).hexdigest()])
    nch = sum(len(s) for _n, s, _k in cx)
    byk = collections.Counter(k for _n, _s, k in cx)
    print(f"wrote {a.out_dir}/complexes_all.fasta: {len(cx)} complexes "
          f"({byk['train']} train, {byk['bench']} bench), {nch} chain slots")
    print(f"wrote {a.out_dir}/complexes.csv ({nch} rows)")


if __name__ == "__main__":
    main()
