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
    --excluded: the homology-exclusion csv from exclude_homologs.py. Default under extra_training.
    --out-dir:  where the fastas go. Default extra_training/msa_queries.
    --min-len:  skip sequences shorter than this. Default 20.
    --which:    train, bench, or both. Default both.

    COMPLEX-LEVEL QUERIES ARE THE ONES ACTUALLY SEARCHED. Protenix needs a paired MSA per chain
    (`pairedMsaPath` in every job json, and our existing benchmark MSAs all have `pairing.a3m`), and
    colabfold_search only runs its pairing path when `get_queries` reports `is_complex` -- which
    happens only for FASTA records whose chains are ':'-joined. Passing --pair-mode on a file of
    single sequences is SILENTLY IGNORED. So `complexes_all.fasta` is the query set; with
    --pair-mode unpaired_paired one pass yields both the paired and the unpaired MSAs.

    The per-sequence fastas are kept for the dedup statistics and the train-vs-bench leakage check
    (verified: 0 sequences shared), not for searching.

    TRAIN AND BENCH GO IN ONE FILE. Rebuilding the benchmark MSAs on 2302 exists precisely to remove a
    train/test database-version difference, so running them through the same pass removes any chance
    of the two sets being treated differently, and halves database load overhead.

OUTPUTS
    <out-dir>/complexes_all.fasta   THE QUERY SET: one record per complex, chains ':'-joined
    <out-dir>/complexes.csv         complex id -> chain order, lengths, source; needed to map the
                                    per-complex a3m back onto Protenix's per-chain layout
    <out-dir>/train.fasta           unique training sequences (stats and leakage check only)
    <out-dir>/bench.fasta           unique benchmark sequences (same)
    <out-dir>/manifest.csv          sequence id -> sha1, length, source

FUNCTIONS
    training_clusters(): re-derive the surviving cluster set
    train_records():     the splits csv -> unique training sequences
    bench_records():     our pool csvs + reference cifs -> unique benchmark sequences
    train_complexes():   the splits csv -> one (name, [chain seqs]) per training structure
    bench_complexes():   reference cifs -> one (name, [chain seqs]) per benchmark complex
    main():              write the fastas, the complex table and the manifest
"""
import argparse
import collections
import csv
import hashlib
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paths              # noqa: E402
import exclude_homologs as EH  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = f"{REPO}/extra_training/data/zenodo_splits/splits_final/abag_split.csv"
EXCLUDED = f"{REPO}/extra_training/results/excluded_homolog_clusters.csv"
OUTDIR = f"{REPO}/extra_training/msa_queries"


def training_clusters(splits, excluded):
    """in: the splits csv, the homology-exclusion csv.  out: (surviving cluster set, the splits rows)

    Clusters holding one of our benchmark complexes go by id lookup; the homology csv removes the few
    more that match the 39 unlisted complexes by sequence.
    """
    _miss, _listed, by_id = EH.unlisted(splits)
    hom = set()
    if os.path.exists(excluded):
        with open(excluded) as fh:
            hom = {r["cluster"] for r in csv.DictReader(fh)}
    s = pd.read_csv(splits, low_memory=False,
                    usecols=["INSTANCE", "PDB_ID", "agtypes", "ab_ag_cluster", "Hseq", "Lseq",
                             "agexpectedseqs"])
    s = s[s.agtypes.apply(lambda v: any(t in EH.POLY for t in str(v).split("/")))]
    s = s[~s.ab_ag_cluster.isin(by_id) & ~s.ab_ag_cluster.astype(str).isin(hom)]
    return set(s.ab_ag_cluster.dropna()), s


def train_records(rows, min_len):
    """in: the surviving splits rows, a minimum length.  out: [(name, seq)] deduplicated by sequence

    The antibody chains come from Hseq/Lseq and the antigens from agexpectedseqs (the full construct,
    not agresolvedseqs, so an MSA is not built against a sequence with unmodelled gaps).
    """
    seen, out = set(), []
    for r in rows.itertuples(index=False):
        pdb = str(r.PDB_ID)[-4:]
        items = [("H", r.Hseq), ("L", r.Lseq)]
        items += [(f"ag{i}", p) for i, p in enumerate(str(r.agexpectedseqs).split("/"))]
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


def bench_records(min_len):
    """in: a minimum length.  out: [(name, seq)] unique chain sequences of our benchmark complexes

    Read from the reference cifs rather than the job jsons, so the sequence is the one the structure
    actually contains. Every polymer chain is taken -- which chain is antibody or antigen does not
    matter for an MSA.
    """
    seen, out = set(), []
    for ds in paths.DATASET_NAMES:
        D = paths.DATASETS[ds]
        with open(str(D.pool_csv)) as fh:
            pdbs = sorted({r["pdb"].lower() for r in csv.DictReader(fh)})
        for pdb in pdbs:
            ref = D.ref(pdb)
            if not os.path.exists(ref):
                continue
            try:
                chains = EH.chains_of(ref)
            except Exception:                           # noqa: BLE001
                continue
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
        for p in str(r.agexpectedseqs).split("/"):
            q = p.strip().upper()
            if len(q) >= min_len and q != "NAN":
                chains.append(q)
        key = (pdb, tuple(chains))
        if len(chains) < 2 or key in seen:
            continue
        seen.add(key)
        out.append((f"tr_c{r.ab_ag_cluster}_{pdb}", chains))
    return out


def bench_complexes(min_len):
    """in: a minimum length.  out: [(name, [chain seqs])] per benchmark complex

    Chains come from the reference cif in its own chain order, so the mapping back to Protenix's
    per-chain msa/<n>/ directories follows the same order the folds already use.
    """
    out = []
    for ds in paths.DATASET_NAMES:
        D = paths.DATASETS[ds]
        with open(str(D.pool_csv)) as fh:
            pdbs = sorted({r["pdb"].lower() for r in csv.DictReader(fh)})
        for pdb in pdbs:
            ref = D.ref(pdb)
            if not os.path.exists(ref):
                continue
            try:
                chains = EH.chains_of(ref)
            except Exception:                           # noqa: BLE001
                continue
            seqs = [q.upper() for _c, q in sorted(chains.items()) if len(q) >= min_len]
            if len(seqs) >= 2:
                out.append((f"bn_{ds[:2]}_{pdb}", seqs))
    return out


def write_fasta(path, records):
    with open(path, "w") as fh:
        for n, q in records:
            fh.write(f">{n}\n{q}\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default=SPLITS)
    ap.add_argument("--excluded", default=EXCLUDED)
    ap.add_argument("--out-dir", default=OUTDIR)
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--which", choices=["train", "bench", "both"], default="both")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    manifest = []
    if a.which in ("train", "both"):
        clusters, rows = training_clusters(a.splits, a.excluded)
        recs = train_records(rows, a.min_len)
        write_fasta(f"{a.out_dir}/train.fasta", recs)
        lens = sorted(len(q) for _n, q in recs)
        print(f"train.fasta: {len(recs)} unique sequences from {len(clusters)} clusters, "
              f"{rows.PDB_ID.nunique()} structures; len median {lens[len(lens)//2]}, "
              f"total {sum(lens):,} residues")
        manifest += [(n, hashlib.sha1(q.encode()).hexdigest(), len(q), "train") for n, q in recs]

    if a.which in ("bench", "both"):
        recs = bench_records(a.min_len)
        write_fasta(f"{a.out_dir}/bench.fasta", recs)
        lens = sorted(len(q) for _n, q in recs)
        print(f"bench.fasta: {len(recs)} unique sequences; len median {lens[len(lens)//2]}, "
              f"total {sum(lens):,} residues")
        manifest += [(n, hashlib.sha1(q.encode()).hexdigest(), len(q), "bench") for n, q in recs]

    with open(f"{a.out_dir}/manifest.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "sha1", "length", "source"])
        w.writerows(manifest)
    print(f"wrote {a.out_dir}/manifest.csv ({len(manifest)} rows)")

    # the query set actually searched: one record per complex, chains ':'-joined
    cx = []
    if a.which in ("train", "both"):
        clusters, rows = training_clusters(a.splits, a.excluded)
        cx += [(n, s, "train") for n, s in train_complexes(rows, a.min_len)]
    if a.which in ("bench", "both"):
        cx += [(n, s, "bench") for n, s in bench_complexes(a.min_len)]

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
