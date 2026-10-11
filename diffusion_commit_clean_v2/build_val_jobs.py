#!/usr/bin/env python
"""Write Protenix job jsons for the validation clusters, one representative structure per cluster.

SUMMARY
    Validation for the confidence-head finetune is held out BY CLUSTER (pick_val_clusters.py: 50
    clusters, 291 structures, spread across release dates). Only ONE structure per cluster is folded:
    structures inside a cluster are near-duplicates by construction, so the other 241 would cost GPU
    hours without adding information, and the cluster is the unit generalisation is measured over.

    THE REPRESENTATIVE is the structure with the median token count in its cluster -- not the largest
    (worst case for memory) and not the smallest (often a partial construct). Ties break on PDB id so
    the choice is deterministic.

    MSA PATHS come from Protenix's own seq_to_pdb_index.json, which maps a sequence to its directory
    under mmcif_msa_template/. That file is written by scripts/msa/step1-get_prot_seq.py, so this
    script cannot run until the MSA pipeline has produced it; it fails loudly rather than emitting a
    json with dangling paths.

    JSON SHAPE matches what our existing folds use (experimental_data/protenix/pxm2025/jsons):
        {"name": "<pdb>", "sequences": [{"proteinChain": {sequence, count,
          pairedMsaPath, unpairedMsaPath, templatesPath}}, ...]}

INPUTS
    --val-clusters: pick_val_clusters.py output. Default extra_training/results/val_clusters.csv.
    --msa-root:     directory holding mmcif_msa_template/ and seq_to_pdb_index.json.
                    Default extra_training/msa_work/mmcif_msa.
    --seq-index:    seq_to_pdb_index.json. Default <msa-root>/../pdb_seqs/seq_to_pdb_index.json.
    --out-dir:      where the jsons go. Default extra_training/val_jobs.
    --require-msa:  fail if any chain has no MSA directory. Default on; --no-require-msa to survey.

OUTPUTS
    <out-dir>/<pdb>.json   one per validation cluster
    <out-dir>/manifest.csv cluster, pdb, n_chains, n_tokens, msa_ok -- the record of what was folded

FUNCTIONS
    representatives(): the validation clusters -> one (cluster, pdb, chain sequences) each
    main():            resolve MSA paths, write jsons and the manifest
"""
import argparse
import csv
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import extract_msa_queries as Q  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VAL = f"{REPO}/extra_training/results/val_clusters.csv"
MSA_ROOT = f"{REPO}/extra_training/msa_work/mmcif_msa"
OUT = f"{REPO}/extra_training/val_jobs"


def _chains(row, min_len=20):
    """in: a splits row.  out: [sequence] in chain order -- heavy, light, then antigens"""
    out = []
    for v in (row.Hseq, row.Lseq):
        if isinstance(v, str) and len(v.strip()) >= min_len:
            out.append(v.strip().upper())
    for p in str(row.agexpectedseqs).split("/"):
        q = p.strip().upper()
        if len(q) >= min_len and q != "NAN":
            out.append(q)
    return out


def representatives(val_csv, reps=1):
    """in: the validation-cluster csv, structures to take per cluster.
    out: [(cluster, pdb, [sequences])], up to `reps` per cluster

    Structures are sorted by token count and `reps` evenly spaced picks are taken, so one rep is the
    cluster median and two span its size range rather than both being tiny partial constructs. Ties
    break on pdb id, so re-running picks the same structures. A cluster with fewer structures than
    `reps` contributes all it has -- the median across validation clusters is only 2.
    """
    want = set(pd.read_csv(val_csv).cluster)
    _clusters, rows = Q.training_clusters(Q.SPLITS, Q.EXCLUDED)
    rows = rows[rows.ab_ag_cluster.isin(want)].copy()
    rows["chains"] = [_chains(r) for r in rows.itertuples(index=False)]
    rows["instance"] = rows.INSTANCE.astype(str)
    rows["ntok"] = rows.chains.apply(lambda c: sum(len(x) for x in c))
    rows = rows[rows.chains.apply(len) >= 2]

    picks, skipped = [], sorted(want - set(rows.ab_ag_cluster))
    for cl, grp in rows.groupby("ab_ag_cluster"):
        g = (grp.drop_duplicates("PDB_ID").sort_values(["ntok", "PDB_ID"]).reset_index(drop=True))
        k = min(reps, len(g))
        idx = [int(round((i + 0.5) * len(g) / k)) for i in range(k)]
        for i in sorted(set(min(j, len(g) - 1) for j in idx)):
            r = g.iloc[i]
            picks.append((cl, str(r.PDB_ID)[-4:].lower(), r.chains, r.instance))
    return picks, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-clusters", default=VAL)
    ap.add_argument("--msa-root", default=MSA_ROOT)
    ap.add_argument("--seq-index", default=None)
    ap.add_argument("--out-dir", default=OUT)
    ap.add_argument("--reps", type=int, default=2,
                    help="structures per cluster; median of validation clusters has only 2")
    ap.add_argument("--require-msa", action=argparse.BooleanOptionalAction, default=True)
    a = ap.parse_args()

    seq_index_path = a.seq_index or os.path.join(
        os.path.dirname(a.msa_root.rstrip("/")), "pdb_seqs", "seq_to_pdb_index.json")
    seq_to_idx = {}
    if os.path.exists(seq_index_path):
        seq_to_idx = json.load(open(seq_index_path))
        print(f"seq_to_pdb_index.json: {len(seq_to_idx)} sequences")
    elif a.require_msa:
        raise SystemExit(
            f"missing {seq_index_path}\n"
            "It comes from Protenix scripts/msa/step1-get_prot_seq.py, so the MSA pipeline must\n"
            "finish first. Re-run with --no-require-msa only to survey which structures would be used."
        )
    else:
        print(f"NOTE {seq_index_path} absent -- surveying only, no jsons written")

    picks, skipped = representatives(a.val_clusters, a.reps)
    print(f"structures picked: {len(picks)} from {len(set(c for c, *_ in picks))} clusters"
          + (f"   SKIPPED (fewer than 2 chains): {skipped}" if skipped else ""))

    os.makedirs(a.out_dir, exist_ok=True)
    rows, missing = [], 0
    for cl, pdb, chains, inst in picks:
        seqs, ok = [], True
        for q in chains:
            idx = seq_to_idx.get(q)
            d = os.path.join(a.msa_root, str(idx)) if idx is not None else None
            if d is None or not os.path.isdir(d):
                ok = False
            seqs.append({"proteinChain": {
                "sequence": q, "count": "1",
                "pairedMsaPath": f"{d}/pairing.a3m" if d else "",
                "unpairedMsaPath": f"{d}/non_pairing.a3m" if d else "",
                "templatesPath": f"{d}/hmmsearch.a3m" if d else "",
            }})
        if not ok:
            missing += 1
        # Stage the chosen instance's cif as the DockQ reference. folder.py resolves references via
        # paths.DATASETS[...].ref(pdb), so one file per pdb under val_refs/ keeps that lookup trivial
        # and unambiguous when a pdb has several instances in the cluster.
        src = f"{REPO}/extra_training/data/training_cifs/{inst}.cif"
        ref_ok = os.path.exists(src)
        if ref_ok:
            os.makedirs(f"{REPO}/extra_training/val_refs", exist_ok=True)
            dst = f"{REPO}/extra_training/val_refs/{pdb}.cif"
            if not os.path.exists(dst):
                os.link(src, dst) if os.stat(src).st_dev == os.stat(
                    os.path.dirname(dst)).st_dev else __import__("shutil").copyfile(src, dst)
        rows.append(dict(cluster=cl, pdb=pdb, instance=inst, n_chains=len(chains),
                         n_tokens=sum(len(x) for x in chains), msa_ok=ok, ref_ok=ref_ok))
        if seq_to_idx and ok:
            with open(f"{a.out_dir}/{pdb}.json", "w") as fh:
                json.dump([{"name": pdb, "sequences": seqs}], fh, indent=2)

    with open(f"{a.out_dir}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    tok = sorted(r["n_tokens"] for r in rows)
    print(f"tokens: min {tok[0]} median {tok[len(tok)//2]} max {tok[-1]}")
    print(f"jsons written: {sum(1 for r in rows if r['msa_ok']) if seq_to_idx else 0}"
          f"   without complete MSAs: {missing}"
          f"   without a reference cif: {sum(1 for r in rows if not r['ref_ok'])}")
    if missing and a.require_msa:
        raise SystemExit(f"{missing} clusters are missing MSA directories -- not folding a partial set")
    print(f"wrote {a.out_dir}/manifest.csv")


if __name__ == "__main__":
    main()
