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


def _chains(row, min_len=5):
    """in: a splits row.  out: [(role, sequence)] in chain order -- heavy, light, then antigens

    The ORDER IS THE CONTRACT. Protenix labels predicted chains A, B, C... in the order of the job
    json's `sequences` array -- verified against an existing fold: 8ror's json order (1457, 218, 222
    residues) comes back as chains A, B, C with those exact lengths. pool.csv below depends on it.
    """
    out = []
    for role, v in (("H", row.Hseq), ("L", row.Lseq)):
        if isinstance(v, str) and len(v.strip()) >= min_len:
            out.append((role, v.strip().upper()))
    for i, p in enumerate(str(row.agexpectedseqs).split("/")):
        q = p.strip().upper()
        if len(q) >= min_len and q != "NAN":
            out.append((f"ag{i}", q))
    return out


def representatives(val_csv, reps=1, min_len=5):
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
    rows["chains"] = [_chains(r, min_len) for r in rows.itertuples(index=False)]
    rows["instance"] = rows.INSTANCE.astype(str)
    rows["ntok"] = rows.chains.apply(lambda c: sum(len(q) for _, q in c))
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
    ap.add_argument("--min-len", type=int, default=5,
                    help="shortest antigen chain to include. 5, not 20: FoldBench contains protein "
                         "chains under 20 aa and searches them normally, and phase1b_short.sh built "
                         "MSAs for the 235 such antigen chains the first pass skipped.")
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

    picks, skipped = representatives(a.val_clusters, a.reps, a.min_len)
    print(f"structures picked: {len(picks)} from {len(set(c for c, *_ in picks))} clusters"
          + (f"   SKIPPED (fewer than 2 chains): {skipped}" if skipped else ""))

    os.makedirs(a.out_dir, exist_ok=True)
    rows, missing, pool, roles = [], 0, [], []
    for cl, pdb, chains, inst in picks:
        seqs, ok = [], True
        # Chain letters follow the json order, so enumerate(chains) gives A, B, C... Interfaces are
        # every (antibody chain, antigen chain) pair, alphabetically ordered, which is the convention
        # the pxm2025 pool uses (8ror: A-B and A-C, antigen against each antibody chain) and excludes
        # the intra-antibody H-L interface, which is not what these folds are about.
        letters = {i: chr(ord("A") + i) for i in range(len(chains))}
        ab = [letters[i] for i, (r, _) in enumerate(chains) if r in ("H", "L")]
        ag = [letters[i] for i, (r, _) in enumerate(chains) if r.startswith("ag")]
        for i, (r, _) in enumerate(chains):
            roles.append(dict(pdb=pdb, chain=letters[i], role=r))
        for abc in ab:
            for agc in ag:
                c1, c2 = sorted((abc, agc))
                pool.append(dict(pdb=pdb, chain1=c1, chain2=c2))
        for _role, q in chains:
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
                         n_tokens=sum(len(q) for _, q in chains), msa_ok=ok, ref_ok=ref_ok,
                         n_ifaces=len(ab) * len(ag)))
        if seq_to_idx and ok:
            with open(f"{a.out_dir}/{pdb}.json", "w") as fh:
                json.dump([{"name": pdb, "sequences": seqs}], fh, indent=2)

    # pool.csv is what folder.py's interfaces_by_pdb() reads to decide which interfaces a target=*
    # unit is scored on. Without it the fold runs and scores NOTHING, and paths.py already points
    # sabdab_val's pool_csv here.
    os.makedirs(f"{REPO}/extra_training/val_fold", exist_ok=True)
    with open(f"{REPO}/extra_training/val_fold/pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "chain1", "chain2"])
        w.writeheader()
        w.writerows(pool)
    with open(f"{REPO}/extra_training/val_fold/chain_roles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "chain", "role"])
        w.writeheader()
        w.writerows(roles)
    print(f"pool.csv: {len(pool)} interfaces over {len(set(r['pdb'] for r in pool))} structures")

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
