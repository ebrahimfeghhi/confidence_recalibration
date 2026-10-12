#!/usr/bin/env python
"""PHASE G -- write one Protenix job json per validation structure.

SUMMARY
    Validation for the confidence-head finetune is held out BY CLUSTER
    (phase_b_define_val_clusters.py: 50 clusters, 291 structures, spread across release dates).

    TWO structures per cluster are folded, not one (--reps, default 2), and only where two distinct
    PDB entries exist: 29 clusters supply two and 21 supply one, giving 79 picks. Two more are then
    excluded for an ambiguous reference labelling, so **77 structures and 150 interfaces are folded**
    -- that is the number to quote, and it is printed on every run rather than kept here.

    Structures inside a cluster are near-duplicates by construction, so folding all 291 would cost
    GPU hours without adding information, and the cluster is the unit generalisation is measured over.

    THE REPRESENTATIVES are spread over the INTERIOR of the cluster's size range, deliberately
    avoiding both the largest structure (the memory worst case) and the smallest (often a partial
    construct). Ties break on PDB id, so the choice is deterministic.

    MSA PATHS come from Protenix's own seq_to_pdb_index.json, which maps a sequence to its directory
    under mmcif_msa_template/. That file is written by scripts/msa/step1-get_prot_seq.py, so this
    script cannot run until the MSA pipeline has produced it; it fails loudly rather than emitting a
    json with dangling paths.

    JSON SHAPE matches what our existing folds use (experimental_data/protenix/pxm2025/jsons):
        {"name": "<pdb>", "sequences": [{"proteinChain": {sequence, count,
          pairedMsaPath, unpairedMsaPath, templatesPath}}, ...]}

INPUTS
    --val-clusters: phase B output. Default extra_training/results/val_clusters.csv.
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

import gemmi
import pandas as pd

# This file lives at <repo>/extra_training/phases/, so the repo root is three levels up. paths.py
# stays at the repo root: it is shared infrastructure that 20+ files import.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import phase_a_define_train_set as A  # noqa: E402

VAL = f"{REPO}/extra_training/results/val_clusters.csv"
MSA_ROOT = f"{REPO}/extra_training/msa_work/mmcif_msa"
OUT = f"{REPO}/extra_training/val_jobs"


def ref_auth_to_label(cif):
    """in: a reference mmCIF.  out: ({auth chain: label_asym_id}, {shared label: [auth chains]})

    An empty first element with a non-empty second means the reference's labelling is ambiguous and
    the structure cannot be used -- see the reverse check at the end.

    WHY THE LABEL NAME IS THE ONE THAT MATTERS. The scorer keys every interface by the REFERENCE
    chain name, and the reference name it sees is label_asym_id, not auth_asym_id: an actual scorer
    output for 8ror carries ref_chain_info keys ['A','B','C'] and
    ref_to_model_chain_mapping {'A':'A','B':'B','C':'C'}, while gemmi reports that same cif's auth
    chains as ['1','L','H'].

    pxm2025 happens to have label chains A, B, C, which is why pool.csv's A/B/C letters have always
    worked there. The validation references do not: 1him is D/E/F, 7wur is B/E/F, 1rzk is A/C/D. An
    interface written as "AC" against those is looked up with res["A,C"], which does not exist, and
    ost_dockq.interface() has no positional fallback -- so it returns None and NO ROW IS WRITTEN. On
    8 of 8 sampled validation structures that is every interface.

    This class of error has already cost this project once: archive/.../iface_chains.py records
    scripts that read these labels as MODEL names, which silently measured an antibody-internal
    contact as the epitope on 8gy5 and affected 72 of 161 interfaces.
    """
    st = gemmi.read_structure(cif)
    st.setup_entities()
    out = {}
    for ch in st[0]:
        subs = [sub.subchain_id() for sub in ch.subchains() if len([r for r in sub]) >= 5]
        if not subs:
            continue
        if len(subs) > 1:
            # Measured 0 of 70 on the validation references, but a Fab split into variable and
            # constant label chains would land here, and guessing which one the role means would be
            # exactly the kind of assumption that caused the 8gy5 error.
            raise SystemExit(
                f"STOP: in {cif}, auth chain {ch.name!r} covers several polymer label chains "
                f"{subs}. The reference chain name for a role is then ambiguous; resolve it "
                "explicitly rather than guessing.")
        out[ch.name] = subs[0]

    # AND THE REVERSE, which is the direction that actually occurs. label_asym_id is supposed to be
    # unique per chain instance, and in two of the validation references it is not: 8evg carries
    # (auth N1, label L) AND (auth N2, label L), 8vfx the same with J. The heavy and light chains are
    # then indistinguishable by reference name, so the H-antigen and L-antigen interfaces collapse
    # into one pair and neither can be scored as itself. Returning an empty mapping makes the caller
    # exclude the structure, which is the only honest option -- the alternative is silently scoring
    # one interface twice and calling it two.
    back = {}
    for an, ln in out.items():
        back.setdefault(ln, []).append(an)
    shared = {ln: a for ln, a in back.items() if len(a) > 1}
    if shared:
        return {}, shared
    return out, {}


def relabel_unique(src, dst):
    """in: a cif whose label_asym_id is shared by several auth chains, a path to write.
    out: the written path, with one unique label per auth chain

    Assigns each auth chain's residues a fresh subchain id -- gemmi's own field, written by its own
    mmCIF writer -- restoring the invariant the format requires. Auth names, coordinates and residue
    numbering are untouched; only the label is changed, and only where it was already degenerate.
    `setup_entities()` alone does not fix this: it preserves the duplicate labels.
    """
    st = gemmi.read_structure(src)
    st.setup_entities()
    nxt = iter("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")
    for ch in st[0]:
        lab = next(nxt)
        for res in ch:
            res.subchain = lab
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    st.make_mmcif_document().write_file(dst)
    return dst


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
    # BY ANTIGEN TYPE, the same rule phase C applies. Taking every agexpectedseqs entry emitted
    # non-polypeptide antigens as proteinChain: a mixed protein/RNA row shipped its RNA antigen into
    # the job json, where Protenix would fold it as a protein. agtypes is '/'-joined in the same
    # order, and a length disagreement between the two fields means the positional pairing is not
    # safe, so it is an error rather than a silent truncation by zip().
    types = str(row.agtypes).split("/")
    seqs = str(row.agexpectedseqs).split("/")
    if len(types) != len(seqs):
        raise SystemExit(f"STOP: {row.PDB_ID} lists {len(types)} antigen types but {len(seqs)} "
                         "antigen sequences; they are positional and must correspond.")
    for i, (t, p) in enumerate(zip(types, seqs)):
        q = p.strip().upper()
        if len(q) >= min_len and q != "NAN" and t.strip().upper() in A.POLY:
            out.append((f"ag{i}", q))
    return out


def representatives(val_csv, reps=1, min_len=5, splits=None, excluded=None):
    """in: the validation-cluster csv, structures to take per cluster.
    out: [(cluster, pdb, [sequences])], up to `reps` per cluster

    Structures are sorted by token count and `reps` evenly spaced picks are taken, so one rep is the
    cluster median and two span its size range rather than both being tiny partial constructs. Ties
    break on pdb id, so re-running picks the same structures. A cluster with fewer structures than
    `reps` contributes all it has -- the median across validation clusters is only 2.
    """
    want = set(pd.read_csv(val_csv).cluster)
    # --splits AND --excluded PROPAGATED. These used phase A's module defaults while --splits
    # controlled only the role-name lookup in main(), so pointing the phase at an alternative release
    # mixed sequences from one dataset with roles from another.
    _clusters, rows = A.training_clusters(splits or A.SPLITS, excluded or A.EXCLUDED)
    rows = rows[rows.ab_ag_cluster.isin(want)].copy()
    rows["chains"] = [_chains(r, min_len) for r in rows.itertuples(index=False)]
    rows["instance"] = rows.INSTANCE.astype(str)
    rows["ntok"] = rows.chains.apply(lambda c: sum(len(q) for _, q in c))
    rows = rows[rows.chains.apply(len) >= 2]

    picks, skipped = [], sorted(want - set(rows.ab_ag_cluster))
    for cl, grp in rows.groupby("ab_ag_cluster"):
        g = (grp.drop_duplicates("PDB_ID").sort_values(["ntok", "PDB_ID"]).reset_index(drop=True))
        k = min(reps, len(g))
        # SPREAD OVER THE INTERIOR, not over the full range. The old arithmetic,
        # round((i+0.5)*len/k), lands on the LARGEST structure whenever k=2: indices 1 and 2 for a
        # cluster of 3, 1 and 4 for a cluster of 5. The largest structure is the memory worst case,
        # which is exactly what picking the median avoided at reps=1, and phase I's only backstop for
        # an OOM is a round that completes nothing. Scaling by (len-1) instead keeps both picks away
        # from the extreme: 0 and 1 for a cluster of 3, 1 and 3 for a cluster of 5.
        span = max(1, len(g) - 1)
        idx = [int(round((i + 0.5) * span / k)) for i in range(k)]
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
    ap.add_argument("--excluded", default=None,
                    help="phase A's exclusion csv; default phase A's own path")
    ap.add_argument("--splits", default=f"{REPO}/extra_training/data/zenodo_splits/"
                                        "splits_final/abag_split.csv",
                    help="SAbDab2's own chain-role assignment, for the auth chain names")
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

    # THE AUTH CHAIN NAMES, from SAbDab2's own role assignment, keyed by instance. agchains is
    # '/'-joined in the same order as agtypes and agexpectedseqs, so ag<i> is agchains[i].
    sp = pd.read_csv(a.splits, low_memory=False,
                     usecols=["INSTANCE", "Hchain", "Lchain", "agchains"])
    auth_names = {}
    for r in sp.itertuples(index=False):
        d = {}
        if isinstance(r.Hchain, str) and r.Hchain.strip():
            d["H"] = r.Hchain.strip()
        if isinstance(r.Lchain, str) and r.Lchain.strip():
            d["L"] = r.Lchain.strip()
        if isinstance(r.agchains, str):
            for i, c in enumerate(r.agchains.split("/")):
                if c.strip():
                    d[f"ag{i}"] = c.strip()
        auth_names[str(r.INSTANCE)] = d

    # deposition dates, by instance and by pdb, for the manifest's per-structure cutoff flag
    _dp = pd.read_csv(a.splits, low_memory=False, usecols=["INSTANCE", "PDB_ID", "PDBdepo"])
    depo_of = {}
    for r in _dp.itertuples(index=False):
        if isinstance(r.PDBdepo, str):
            depo_of[str(r.INSTANCE)] = r.PDBdepo[:10]
            depo_of.setdefault(str(r.PDB_ID)[-4:].lower(), r.PDBdepo[:10])

    unresolved, ambiguous, relabelled = [], [], []
    picks, skipped = representatives(a.val_clusters, a.reps, a.min_len, a.splits, a.excluded)
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
        # MODEL letters follow the json order; REFERENCE labels come from the cif. Both are recorded,
        # and pool.csv carries the REFERENCE ones because that is what the scorer looks up.
        letters = {i: chr(ord("A") + i) for i in range(len(chains))}
        # THE SOURCE CIF, not the staged copy. The staged val_refs/<pdb>.cif is written further down
        # this same loop, so reading it here saw only files left by a previous run -- every newly
        # picked structure came back "cif offers []" and looked like a data problem when it was an
        # ordering bug.
        src_cif = f"{REPO}/extra_training/data/training_cifs/{inst}.cif"
        a2l, shared_lbl = ref_auth_to_label(src_cif) if os.path.exists(src_cif) else ({}, {})
        if shared_lbl:
            # RELABELLED, NOT EXCLUDED. label_asym_id must be unique per chain instance and in a
            # couple of SAbDab2's processed cifs it is not (8evg: auth N1 and N2 both labelled L),
            # which collapses the heavy-antigen and light-antigen interfaces into one unscoreable
            # pair. The deposited RCSB entries are clean, but SAbDab2 renamed the chains, so their
            # names cannot bridge the two files. Instead the staged reference gets unique labels
            # assigned here, and pool.csv is written to match -- both sides come from this function,
            # so they cannot disagree.
            src_cif = relabel_unique(src_cif, f"{REPO}/extra_training/val_refs/{pdb}.relabelled.cif")
            a2l, shared_lbl = ref_auth_to_label(src_cif)
            if shared_lbl:
                raise SystemExit(f"STOP: {pdb} still has shared reference labels after relabelling: "
                                 f"{shared_lbl}")
            relabelled.append(pdb)
        auth = auth_names.get(inst, {})
        ref_of = {}
        for i, (r, _) in enumerate(chains):
            an = auth.get(r)
            ln = a2l.get(an) if an else None
            ref_of[i] = ln
            roles.append(dict(pdb=pdb, model_chain=letters[i], ref_chain=ln or "",
                              auth_chain=an or "", role=r))
        # NOT named `missing`: that counter already exists below for chains lacking an MSA.
        miss_ref = [(letters[i], r) for i, (r, _) in enumerate(chains) if not ref_of[i]]
        if miss_ref:
            unresolved.append((pdb, miss_ref, sorted(a2l.items())))
            continue
        ab = [ref_of[i] for i, (r, _) in enumerate(chains) if r in ("H", "L")]
        ag = [ref_of[i] for i, (r, _) in enumerate(chains) if r.startswith("ag")]
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
                # count IS AN INT, not "1". Protenix's featuriser does
                # `range(1, entity["count"] + 1)` (json_to_feature.py:136), which raises
                # TypeError: can only concatenate str (not "int") to str on a string -- and it
                # raises inside the dataloader, so the fold "completes" in 0.04s having produced
                # nothing. The benchmark jsons we were copying use ints; this was a transcription
                # error that cost the first smoke attempt.
                "sequence": q, "count": 1,
                "pairedMsaPath": f"{d}/pairing.a3m" if d else "",
                "unpairedMsaPath": f"{d}/non_pairing.a3m" if d else "",
                "templatesPath": f"{d}/hmmsearch.a3m" if d else "",
            }})
        if not ok:
            missing += 1
        # Stage the chosen instance's cif as the DockQ reference. folder.py resolves references via
        # paths.DATASETS[...].ref(pdb), so one file per pdb under val_refs/ keeps that lookup trivial
        # and unambiguous when a pdb has several instances in the cluster.
        src = src_cif
        ref_ok = os.path.exists(src)
        if ref_ok:
            os.makedirs(f"{REPO}/extra_training/val_refs", exist_ok=True)
            dst = f"{REPO}/extra_training/val_refs/{pdb}.cif"
            # ALWAYS RE-STAGE. The old `if not os.path.exists(dst)` meant a file left by an earlier
            # run was never replaced -- which silently defeated the relabelling above: 8evg's
            # corrected reference was written, and then the stale copy with the degenerate labels was
            # kept, so its four interfaces still named chains the reference did not have. The source
            # is authoritative, so overwriting is always right and removes a whole staleness class.
            if os.path.exists(dst) or os.path.islink(dst):
                os.remove(dst)
            same_fs = os.stat(src).st_dev == os.stat(os.path.dirname(dst)).st_dev
            os.link(src, dst) if same_fs else __import__("shutil").copyfile(src, dst)
        # EACH STRUCTURE'S OWN DATE, not its cluster's median. Pre/post-cutoff performance has to be
        # reported per folded structure: a cluster whose median date is after 2021-09-30 can still
        # have picked a representative from before it, and freezing the trunk does not make that
        # representative unseen by the pretrained weights.
        dep = depo_of.get(inst) or depo_of.get(pdb) or ""
        rows.append(dict(cluster=cl, pdb=pdb, instance=inst, deposited=dep,
                         post_cutoff=bool(dep and dep[:10] > "2021-09-30"), n_chains=len(chains),
                         n_tokens=sum(len(q) for _, q in chains), msa_ok=ok, ref_ok=ref_ok,
                         n_ifaces=len(ab) * len(ag)))
        if seq_to_idx and ok:
            # THE NAME folder.py LOADS, not a name of our choosing. folder.py:784 opens
            # f"{jobs}/{pdb}-final-updated.json" and :985 builds the runner from the same pattern, and
            # pxm2025 and foldbench both follow it (8ror-final-updated.json, 7st5-final-updated.json).
            # Writing <pdb>.json meant zero matches: the fold would have died at runner
            # initialisation, before folding anything -- and phase I's input check globs THESE files,
            # so it validated them happily and could not see the mismatch.
            with open(f"{a.out_dir}/{pdb}-final-updated.json", "w") as fh:
                json.dump([{"name": pdb, "sequences": seqs}], fh, indent=2)

    # pool.csv is what folder.py's interfaces_by_pdb() reads to decide which interfaces a target=*
    # unit is scored on. Without it the fold runs and scores NOTHING, and paths.py already points
    # sabdab_val's pool_csv here.
    os.makedirs(f"{REPO}/extra_training/val_fold", exist_ok=True)
    # lineterminator="\n". csv.writer defaults to CRLF, and a shell reading the file with
    # `IFS=, read -r pdb c1 c2` keeps the \r in the last field -- producing interfaces like
    # "6mnq_AB\r", which scorer.py rejects because the chain pair is then three characters. Every
    # consumer here is POSIX, so LF is the right output and no reader needs to strip anything.
    with open(f"{REPO}/extra_training/val_fold/pool.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "chain1", "chain2"], lineterminator="\n")
        w.writeheader()
        w.writerows(pool)
    with open(f"{REPO}/extra_training/val_fold/chain_roles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["pdb", "model_chain", "ref_chain", "auth_chain", "role"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(roles)
    print(f"pool.csv: {len(pool)} interfaces over {len(set(r['pdb'] for r in pool))} structures")

    with open(f"{a.out_dir}/manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    if relabelled:
        print(f"  relabelled {len(relabelled)} reference(s) whose label_asym_id was shared by "
              f"several auth chains: {relabelled}")

    if ambiguous:
        # EXCLUDED, NOT FAILED. Two structures with a defective reference labelling should not block
        # the other 77, and scoring what cannot be named is worse than not scoring it.
        for pdb, sh in ambiguous:
            print(f"  EXCLUDED {pdb}: reference label(s) shared by several auth chains {sh}; its "
                  "antibody-antigen interfaces cannot be named distinctly")
        print(f"  {len(ambiguous)} structure(s) excluded for ambiguous reference labelling")

    if unresolved:
        for pdb, miss, labels in unresolved[:5]:
            print(f"  UNRESOLVED {pdb}: no reference label for {miss}; cif offers {labels}")
        raise SystemExit(
            f"STOP: {len(unresolved)} structures have a chain with no reference label. pool.csv would "
            "name an interface the scorer cannot find, and the fold would score NOTHING for them.")

    # ONE JOB PER PDB ID. Two instances of the same entry picked from different clusters would write
    # the same <pdb>-final-updated.json, and the second would overwrite the first -- one fold, two
    # clusters' worth of expectations, and a pool.csv naming interfaces from both.
    seen_pdb = {}
    for r in rows:
        if r["pdb"] in seen_pdb and seen_pdb[r["pdb"]] != r["instance"]:
            raise SystemExit(
                f"STOP: pdb {r['pdb']} was picked twice, as instances {seen_pdb[r['pdb']]} and "
                f"{r['instance']}. Both would write the same job file.")
        seen_pdb[r["pdb"]] = r["instance"]

    # OBSOLETE JOBS REMOVED. A regenerated set with different picks left the previous run's job files
    # behind, and phase I folds everything matching *-final-updated.json -- so it would fold
    # structures this run did not choose, against a pool.csv that does not describe them.
    want_jobs = {f"{r['pdb']}-final-updated.json" for r in rows}
    stale = [f for f in os.listdir(a.out_dir)
             if f.endswith("-final-updated.json") and f not in want_jobs] if os.path.isdir(a.out_dir) else []
    for f in stale:
        os.remove(os.path.join(a.out_dir, f))
    if stale:
        print(f"  removed {len(stale)} obsolete job file(s) from a previous selection: {stale[:5]}")

    # THE CLUSTER SET IS WHAT WAS ASKED FOR.
    want_cl = set(pd.read_csv(a.val_clusters).cluster)
    got_cl = {r["cluster"] for r in rows}
    if got_cl != want_cl:
        miss_cl = sorted(want_cl - got_cl)
        raise SystemExit(f"STOP: {len(miss_cl)} validation cluster(s) contributed no structure: "
                         f"{miss_cl[:10]}")

    if not rows:
        raise SystemExit(
            "STOP: no structure survived. Nothing was picked, or every pick was excluded for an "
            "ambiguous reference labelling or an unresolvable chain. Check the messages above.")

    tok = sorted(r["n_tokens"] for r in rows)
    npost = sum(1 for r in rows if r["post_cutoff"])
    print(f"tokens: min {tok[0]} median {tok[len(tok)//2]} max {tok[-1]}")
    print(f"deposited after Protenix's 2021-09-30 cutoff: {npost} of {len(rows)} folded structures "
          "(manifest.csv carries each one's own date; a cluster's median date does not settle it)")
    print(f"jsons written: {sum(1 for r in rows if r['msa_ok']) if seq_to_idx else 0}"
          f"   without complete MSAs: {missing}"
          f"   without a reference cif: {sum(1 for r in rows if not r['ref_ok'])}")
    if missing and a.require_msa:
        raise SystemExit(f"{missing} clusters are missing MSA directories -- not folding a partial set")
    print(f"wrote {a.out_dir}/manifest.csv")


if __name__ == "__main__":
    main()
