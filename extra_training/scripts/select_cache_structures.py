#!/usr/bin/env python
"""Freeze the structure list for the FT3 training cache.

Selection, in order (see docs/cached_training.md):
  0. EXCLUDE every cluster in the live validation split. results/train_clusters.csv is STALE: it was
     built against the ORIGINAL 50 phase-B validation clusters, and the 2026-10-13 redesign moved
     validation to 40 clusters, 22 of which that file still lists as training. Without this step the
     selection contained 44 structures from validation clusters, 28 of them structures actually
     folded into the validation set. Never trust train_clusters.csv alone.
  1. training clusters (results/train_clusters.csv, minus step 0)
  2. keep clusters holding >=1 post-2021-09-30 structure                     -> 224 clusters
  3. keep only the post-cutoff structures in them                            -> 1,594
  4. cap at --cap structures per (ab_cluster, agclusters) pairing            -> 666 at cap 2

Deterministic: candidates are visited in sorted order, so the same input gives the same list.
A structure already taken for its cluster is never taken twice; a structure can still appear under
two clusters, which is why rows (668) exceed unique structures (666).

Writes results/cache_train_structures.csv and prints the coverage gates. Nothing is folded here.
"""
import argparse, collections, csv, hashlib, json, os

CUT = "2021-09-30"
POLY = ("PROTEIN", "PEPTIDE")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job_tokens(r):
    n = len(r["Hseq"] or "") + len(r["Lseq"] or "")
    types = str(r["agtypes"]).split("/")
    seqs = str(r["agexpectedseqs"]).split("/")
    for i, t in enumerate(types):
        if t in POLY and i < len(seqs) and seqs[i]:
            n += len(seqs[i])
    return n


def eligible(r):
    """A training job must be an ANTIBODY plus at least one POLYPEPTIDE ANTIGEN.

    phase_a_define_train_set.py:263 applies exactly this filter when it builds the pool, and
    phases C and G apply it again when they emit sequences. This selector read abag_split.csv
    directly and applied neither, so the first frozen list carried 186 rows with no polypeptide
    antigen at all (175 SUGAR, 121 HAPTEN, 63 ION, 19 RNA among the types present), 29 jobs of a
    single chain, and 143 with no antigen cluster recorded. A hapten-only or sugar-only row is not
    an antibody-antigen complex for this purpose: there is no antigen chain to fold, so the fold
    produces no interface and the confidence head has no interface to be calibrated on.
    """
    has_ab = bool((r["Hseq"] or "").strip()) or bool((r["Lseq"] or "").strip())
    types = str(r["agtypes"]).split("/")
    seqs = str(r["agexpectedseqs"]).split("/")
    if len(types) != len(seqs):
        return False, "agtypes/agexpectedseqs length mismatch"
    poly = [s for t, s in zip(types, seqs) if t in POLY and s and s.strip().upper() != "NAN"]
    if not has_ab:
        return False, "no antibody chain"
    if not poly:
        return False, f"no polypeptide antigen (types={'/'.join(t for t in types if t)})"
    return True, ""


def chain_seqs(r):
    out = [s for s in (r["Hseq"], r["Lseq"]) if s]
    types = str(r["agtypes"]).split("/")
    seqs = str(r["agexpectedseqs"]).split("/")
    return out + [seqs[i] for i, t in enumerate(types)
                  if t in POLY and i < len(seqs) and seqs[i]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=f"{ROOT}/data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--train-clusters", default=f"{ROOT}/results/train_clusters.csv")
    ap.add_argument("--val-clusters", default=f"{ROOT}/results/val_split_proposal_clusters.csv",
                    help="the LIVE validation split; its clusters are removed from training")
    ap.add_argument("--seq-index", default=f"{ROOT}/msa_work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--cap", type=int, default=2, help="structures per (ab, ag) pairing")
    ap.add_argument("--out", default=f"{ROOT}/results/cache_train_structures.csv")
    a = ap.parse_args()

    # STALENESS GATE. train_clusters.csv is derived from a validation split and carries no record of
    # which one; it outlived its split once already. phase B now writes train_clusters.meta.json
    # beside it naming the validation file and hashing it, and that is checked here.
    meta_p = a.train_clusters.replace(".csv", ".meta.json")
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        live = hashlib.sha256(open(a.val_clusters, "rb").read()).hexdigest()
        if meta.get("val_sha256") != live:
            raise SystemExit(
                f"STOP: {os.path.basename(a.train_clusters)} was built from "
                f"{meta.get('val_file')} (sha {str(meta.get('val_sha256'))[:12]}, "
                f"{meta.get('n_val')} clusters), but the live validation split is {a.val_clusters} "
                f"(sha {live[:12]}). Re-run phase_b_define_val_clusters.py --prune first.")
    else:
        print(f"WARNING no {os.path.basename(meta_p)}; cannot verify which validation split "
              f"{os.path.basename(a.train_clusters)} was built from")

    train = {r["cluster"] for r in csv.DictReader(open(a.train_clusters))}
    val = {str(r["cluster"]) for r in csv.DictReader(open(a.val_clusters))}
    leaked = train & val
    train -= val
    print(f"train clusters {len(train) + len(leaked)} -> {len(train)} after removing "
          f"{len(leaked)} that are in the live validation split")
    rows = [r for r in csv.DictReader(open(a.split))
            if str(r["ab_ag_cluster"]) in train and (r["PDBdepo"] or "")[:10] > CUT]
    if not rows:
        raise SystemExit("STOP: no post-cutoff rows in the training clusters")

    cand = collections.defaultdict(set)
    byinst = {}
    dropped = collections.Counter()
    for r in rows:
        ok, why = eligible(r)
        if not ok:
            dropped[why.split(" (")[0]] += 1
            continue
        key = (str(r["ab_cluster"]), str(r["agclusters"]), r["PDB_ID"][-4:].lower(), r["INSTANCE"])
        cand[str(r["ab_ag_cluster"])].add(key)
        byinst[r["INSTANCE"]] = r

    idx = json.load(open(a.seq_index))
    picked = []
    seen_pdb = set()        # ONE ROW PER STRUCTURE: 9lsi and 9lsj sit in two ab_ag_clusters each,
                            # and a per-row fold queue would fold them twice for no extra signal.
    for cl in sorted(cand):
        seen, taken = collections.Counter(), set()
        for ab, ag, pdb, inst in sorted(cand[cl]):
            if pdb in taken or pdb in seen_pdb or seen[(ab, ag)] >= a.cap:
                continue
            seen_pdb.add(pdb)
            seen[(ab, ag)] += 1
            taken.add(pdb)
            r = byinst[inst]
            ss = chain_seqs(r)
            picked.append({
                "cluster": cl, "pdb": pdb, "instance": inst,
                "ab_cluster": ab, "agclusters": ag,
                "tokens": job_tokens(r), "n_chains": len(ss),
                "msa_ok": int(bool(ss) and all(s in idx for s in ss)),
            })

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(picked[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(picked)

    assert not ({r["cluster"] for r in picked} & val), "validation cluster survived the filter"
    pdbs = {r["pdb"] for r in picked}
    blocked = [r for r in picked if not r["msa_ok"]]
    need = {s for r in blocked for s in chain_seqs(byinst[r["instance"]]) if s not in idx}
    print(f"eligibility: dropped {sum(dropped.values())} row(s) -- {dict(dropped)}")
    print(f"rows {len(picked)}  unique structures {len(pdbs)}  clusters {len({r['cluster'] for r in picked})}")
    print(f"MSA ready {len(picked) - len(blocked)}  blocked {len(blocked)}  distinct sequences to search {len(need)}")
    print(f"tokens: median {sorted(r['tokens'] for r in picked)[len(picked)//2]}  max {max(r['tokens'] for r in picked)}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
