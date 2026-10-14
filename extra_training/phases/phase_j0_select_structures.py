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
import argparse, collections, csv, hashlib, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
import phase_a_define_train_set as A          # noqa: E402

CUT = "2021-09-30"
POLY = A.POLY
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def job_tokens(r):
    """Token count of the job: antibody chains plus every polypeptide antigen."""
    n = len(r["Hseq"] or "") + len(r["Lseq"] or "")
    for t, q in A.antigen_chains(r["PDB_ID"], r["agtypes"], r["agexpectedseqs"]):
        if t in POLY and q:
            n += len(q)
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


def ag_key(v):
    """Normalise SAbDab2's agclusters into a stable key.

    The field is '/'-joined PER ANTIGEN CHAIN, with an empty slot for every non-polypeptide antigen,
    so one antigen appears as "P220", "P220/", "/P220" and "/P220/" depending on what else is bound,
    and a repeated chain as "P162/P162". Those are the SAME antigen and were four different cap keys,
    so the cap did not bind and a pairing could keep more than --cap structures. Measured on the
    previous selection: 325 raw keys collapse to 243.
    """
    return "/".join(sorted({x.strip() for x in str(v).split("/")
                            if x.strip() and x.strip().lower() != "nan"}))


def resolution_of(r):
    """Lower is better; an unparseable or absent resolution sorts last."""
    try:
        x = float(str(r.get("resolution", "")).strip())
        return x if x > 0 else 99.0
    except (TypeError, ValueError):
        return 99.0


def input_key(r):
    """What makes two jobs the same FOLD: the antibody chains and the polypeptide antigens.

    Two rows of one PDB entry can be different complexes -- a different antibody, or the same
    antibody against a different antigen chain -- and deduplicating on the PDB id treats them as one.
    """
    types = str(r["agtypes"]).split("/")
    seqs = str(r["agexpectedseqs"]).split("/")
    poly = tuple(sorted(x.strip().upper() for t, x in zip(types, seqs)
                        if t in POLY and x and x.strip().upper() != "NAN"))
    return ((r["Hseq"] or "").strip().upper(), (r["Lseq"] or "").strip().upper(), poly)


def chain_seqs(r):
    """Every chain sequence the fold needs an MSA for.

    THROUGH A.antigen_chains, which RAISES on a length disagreement, because this list decides
    `msa_ok`. The private copy here paired with `i < len(seqs)`, i.e. the silent truncation phase A
    refuses: a short sequence field dropped trailing antigens, `all(s in idx for s in ss)` then
    passed over a chain that was never checked, and the structure was marked MSA-ready with an
    antigen that has no MSA. The fold would fail later with nothing pointing back here.
    """
    out = [s for s in (r["Hseq"], r["Lseq"]) if s]
    return out + [q for t, q in A.antigen_chains(r["PDB_ID"], r["agtypes"], r["agexpectedseqs"])
                  if t in POLY and q]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=f"{ROOT}/data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--train-clusters", default=f"{ROOT}/splits/train_clusters.csv")
    ap.add_argument("--val-clusters", default=f"{ROOT}/splits/val_split_proposal_clusters.csv",
                    help="the LIVE validation split; its clusters are removed from training")
    ap.add_argument("--seq-index", default=f"{ROOT}/msa_work/pdb_seqs/seq_to_pdb_index.json")
    ap.add_argument("--cap", type=int, default=2, help="structures per (ab, ag) pairing")
    ap.add_argument("--out", default=f"{ROOT}/splits/cache_train_structures.csv")
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
    seen_pdb = set()
    collapsed = 0
    for cl in sorted(cand):
        # ONE PASS PER PAIRING, so the second pick is chosen against the structure actually taken
        # first. The previous version ranked candidates against `first_inp`, the first in SORTED
        # order, which is not the one the loop ends up taking once pdbs already used elsewhere are
        # skipped -- so a pairing could take two structures with IDENTICAL input while a genuinely
        # different one went unpicked. Measured on that selection: 28 of 69 two-pick pairings held
        # duplicate inputs (9b70/9b71, 7tmz/7u9f, 9cz7/9cza, 8hbv/8hbw among them).
        bypair = collections.defaultdict(list)
        for ab, ag, pdb, inst in sorted(cand[cl]):
            bypair[(ab, ag_key(ag))].append((pdb, inst))
        for pair in sorted(bypair):
            taken_inputs = []
            for _ in range(a.cap):
                best = None
                for pdb, inst in bypair[pair]:
                    if pdb in seen_pdb:
                        continue
                    r = byinst[inst]
                    inp = input_key(r)
                    # NEVER A DUPLICATE INPUT. A second structure identical in antibody and antigen
                    # sequence to one already taken adds a crystal form, not a training example.
                    if inp in taken_inputs:
                        collapsed += 1
                        continue
                    # WELL-RESOLVED NATIVE INTERFACES FIRST, then difference from what is already
                    # taken, then size. Resolution leads because a 3.5 A interface teaches the head
                    # a contact map the crystal barely determines.
                    diff = 0 if not taken_inputs else (
                        0 if all(inp[2] != t[2] for t in taken_inputs) else
                        1 if all(inp[:2] != t[:2] for t in taken_inputs) else 2)
                    k = (diff, resolution_of(r), job_tokens(r), pdb)
                    if best is None or k < best[0]:
                        best = (k, pdb, inst, r, inp)
                if best is None:
                    break
                _k, pdb, inst, r, inp = best
                seen_pdb.add(pdb)
                taken_inputs.append(inp)
                ss = chain_seqs(r)
                picked.append({
                    "cluster": cl, "pdb": pdb, "instance": inst,
                    "ab_cluster": pair[0], "agclusters": pair[1],
                    "tokens": job_tokens(r), "n_chains": len(ss),
                    "resolution": f"{resolution_of(r):.2f}", "method": r.get("method", ""),
                    "msa_ok": int(bool(ss) and all(x in idx for x in ss)),
                })
    print(f"duplicate-input candidates passed over: {collapsed}")

    # GATE THEN WRITE, and never with `assert`. This check is the only thing standing between the
    # cache and training on the validation set -- the failure it exists for already happened once,
    # putting 28 folded validation structures into an earlier selection. It used to sit BELOW the
    # writer, so a contaminated list reached disk before firing, and `assert` is stripped entirely
    # by `python -O`, which would have removed it. Same pattern phase G was restructured for.
    leaked_cl = sorted({r["cluster"] for r in picked} & val)
    if leaked_cl:
        raise SystemExit(f"STOP: {len(leaked_cl)} validation cluster(s) survived the filter and "
                         f"would be trained on: {leaked_cl[:10]}")
    if not picked:
        raise SystemExit("STOP: no structure was selected. Nothing is eligible, or the cap and the "
                         "per-pdb filter removed everything -- writing a header-only selection "
                         "would make the next phase prepare an empty cache.")

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(picked[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(picked)
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
