#!/usr/bin/env python
"""Choose the second representative for each validation cluster that has one.

    ./phase_b3_pick_second_reps.py                       # dry run
    ./phase_b3_pick_second_reps.py --out ../results/val_split_second_reps.csv

RULE, fixed in advance and blind to predictions: among a cluster's other eligible post-cutoff
structures, prefer one whose POLYPEPTIDE ANTIGEN SEQUENCES DIFFER from the first representative's,
then the smallest by token count. Antigen identity and token count are properties of the input and
the reference; no DockQ, ipTM or other prediction enters, so the choice cannot be tuned to a result.

ANCHORED TO THE FROZEN FIRST INSTANCES, by exact INSTANCE id. Two bugs in the first version made that
necessary, both reproduced in review:

  1. The first representative's antigen was read from the first row encountered for its PDB_ID. One
     PDB entry can hold several antibody-antigen instances with DIFFERENT antigens, so the comparison
     could use an antigen the fold never saw -- labelling an identical antigen "different" and
     passing over a genuinely different one.
  2. The first representative was read as {cluster: pdb} from val2_jobs/manifest.csv. Once that
     manifest holds 53 structures the dict keeps whichever row comes LAST per cluster, so a re-run
     silently promoted the second representative to first and lost the original.

So the first instances come from --first (the frozen proposal), exactly one per cluster is required,
and every one must be present in the built manifest. Antigens are resolved by INSTANCE, and only the
agexpectedseqs entries whose agtypes is polypeptide are compared -- those are what phase G folds.
"""
import argparse
import os
import collections
import csv
import sys

# __file__-RELATIVE, like the phases. A hardcoded absolute root cannot run in a review
# clone or against a scratch directory, which is how phase G's refactor was proved
# byte-identical.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLY = ("PROTEIN", "PEPTIDE")
CUT = "2021-09-30"


def poly_parts(r):
    """(token count, polypeptide antigen sequences) exactly as phase G folds them."""
    n = len(r["Hseq"] or "") + len(r["Lseq"] or "")
    types = str(r["agtypes"]).split("/")
    seqs = str(r["agexpectedseqs"]).split("/")
    ag = set()
    for i, t in enumerate(types):
        if t in POLY and i < len(seqs) and seqs[i]:
            n += len(seqs[i])
            ag.add(seqs[i])
    return n, ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", default=f"{REPO}/train_val_splits/val_split_proposal.csv",
                    help="the frozen split: one row per cluster with cluster and instance")
    ap.add_argument("--manifest", default=f"{REPO}/val2_jobs/manifest.csv",
                    help="checked only -- every frozen first instance must appear here")
    ap.add_argument("--splits", default=f"{REPO}/SAbDab2_data/zenodo_splits/splits_final/abag_split.csv")
    ap.add_argument("--min-tokens", type=int, default=150)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    # ---- the frozen first representatives: exactly one per cluster, by INSTANCE
    rows = list(csv.DictReader(open(a.first)))
    per = collections.defaultdict(list)
    for r in rows:
        per[r["cluster"]].append(r["instance"])
    dupes = {c: v for c, v in per.items() if len(v) != 1}
    if dupes:
        raise SystemExit(f"STOP: {len(dupes)} cluster(s) in {a.first} do not have exactly one "
                         f"instance: {dict(list(dupes.items())[:3])}. The first representatives must "
                         f"be frozen and unique; a file that already contains second "
                         f"representatives is not a valid --first.")
    first_inst = {c: v[0] for c, v in per.items()}
    built = {m["instance"] for m in csv.DictReader(open(a.manifest))}
    missing = sorted(c for c, i in first_inst.items() if i not in built)
    if missing:
        raise SystemExit(f"STOP: {len(missing)} frozen first instance(s) are absent from "
                         f"{a.manifest}: clusters {missing[:5]}. Refusing to pick seconds against a "
                         f"set that was not built.")

    # ---- candidates, keyed by INSTANCE so an antigen is never taken from a sibling instance
    by_inst, cand = {}, collections.defaultdict(dict)
    for r in csv.DictReader(open(a.splits)):
        c = str(r["ab_ag_cluster"])
        if c not in first_inst:
            continue
        if not any(t in POLY for t in str(r["agtypes"]).split("/")):
            continue
        n, ag = poly_parts(r)
        rec = (n, ag, r["INSTANCE"], (r["PDBdepo"] or "")[:10], r["PDB_ID"][-4:].lower())
        by_inst[r["INSTANCE"]] = rec
        if (r["PDBdepo"] or "")[:10] > CUT and n >= a.min_tokens:
            # one entry per PDB: a second structure, not a second instance of the same crystal
            cand[c].setdefault(rec[4], rec)

    gib = lambda t: t * t * 512 / 1024 ** 3 * a.seeds * 1.04
    secs = lambda n: 0.00182 * n ** 1.57 * a.seeds

    out, tot_gib, tot_s, nofirst = [], 0.0, 0.0 , []
    for c in sorted(first_inst, key=int):
        fi = first_inst[c]
        if fi not in by_inst:
            nofirst.append(c)
            continue
        f_n, f_ag, _, _, f_pdb = by_inst[fi]
        others = {p: v for p, v in cand[c].items() if p != f_pdb}
        if not others:
            continue
        p, (n, ag, inst, dep, _) = sorted(others.items(),
                                          key=lambda kv: (bool(kv[1][1] & f_ag), kv[1][0]))[0]
        out.append(dict(cluster=c, instance=inst, pdb=p, rank="second", tokens=n, deposited=dep,
                        diff_antigen="yes" if not (ag & f_ag) else "no"))
        tot_gib += gib(n)
        tot_s += secs(n)
    if nofirst:
        raise SystemExit(f"STOP: {len(nofirst)} frozen first instance(s) are not in {a.splits} or "
                         f"carry no polypeptide antigen: clusters {nofirst[:5]}")

    nd = sum(1 for r in out if r["diff_antigen"] == "yes")
    print(f"{len(out)} of {len(first_inst)} clusters have a second eligible structure")
    print(f"   {nd} get a DIFFERENT antigen, {len(out)-nd} only the same one")
    print(f"   cache at {a.seeds} seeds: {tot_gib:.0f} GiB")
    print(f"   fold time: {tot_s/3600:.0f} GPU-h -> {tot_s/3/3600:.1f} h on 3 cards")
    for r in out:
        fp = by_inst[first_inst[r["cluster"]]][4]
        print(f"   cluster {r['cluster']:>5}  first {fp}  second {r['pdb']}  {r['tokens']:4d} tok  "
              f"diff-antigen {r['diff_antigen']}")

    if a.out:
        # EVERY instance, first and second. phase G --instances rebuilds pool.csv for the whole job
        # set; passing only the seconds would strip the 40 folded structures of their interfaces.
        allrows = [dict(cluster=c, instance=first_inst[c], pdb=by_inst[first_inst[c]][4],
                        rank="first", tokens=by_inst[first_inst[c]][0], deposited="",
                        diff_antigen="") for c in sorted(first_inst, key=int)]
        allrows += out
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cluster", "instance", "pdb", "rank", "tokens",
                                               "deposited", "diff_antigen"], lineterminator="\n")
            w.writeheader()
            w.writerows(allrows)
        print(f"\nwrote {a.out}: {len(first_inst)} first + {len(out)} second = {len(allrows)}")


if __name__ == "__main__":
    main()
