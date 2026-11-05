#!/usr/bin/env python
"""Write the fold job json for each Wave 1 structure, in the format the existing 501 use.

WHY THIS IS NEEDED AT ALL. The fold pipeline consumes a job json per structure -- `name` plus a
`sequences` list carrying each chain's sequence and the absolute paths to its alignments. None of
the 386 Wave 1 structures has one; training/jobs holds 502 and validation/jobs 54, all of them for
structures already folded. This is the only thing standing between the queue and the GPUs.

THE FORMAT IS COPIED, NOT INVENTED. Measured across all 501 existing training job jsons: a
one-element list, keys `name` and `sequences`, one entry per chain under `proteinChain` with
exactly {sequence, count, pairedMsaPath, unpairedMsaPath, templatesPath}, and `count` is 1 on all
1,535 of them -- a repeated sequence gets a second entry rather than count 2. Deviating would make
these structures different inputs from the existing set, which is the one thing this whole
expansion must not do.

THE INSTANCE, NOT THE PDB. A pdb carries one row per antibody/antigen chain pairing and they do
not all pass the gates: 7pa7 has 10 instances, 4 with a protein antigen and the 6 cheapest
ion-only. The queue records which instance it chose and this reads that, because re-deriving "the
row for this pdb" is how the preflight produced 21 false failures.

TEMPLATES ARE REFERENCED, NOT CREATED. templatesPath points at <msa_dir>/hmmsearch.a3m whether or
not it exists yet; phase_h_search_templates.py is what creates it and rewrites these files. 734 of
the 1,010 Wave 1 chains already have one, because MSA directories are keyed by SEQUENCE and those
chains are shared with already-folded structures. The other 276 are sequences new to the pipeline.
Protenix will NOT search for them at fold time -- runner/template_search.py is called from
batch_inference.py only, and our folds go through runner/inference.py -- so phase_h must run first
or those structures fold template-free and stop being comparable to the existing 500.

Usage:
  python -I run_logging/build_wave1_jobs.py --queue training/full_train/wave1_queue.json \
      --jobs-dir training/jobs --dry-run
  python -I run_logging/build_wave1_jobs.py --pdb-list <file> --jobs-dir validation/jobs
"""
import argparse, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
_src = open(os.path.join(ROOT, "phases", "phase_j0_select_structures.py")).read()
_g = {"__file__": os.path.join(ROOT, "phases", "phase_j0_select_structures.py"), "__name__": "nm"}
exec(compile(_src.split("def main(")[0], "j0", "exec"), _g)
chain_seqs, eligible, job_tokens = _g["chain_seqs"], _g["eligible"], _g["job_tokens"]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="")
ap.add_argument("--pdb-list", default="")
ap.add_argument("--jobs-dir", required=True)
ap.add_argument("--msa-index", default="building_msa/work/pdb_seqs/seq_to_pdb_index.json")
ap.add_argument("--msa-root", default="building_msa/work/mmcif_msa")
ap.add_argument("--suffix", default="-final-updated.json")
ap.add_argument("--dry-run", action="store_true")
ap.add_argument("--overwrite", action="store_true", help="rewrite a json that already exists")
a = ap.parse_args()

if a.queue:
    q = json.load(open(os.path.join(ROOT, a.queue)))["queue"]
    want = [(x["pdb"], x.get("instance")) for x in q]
elif a.pdb_list:
    want = [(p, None) for p in open(a.pdb_list).read().split()]
else:
    sys.exit("STOP: pass --queue or --pdb-list")

idx = json.load(open(os.path.join(ROOT, a.msa_index)))
MR = os.path.abspath(os.path.join(ROOT, a.msa_root))
byinst, bypdb = {}, {}
for r in csv.DictReader(open(os.path.join(ROOT, "SAbDab2_data/zenodo_splits/splits_final/abag_split.csv"))):
    byinst[r["INSTANCE"]] = r
    bypdb.setdefault(r["PDB_ID"][-4:].lower(), []).append(r)

JD = os.path.join(ROOT, a.jobs_dir)
written = skipped = failed = 0
problems = []
for pdb, inst in want:
    out = os.path.join(JD, f"{pdb}{a.suffix}")
    if os.path.exists(out) and not a.overwrite:
        skipped += 1
        continue
    if inst:
        r = byinst.get(inst)
        if r is None:
            problems.append((pdb, f"instance {inst} absent from abag_split")); failed += 1; continue
    else:
        # CHEAPEST INSTANCE THAT IS ELIGIBLE *AND* FULLY COVERED BY MSAs. Eligibility alone is not
        # enough: 8weo's cheapest eligible instance carries a chain absent from the MSA index
        # while another of its instances is fully covered, so selecting on tokens-among-eligible
        # rejected a structure that is perfectly foldable. Same class of error as deriving "the
        # row for this pdb" by token count in the preflight.
        rs = bypdb.get(pdb) or []
        elig = [x for x in rs if eligible(x)[0]]
        if not elig:
            problems.append((pdb, "no eligible instance")); failed += 1; continue

        def covered(x):
            ss_ = chain_seqs(x)
            if not ss_:
                return False
            for q in ss_:
                d_ = idx.get(q)
                if not d_:
                    return False
                for n_ in ("pairing.a3m", "non_pairing.a3m"):
                    try:
                        if os.path.getsize(os.path.join(MR, d_, n_)) <= 0:
                            return False
                    except OSError:
                        return False
            return True

        full = [x for x in elig if covered(x)]
        if not full:
            problems.append((pdb, "no eligible instance with complete MSAs")); failed += 1; continue
        r = min(full, key=job_tokens)
    ss = chain_seqs(r)
    if not ss:
        problems.append((pdb, "no chain sequences")); failed += 1; continue
    seqs, bad = [], []
    for s in ss:
        d = idx.get(s)
        if not d:
            bad.append("seq-not-in-msa-index"); continue
        base = os.path.join(MR, d)
        for n in ("pairing.a3m", "non_pairing.a3m"):
            p = os.path.join(base, n)
            try:
                if os.path.getsize(p) <= 0:
                    bad.append(f"{n}-empty")
            except OSError:
                bad.append(f"{n}-missing")
        seqs.append({"proteinChain": {
            "sequence": s,
            "count": 1,
            "pairedMsaPath": os.path.join(base, "pairing.a3m"),
            "unpairedMsaPath": os.path.join(base, "non_pairing.a3m"),
            "templatesPath": os.path.join(base, "hmmsearch.a3m"),
        }})
    if bad:
        problems.append((pdb, ",".join(sorted(set(bad))))); failed += 1; continue
    doc = [{"name": pdb, "sequences": seqs}]
    if a.dry_run:
        written += 1
        continue
    os.makedirs(JD, exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1)
    os.replace(tmp, out)
    written += 1

print(f"{'DRY RUN -- ' if a.dry_run else ''}jobs dir {a.jobs_dir}")
print(f"  written {written}   already present (skipped) {skipped}   failed {failed}")
for p, why in problems[:12]:
    print(f"    {p}: {why}")
if len(problems) > 12:
    print(f"    ... and {len(problems)-12} more")
if not a.dry_run and written:
    nt = 0
    for pdb, _ in want:
        out = os.path.join(JD, f"{pdb}{a.suffix}")
        if not os.path.exists(out):
            continue
        for sq in json.load(open(out))[0]["sequences"]:
            if not os.path.exists(sq["proteinChain"]["templatesPath"]):
                nt += 1
    print(f"\n  chains still lacking hmmsearch.a3m: {nt}")
    print("  run phases/phase_h_search_templates.py over this jobs dir BEFORE folding, "
          "single-process -- MSA dirs are shared by sequence and two workers would race.")
sys.exit(1 if failed else 0)
