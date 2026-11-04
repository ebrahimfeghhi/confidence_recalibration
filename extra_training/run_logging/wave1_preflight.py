#!/usr/bin/env python
"""EXACT input preflight for every structure in the Wave 1 queues. Read-only; folds nothing.

WHY A SEPARATE GATE. build_wave1_queue.py decides which structures to fold from the split table.
That is a different question from whether the folding pipeline can actually consume them, and the
difference is 36 GPU-hours. This checks the real files the real pipeline opens, per structure, and
refuses to report a queue as launchable while anything is missing.

WHAT IT CHECKS, and which stage each one gates:

  FOLD STAGE -- needed before a single unit can be folded
    1. job json          training/jobs/<pdb>-final-updated.json, parseable, with `sequences`
    2. MSA files         for EVERY chain sequence: pairing.a3m AND non_pairing.a3m, non-empty,
                         resolved through seq_to_pdb_index.json, not by guessing a directory name
    3. token budget      job_tokens <= 2560, Protenix v2's inference ceiling (runner/inference.py)
    4. chain mapping     an antibody chain plus >=1 PROTEIN/PEPTIDE antigen, and the agtypes /
                         agexpectedseqs lists the same length -- a mismatch means the row's own
                         antigen record is inconsistent

  LABEL STAGE -- needed before DockQ exists, so before the pilot can say anything
    5. mmcif             <mmcif-dir>/<pdb>.cif
    6. bioassembly       training/prep/job_bioassembly/<pdb>.pkl.gz

Separating the two matters: folding needs nothing from the label stage, so a missing mmcif must
not hold up the queue. It must, however, be fixed before the pilot report can be produced.

Exit 0 only when every FOLD-stage check passes for every structure.

Usage:
  python -I run_logging/wave1_preflight.py --queue training/full_train/wave1_queue.json
  python -I run_logging/wave1_preflight.py --pdb-list <file> --fold-root validation/fold \
      --jobs-dir validation/jobs --bio-dir validation/prep/job_bioassembly
"""
import argparse, collections, csv, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "phases"))
_src = open(os.path.join(ROOT, "phases", "phase_j0_select_structures.py")).read()
_g = {"__file__": os.path.join(ROOT, "phases", "phase_j0_select_structures.py"), "__name__": "nm"}
exec(compile(_src.split("def main(")[0], "j0", "exec"), _g)
chain_seqs, eligible, job_tokens = _g["chain_seqs"], _g["eligible"], _g["job_tokens"]

ap = argparse.ArgumentParser()
ap.add_argument("--queue", default="")
ap.add_argument("--pdb-list", default="")
ap.add_argument("--jobs-dir", default="training/jobs")
ap.add_argument("--jobs-dir-corrected", default="training/jobs_corrected")
ap.add_argument("--bio-dir", default="training/prep/job_bioassembly")
ap.add_argument("--mmcif-dir", default="/home/ebrahim/mmcif")
ap.add_argument("--msa-index", default="building_msa/work/pdb_seqs/seq_to_pdb_index.json")
ap.add_argument("--msa-root", default="building_msa/work/mmcif_msa")
ap.add_argument("--max-tokens", type=int, default=2560)
ap.add_argument("--show", type=int, default=8, help="example failures to print per check")
a = ap.parse_args()

if a.queue:
    want = [x["pdb"] for x in json.load(open(os.path.join(ROOT, a.queue)))["queue"]]
elif a.pdb_list:
    want = open(a.pdb_list).read().split()
else:
    sys.exit("STOP: pass --queue or --pdb-list")

idx = json.load(open(os.path.join(ROOT, a.msa_index)))
MR = os.path.join(ROOT, a.msa_root)
rows = collections.defaultdict(list)
for r in csv.DictReader(open(os.path.join(ROOT, "SAbDab2_data/zenodo_splits/splits_final/abag_split.csv"))):
    rows[r["PDB_ID"][-4:].lower()].append(r)

jobs = {}
for d in (a.jobs_dir, a.jobs_dir_corrected):
    p = os.path.join(ROOT, d)
    if os.path.isdir(p):
        for f in os.listdir(p):
            jobs.setdefault(f.split("-")[0].lower(), os.path.join(p, f))


def nonempty(path):
    try:
        return os.path.getsize(path) > 0
    except OSError:
        return False


FOLD = ["job_json", "msa_files", "token_budget", "chain_mapping"]
LABEL = ["mmcif", "bioassembly"]
fail = collections.defaultdict(list)
for pdb in want:
    rs = rows.get(pdb) or []
    if not rs:
        for c in FOLD + LABEL:
            fail[c].append((pdb, "not in abag_split"))
        continue
    r = min(rs, key=job_tokens)

    jp = jobs.get(pdb)
    if not jp or not nonempty(jp):
        fail["job_json"].append((pdb, "absent" if not jp else "empty"))
    else:
        try:
            j = json.load(open(jp))
            j0 = j[0] if isinstance(j, list) else j
            if not j0.get("sequences"):
                fail["job_json"].append((pdb, "no `sequences` key"))
        except (ValueError, OSError) as e:
            fail["job_json"].append((pdb, f"unparseable: {type(e).__name__}"))

    ss = chain_seqs(r)
    if not ss:
        fail["msa_files"].append((pdb, "no chain sequences"))
    else:
        bad = []
        for s in ss:
            if s not in idx:
                bad.append("seq-not-in-index"); continue
            d = os.path.join(MR, idx[s])
            if not os.path.isdir(d):
                bad.append("msa-dir-missing"); continue
            for n in ("pairing.a3m", "non_pairing.a3m"):
                if not nonempty(os.path.join(d, n)):
                    bad.append(f"{n}-missing-or-empty")
        if bad:
            fail["msa_files"].append((pdb, ",".join(sorted(set(bad)))))

    t = job_tokens(r)
    if t > a.max_tokens:
        fail["token_budget"].append((pdb, f"{t} > {a.max_tokens}"))

    ok, msg = eligible(r)
    if not ok:
        fail["chain_mapping"].append((pdb, msg))
    elif len(str(r["agtypes"]).split("/")) != len(str(r["agexpectedseqs"]).split("/")):
        fail["chain_mapping"].append((pdb, "agtypes/agexpectedseqs length mismatch"))

    if not nonempty(os.path.join(a.mmcif_dir, f"{pdb}.cif")):
        fail["mmcif"].append((pdb, "absent"))
    if not nonempty(os.path.join(ROOT, a.bio_dir, f"{pdb}.pkl.gz")):
        fail["bioassembly"].append((pdb, "absent"))

print(f"preflight over {len(want)} structure(s)\n")
print("FOLD STAGE -- blocks the queue")
for c in FOLD:
    n = len(fail[c])
    print(f"  {'PASS' if not n else 'FAIL'}  {c:<14} {len(want)-n}/{len(want)} ok")
    for pdb, why in fail[c][:a.show]:
        print(f"          {pdb}: {why}")
    if n > a.show:
        print(f"          ... and {n - a.show} more")
print("\nLABEL STAGE -- blocks DockQ and therefore the pilot report, NOT the fold")
for c in LABEL:
    n = len(fail[c])
    print(f"  {'PASS' if not n else 'FAIL'}  {c:<14} {len(want)-n}/{len(want)} ok")
    for pdb, why in fail[c][:a.show]:
        print(f"          {pdb}: {why}")
    if n > a.show:
        print(f"          ... and {n - a.show} more")

blocked = sorted({p for c in FOLD for p, _ in fail[c]})
print(f"\nfold-blocked structures: {len(blocked)} of {len(want)}")
print(f"label-blocked structures: {len({p for c in LABEL for p, _ in fail[c]})} of {len(want)}")
sys.exit(0 if not blocked else 1)
