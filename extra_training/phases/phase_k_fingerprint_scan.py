"""PRE-LAUNCH GATE: scan every cached label's PER-STRUCTURE provenance before training starts.

WHY THIS EXISTS. phase_k_train verifies a label's fingerprint when it REACHES that unit, which is
correct but late: run iabg5i trained for 178 minutes and 65 steps, then died on 8gpu because that
structure's label carried an `idx` hash from a different job_indices.csv. One structure of 491, and
it cost three hours and an epoch-1 validation number we never got. Run this first and the same fault
surfaces in two minutes.

Exit code is 1 if any label differs in a field the trainer will NOT tolerate, so a launch script can
gate on it. `code` alone is tolerated, because --allow-code-drift exists for exactly that and the
data provenance is intact; anything else is a real mismatch between a label and its inputs.

Reuses phase_j_build_cache's own fingerprint_context and helpers rather than reimplementing them --
a reimplementation that drifted would either reject every label or accept every label, which is the
exact failure review round 8 found. Skips the per-unit `trunk=` field: trunks are 340 MB each and
1.4 TB in total, so hashing them is hours of I/O, and the trainer verifies that field per unit
anyway. The field that killed iabg5i (`idx`) is per-structure and is covered here.
"""
import csv, os, sys, types
from collections import defaultdict
import numpy as np

ROOT = "/home/ebrahim/confidence_recalibration/extra_training"
sys.path.insert(0, os.path.join(ROOT, "phases"))
sys.path.insert(0, "/home/ebrahim/Protenix")
import phase_j_build_cache as BC
import phase_j_dataset as DS

import argparse
_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("--manifest-skip",
                 default="9mx1,9q5m,9dzf,9b9j,9e7x,9q5v,9q7b,8vba,9bbn,9dc3",
                 help="the same exclusions the launch passes to phase_k_train")
_ap.add_argument("--seeds", default="500000-500009",
                 help="ignored when --seed-manifest is given")
# THE INDICES FILE IS PART OF THE FINGERPRINT, so scanning with a different one than the labels
# were built with reports every label as stale. These were hardcoded to job_indices.csv while the
# 853-structure population's labels are built against all857_indices.csv, and the scan then
# declared 2,010 labels over 223 structures mismatched on `bio`+`cif` -- with `code` and `loss`
# matching exactly, which is the tell: the recipe agreed and only the per-structure inputs differed.
# Nothing was wrong with the labels; the gate was comparing them against the wrong context.
_ap.add_argument("--indices", default="training/prep/job_indices.csv")
_ap.add_argument("--bio-dir", default="training/prep/job_bioassembly")
_ap.add_argument("--mmcif-dir", default="training/prep/job_cifs")
_ap.add_argument("--selection", default="train_val_splits/cache_train_structures.csv",
                 help="which structures to scan")
# AND THE SEED RANGE IS PER STRUCTURE NOW. A flat lo-hi scans seeds a shallow structure does not
# have and never trains on: of the 223 structures reported above, 220 are declared at ONE seed, so
# 9 of every 10 labels the scan judged are units the trainer will never read.
_ap.add_argument("--seed-manifest", default="",
                 help="per-structure seed declaration; overrides --seeds")
# THE LABEL ROOT WAS HARDCODED TO training/fold/labels, which made this gate silently wrong for
# any fold but the training one: given a validation --selection it looked each val structure up
# under the TRAINING labels, found nothing there, and reported every one of them as "missing"
# -- a message that reads like absent labels rather than "you pointed the scan at the wrong
# tree". phase_k_train takes --fold and derives labels from it, so this takes the same shape.
_ap.add_argument("--fold", default="training/fold",
                 help="fold whose labels/ subtree is scanned; must be the SAME --fold the run "
                      "will train with, since the stored fingerprint is per-label-file")
_ap.add_argument("--labels", default="",
                 help="label root, overriding <fold>/labels")
_a = _ap.parse_args()
SKIP = {x.strip().lower() for x in _a.manifest_skip.split(",") if x.strip()}
_lo, _hi = (int(x) for x in _a.seeds.split("-"))
a = types.SimpleNamespace(
    indices=os.path.join(ROOT, _a.indices),
    bio_dir=os.path.join(ROOT, _a.bio_dir),
    mmcif_dir=os.path.join(ROOT, _a.mmcif_dir))
LABEL_ROOT = os.path.join(ROOT, _a.labels or os.path.join(_a.fold, "labels"))
if not os.path.isdir(LABEL_ROOT):
    raise SystemExit(f"STOP: label root {LABEL_ROOT} is not a directory; every label would be "
                     f"reported 'missing' for a reason that has nothing to do with the labels.")
print(f"indices {_a.indices}", flush=True)
print(f"labels  {os.path.relpath(LABEL_ROOT, ROOT)}", flush=True)

cfg = DS.build_configs(3400, a.indices, a.bio_dir, a.mmcif_dir,
                       os.path.join(ROOT, "building_msa/work/mmcif_msa"),
                       os.path.join(ROOT, "building_msa/work/pdb_seqs/seq_to_pdb_index.json"),
                       os.path.join(ROOT, "building_msa/work/mmcif_msa"))
cfg.train_confidence_only = True
code_fp, loss_fp = BC.fingerprint_context(cfg, verbose=False)
print(f"current context: code={code_fp} loss={loss_fp}", flush=True)

idx_by_pdb = BC._indices_by_pdb(a.indices)
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, _a.selection)))
        if r["tokens"].isdigit() and r["pdb"] not in SKIP]
pdbs = [r["pdb"] for r in rows]

def _parse_seeds(txt):
    out = set()
    for part in str(txt).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            _a1, _b1 = part.split("-", 1)
            out.update(range(int(_a1), int(_b1) + 1))
        else:
            out.add(int(part))
    return out

SEEDS_OF = {}
if _a.seed_manifest:
    for r in csv.DictReader(open(os.path.join(ROOT, _a.seed_manifest))):
        t = str(r.get("seeds", "") or "").strip()
        if t:
            SEEDS_OF[r["pdb"].strip().lower()] = sorted(_parse_seeds(t))
    miss = [p for p in pdbs if p.lower() not in SEEDS_OF]
    if miss:
        sys.exit(f"STOP: {len(miss)} scanned structure(s) are not declared in --seed-manifest, "
                 f"e.g. {miss[:5]}")
    n = sum(len(SEEDS_OF[p.lower()]) for p in pdbs)
    print(f"scanning {len(pdbs)} structures, per-structure seeds = {n} labels", flush=True)
else:
    print(f"scanning {len(pdbs)} structures x {_hi-_lo+1} seeds = "
          f"{len(pdbs)*(_hi-_lo+1)} labels", flush=True)

def want_fields(pdb):
    return {"schema": str(BC.SCHEMA), "code": code_fp, "loss": loss_fp,
            "cif": BC.sha(os.path.join(a.mmcif_dir, f"{pdb}.cif")),
            "bio": BC.sha(os.path.join(a.bio_dir, f"{pdb}.pkl.gz")),
            "idx": idx_by_pdb.get(pdb, "absent")}

bad = defaultdict(list)
missing, checked = [], 0
for i, pdb in enumerate(pdbs):
    w = want_fields(pdb)
    for seed in (SEEDS_OF[pdb.lower()] if SEEDS_OF else range(_lo, _hi + 1)):
        lp = os.path.join(LABEL_ROOT, pdb, f"free_{seed}.npz")
        if not os.path.exists(lp):
            missing.append(f"{pdb}/{seed}"); continue
        try:
            stored = str(np.asarray(np.load(lp, allow_pickle=True)["fingerprint"]))
        except Exception as e:
            bad["UNREADABLE"].append(f"{pdb}/{seed} [{type(e).__name__}]"); continue
        got = dict(kv.split("=", 1) for kv in stored.split("|") if "=" in kv)
        diff = sorted(k for k, v in w.items() if got.get(k) != v)
        checked += 1
        if diff:
            bad["+".join(diff)].append(f"{pdb}/{seed}")
    if (i + 1) % 100 == 0:
        print(f"  ...{i+1}/{len(pdbs)} structures", flush=True)

print(f"\nchecked {checked} labels, {len(missing)} missing")
print("MISMATCHES BY FIELD SET (the trainer tolerates 'code' alone, nothing else):")
for k in sorted(bad, key=lambda k: -len(bad[k])):
    structs = sorted({x.split("/")[0] for x in bad[k]})
    tol = "  <- TOLERATED by --allow-code-drift" if k == "code" else "  <- WILL KILL THE RUN"
    print(f"  {k:24s} {len(bad[k]):5d} labels, {len(structs):4d} structures{tol}")
    if k != "code":
        print(f"      structures: {' '.join(structs)}")
if missing:
    print(f"\nmissing labels: {missing[:10]}")

# A MISSING LABEL IS A FAILURE, NOT A FOOTNOTE. The gate used to print the missing list and then
# decide only on the mismatches, so a fixture with no label file at all exited 0 saying "GATE
# PASSED" (review 2026-10-16). The trainer does catch it later, but the whole point of a preflight
# is that its assurance means something.
if missing:
    print(f"\nGATE FAILED: {len(missing)} label(s) do not exist, e.g. {missing[:6]}")
    print("The manifest is the frozen population: a label that was never written is an error, not "
          "a smaller training set. Build them with phases/phase_j_build_cache.py.")
    sys.exit(1)
fatal = {k: v for k, v in bad.items() if k != "code"}
if fatal:
    print(f"\nGATE FAILED: {sum(len(v) for v in fatal.values())} label(s) the trainer will refuse.")
    print("Rebuild each named structure with:  phases/phase_j_build_cache.py --only <pdb> --overwrite")
    sys.exit(1)
print("\nGATE PASSED: every label's inputs match, or differ only in `code` (--allow-code-drift).")
sys.exit(0)
