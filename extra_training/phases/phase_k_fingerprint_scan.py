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
_ap.add_argument("--seeds", default="500000-500009")
_a = _ap.parse_args()
SKIP = {x.strip().lower() for x in _a.manifest_skip.split(",") if x.strip()}
_lo, _hi = (int(x) for x in _a.seeds.split("-"))
a = types.SimpleNamespace(
    indices=os.path.join(ROOT, "training/prep/job_indices.csv"),
    bio_dir=os.path.join(ROOT, "training/prep/job_bioassembly"),
    mmcif_dir=os.path.join(ROOT, "training/prep/job_cifs"))

cfg = DS.build_configs(3400, a.indices, a.bio_dir, a.mmcif_dir,
                       os.path.join(ROOT, "building_msa/work/mmcif_msa"),
                       os.path.join(ROOT, "building_msa/work/pdb_seqs/seq_to_pdb_index.json"),
                       os.path.join(ROOT, "building_msa/work/mmcif_msa"))
cfg.train_confidence_only = True
code_fp, loss_fp = BC.fingerprint_context(cfg, verbose=False)
print(f"current context: code={code_fp} loss={loss_fp}", flush=True)

idx_by_pdb = BC._indices_by_pdb(a.indices)
rows = [r for r in csv.DictReader(open(os.path.join(ROOT, "train_val_splits/cache_train_structures.csv")))
        if r["tokens"].isdigit() and r["pdb"] not in SKIP]
pdbs = [r["pdb"] for r in rows]
print(f"scanning {len(pdbs)} structures x {_hi-_lo+1} seeds = {len(pdbs)*(_hi-_lo+1)} labels", flush=True)

def want_fields(pdb):
    return {"schema": str(BC.SCHEMA), "code": code_fp, "loss": loss_fp,
            "cif": BC.sha(os.path.join(a.mmcif_dir, f"{pdb}.cif")),
            "bio": BC.sha(os.path.join(a.bio_dir, f"{pdb}.pkl.gz")),
            "idx": idx_by_pdb.get(pdb, "absent")}

bad = defaultdict(list)
missing, checked = [], 0
for i, pdb in enumerate(pdbs):
    w = want_fields(pdb)
    for seed in range(_lo, _hi + 1):
        lp = os.path.join(ROOT, "training/fold/labels", pdb, f"free_{seed}.npz")
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

fatal = {k: v for k, v in bad.items() if k != "code"}
if fatal:
    print(f"\nGATE FAILED: {sum(len(v) for v in fatal.values())} label(s) the trainer will refuse.")
    print("Rebuild each named structure with:  phases/phase_j_build_cache.py --only <pdb> --overwrite")
    sys.exit(1)
print("\nGATE PASSED: every label's inputs match, or differ only in `code` (--allow-code-drift).")
sys.exit(0)
