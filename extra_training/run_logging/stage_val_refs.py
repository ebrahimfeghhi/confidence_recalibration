#!/usr/bin/env python
"""Stage the DockQ reference for each new validation structure, the way phase_g does.

paths.py routes the scorer to validation/refs/<pdb>.cif, and phase_g_build_val_jobs stages the
chosen instance's cif there. This does the same for the Wave 1 validation structures, which were
selected outside phase_g and so never had a reference staged -- 0 of 30 existed.

THE RELABELLED COPY WINS WHERE THERE IS ONE. build_wave1_pool runs relabel_unique when an auth
chain's reference labels are not unique and writes <refs-dir>/<pdb>.relabelled.cif; the pool rows
then name the RELABELLED labels. Staging the raw instance cif for such a structure would name
chains the reference does not have, which is the failure phase_g records on 8evg -- its corrected
reference was written and then a stale copy with the degenerate labels was kept, so all four of
its interfaces named chains the reference lacked. 8z7j is that case here.

VERIFIED, NOT ASSUMED: every pool row's chain1 and chain2 must appear as a label_asym_id in the
reference actually staged, or the structure is reported and nothing is staged for it.
"""
import argparse, csv, hashlib, math, os, shutil, sys
import gemmi

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="validation/full_val/pool_val30.csv")
ap.add_argument("--selection", default="validation/full_val/val30_structures.csv")
ap.add_argument("--instance-cifs", default="SAbDab2_data/training_cifs")
ap.add_argument("--relabelled", default="validation/fold/refs")
ap.add_argument("--out", default="validation/refs")
ap.add_argument("--apply", action="store_true", help="without this, report and change nothing")
ap.add_argument("--csv", default=None,
                help="results csv. A structure with recorded DockQ rows is NOT re-staged -- "
                     "replacing the reference those numbers were measured against would silently "
                     "change what they mean -- and the difference is reported instead. A structure "
                     "with no rows is always re-staged, because nothing depends on the old file.")
a = ap.parse_args()

inst = {r["pdb"]: r["instance"] for r in csv.DictReader(open(a.selection))}
pool = {}
for r in csv.DictReader(open(a.pool)):
    pool.setdefault(r["pdb"], set()).update((r["chain1"], r["chain2"]))


def labels(cif):
    st = gemmi.read_structure(cif)
    st.setup_entities()
    out = set()
    for ch in st[0]:
        for sub in ch.subchains():
            if len([x for x in sub]) >= 5:
                out.add(sub.subchain_id())
    return out


# THE POOL IS SHARED, THE SELECTION IS NOT. training/fold/pool.csv covers all 856 training
# structures; only the selected ones are being staged here, so a pdb outside the selection is
# skipped rather than treated as a missing instance.
skipped = sorted(set(pool) - set(inst))
if skipped:
    print(f"  {len(skipped)} pooled structure(s) outside the selection, not touched")
plan, problems = [], []
for pdb in sorted(set(pool) & set(inst)):
    rel = os.path.join(a.relabelled, f"{pdb}.relabelled.cif")
    raw = os.path.join(a.instance_cifs, f"{inst[pdb]}.cif")
    src = rel if os.path.exists(rel) else raw
    if not os.path.exists(src):
        problems.append((pdb, f"no source cif at {src}")); continue
    have = labels(src)
    missing = sorted(pool[pdb] - have)
    if missing:
        problems.append((pdb, f"pool names chain(s) {missing} that {os.path.basename(src)} "
                              f"does not have (it has {sorted(have)})")); continue
    plan.append((pdb, src, "relabelled" if src == rel else "instance"))

print(f"{len(set(pool) & set(inst))} selected structure(s) in the pool; {len(plan)} verified, "
      f"{len(problems)} problem(s)")
for pdb, src, kind in plan:
    if kind == "relabelled":
        print(f"  {pdb}: staging the RELABELLED reference")
for pdb, why in problems:
    print(f"  PROBLEM {pdb}: {why}")
if problems:
    raise SystemExit("STOP: nothing staged. Every pool chain must exist in the reference.")

if not a.apply:
    print("report only; pass --apply to stage")
    sys.exit(0)

os.makedirs(a.out, exist_ok=True)
scored = set()
if a.csv and os.path.exists(a.csv):
    for r in csv.DictReader(open(a.csv)):
        try:
            dq, ip = float(r["dockq"]), float(r["iptm"])
        except (TypeError, ValueError, KeyError):
            continue
        if math.isfinite(dq) and math.isfinite(ip):
            scored.add(r["iface"].split("_")[0])
print(f"  structures with recorded DockQ rows: {len(scored & set(inst))} of {len(inst)}")

n = kept = 0
for pdb, src, _kind in plan:
    dst = os.path.join(a.out, f"{pdb}.cif")
    # THE GUARD IS RECORDED SCORES, NOT A FILE ON DISK. The first version skipped any pdb whose
    # reference already existed, assuming an existing reference had produced data. 9ofp disproved
    # it: its staged reference came from the OTHER instance of the same entry
    # (pdb_00009ofp_H_L, labels A/D/E) while the pool names B/F/G from pdb_00009ofp_M_N -- and it
    # had 0 trunks, 0 poses and 0 rows. Skipping it would have left a reference that cannot match
    # the pool, so the structure folds and scores nothing.
    if pdb in scored and os.path.exists(dst):
        if hashlib.sha256(open(dst, "rb").read()).hexdigest() != \
           hashlib.sha256(open(src, "rb").read()).hexdigest():
            print(f"  LEFT IN PLACE, AND IT DIFFERS: {pdb} has recorded DockQ rows measured "
                  f"against the current reference, which is not {os.path.basename(src)}. "
                  f"Replacing it would change what those rows mean; decide deliberately.")
        kept += 1
        continue
    # ALWAYS RE-STAGE, as phase_g does: a file left by an earlier run must not survive
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    shutil.copy2(src, dst)
    os.chmod(dst, 0o664)
    n += 1
print(f"staged {n} reference(s) into {a.out}"
      + (f"; left {kept} with recorded scores in place" if kept else ""))

# VERIFY WHAT IS ON DISK, not what was intended. Every check above ran against the SOURCE file,
# while the thing the scorer opens is <out>/<pdb>.cif -- and 9ofp is exactly the case where those
# two differed. A reference that does not carry the pool's chains means the fold scores nothing.
bad = []
sel = sorted(set(pool) & set(inst))
for pdb in sel:
    dst = os.path.join(a.out, f"{pdb}.cif")
    if not os.path.exists(dst):
        bad.append((pdb, "no reference on disk", [])); continue
    have = labels(dst)
    miss = sorted(pool[pdb] - have)
    if miss:
        bad.append((pdb, f"pool names {miss}", sorted(have)))
print(f"ON DISK: {len(sel) - len(bad)} of {len(sel)} reference(s) carry every pool chain")
for pdb, why, have in bad[:10]:
    print(f"  MISMATCH {pdb}: {why}; reference has {have}")
if bad:
    raise SystemExit(f"STOP: {len(bad)} staged reference(s) cannot match the pool; those "
                     f"structures would fold and score nothing.")
