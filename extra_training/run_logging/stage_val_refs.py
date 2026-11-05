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
import argparse, csv, os, shutil, sys
import gemmi

ap = argparse.ArgumentParser()
ap.add_argument("--pool", default="validation/full_val/pool_val30.csv")
ap.add_argument("--selection", default="validation/full_val/val30_structures.csv")
ap.add_argument("--instance-cifs", default="SAbDab2_data/training_cifs")
ap.add_argument("--relabelled", default="validation/fold/refs")
ap.add_argument("--out", default="validation/refs")
ap.add_argument("--apply", action="store_true", help="without this, report and change nothing")
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


plan, problems = [], []
for pdb in sorted(pool):
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

print(f"{len(pool)} structure(s) in the pool; {len(plan)} reference(s) verified, "
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
n = 0
for pdb, src, _kind in plan:
    dst = os.path.join(a.out, f"{pdb}.cif")
    # ALWAYS RE-STAGE, as phase_g does: a file left by an earlier run must not survive
    if os.path.exists(dst) or os.path.islink(dst):
        os.remove(dst)
    shutil.copy2(src, dst)
    os.chmod(dst, 0o664)
    n += 1
print(f"staged {n} reference(s) into {a.out}")
