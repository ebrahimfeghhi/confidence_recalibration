#!/usr/bin/env python
"""Give pxm2025 the same two-directory shape FoldBench has, and retire its mask-OFF arms.

    ./organize_pxm.py [--apply]

  free_compute_matched/ -> free/        one pool: 300000-300049 AND the matched extras 300050-300149
  self_template_fixed_50_seed/          -> archive   MASK OFF
  self_template_fixed_50_seed_old_crystal_iptm/ -> archive   MASK OFF, already superseded
  self_template_variable_50_seed/       -> archive   MASK OFF

NO free_other_seeds FOR PXM. Unlike FoldBench, pxm's free/ holds nothing but the two blocks the
experiment uses -- 6,367 pool pose dirs and 4,664 extras, no stray leftovers -- so there is nothing
to park. Verified, not assumed.

THE TEMPLATE NAMESPACE DOES NOT GO TO THE ARCHIVE. `self_template_fixed_50_seed/templates/` (3,524
jobs) is the LIVE input for the mask-ON arm: pxm2025/selftmpl_mask_on_nc10/launch.sh and
selftmpl_mask_on_9g9p/launch.sh both pass it as --templates, and that arm has 2,937 folds queued and
0 done. Archiving the directory wholesale would break the one arm we are keeping. So the namespace is
promoted to `pxm2025/templates/` -- an input beside jsons/ and cif/, which is what it always was --
and both launch scripts are repointed.
"""
import argparse, csv, glob, os, shutil

P = "/home/ebrahim/tts_abag/experimental_data/protenix/pxm2025"
ARCHIVE = "/home/ebrahim/tts_abag/experimental_data/archive"
MASK_OFF = ["self_template_fixed_50_seed", "self_template_fixed_50_seed_old_crystal_iptm",
            "self_template_variable_50_seed"]
CONSUMERS = [f"{P}/selftmpl_mask_on_nc10/launch.sh",
             "/home/ebrahim/tts_abag/experimental_data/selftmpl_mask_on_9g9p/launch.sh"]


def dirs(root, sub=""):
    base = f"{root}/{sub}" if sub else root
    return [d for d in glob.glob(f"{base}/*/*") if os.path.isdir(d)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    src, dst = f"{P}/free_compute_matched", f"{P}/free"
    sd, dd = dirs(src), dirs(dst)
    sk = {(d.split("/")[-2], d.split("/")[-1]) for d in sd}
    dk = {(d.split("/")[-2], d.split("/")[-1]) for d in dd}
    ss = {(p, n.split("-")[0]) for p, n in sk}; ds_ = {(p, n.split("-")[0]) for p, n in dk}
    assert not sk & dk and not ss & ds_, "collision between free and free_compute_matched"
    rows = list(csv.DictReader(open(f"{src}/results.csv")))
    print(f"merge: {len(sd)} pose dirs, {len(rows)} rows -> free/")
    print(f"archive mask-OFF: {', '.join(MASK_OFF)}")
    print(f"templates promoted to {P}/templates (consumers repointed: {len(CONSUMERS)})")
    if not A.apply:
        print("\ndry run -- pass --apply"); return

    for d in sd:
        p, n = d.split("/")[-2], d.split("/")[-1]
        os.makedirs(f"{dst}/{p}", exist_ok=True)
        if not os.path.exists(f"{dst}/{p}/{n}"): shutil.move(d, f"{dst}/{p}/{n}")
    cols = csv.DictReader(open(f"{dst}/results.csv")).fieldnames
    with open(f"{dst}/results.csv", "a", newline="") as fh:
        csv.DictWriter(fh, fieldnames=cols).writerows(rows)
    for c in glob.glob(f"{src}/claims/*"):
        os.makedirs(f"{dst}/claims", exist_ok=True)
        t = f"{dst}/claims/{os.path.basename(c)}"
        if not os.path.exists(t): shutil.move(c, t)
    for f in glob.glob(f"{src}/queue*.tsv"):
        shutil.copy2(f, f"{dst}/{os.path.basename(f)}")
    shutil.move(src, f"{ARCHIVE}/pxm2025_free_compute_matched_merged_2026-10-06")

    # templates out of the mask-off arm BEFORE it is archived
    tsrc = f"{P}/self_template_fixed_50_seed/templates"
    if os.path.isdir(tsrc) and not os.path.exists(f"{P}/templates"):
        shutil.move(tsrc, f"{P}/templates")
        print(f"templates -> {P}/templates")
    for c in CONSUMERS:
        if not os.path.exists(c): continue
        s = open(c).read()
        s = s.replace("$FIXED/templates", f"{P}/templates")
        if not os.path.exists(c + ".premask.bak"): shutil.copy2(c, c + ".premask.bak")
        open(c, "w").write(s)
    for a in MASK_OFF:
        s = f"{P}/{a}"
        if os.path.isdir(s):
            shutil.move(s, f"{ARCHIVE}/pxm2025_maskoff_{a}")
    print("done")


if __name__ == "__main__":
    main()
