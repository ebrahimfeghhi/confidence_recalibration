#!/usr/bin/env python
"""Give pxm2025 the same layout FoldBench already has: one directory per benchmark.

    ./migrate_pxm_layout.py            # dry run: prints every move and every rewrite
    ./migrate_pxm_layout.py --apply

BEFORE, pxm2025's `data` WAS `experimental_data` itself, so its arms sat loose at the repo's data
root while FoldBench had `experimental_data/protenix/foldbench/`. Two pools were both called `free/`
at different depths, and one newer pxm arm had already landed under `protenix/pxm2025/`, so the
convention was split mid-benchmark.

AFTER:

    experimental_data/protenix/pxm2025/      inputs (jsons, cif, pool csv, chain maps) AND arms
    experimental_data/protenix/foldbench/    unchanged

WHAT IS NOT MOVED, deliberately: `analysis/`, `archive/` and `pose_dockq_matrices/` are shared across
benchmarks, and `redock_branch/`, `redock_probe/`, `cost_model/` and the `*_9g9p` directories are
one-off experiments whose own scripts carry absolute paths -- redock_branch is paused mid-run at
11/20 complexes and must stay resumable. Moving those buys nothing and risks a resume.

THE DANGEROUS PART is not the move, it is the references. A templated fold records its donor as an
absolute path, so `free/` cannot move without rewriting every queue and every `donor_pose` column
that points into it. This script rewrites exactly the prefixes it moves, in the queues, the results
csvs and the code, and verifies afterwards that a sample of donor paths still resolves on disk. It
refuses to touch `archive/`, whose whole purpose is to record where things used to be.
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess

ROOT = "/home/ebrahim/tts_abag"
DATA = f"{ROOT}/experimental_data"
DEST = f"{DATA}/protenix/pxm2025"

# directory name -> it moves to DEST/<same name>
ARMS = ["free", "free_compute_matched", "self_template_fixed_50_seed",
        "self_template_fixed_50_seed_old_crystal_iptm", "self_template_variable_50_seed",
        "seed_scatter_50"]
INPUTS = "pxm2025"          # its CONTENTS move to DEST/, mirroring foldbench's flat inputs


def moves():
    """out: [(src, dst)] every directory move, skipping anything already migrated"""
    out = []
    for a in ARMS:
        s = f"{DATA}/{a}"
        if os.path.isdir(s):
            out.append((s, f"{DEST}/{a}"))
    src = f"{DATA}/{INPUTS}"
    if os.path.isdir(src):
        for e in sorted(os.listdir(src)):
            out.append((f"{src}/{e}", f"{DEST}/{e}"))
    return out


def rewrites():
    """out: [(old, new)] the literal path prefixes to rewrite, longest first

    Longest first matters: `experimental_data/free` is a prefix of
    `experimental_data/free_compute_matched`, and rewriting the short one first would turn the long
    one into `protenix/pxm2025/free_compute_matched` only by luck of the trailing text.
    """
    pairs = [(f"{DATA}/{a}", f"{DEST}/{a}") for a in ARMS]
    pairs += [(f"{DATA}/{INPUTS}/", f"{DEST}/")]
    return sorted(pairs, key=lambda p: -len(p[0]))


def targets():
    """out: the files whose contents mention a moved path -- queues, result csvs, code. Never archive."""
    pats = [f"{DATA}/**/queue*.tsv", f"{DATA}/**/*.tsv", f"{DATA}/**/results*.csv",
            f"{ROOT}/*.py", f"{ROOT}/diffusion_commit_clean_v2/**/*.py",
            f"{ROOT}/diffusion_commit_clean_v2/**/*.sh", f"{DATA}/**/*.sh"]
    found = set()
    for p in pats:
        for f in glob.glob(p, recursive=True):
            if "/archive/" in f or not os.path.isfile(f):
                continue
            found.add(f)
    olds = [o for o, _ in rewrites()]
    hit = []
    for f in sorted(found):
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if any(o in s for o in olds):
            hit.append(f)
    return hit


def sample_donors(n=5):
    """out: a few absolute donor paths from the pxm arms, to check they resolve after the move"""
    out = []
    for c in glob.glob(f"{DEST}/*/results.csv") + glob.glob(f"{DATA}/*/results.csv"):
        if "/archive/" in c:
            continue
        try:
            for r in csv.DictReader(open(c)):
                d = r.get("donor_pose", "")
                if d and d != "-1" and "/free/" in d:
                    out.append(d.split()[0])
                    break
        except OSError:
            pass
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    A = ap.parse_args()
    mv = moves()
    files = targets()
    print(f"{len(mv)} directory moves into {DEST}")
    for s, d in mv:
        print(f"   {os.path.relpath(s, DATA)}  ->  {os.path.relpath(d, DATA)}")
    print(f"\n{len(files)} files mention a moved path and will be rewritten:")
    for f in files:
        print(f"   {os.path.relpath(f, ROOT)}")
    before = sample_donors()
    print(f"\ndonor paths sampled for the after-check: {len(before)}")
    if not A.apply:
        print("\ndry run -- pass --apply")
        return

    os.makedirs(DEST, exist_ok=True)
    for s, d in mv:
        if os.path.exists(d):
            print(f"   SKIP (exists): {d}")
            continue
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.move(s, d)
    # the inputs directory is left behind empty by the per-entry moves
    if os.path.isdir(f"{DATA}/{INPUTS}") and not os.listdir(f"{DATA}/{INPUTS}"):
        os.rmdir(f"{DATA}/{INPUTS}")

    # RECOMPUTE after the moves. The first scan listed these files at their OLD paths, and most of
    # them live INSIDE the directories just moved -- rewriting that stale list would open nothing.
    files = targets()
    print(f"\nafter the move, {len(files)} files mention a moved path")
    for f in files:
        s = open(f, encoding="utf-8", errors="ignore").read()
        for old, new in rewrites():
            s = s.replace(old, new)
        if not os.path.exists(f + ".premigration.bak"):
            shutil.copy2(f, f + ".premigration.bak")
        open(f, "w", encoding="utf-8").write(s)
    print(f"rewrote {len(files)} files (each with a .premigration.bak)")

    # paths.py builds these from DATA_DIR with `/`, never as literal strings, so the content rewrite
    # above cannot reach it. Repoint the two constants by hand.
    pp = f"{ROOT}/paths.py"
    t = open(pp).read()
    t = t.replace('PXM2025_DIR = _path("PXM2025_DIR", DATA_DIR / "pxm2025")',
                  'PXM2025_DATA = _path("PXM2025_DATA", DATA_DIR / "protenix" / "pxm2025")\n'
                  '# pxm2025 now mirrors foldbench: inputs AND arms under one benchmark directory.\n'
                  '# Before 2026-10-06 its data root WAS experimental_data itself, which is why two\n'
                  '# different pools were both called free/ at different depths.\n'
                  'PXM2025_DIR = _path("PXM2025_DIR", PXM2025_DATA)')
    t = t.replace('FREE_DIR = _path("FREE_DIR", DATA_DIR / "free")',
                  'FREE_DIR = _path("FREE_DIR", PXM2025_DATA / "free")')
    t = t.replace('name="pxm2025", data=DATA_DIR, free=FREE_DIR, jobs=PXM2025_JOBS,',
                  'name="pxm2025", data=PXM2025_DATA, free=FREE_DIR, jobs=PXM2025_JOBS,')
    t = t.replace('"PXM2025_DIR", "PXM2025_JOBS", "PXM2025_CIF", "PXM2025_POOL_CSV",',
                  '"PXM2025_DATA", "PXM2025_DIR", "PXM2025_JOBS", "PXM2025_CIF", "PXM2025_POOL_CSV",')
    if not os.path.exists(pp + ".premigration.bak"):
        shutil.copy2(pp, pp + ".premigration.bak")
    open(pp, "w").write(t)
    print("paths.py: PXM2025_DATA added, PXM2025_DIR and FREE_DIR repointed, Dataset data= updated")

    bad = [d for d in sample_donors() if not os.path.exists(d)]
    print(f"donor-path check: {len(bad)} broken of {len(sample_donors())}")
    for b in bad[:5]:
        print("   BROKEN:", b)


if __name__ == "__main__":
    main()
