#!/usr/bin/env python
"""WRITE PHASE: turn every pose a sweep will use into a template file, once, before any GPU starts.

WHY THIS EXISTS -- the bug it makes impossible
----------------------------------------------
Protenix reads templates as mmCIF files named by a 4-character id (it treats them as PDB ids). To
self-template we therefore have to write our poses into a template directory and let the fold read
them back. Doing that DURING the run cost us a day:

  * The id was keyed on (interface, seed) plus the pose's index within the arm's own pose list. A
    per-pose arm has one pose, so its index was always 0 -- every per-pose arm of a seed wrote to the
    SAME file. Two GPUs working adjacent arms overwrote each other, and because both poses come from
    the same complex the sequence still matched, so the featuriser accepted the wrong pose silently.
    35 of 40 arms in one sweep overlapped that way.
  * The shared directory holds >10k of these 4-char ids. 36^4 is only 1.68M, so by the birthday bound
    a collision there is near-certain rather than unlikely. No hashing scheme fixes that.
  * Protenix caches the release-date file in a module-level dict keyed by path and populates it once
    per process (template_utils.py:838). Appending ids to that file mid-run means later arms write ids
    the cached snapshot cannot see, and the prefilter rejects them as "date unknown"
    (template_utils.py:369) -- a silent zero-template fold.

This script removes all three by construction:

  1. An id identifies A POSE FILE: key = its absolute path plus its md5. Two arms that want the same
     pose share one file legitimately, because they want identical content -- while a re-fold of the
     same seed, whose poses differ, gets fresh ids instead of silently reusing the old ones.
  2. Everything is written into a PRIVATE directory for this sweep, so the id namespace holds only
     these few ids and cannot collide with anything else on the machine.
  3. All writing finishes before any worker starts, so the release-date file is complete and static
     when Protenix caches it, and nothing is ever mutated underneath a running fold.

WHAT IT PRODUCES
----------------
    <ns>/mmcif/<tid>.cif        one file per pose
    <ns>/release_dates.json     every tid registered, written once
    <ns>/jobs/<iface>_<tag>/    a job.json + a3m per ARM, pointing at the tids it needs
    <ns>/index.tsv             provenance: arm -> tids -> source pose paths + md5

The run phase then only READS. Point the driver at <ns> and it cannot write a template even by mistake.

Usage:
    write_templates.py --manifest <arms.tsv> --ns <dir>
    write_templates.py --manifest <arms.tsv> --ns <dir> --verify   # re-check an existing namespace
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

DC = "/home/ebrahim/tts_abag/diffusion_commit"
PY = "/home/ebrahim/Protenix/.venv/bin/python"
JOBS = str(paths.PXM2025_JOBS)
B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def pose_tid(path):
    """4-char id for ONE POSE FILE, derived from its path AND its contents.

    Keyed on the file, not on labels, which handles the cases labels cannot:
      * two arms referencing the same pose get the same id -- correct, they want identical content
      * different seeds of an interface get different ids -- the paths differ
      * THE SAME SEED RE-FOLDED gets a different id -- folds are not reproducible, so run A's pose 3
        and run B's pose 3 are different structures; keying on (iface, seed, index) would give them
        one id and silently conflate them
      * the same pose copied to a second location still maps to one id, because the md5 matches
    """
    ap = os.path.abspath(path)
    key = f"{ap}:{hashlib.md5(open(ap, 'rb').read()).hexdigest()}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16) % (36 ** 4)
    return "".join(B36[(h // 36 ** k) % 36] for k in (3, 2, 1, 0))


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--ns", required=True, help="private namespace dir for this sweep")
    ap.add_argument("--verify", action="store_true", help="check an existing namespace, write nothing")
    A = ap.parse_args()

    arms = []
    for line in open(A.manifest):
        if not line.strip():
            continue
        iface, pdb, tag, seed, poses = line.rstrip("\n").split("\t")
        arms.append((iface, pdb, tag, seed, poses.split()))

    # ---- which pose files does this sweep touch, and what id does each get? -------------------
    # Content-addressed: the id comes from the file, so nothing about an arm's shape can influence it.
    # The old scheme keyed on the pose's position in an ARM's list, and a per-pose arm has one pose, so
    # every such arm got index 0 and they all wrote the same file.
    pose_of = {}
    for iface, pdb, tag, seed, poses in arms:
        for p in poses:
            tid = pose_tid(p)
            prev = pose_of.get(tid)
            if prev and prev[0] != os.path.abspath(p):
                # 36^4 is 1.68M, so a real hash clash is possible if a sweep ever gets huge; refuse
                # rather than write one file under an id another pose also claims.
                sys.exit(f"ID CLASH: {tid} wanted by two different files:\n  {prev[0]}\n  {p}")
            pose_of[tid] = (os.path.abspath(p), iface, seed, p)

    mm = f"{A.ns}/mmcif"
    rd = f"{A.ns}/release_dates.json"
    if A.verify:
        bad = [t for t in pose_of if not os.path.exists(f"{mm}/{t}.cif")]
        reg = json.load(open(rd)) if os.path.exists(rd) else {}
        missing_reg = [t for t in pose_of if t not in reg]
        print(f"  namespace {A.ns}")
        print(f"  poses needed {len(pose_of)} | cif present {len(pose_of)-len(bad)} | registered "
              f"{len(pose_of)-len(missing_reg)}")
        if bad:
            print(f"  MISSING CIFS: {bad[:6]}")
        if missing_reg:
            print(f"  UNREGISTERED: {missing_reg[:6]}")
        return 1 if (bad or missing_reg) else 0

    os.makedirs(mm, exist_ok=True)
    os.makedirs(f"{A.ns}/jobs", exist_ok=True)
    # release dates first and complete: Protenix caches this file once per process, so it must be
    # final before any fold runs.
    json.dump({t: {"release_date": "2000-01-01"} for t in sorted(pose_of)}, open(rd, "w"))
    print(f"  registered {len(pose_of)} template ids in {rd}")

    # ---- one build per ARM, run SERIALLY here so nothing races ---------------------------------
    env = dict(os.environ, SELFTMPL_MMCIF_DIR=mm, SELFTMPL_RELEASE_DATES=rd, SELFTMPL_JOBS=JOBS)
    index = []
    for iface, pdb, tag, seed, poses in arms:
        out = f"{A.ns}/jobs/{iface}_{tag}"
        tids = [pose_tid(p) for p in poses]
        r = subprocess.run([PY, "/home/ebrahim/tts_abag/diffusion_commit_clean/build_selftmpl_general.py", "--pdb", pdb, "--out", out,
                            "--poses", *[os.path.abspath(p) for p in poses], "--tids", *tids],
                           capture_output=True, text=True, env=env, cwd=DC)
        if not os.path.exists(f"{out}/job.json"):
            print(f"  FAILED {iface} {tag}: {(r.stderr or '')[-300:]}")
            return 1
        for p, t in zip(poses, tids):
            index.append(f"{iface}\t{tag}\t{seed}\t{t}\t{md5(p)}\t{os.path.abspath(p)}")
        print(f"  built {iface} {tag}: {len(tids)} template(s) {tids}")

    with open(f"{A.ns}/index.tsv", "w") as fh:
        fh.write("iface\ttag\tseed\ttid\tpose_md5\tsource_pose\n")
        fh.write("\n".join(index) + "\n")
    print(f"\n  WRITE PHASE COMPLETE: {len(pose_of)} pose files, {len(arms)} arm jobs")
    print(f"  namespace: {A.ns}   provenance: {A.ns}/index.tsv")
    print("  the run phase only reads this directory")
    return 0


if __name__ == "__main__":
    sys.exit(main())
sys.path.insert(0, "/home/ebrahim/tts_abag")

import paths  # noqa: E402  (needs the path insert above)
