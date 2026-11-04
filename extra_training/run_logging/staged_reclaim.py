#!/usr/bin/env python
"""Reclaim disk in stages, verifying at every step. Dry-run by default.

THE THING REVIEW CAUGHT. Moving files to a quarantine directory on the SAME filesystem reclaims
nothing -- the blocks are still charged to the user's quota, because quota is per uid over the
whole device, not per directory. /home is one ext4 volume with usrquota, so a quarantine on it is
a rename and the quota number does not move. Any plan that budgets folding against
"space freed by quarantining" is budgeting against space it does not have.

SO QUARANTINE IS FOR REVERSIBILITY, NOT FOR SPACE, and this tool keeps the two apart:

  stage 1  INVENTORY   record every target file with size, mtime and sha256 of the first 1 MiB,
                       plus the quota reading before anything moves. Written to the journal.
  stage 2  QUARANTINE  rename targets under --quarantine. Reversible, reclaims NOTHING. The
                       journal records the old and new path for every file.
  stage 3  VERIFY      run the caller's verification command. It must exit 0. This is where
                       "the retained manifest still resolves" is proven, with the data still
                       recoverable by rename.
  stage 4  UNLINK      delete the quarantined files. THIS is the step that moves the quota, and
                       the tool reads the quota before and after and refuses to report success
                       unless the delta is within --tolerance of the inventoried bytes.

A missing stage-4 delta means something else holds the blocks -- an open file descriptor on a
deleted inode is the usual cause, and the space returns only when the holder exits. The tool says
so rather than reporting space it cannot see.

Usage:
  python -I run_logging/staged_reclaim.py --targets <file> --journal <path> --stage inventory
  ... --stage quarantine --quarantine training/.reclaim_q
  ... --stage verify --verify-cmd "python -I run_logging/verify_val_manifest.py"
  ... --stage unlink --i-have-verified
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def quota_kib():
    """The user's charged blocks, from quota(1). None if quotas are not enabled."""
    try:
        out = subprocess.run(["quota", "-v"], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for ln in out.splitlines():
        parts = ln.split()
        if len(parts) >= 3 and parts[0].startswith("/dev/"):
            try:
                return int(parts[1])
            except ValueError:
                continue
    return None


def head_digest(p, n=1 << 20):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()[:16]


ap = argparse.ArgumentParser()
ap.add_argument("--targets", required=True, help="one path per line, relative to extra_training")
ap.add_argument("--journal", required=True)
ap.add_argument("--stage", required=True,
                choices=["inventory", "quarantine", "verify", "unlink", "restore"])
ap.add_argument("--quarantine", default="training/.reclaim_quarantine")
ap.add_argument("--verify-cmd", default="")
ap.add_argument("--tolerance", type=float, default=0.05, help="allowed quota-delta shortfall")
ap.add_argument("--i-have-verified", action="store_true", help="required for --stage unlink")
a = ap.parse_args()

JP = os.path.join(ROOT, a.journal)
J = json.load(open(JP)) if os.path.exists(JP) else {"stages": {}, "files": []}


def save():
    os.makedirs(os.path.dirname(JP), exist_ok=True)
    tmp = JP + ".tmp"
    with open(tmp, "w") as f:
        json.dump(J, f, indent=1)
    os.replace(tmp, JP)


if a.stage == "inventory":
    paths = [l.strip() for l in open(a.targets) if l.strip()]
    files, total = [], 0
    for rel in paths:
        p = os.path.join(ROOT, rel)
        if not os.path.isfile(p):
            print(f"  SKIP (not a file): {rel}")
            continue
        sz = os.path.getsize(p)
        files.append({"rel": rel, "bytes": sz, "mtime": os.path.getmtime(p),
                      "head_sha256_1MiB": head_digest(p)})
        total += sz
    J["files"] = files
    J["stages"]["inventory"] = {"at": time.time(), "n": len(files), "bytes": total,
                                "quota_kib_before": quota_kib()}
    save()
    print(f"inventoried {len(files)} file(s), {total/1024**3:.1f} GiB")
    print(f"  quota before: {J['stages']['inventory']['quota_kib_before']} KiB")
    print("  NOTE: nothing has moved. Quarantine does not reclaim quota; only unlink does.")

elif a.stage == "quarantine":
    if not J.get("files"):
        sys.exit("STOP: run --stage inventory first")
    q = os.path.join(ROOT, a.quarantine)
    moved = 0
    for rec in J["files"]:
        src = os.path.join(ROOT, rec["rel"])
        dst = os.path.join(q, rec["rel"])
        if rec.get("quarantined"):
            continue
        if not os.path.isfile(src):
            rec["quarantine_error"] = "source vanished"
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)
        rec["quarantined"] = True
        rec["quarantine_path"] = os.path.relpath(dst, ROOT)
        moved += 1
    J["stages"]["quarantine"] = {"at": time.time(), "moved": moved,
                                 "quota_kib_after": quota_kib()}
    save()
    print(f"quarantined {moved} file(s) under {a.quarantine}")
    print(f"  quota now: {J['stages']['quarantine']['quota_kib_after']} KiB")
    print("  EXPECTED TO BE UNCHANGED -- a same-filesystem rename reclaims no quota.")

elif a.stage == "verify":
    if not a.verify_cmd:
        sys.exit("STOP: --verify-cmd is required; the point of this stage is an external check")
    print(f"running: {a.verify_cmd}")
    rc = subprocess.run(a.verify_cmd, shell=True, cwd=ROOT).returncode
    J["stages"]["verify"] = {"at": time.time(), "cmd": a.verify_cmd, "returncode": rc}
    save()
    print(f"\nverification exit {rc} -- {'OK, unlink may proceed' if rc == 0 else 'FAILED, do not unlink'}")
    sys.exit(rc)

elif a.stage == "unlink":
    if not a.i_have_verified:
        sys.exit("STOP: --i-have-verified is required. This is the irreversible stage.")
    v = J["stages"].get("verify")
    if not v or v.get("returncode") != 0:
        sys.exit("STOP: no passing verify stage in the journal. Run --stage verify first.")
    before = quota_kib()
    freed = 0
    for rec in J["files"]:
        if not rec.get("quarantined"):
            continue
        p = os.path.join(ROOT, rec["quarantine_path"])
        if os.path.isfile(p):
            os.unlink(p)
            freed += rec["bytes"]
            rec["unlinked"] = True
    after = quota_kib()
    delta = (before - after) * 1024 if (before is not None and after is not None) else None
    J["stages"]["unlink"] = {"at": time.time(), "bytes_expected": freed,
                             "quota_kib_before": before, "quota_kib_after": after,
                             "quota_bytes_delta": delta}
    save()
    print(f"unlinked {freed/1024**3:.1f} GiB of inventoried data")
    if delta is None:
        print("  quota unreadable; cannot confirm the space was actually released")
        sys.exit(2)
    print(f"  quota delta: {delta/1024**3:.1f} GiB")
    short = freed - delta
    if short > freed * a.tolerance:
        print(f"  SHORTFALL {short/1024**3:.1f} GiB -- the blocks are still charged. Usual cause")
        print("  is an open descriptor on a deleted inode; the space returns when the holder")
        print("  exits. DO NOT budget folding against this space until the delta matches.")
        sys.exit(3)
    print("  delta matches the inventory; the space is genuinely available.")

elif a.stage == "restore":
    n = 0
    for rec in J["files"]:
        if rec.get("unlinked"):
            print(f"  CANNOT RESTORE (already unlinked): {rec['rel']}")
            continue
        if not rec.get("quarantined"):
            continue
        src = os.path.join(ROOT, rec["quarantine_path"])
        dst = os.path.join(ROOT, rec["rel"])
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            rec["quarantined"] = False
            n += 1
    J["stages"]["restore"] = {"at": time.time(), "restored": n}
    save()
    print(f"restored {n} file(s)")
